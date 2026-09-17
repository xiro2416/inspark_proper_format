"""Independent model processes on ONE GPU; explicit incremental input messages."""
import multiprocessing as mp
import os,time,traceback,sys
from contextlib import redirect_stdout

def _worker(gpu,config,pipe):
    from .device import select_gpu
    select_gpu(gpu)
    from acc_infer_clear.streaming.engine import Engine
    engine=None
    try:
        with redirect_stdout(sys.stderr):engine=Engine(config)
        pipe.send(dict(ok=True,pid=os.getpid(),identity=engine.student.identity,ready_heads=False,ready_tails=False))
        while True:
            cmd,args,kwargs=pipe.recv()
            if cmd=='close':break
            try:
                if cmd=='stats':result=dict(sessions=len(engine.sessions),draft_calls=engine.rt.proposal.calls,target_calls=engine.rt.target.calls)
                elif cmd=='result':
                    s=engine.sessions[args[0]]
                    result=dict(id=args[0],text=s['text'],parts=s['parts'],codes=s['codes'],complete=s['complete'],error=s['error'],chunks=s['chunks'],arrival=s['arrival'])
                elif cmd in ('run_ready','tick'):
                    getattr(engine,cmd)(on_chunk=lambda event:pipe.send(dict(ok=True,kind='chunk',event=event)))
                    result=[]
                elif cmd=='release':engine.release(*args);result=None
                elif cmd=='cancel':engine.cancel(*args);result=None
                else:result=getattr(engine,cmd)(*args,**kwargs)
                ready=engine.ready()
                pipe.send(dict(ok=True,result=result,pid=os.getpid(),ready_heads=any(not s['chunks'] for s in ready),ready_tails=any(s['chunks'] for s in ready)))
            except Exception:pipe.send(dict(ok=False,error=traceback.format_exc()))
    except BaseException:
        try:pipe.send(dict(ok=False,error=traceback.format_exc()))
        except Exception:pass
    finally:
        if engine:engine.close()

class Pool:
    def __init__(self,config,gpu=4,workers=1):
        if workers not in (1,2,4):raise ValueError('Worker count must be1/2/4')
        self.pipes=[];self.processes=[];self.states={};self.owners={};self.closed=False
        ctx=mp.get_context('spawn')
        try:
            for w in range(workers):
                a,b=ctx.Pipe();p=ctx.Process(target=_worker,args=(gpu,config,b));p.start()
                self.pipes.append(a);self.processes.append(p);self._receive(w)
        except BaseException:self.close();raise
    def _receive(self,w):
        if not self.pipes[w].poll(600):raise TimeoutError('Worker did not respond within600 seconds')
        msg=self.pipes[w].recv()
        if not msg['ok']:raise RuntimeError(msg['error'])
        self.states[w]=msg;return msg.get('result')
    def _call(self,w,cmd,*args,**kwargs):
        if self.closed:raise RuntimeError('Pool closed')
        self.pipes[w].send((cmd,args,kwargs));return self._receive(w)
    def prepare_reference(self,voice_id,path):
        return [self._call(w,'prepare_reference',voice_id,path) for w in range(len(self.pipes))]
    def prepare_deployment(self,plan):
        # Sequential preparation; there are no admitted requests at this point.
        return [self._call(w,'prepare_deployment',plan) for w in range(len(self.pipes))]
    def create_session(self,request_id,voice_id,seed=0,emotion=None,arrival=None):
        if request_id in self.owners:raise ValueError('Duplicate request id')
        w=len(self.owners)%len(self.pipes)
        self._call(w,'create_session',request_id,voice_id,seed,emotion,arrival)
        self.owners[request_id]=w
    def push_text(self,request_id,delta):return self._call(self.owners[request_id],'push_text',request_id,delta)
    def finish_input(self,request_id):return self._call(self.owners[request_id],'finish_input',request_id)
    def run_ready(self,on_chunk=None):
        return self._advance('run_ready',on_chunk)
    def tick(self,on_chunk=None):
        return self._advance('tick',on_chunk)
    def _advance(self,command,on_chunk):
        heads=[w for w,s in self.states.items() if s['ready_heads']]
        active=heads or [w for w,s in self.states.items() if s['ready_tails']]
        for w in active:self.pipes[w].send((command,(),{}))
        # Receive whichever worker becomes ready first, not in worker index order.
        from multiprocessing.connection import wait
        pending={self.pipes[w]:w for w in active};events=[]
        while pending:
            ready=wait(list(pending),timeout=600)
            if not ready:raise TimeoutError('Inference workers stalled')
            for pipe in ready:
                w=pending[pipe];msg=pipe.recv();received=time.perf_counter()
                if not msg['ok']:raise RuntimeError(msg['error'])
                if msg.get('kind')=='chunk':
                    rows=[msg['event']]
                else:
                    pending.pop(pipe);self.states[w]=msg;rows=msg.get('result') or []
                for row in rows:
                    row.update(worker=w,received=received)
                    if on_chunk is not None:on_chunk(row)
                events.extend(rows)
        return events
    def result(self,request_id):return self._call(self.owners[request_id],'result',request_id)
    def release(self,request_id):
        self._call(self.owners[request_id],'release',request_id);self.owners.pop(request_id)
    def cancel(self,request_id):
        self._call(self.owners[request_id],'cancel',request_id);self.owners.pop(request_id)
    def close(self):
        if self.closed:return
        for pipe,p in zip(self.pipes,self.processes):
            if p.is_alive():
                try:pipe.send(('close',(),{}))
                except (BrokenPipeError,OSError):pass
        for p in self.processes:
            p.join(timeout=10)
            if p.is_alive():p.terminate();p.join()
        for pipe in self.pipes:pipe.close()
        self.closed=True
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
