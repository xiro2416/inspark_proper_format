"""Explicit deployment-time capture. No lazy capture or online shape learning."""
from dataclasses import dataclass
from copy import deepcopy
import json
import torch
from .graph_policy import batches

@dataclass
class Captured:
    inputs:tuple
    outputs:object
    graph:object
    def __call__(self,*args):
        if len(args)!=len(self.inputs):raise ValueError('Graph argument count changed')
        for static,value in zip(self.inputs,args):
            if static.shape!=value.shape or static.dtype!=value.dtype or static.device!=value.device:
                raise ValueError('Graph input signature changed')
            static.copy_(value)
        self.graph.replay()
        return self.outputs

def capture(fn,args):
    static=tuple(x.clone() for x in args)
    for _ in range(3):fn(*static)
    torch.cuda.current_stream().synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):outputs=fn(*static)
    return Captured(static,outputs,graph)

class HeadGraphs:
    def __init__(self):
        self.cfm={};self.vocoder={};self.hits={'cfm':0,'vocoder':0};self.sealed=False
        self.routes={'cfm':{},'vocoder':{}}
        self.route_counts={}
        self.wrapper_prepare_counts={}
        self._direct={}
    @staticmethod
    def _describe_route(component,fn,args):
        describe=getattr(fn,'describe_route',None)
        if describe is None:describe=getattr(fn,'route_for_signature',None)
        if describe is not None:
            route=deepcopy(describe(*args))
        else:
            route=dict(backend='eager',kind='eager',reason=None,plan=None,sha256=None,plugins=[])
        route.setdefault('component',component)
        route.setdefault('batch',int(args[0].shape[0]))
        route.setdefault('frames',int(args[0].shape[-1]))
        route.setdefault('fallback',describe is not None and route.get('backend')=='eager')
        return route
    def _record(self,route,operation):
        # A route is copied before capture; later wrapper changes cannot relabel
        # an already captured eager graph as a TensorRT graph.
        key=json.dumps(route,sort_keys=True)
        if key not in self.route_counts:
            self.route_counts[key]=dict(route=deepcopy(route),prepare=0,direct=0,replay=0,fallback=0)
        row=self.route_counts[key]
        row[operation]+=1
        if operation!='prepare' and route.get('fallback',False):row['fallback']+=1
    @staticmethod
    def _wrapper_counts(fn):
        return dict(calls=int(getattr(fn,'calls',0)),fallbacks=int(getattr(fn,'fallbacks',0)))
    def prepare(self,engine):
        if engine.sessions:raise RuntimeError('Capture only before admitting requests')
        if self.sealed:raise RuntimeError('Already prepared; create a new bank explicitly')
        self._direct={'cfm':engine.student,'vocoder':engine.vocoder}
        for voice in list(engine.model.bank.entries):
            v=engine.model.bank.get(voice)['values']
            plen=v['voice.cache_mel'].shape[-1]
            for batch in batches(engine.config['max_batch']):
                key=(batch,plen)
                if key in self.cfm:continue
                mu=torch.cat((v['voice.cache_s2mel_prompt'],v['voice.cache_s2mel_prompt'].new_zeros(1,52,v['voice.cache_s2mel_prompt'].shape[-1])),1)
                assert mu.shape[1]==plen+52,'Reference feature lengths differ'
                mu=mu.repeat(batch,1,1)
                x=mu.new_zeros(batch,80,plen+52)
                prompt=x.clone();prompt[:,:,:plen]=v['voice.cache_mel']
                lengths=torch.full((batch,),plen+52,device=x.device,dtype=torch.long)
                style=v['voice.cache_s2mel_style'].repeat(batch,1)
                mask=(torch.arange(plen+52,device=x.device)[None,None]<plen).expand(batch,1,-1).clone()
                args=(x,prompt,lengths,style,mu,mask)
                route=self._describe_route('cfm',engine.student,args)
                self.cfm[key]=capture(engine.student,args)
                self.routes['cfm'][key]=route
                self._record(route,'prepare')
                if batch not in self.vocoder:
                    args=(x[:,:,:52].clone(),)
                    route=self._describe_route('vocoder',engine.vocoder,args)
                    self.vocoder[batch]=capture(engine.vocoder,args)
                    self.routes['vocoder'][batch]=route
                    self._record(route,'prepare')
        self.wrapper_prepare_counts={name:self._wrapper_counts(fn) for name,fn in self._direct.items()}
        self.sealed=True
    def _run(self,component,key,args):
        bank=self.cfm if component=='cfm' else self.vocoder
        graph=bank.get(key)
        # A known batch can still have another extent/dtype. Do not replay its
        # static bindings unless the complete tensor signature matches.
        compatible=graph is not None and len(args)==len(graph.inputs) and all(
            value.shape==static.shape and value.dtype==static.dtype and value.device==static.device
            for static,value in zip(graph.inputs,args))
        if compatible:
            result=graph(*args)
            self.hits[component]+=1
            self._record(self.routes[component][key],'replay')
            return result
        if component not in self._direct:raise RuntimeError('Head graphs have not been prepared')
        fn=self._direct[component]
        route=self._describe_route(component,fn,args)
        route['graph_fallback']='missing_graph' if graph is None else 'input_signature'
        route['fallback']=True
        result=fn(*args)
        self._record(route,'direct')
        return result
    def run_cfm(self,args,plen):
        return self._run('cfm',(args[0].shape[0],plen),args)
    def run_vocoder(self,mel):
        return self._run('vocoder',mel.shape[0],(mel,))
    def stats(self):
        routes={name:[dict(key=list(key) if isinstance(key,tuple) else key,route=deepcopy(route))
                      for key,route in rows.items()] for name,rows in self.routes.items()}
        return dict(cfm_keys=[list(k) for k in self.cfm],vocoder_batches=list(self.vocoder),
                    hits=dict(self.hits),sealed=self.sealed,online_capture=False,tail_graphs=False,
                    captured_routes=routes,route_counts=deepcopy(list(self.route_counts.values())),
                    wrapper_prepare_counts=deepcopy(self.wrapper_prepare_counts),
                    counter_semantics=dict(prepare='successful graph captures, excluding warmup calls',
                        replay='successful graph replays, classified by the frozen capture route',
                        direct='successful uncaptured calls through this graph bank',
                        fallback='runtime calls using eager backend fallback or an unavailable graph',
                        wrapper_prepare_counts='cumulative wrapper calls/fallbacks at preparation completion'))
