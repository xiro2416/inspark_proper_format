"""CUDA Graphs for exact-length acoustic groups; only batch rows are padded."""
import torch
from inspark_infer.runtime.unified_conditions import _validate_modules


class ConditionGraph:
    def __init__(self,tts,batch,count):
        self.batch,self.count=batch,count;self.frames=int(count*1.72)
        self.q=tts.semantic_codec.quantizer;self.proj=tts.s2mel.models['gpt_layer'];self.reg=tts.s2mel.models['length_regulator']
        device=next(self.proj.parameters()).device
        self.codes=torch.zeros(batch,count,device=device,dtype=torch.long)
        self.latents=torch.zeros(batch,count,1280,device=device)
        self.lengths=torch.full((batch,),self.frames,device=device,dtype=torch.long)
        # Graphs replay concurrently on condition streams. A shared default
        # capture stream also shares cuBLAS per-stream scratch across captures;
        # that scratch is not made private by each Graph's allocator pool.
        # Give every Graph its own library stream/workspace during capture.
        self.capture_stream=torch.cuda.Stream(device=device)
        parent=torch.cuda.current_stream(device)
        self.capture_stream.wait_stream(parent)
        def body():
            semantic=self.q.vq2emb(self.codes.unsqueeze(0)).transpose(1,2)+self.proj(self.latents)
            self.output=self.reg(semantic,ylens=self.lengths,n_quantizers=3,f0=None,length_hint=self.frames)[0]
        with torch.cuda.stream(self.capture_stream):
            for _ in range(2):body()
            self.capture_stream.synchronize();self.graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph,stream=self.capture_stream):body()
        parent.wait_stream(self.capture_stream)

    def run(self,codes,latents):
        actual=len(codes)
        torch.cat(codes,0,out=self.codes[:actual]);torch.cat(latents,0,out=self.latents[:actual])
        self.codes[actual:].zero_();self.latents[actual:].zero_()
        self.graph.replay();return self.output[:actual]


class ConditionGraphBank:
    def __init__(self,tts,max_batch=64,max_count=40,exact_small_batches=False):
        _validate_modules(tts.semantic_codec.quantizer,tts.s2mel.models['gpt_layer'],tts.s2mel.models['length_regulator'])
        self.graphs={};self.hits=0;self.misses=0
        choices=(*range(1,17),32,64) if exact_small_batches else (1,4,8,16,32,64)
        self.buckets=[b for b in choices if b<=max_batch]
        for count in range(1,max_count+1):
            for b in self.buckets:self.graphs[(b,count)]=ConditionGraph(tts,b,count)
            print(f'Condition Graph preparation: count={count}/{max_count}, graphs={len(self.graphs)}',flush=True)

    def run(self,codes,latents,count):
        b=next((b for b in self.buckets if b>=len(codes)),None)
        graph=self.graphs.get((b,count))
        if graph is None:self.misses+=1;return None
        self.hits+=1;return graph.run(codes,latents)
