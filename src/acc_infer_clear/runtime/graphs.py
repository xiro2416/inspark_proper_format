"""Explicit deployment-time capture. No lazy capture or online shape learning."""
from dataclasses import dataclass
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
    def prepare(self,engine):
        if engine.sessions:raise RuntimeError('Capture only before admitting requests')
        if self.sealed:raise RuntimeError('Already prepared; create a new bank explicitly')
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
                self.cfm[key]=capture(engine.student,(x,prompt,lengths,style,mu,mask))
                if batch not in self.vocoder:
                    self.vocoder[batch]=capture(engine.vocoder,(x[:,:,:52].clone(),))
        self.sealed=True
    def run_cfm(self,args,plen):
        self.hits['cfm']+=1
        return self.cfm[(args[0].shape[0],plen)](*args)
    def run_vocoder(self,mel):
        self.hits['vocoder']+=1
        return self.vocoder[mel.shape[0]](mel)
    def stats(self):
        return dict(cfm_keys=[list(k) for k in self.cfm],vocoder_batches=list(self.vocoder),
                    hits=dict(self.hits),sealed=self.sealed,online_capture=False,tail_graphs=False)
