"""Causal latent suffix replay using the existing TRT Target and prefill KV."""
import torch
from torch.nn import functional as F


class LatentPrefixReplay:
    def __init__(self,target,prefill,norm,steps=5):
        self.target,self.prefill,self.norm=target,prefill,norm
        b=target.inputs['x'][0][0];self.batch=b;self.capacity=target.inputs['k_cache_in_0'][0][2]
        self.steps=steps;self.device=prefill.inputs['x'].device
        self.target_steps=steps
        self.prefix_lengths=torch.full((b,),38,device=self.device,dtype=torch.long)
        self.suffix_lengths=torch.full((b,),steps*8,device=self.device,dtype=torch.long)
        self.suffix=torch.zeros(b,steps*8,1280,device=self.device)
        self.cache=torch.zeros(24,2,b,20,self.capacity,64,device=self.device,dtype=torch.bfloat16)
        self.append=torch.empty(24,2,b,20,8,64,device=self.device,dtype=torch.bfloat16)
        # Restore output bindings after capture; cached graph nodes retain these
        # private append buffers, independently of AR's output bindings.
        self.output=torch.empty(b,1+steps*8,1280,device=self.device)
        self.positions=torch.arange(self.capacity,device=self.device)
        self.query=torch.arange(8,device=self.device)
        self.causal=(self.query[:,None]>=self.query[None,:])[None,None]
        self.graph=None

    def body(self):
        self.cache.zero_()
        self.cache[:,:,:,:,:self.prefill.extent].copy_(self.prefill.outputs['packed_kv'])
        keep=torch.zeros(self.batch,self.capacity,device=self.device,dtype=torch.bool)
        keep[:,:self.prefill.extent].copy_(self.prefill.inputs['keep'].bool())
        bos=self.prefill.outputs['final'].gather(1,(self.prefix_lengths-1)[:,None,None].expand(-1,1,1280))
        self.output[:,:1].copy_(self.norm(bos))
        for step in range(self.steps):
            logical=self.prefix_lengths[:,None]+step*8+self.query[None]
            bindings={'x':self.suffix[:,step*8:(step+1)*8].contiguous(),
                      'mask':torch.cat((keep[:,None,None].expand(-1,1,8,-1),self.causal.expand(self.batch,-1,-1,-1)),-1)}
            for layer in range(24):
                bindings[f'k_cache_in_{layer}']=self.cache[layer,0]
                bindings[f'v_cache_in_{layer}']=self.cache[layer,1]
            result=self.target(bindings)
            self.output[:,1+step*8:1+(step+1)*8].copy_(self.norm(result['final']))
            valid=((step*8+self.query[None]<self.suffix_lengths[:,None])&(logical<self.capacity))
            # Modulo sends unused out-of-range writes to distinct old prefix
            # cells, preserving them; clamp would alias a valid final cell.
            indices=(logical%self.capacity)[None,None,:,None,:,None].expand(24,2,-1,20,-1,64)
            old=self.cache.gather(4,indices)
            self.cache.scatter_(4,indices,torch.where(valid[None,None,:,None,:,None],self.append,old))
            ki=logical%self.capacity
            keep.scatter_(1,ki,torch.where(valid,torch.ones_like(valid),keep.gather(1,ki)))
        return self.output

    def capture(self):
        old=dict(self.target.outputs)
        try:
            for layer in range(24):
                for plane,letter in enumerate(('k','v')):
                    self.target.outputs[f'{letter}_append_{layer}']=self.append[layer,plane]
            for _ in range(2):self.body()
            torch.cuda.synchronize()
            self.graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):self.body()
        finally:self.target.outputs=old

    def run_embedded(self,suffix,prefix_lengths,suffix_lengths):
        if suffix.shape!=self.suffix.shape:raise ValueError('Exact padded suffix shape required')
        self.suffix.copy_(suffix);self.prefix_lengths.copy_(prefix_lengths);self.suffix_lengths.copy_(suffix_lengths)
        self.graph.replay()
        return self.output
