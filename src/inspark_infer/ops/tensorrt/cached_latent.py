"""Dedicated one-pass latent suffix TRT; readonly prefill cache bindings."""
import torch


class OnePassLatentPrefixReplay:
    def __init__(self,engine,prefill,norm):
        self.engine,self.prefill,self.norm=engine,prefill,norm
        self.batch=prefill.batch;self.steps=5;self.capacity=80;self.device=prefill.device
        self.target_steps=1
        b=self.batch
        expected={'x':((b,40,1280),torch.float32),'prefix_keep':((b,48),torch.int64),
                  'past':((24,2,b,20,48,64),torch.float32)}
        if engine.inputs!=expected or tuple(engine.outputs['latent_suffix'].shape)!=(b,40,1280):
            raise ValueError('Dedicated latent cache engine binding contract differs')
        if (engine.plan['kind']!='latent_cached_suffix' or engine.plan['frames']!=40
                or engine.plan.get('plugins')!=[] or engine.plan.get('tf32') is not False):
            raise ValueError('Unexpected cached-latent implementation')
        a=engine.plan['quantization_recipe'];r=prefill.plan['quantization_recipe']
        if a['scheme']!=r['scheme'] or a['calibration']['sha256']!=r['calibration']['sha256']:
            raise ValueError('Cached-latent calibration differs from prefill')
        if a.get('role_specs_sha256')!=r.get('role_specs_sha256'):
            raise ValueError('Cached-latent precision role manifest differs from prefill')
        if engine.plan.get('precision')!='fp8_static_qdq_fp32_attention_and_interfaces':
            raise ValueError('Cached-latent attention/interface precision differs from the approved graph')
        self.suffix=torch.zeros(b,40,1280,device=self.device)
        self.prefix_lengths=torch.full((b,),38,device=self.device,dtype=torch.long)
        self.output=torch.empty(b,41,1280,device=self.device);self.graph=None

    def body(self):
        bos=self.prefill.outputs['final'].gather(1,(self.prefix_lengths-1)[:,None,None].expand(-1,1,1280))
        self.output[:,:1].copy_(self.norm(bos))
        result=self.engine({'x':self.suffix,'prefix_keep':self.prefill.inputs['keep'],
                            'past':self.prefill.outputs['packed_kv']})
        self.output[:,1:].copy_(result['latent_suffix'])

    def capture(self):
        for _ in range(2):self.body()
        torch.cuda.synchronize();self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):self.body()

    def run_embedded(self,suffix,prefix_lengths,suffix_lengths):
        if suffix.shape!=self.suffix.shape:raise ValueError('Exact padded suffix shape required')
        self.suffix.copy_(suffix);self.prefix_lengths.copy_(prefix_lengths)
        self.graph.replay();return self.output
