"""Seven proposed speech tokens; request-local Philox sampling; eager only."""
import torch

class Proposal:
    def __init__(self,backbone):
        self.backbone=backbone;self.model=backbone.model;self.calls=0
        model=self.model;param=next(model.parameters())
        prior=torch.backends.cuda.matmul.allow_tf32;torch.backends.cuda.matmul.allow_tf32=False
        try:self.weight,self.table,_=model._prepare_optimized_linear_rnn(param.new_zeros(1,7,model.interface_size))
        finally:torch.backends.cuda.matmul.allow_tf32=prior
        self.out_weight=model._effective_markov_output_weight()
        self.graphs={};self.graph_hits=0
        self.state_linear=None;self.output_linear=None;self.hidden_linear=None
        self.batched_rng=False
        self.batched_rng_min_batch=1
        self.batch_generator=torch.Generator(device=param.device).manual_seed(0x5EED5EED)
    def prepare_precision(self,mode,plans=None):
        if self.graphs:raise RuntimeError('Set RNN precision before capture')
        from types import SimpleNamespace
        from acc_infer_clear.ops.eager.matrix import MatrixLinear
        from acc_infer_clear.ops.planning.planner import DeviceCaps
        caps=DeviceCaps.current();plans={} if plans is None else plans;m=self.model
        self.state_linear=MatrixLinear(SimpleNamespace(weight=self.weight,bias=None),mode,caps,plans,extents=(8,))
        self.output_linear=MatrixLinear(SimpleNamespace(weight=self.out_weight,bias=None),mode,caps,plans,extents=(8,))
        hidden_weight=m.markov_rnn.weight[:,m.markov_state_size+m.markov_rank:]
        self.hidden_linear=MatrixLinear(SimpleNamespace(weight=hidden_weight,bias=m.markov_rnn.bias),mode,caps,plans,extents=(8,64))
        return dict(mode=mode,matrices=['state','hidden','output'],folded_token_table='FP32',nonlinear_math='FP32',online_tuning=False,
                    cache=plans.save() if hasattr(plans,'save') else None)
    def math(self,base,terms,noise,previous):
        model=self.model
        state=base.new_zeros(base.shape[0],model.markov_state_size);tokens=[];probabilities=[];logits=[]
        for j in range(7):
            if self.state_linear is not None:
                raw=self.state_linear(state)+torch.nn.functional.embedding(previous.long(),self.table)+terms[:,j]
                gate,candidate,output=raw.split((model.markov_state_size,model.markov_state_size,model.markov_rank),dim=-1)
                gate=gate.sigmoid();state=gate*state+(1-gate)*candidate.tanh()
                delta=self.output_linear(output.tanh())
            else:state,delta=model._optimized_linear_rnn_step(state,previous,self.weight,self.table,terms[:,j],self.out_weight)
            logit=base[:,j]+delta;p=torch.softmax(logit/.8,dim=-1);previous=(p/noise[:,j]).argmax(-1)
            tokens.append(previous);probabilities.append(p);logits.append(logit)
        return torch.stack(tokens,1),torch.stack(probabilities,1),torch.stack(logits,1)
    def captured_math(self,hidden,base,noise,previous):
        if self.hidden_linear is not None:terms=self.hidden_linear(hidden)
        else:_,_,terms=self.model._prepare_optimized_linear_rnn(hidden)
        return self.math(base,terms,noise,previous)
    def prepare_graphs(self,max_batch):
        from acc_infer_clear.runtime.graphs import capture
        from acc_infer_clear.runtime.graph_policy import batches
        if self.graphs:raise RuntimeError('Proposal already captured')
        prior=torch.backends.cuda.matmul.allow_tf32;torch.backends.cuda.matmul.allow_tf32=False
        try:
            for b in batches(max_batch):
                hidden=self.weight.new_zeros(b,7,self.model.interface_size);base=self.weight.new_zeros(b,7,self.model.vocab_size)
                noise=torch.ones_like(base);previous=torch.zeros(b,device=base.device,dtype=torch.long)
                self.graphs[b]=capture(self.captured_math,(hidden,base,noise,previous))
        finally:torch.backends.cuda.matmul.allow_tf32=prior
        return dict(batches=list(self.graphs),online_capture=False)
    def __call__(self,jobs,tasks):
        model=self.model;rows=self.backbone([{k:j[k] for k in ('cache','anchor_token','first_position')} for j in jobs])
        hidden=torch.cat([r[0] for r in rows]);base=torch.cat([r[1] for r in rows])
        previous=torch.cat([j['anchor_token'].reshape(1) for j in jobs])
        use_batched_rng=self.batched_rng and len(jobs)>=self.batched_rng_min_batch
        if use_batched_rng:
            noise=None
        else:
            noises=[]
            for task in tasks:
                gen=torch.Generator(device=base.device);gen.set_state(task.state['cuda_rng'])
                noises.append(torch.cat([torch.empty_like(base[0,j:j+1]).exponential_(generator=gen) for j in range(7)]))
                task.state['cuda_rng']=gen.get_state()
            noise=torch.stack(noises)
        prior=torch.backends.cuda.matmul.allow_tf32;torch.backends.cuda.matmul.allow_tf32=False
        try:
            if len(jobs) in self.graphs:
                graph=self.graphs[len(jobs)]
                if use_batched_rng:
                    gh,gb,gn,gp=graph.inputs;gh.copy_(hidden);gb.copy_(base);gp.copy_(previous);gn.exponential_(generator=self.batch_generator);graph.graph.replay();tt,pp,ll=graph.outputs
                else:tt,pp,ll=graph(hidden,base,noise,previous)
                self.graph_hits+=1
            else:
                if self.hidden_linear is not None:terms=self.hidden_linear(hidden)
                else:_,_,terms=model._prepare_optimized_linear_rnn(hidden)
                if noise is None:noise=torch.empty_like(base).exponential_(generator=self.batch_generator)
                tt,pp,ll=self.math(base,terms,noise,previous)
        finally:torch.backends.cuda.matmul.allow_tf32=prior
        self.calls+=1
        return [(tt[i:i+1].clone(),pp[i:i+1].clone(),ll[i:i+1].clone()) for i in range(len(jobs))]
