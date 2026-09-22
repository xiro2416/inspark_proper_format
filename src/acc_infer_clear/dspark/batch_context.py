"""Pack independent committed Target rows for shared Draft context projection.

Only pointwise projections are packed. No attention crosses request boundaries;
each request keeps its own absolute positions and independently owned KV tensors.
"""
import torch

class BatchedContextAppend:

    def __init__(self, model):
        self.model = model
        self.calls = 0
        self.rows = 0
        assert not model.training
        assert model.markov_history_window == 0 and model.markov_final_history_window == 0
        self.persistent=False
        self.pool=None
        self.graphs={}
        self.graph_scatter=False
        self.graph_direct=False
    def release(self,cache):
        if self.pool is not None:self.pool.release(cache)

    def _project(self,prepared,positions):
        """Pure fixed-shape projection/KV body used by optional offline graphs."""
        m=self.model;keys=[];values=[]
        default=None if m.context_fusion_mode=='depth_aligned' else m.project_context(prepared)
        for index,layer in enumerate(m.layers):
            context=m.project_context(prepared,index) if default is None else default
            key,value=(layer.context_kv(context,positions)
                       if m.architecture=='official_qwen3' or m.random_rope_draft
                       else layer.context_kv(context))
            keys.append(key);values.append(value)
        return torch.stack(keys),torch.stack(values)

    def _project_scatter(self,prepared,positions,source,lengths,slots,destinations):
        keys,values=self._project(prepared,positions)
        from acc_infer_clear.kernels.context_scatter import scatter
        scatter(keys,values,self.pool.storage,source,lengths,slots,destinations)
        bank=getattr(self.pool,'native_bank',None)
        if bank is not None and bank.enabled_for_total(prepared.shape[1]):
            scatter(keys,values,bank.cache,source,lengths,slots,destinations)
        return prepared[:, :1, :1]

    def _project_scatter_direct(self,prepared,positions,source,lengths,slots,destinations):
        m=self.model;default=None if m.context_fusion_mode=='depth_aligned' else m.project_context(prepared)
        from acc_infer_clear.kernels.context_scatter import scatter_layer
        for index,layer in enumerate(m.layers):
            context=m.project_context(prepared,index) if default is None else default
            key,value=(layer.context_kv(context,positions) if m.architecture=='official_qwen3' or m.random_rope_draft else layer.context_kv(context))
            scatter_layer(key,value,self.pool.storage,index,source,lengths,slots,destinations)
        return prepared[:, :1, :1]

    def prepare_graphs(self,max_batch,scatter=False,totals=None,direct=False):
        """Capture total-committed-token buckets; no lazy capture in inference."""
        if self.graphs:raise RuntimeError('Context graphs already prepared')
        from acc_infer_clear.runtime.graphs import capture
        width=self.model.num_target_features*self.model.interface_size
        device=next(self.model.parameters()).device
        if scatter and self.pool is None:raise RuntimeError('Context scatter requires DraftPool')
        totals=tuple(totals or range(8,max_batch*8+1,8))
        for total in totals:
            prepared=torch.zeros(1,total,width,device=device,dtype=torch.float32)
            positions=torch.arange(total,device=device)[None]
            if scatter:
                meta=(torch.zeros(max_batch,device=device,dtype=torch.int32),
                      torch.zeros(max_batch,device=device,dtype=torch.int32),
                      torch.zeros(max_batch,device=device,dtype=torch.int32),
                      torch.zeros(max_batch,device=device,dtype=torch.int32))
                fn=self._project_scatter_direct if direct else self._project_scatter
                self.graphs[total]=capture(fn,(prepared,positions,*meta))
            else:self.graphs[total]=capture(self._project,(prepared,positions))
        self.graph_scatter=bool(scatter)
        self.graph_direct=bool(direct)
        return dict(total_token_buckets=sorted(self.graphs),online_capture=False,
                    padding='next_multiple_of_8',precision_unchanged=True,
                    fused_slot_scatter=self.graph_scatter,direct_layer_scatter=self.graph_direct)

    @torch.inference_mode()
    def __call__(self, jobs):
        normalized = []
        for args, kwargs in jobs:
            names = ('cache', 'selected_hidden', 'final_hidden', 'committed_tokens')
            job = dict(zip(names, args))
            job.update(kwargs)
            assert job['selected_hidden'].shape[0] == 1
            normalized.append(job)
        m = self.model
        if self.pool is not None:
            for j in normalized:self.pool.attach(j['cache'])
        lengths = [j['selected_hidden'].shape[1] for j in normalized]
        if self.pool is not None and any(j['cache'].length+n>self.pool.capacity
                                         for j,n in zip(normalized,lengths)):
            raise ValueError('Draft context capacity exceeded')
        prepared = torch.cat([m.prepare_context(j['selected_hidden'], j.get('final_hidden')) for j in normalized], dim=1)
        positions = torch.cat([torch.arange(j['cache'].length, j['cache'].length + n, device=prepared.device) for j, n in zip(normalized, lengths)])[None]
        total=prepared.shape[1];bucket=((total+7)//8)*8
        if bucket in self.graphs:
            if bucket!=total:
                prepared=torch.nn.functional.pad(prepared,(0,0,0,bucket-total))
                positions=torch.nn.functional.pad(positions,(0,bucket-total))
            if self.graph_scatter:
                size=len(normalized);source=[];offset=0
                for n in lengths:source.append(offset);offset+=n
                source+=([0]*(self.pool.max_slots-size));length_meta=lengths+([0]*(self.pool.max_slots-size))
                slots=[j['cache'].pool_slot for j in normalized]+([0]*(self.pool.max_slots-size))
                destinations=[j['cache'].length for j in normalized]+([0]*(self.pool.max_slots-size))
                meta=[torch.tensor(x[:self.pool.max_slots//2],device=prepared.device,dtype=torch.int32)
                      for x in (source,length_meta,slots,destinations)]
                self.graphs[bucket](prepared,positions,*meta);pairs=None
            else:
                keys,values=self.graphs[bucket](prepared,positions);pairs=zip(keys,values)
        else:
            # Preserve the original fallback exactly: do not materialize a
            # stacked layer output merely to share the graph-only helper.
            default=None if m.context_fusion_mode=='depth_aligned' else m.project_context(prepared)
            pairs=[]
            for index,layer in enumerate(m.layers):
                context=m.project_context(prepared,index) if default is None else default
                pairs.append(layer.context_kv(context,positions)
                             if m.architecture=='official_qwen3' or m.random_rope_draft
                             else layer.context_kv(context))
        mirrored_fallback = pairs is not None and self.pool is not None and getattr(self.pool,'native_bank',None) is not None
        if pairs is not None:
         for index, (key,value) in enumerate(pairs):
            offset = 0
            for job, n in zip(normalized, lengths):
                cache = job['cache']
                if self.persistent:
                    if not hasattr(cache,'storage_keys'):
                        capacity=m.position_embedding.num_embeddings
                        cache.storage_keys=[k.new_empty(1,k.shape[1],capacity,k.shape[-1]) for k in cache.keys]
                        cache.storage_values=[v.new_empty(1,v.shape[1],capacity,v.shape[-1]) for v in cache.values]
                        for dest,src in zip(cache.storage_keys,cache.keys):dest[:,:,:cache.length].copy_(src)
                        for dest,src in zip(cache.storage_values,cache.values):dest[:,:,:cache.length].copy_(src)
                    end=cache.length+n
                    if end>cache.storage_keys[index].shape[2]:raise ValueError('Draft context capacity exceeded')
                    cache.storage_keys[index][:,:,cache.length:end].copy_(key[:,:,offset:offset+n])
                    cache.storage_values[index][:,:,cache.length:end].copy_(value[:,:,offset:offset+n])
                    cache.keys[index]=cache.storage_keys[index][:,:,:end]
                    cache.values[index]=cache.storage_values[index][:,:,:end]
                else:
                    cache.keys[index] = torch.cat((cache.keys[index], key[:, :, offset:offset + n]), dim=2)
                    cache.values[index] = torch.cat((cache.values[index], value[:, :, offset:offset + n]), dim=2)
                offset += n
        else:
            for job,n in zip(normalized,lengths):
                cache=job['cache'];end=cache.length+n
                for index in range(len(m.layers)):
                    cache.keys[index]=cache.storage_keys[index][:,:,:end]
                    cache.values[index]=cache.storage_values[index][:,:,:end]
        for job, n in zip(normalized, lengths):
            job['cache'].length += n
            # Captured scatter graphs write both the canonical pool and the
            # compact TensorRT mirror.  The variable-total fallback above used
            # to update only the canonical pool, leaving the initial prompt
            # context absent from TensorRT's cache.  Synchronize only this
            # fallback path; graph hits already mirror their incremental rows.
            if mirrored_fallback:
                self.pool.native_bank.import_slot(job['cache'],job['cache'].pool_slot)
        self.calls += 1
        self.rows += len(jobs)
        return [None] * len(jobs)
