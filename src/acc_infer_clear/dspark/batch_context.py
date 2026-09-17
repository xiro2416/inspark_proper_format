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
    def release(self,cache):
        if self.pool is not None:self.pool.release(cache)

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
        prepared = torch.cat([m.prepare_context(j['selected_hidden'], j.get('final_hidden')) for j in normalized], dim=1)
        positions = torch.cat([torch.arange(j['cache'].length, j['cache'].length + n, device=prepared.device) for j, n in zip(normalized, lengths)])[None]
        default = None if m.context_fusion_mode == 'depth_aligned' else m.project_context(prepared)
        for index, layer in enumerate(m.layers):
            context = m.project_context(prepared, index) if default is None else default
            key, value = layer.context_kv(context, positions) if m.architecture == 'official_qwen3' or m.random_rope_draft else layer.context_kv(context)
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
        for job, n in zip(normalized, lengths):
            job['cache'].length += n
        self.calls += 1
        self.rows += len(jobs)
        return [None] * len(jobs)
