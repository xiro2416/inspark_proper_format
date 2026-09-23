"""Batch the deterministic Draft backbone; keep request-local RNN sampling.

Absolute code positions and context lengths are per-row. Padding never becomes
context. This preserves the existing full-vocabulary RNN and PCG implementation.
"""
import torch

class BatchedDraftBackbone:

    def __init__(self, model):
        assert model.architecture != 'official_qwen3' and model.scratch_absolute_position
        assert not model.midblock_refresh_at
        self.model = model
        self.calls = 0
        self.rows = 0
        self.last_terms = None
        self.step = torch.arange(model.block_size, device=next(model.parameters()).device)[None]
        self.context_uses_positions = any((hasattr(layer, 'rope_inv_freq') for layer in model.layers))
        self.body = self.forward
        self.graphs={};self.graph_hits=0
    def forward_packed(self,anchors,positions,packed,keep,context_positions):
        return self.forward(anchors,positions,tuple(packed[i,0] for i in range(len(self.model.layers))),
                            tuple(packed[i,1] for i in range(len(self.model.layers))),keep,context_positions if self.context_uses_positions else None)
    def prepare_graphs(self,max_batch):
        from inspark_infer.runtime.graphs import capture
        from inspark_infer.runtime.graph_policy import batches, FIRST_KV_LIMITS
        if self.graphs:raise RuntimeError('Draft already captured')
        model=self.model;param=next(model.parameters());layer=model.layers[0]
        for b in batches(max_batch):
            anchors=torch.zeros(b,device=param.device,dtype=torch.long)
            pos=torch.arange(7,device=param.device)[None].expand(b,-1).clone()
            for length in FIRST_KV_LIMITS:
                packed=param.new_zeros(len(model.layers),2,b,layer.num_heads,length,layer.head_dim)
                keep=torch.ones(b,length,device=param.device,dtype=torch.bool)
                ctx=pos+length
                self.graphs[b,length]=capture(self.forward_packed,(anchors,pos,packed,keep,ctx))
        return dict(keys=[list(k) for k in self.graphs],online_capture=False)

    def forward(self, anchors, positions, keys, values, keep, context_positions):
        m = self.model
        hidden = m._noise_embeddings(anchors, positions)
        for index, layer in enumerate(m.layers):
            hidden = layer(hidden, context_k=keys[index], context_v=values[index], context_mask=keep, position_ids=context_positions if hasattr(layer, 'rope_inv_freq') else None)
            hidden = m.apply_query_temporal(hidden, index)
        hidden = m.project_output(hidden)
        return (hidden, m.base_logits(hidden))

    @torch.inference_mode()
    def __call__(self, jobs):
        m = self.model
        b = len(jobs)
        lengths = [j['cache'].length for j in jobs]
        longest = max(lengths)
        assert m.recent_context_window == 0, 'windowed context needs per-layer masks'
        exemplar = jobs[0]['cache'].keys[0]
        layers = len(m.layers)
        h = exemplar.shape[1]
        d = exemplar.shape[-1]
        limit=next(n for n in (64,128,256,512,1024,2048) if n>=longest)
        use_graph=(b,limit) in self.graphs
        if use_graph:
            graph=self.graphs[b,limit]
            packed=graph.inputs[2];keep=graph.inputs[3];keep.zero_()
        else:
            packed = exemplar.new_zeros(layers, 2, b, h, longest, d)
            keep = torch.zeros(b, longest, device=exemplar.device, dtype=torch.bool)
        for row, j in enumerate(jobs):
            length = lengths[row]
            keep[row, :length] = True
            for i, (k, v) in enumerate(zip(j['cache'].keys, j['cache'].values)):
                packed[i, 0, row, :, :length].copy_(k[0])
                packed[i, 1, row, :, :length].copy_(v[0])
        step = self.step
        positions = torch.tensor([j['first_position'] for j in jobs], device=exemplar.device)[:, None] + step
        context_positions = torch.tensor(lengths, device=exemplar.device)[:, None] + step if self.context_uses_positions else None
        anchors = torch.cat([j['anchor_token'].reshape(1) for j in jobs])
        self.last_terms = None
        if use_graph:
            graph.inputs[0].copy_(anchors);graph.inputs[1].copy_(positions)
            if context_positions is not None:graph.inputs[4].copy_(context_positions)
            graph.graph.replay();hidden,base=graph.outputs;self.graph_hits+=1
        else:
            hidden, base = self.body(anchors, positions, tuple((packed[i, 0] for i in range(layers))), tuple((packed[i, 1] for i in range(layers))), keep, context_positions)
        self.calls += 1
        self.rows += b
        return [(hidden[i:i + 1].clone(), base[i:i + 1].clone()) for i in range(b)]

    def stats(self):
        return dict(calls=self.calls, rows=self.rows, mean_batch=self.rows / max(1, self.calls),graph_hits=self.graph_hits,
                    history_packing=True)
