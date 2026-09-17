"""FP32 variable-history verification batches, with request-owned output KV.

The caller adds each request's absolute positions BEFORE calling this module.
History padding is masked; new tokens sit after the common padded history.
Returned caches remove that gap, so existing accept/crop semantics are intact.
No mutable inference_model.cached_mel_emb is read in the batched body.
"""
import torch

class RequestKV:
    """Contiguous per-request storage; crop changes only visible length.

    Grouped packing copies all layers in one launch per request. Accepted KV
    remains in place; only the new Q entries are copied back after verification.
    """
    pooled_dynamic_kv = True

    def __init__(self, packed, length):
        self.packed = packed
        self.length = length

    def __len__(self):
        return self.packed.shape[0]

    def __getitem__(self, i):
        return (self.packed[i, 0, :, :, :self.length], self.packed[i, 1, :, :, :self.length])

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def to_heads_first(self):
        return tuple(self)

    def get_seq_length(self):
        return self.length

    def crop(self, length):
        if not 0 <= length <= self.length:
            raise ValueError('KV crop outside verified extent')
        self.length = int(length)

class BatchedTarget:

    def __init__(self, target):
        self.target = target
        body = target._block_forward_with_hidden_states
        self.body = body
        self.calls = 0
        self.rows = 0
        self.shapes = {}

    @torch.inference_mode()
    def __call__(self, jobs):
        q = jobs[0][0].shape[1]
        assert all((x.shape[0] == 1 and x.shape[1] == q and (pos is None) for x, kv, mask, pos in jobs))
        lengths = [kv[0][0].shape[-2] for x, kv, mask, pos in jobs]
        longest = max(lengths)
        first = jobs[0][1][0][0]
        b = len(jobs)
        layers = len(jobs[0][1])
        h = first.shape[1]
        d = first.shape[-1]
        packed = first.new_zeros(layers, 2, b, h, longest, d)
        mask = torch.zeros(b, longest + q, dtype=jobs[0][2].dtype, device=first.device)
        for row, ((x, kv, keep, _), length) in enumerate(zip(jobs, lengths)):
            assert keep.shape == (1, length + q)
            mask[row, :length] = keep[0, :length]
            mask[row, longest:] = keep[0, -q:]
            if isinstance(kv, RequestKV):
                packed[:, :, row, :, :length].copy_(kv.packed[:, :, 0, :, :length])
            else:
                for layer, (k, v) in enumerate(kv):
                    packed[layer, 0, row, :, :length].copy_(k[0])
                    packed[layer, 1, row, :, :length].copy_(v[0])
        past = tuple(((packed[i, 0], packed[i, 1]) for i in range(layers)))
        logits, kv, selected, final = self.body(torch.cat([r[0] for r in jobs]), past, mask, None)
        new = torch.stack([torch.stack((k[:, :, -q:], v[:, :, -q:])) for k, v in kv])
        outputs = []
        for row, length in enumerate(lengths):
            old = jobs[row][1]
            if isinstance(old, RequestKV) and old.packed.shape[-2] >= length + q:
                own = old.packed
            else:
                capacity = (length + q + 127) // 128 * 128
                own = first.new_empty(layers, 2, 1, h, capacity, d)
                own[:, :, 0, :, :length].copy_(packed[:, :, row, :, :length])
            own[:, :, 0, :, length:length + q].copy_(new[:, :, row])
            outputs.append((logits[row:row + 1].clone(), RequestKV(own, length + q), selected[row:row + 1].clone(), final[row:row + 1].clone()))
        self.calls += 1
        self.rows += b
        key = f'{b}x{q}@{longest}'
        self.shapes[key] = self.shapes.get(key, 0) + 1
        return outputs

    def stats(self):
        return dict(calls=self.calls, rows=self.rows, mean_batch=self.rows / max(self.calls, 1), shapes=self.shapes)

    @torch.inference_mode()
    def prefill(self, jobs):
        """Right-pad embedded prefixes; return only each real causal prefix."""
        lengths = [x.shape[1] for x, mask in jobs]
        longest = max(lengths)
        b = len(jobs)
        first = jobs[0][0]
        x = first.new_zeros(b, longest, first.shape[-1])
        mask = torch.zeros(b, longest, device=first.device, dtype=jobs[0][1].dtype)
        for row, ((src, keep), length) in enumerate(zip(jobs, lengths)):
            x[row, :length].copy_(src[0])
            mask[row, :length].copy_(keep[0])
        prefill_body = getattr(self, 'prefill_body', self.target._block_forward_with_hidden_states)
        logits, kv, selected, final = prefill_body(x, None, mask, None)
        packed = getattr(kv, 'packed', None)
        if packed is None:
            packed = torch.stack([torch.stack((k, v)) for k, v in kv])
        outputs = []
        for row, length in enumerate(lengths):
            importer = getattr(self, 'pool_import', None)
            if importer is not None:
                owned = importer(packed, row, length)
                outputs.append((logits[row:row + 1, :length].clone(), owned, selected[row:row + 1, :length].clone(), final[row:row + 1, :length].clone()))
                continue
            capacity = (length + 127) // 128 * 128
            own = first.new_empty(len(kv), 2, 1, kv[0][0].shape[1], capacity, kv[0][0].shape[-1])
            own[:, :, 0, :, :length].copy_(packed[:, :, row, :, :length])
            outputs.append((logits[row:row + 1, :length].clone(), RequestKV(own, length), selected[row:row + 1, :length].clone(), final[row:row + 1, :length].clone()))
        return outputs

