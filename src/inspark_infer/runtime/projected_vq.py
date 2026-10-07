"""Precompute the fixed, sample-local FVQ output projection before serving."""
import torch
from torch import nn


class ProjectedVQTable:
    def __init__(self, quantizer):
        if quantizer.training or quantizer.num_quantizers != 1 or quantizer.quantizer_type != 'fvq':
            raise ValueError('Projected lookup requires one frozen factorized codebook')
        item = quantizer.quantizers[0]
        projection = item.out_project
        if not (isinstance(projection, nn.Identity) or
                isinstance(projection, nn.Conv1d) and projection.kernel_size == (1,)
                and projection.stride == (1,) and projection.padding == (0,)
                and projection.dilation == (1,) and projection.groups == 1):
            raise ValueError('Code projection must be pointwise and sample independent')
        self.quantizer = quantizer
        self.modules = (item.codebook, projection)
        # The loader may create inference Parameters without version counters.
        # Rebind only lookup dependencies to byte-identical normal Parameters
        # once at setup, so later in-place changes can invalidate the table.
        with torch.inference_mode(False), torch.no_grad():
            for root in self.modules:
                for name, value in list(root.named_parameters()):
                    if value.is_inference():
                        parts=name.rsplit('.',1)
                        parent=root if len(parts)==1 else root.get_submodule(parts[0])
                        parent.register_parameter(parts[-1],nn.Parameter(value.clone(),requires_grad=value.requires_grad))
        self.parameters = tuple(p for root in self.modules for p in root.parameters())
        self.versions = tuple(p._version for p in self.parameters)
        self.dtype = item.codebook.weight.dtype
        if self.dtype != torch.float32:
            raise ValueError('Preserve the original protected FP32 semantic lookup')
        with torch.inference_mode():
            codes = torch.arange(item.codebook.num_embeddings, device=item.codebook.weight.device)[None,None]
            self.table = quantizer.vq2emb(codes)[0].transpose(0,1).contiguous()
        self.calls = 0

    def __call__(self, codes):
        current=tuple(p for root in self.modules for p in root.parameters())
        if (tuple(map(id,current))!=tuple(map(id,self.parameters)) or
                tuple(p._version for p in current) != self.versions):
            raise RuntimeError('FVQ weights changed after projection table preparation')
        self.calls += 1
        return torch.nn.functional.embedding(codes, self.table)
