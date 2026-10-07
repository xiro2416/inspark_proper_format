"""Standard-ONNX Target prefill/latent graphs using the shared static recipe.

This is an export graph built from existing torch operators. It mirrors the
production PrefixGraphs body and does not add a GPU kernel or serving backend.
"""
from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from inspark_infer.build.unified_acoustic_export import ExportWeightOp


EXTENTS = {"prefill": 48, "latent": 80}
PREFILL_OUTPUTS = ("last_logits", "packed_kv", "selected", "final")


class PrefixTransformer(nn.Module):
    """Registered Target blocks plus the correct prefill or latent output head."""
    def __init__(self, transformer, lm_head, latent_final_norm, target_layer_ids, *, kind, sdpa_bf16=False):
        super().__init__()
        if kind not in EXTENTS:
            raise ValueError("Prefix kind must be prefill or latent")
        ids = tuple(int(index) for index in target_layer_ids)
        if not ids or tuple(sorted(set(ids))) != ids or ids[-1] >= len(transformer.h) or ids[0] < 0:
            raise ValueError("Target hidden layer indices must be unique, ordered and in range")
        self.transformer = transformer
        self.lm_head = lm_head
        self.latent_final_norm = latent_final_norm
        self.target_layer_ids = ids
        self.kind = kind
        self.sdpa_bf16 = bool(sdpa_bf16)
        self.arithmetic_overrides = (dict(prefix_sdpa_qkv_dtype='bfloat16',prefix_sdpa_output_dtype='float32',
                                         prefix_sdpa_onnx_mask='bfloat16_0_or_negative_infinity',
                                         prefill_returned_kv_dtype='float32') if self.sdpa_bf16 else {})

    def forward(self, x, keep):
        hidden, selected, caches = x, [], []
        batch, length = x.shape[:2]
        positions = torch.arange(length, device=x.device)
        mask = ((positions[None, :] <= positions[:, None])[None, None]
                & keep[:, None, None, :].bool())
        attention_mask = mask
        if self.sdpa_bf16 and torch.onnx.is_in_onnx_export():
            # A Boolean SDPA mask is otherwise lowered through FP32 constants
            # by the legacy exporter, promoting Softmax/PV back to FP32.
            zero=torch.zeros((),device=x.device,dtype=torch.bfloat16)
            negative_infinity=torch.full((),float('-inf'),device=x.device,dtype=torch.bfloat16)
            attention_mask=torch.where(mask,zero,negative_infinity)
        for index, block in enumerate(self.transformer.h):
            attention = block.attn
            q, k, v = attention.c_attn(block.ln_1(hidden)).split(attention.split_size, dim=-1)
            q = q.view(batch, length, attention.num_heads, attention.head_dim).transpose(1, 2)
            k = k.view(batch, length, attention.num_heads, attention.head_dim).transpose(1, 2)
            v = v.view(batch, length, attention.num_heads, attention.head_dim).transpose(1, 2)
            if self.sdpa_bf16:
                # Preserve the original FP32 k/v for the externally returned
                # prefill cache. Only the attention branch changes precision.
                context=F.scaled_dot_product_attention(q.bfloat16(),k.bfloat16(),v.bfloat16(),
                                                       attn_mask=attention_mask).float()
            else:context = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
            hidden = hidden + attention.c_proj(context.transpose(1, 2).contiguous().view_as(x))
            hidden = hidden + block.mlp(block.ln_2(hidden))
            if self.kind == "prefill":
                caches.append(torch.stack((k, v)))
                if index in self.target_layer_ids:
                    selected.append(hidden)
        final = self.transformer.ln_f(hidden)
        if self.kind == "latent":
            return self.latent_final_norm(final)
        last = (keep.long() * positions[None]).amax(-1)
        last_hidden = final.gather(1, last[:, None, None].expand(-1, 1, final.shape[-1]))
        return self.lm_head(last_hidden), torch.stack(caches), torch.cat(selected, dim=-1), final


def prepare_target_roles(engine, artifact):
    """Replace raw Target block ops; retain explicit weight AND activation Q/DQ.

    Mutates only this export process's raw Target modules. Engine deployment and
    install_reference_recipe are deliberately not used: their folded reference
    weights must never be silently re-exported as ordinary float constants.
    """
    from inspark_infer.quantization.unified import iter_roles
    if getattr(engine, "deployment_state", "raw") != "raw":
        raise ValueError("Prefix export requires a fresh raw model")
    target = engine.rt.engine.target
    if any(getattr(module, "weight_quantization_folded", False)
           for module in target.model.transformer.modules()):
        raise ValueError("Prefix export must start before weight Q/DQ folding")
    bindings = list(iter_roles(engine, artifact["scheme"], ("target",)))
    declared = {name for name in artifact["role_specs"] if name.startswith("target.")}
    expected = {role.path for role in bindings}
    if declared != expected:
        raise ValueError(f"Target recipe roles differ: missing={sorted(expected-declared)}, extra={sorted(declared-expected)}")
    roles = []
    for role in bindings:
        spec = deepcopy(artifact["role_specs"][role.path])
        if spec["precision"] != role.precision:
            raise ValueError(f"Target role protection/precision mismatch: {role.path}")
        module = role.module
        if type(module).__name__ == "Conv1D":
            weight = role.weight()
            linear = nn.Linear(weight.shape[1], weight.shape[0], bias=module.bias is not None,
                               device=weight.device, dtype=torch.float32)
            with torch.no_grad():
                linear.weight.copy_(weight)
                if module.bias is not None:
                    linear.bias.copy_(module.bias.detach().float())
            module = linear
        if not isinstance(module, nn.Linear):
            raise ValueError(f"Unexpected Target block weight operation: {role.path}")
        wrapped = ExportWeightOp(module, spec)
        if getattr(wrapped, "weight_quantization_folded", False):
            raise ValueError("Export wrapper unexpectedly folded weight quantization")
        role.parent.add_module(role.child_name, wrapped)
        roles.append(dict(canonical_path=role.path, precision=spec["precision"],
                          original_type=type(role.module).__name__, exported_type="Linear",
                          weight_shape=list(wrapped.weight.shape), explicit_weight_qdq=wrapped.quantized))
    return dict(component="target", roles=roles, plugins=[], custom_domains=[],
                source="raw_checkpoint_plus_shared_static_quantization_recipe",
                unquantized_boundaries=["token/position embeddings", "norms", "lm_head", "latent_final_norm"],
                kv_dtype="float32", interface_dtype="float32")


def from_engine(engine, kind, *, sdpa_bf16=False):
    target = engine.rt.engine.target
    return PrefixTransformer(target.model.transformer, target.model.lm_head,
                             engine.tts.gpt.final_norm, target.target_layer_ids, kind=kind,sdpa_bf16=sdpa_bf16)
