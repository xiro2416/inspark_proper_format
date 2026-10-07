"""Canonical standard-ONNX Draft backbone with the shared static Q/DQ recipe."""
from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from inspark_infer.build.unified_acoustic_export import ExportWeightOp


def check_draft_contract(model):
    if model.architecture != "shared_context" or model.block_size != 7:
        raise ValueError("Export requires the shared-context Q7 checkpoint")
    if any(getattr(model, name, 0) for name in (
            "query_temporal_kernel", "midblock_refresh_at", "logit_residual_rank", "recent_context_window")):
        raise ValueError("Draft graph does not implement temporal/refresh/logit-residual variants")
    if any(hasattr(layer, "rope_inv_freq") or hasattr(layer, "attention_conv") for layer in model.layers):
        raise ValueError("The x/cache-only signature cannot represent rotary or temporal Draft positions")
    if any(getattr(layer, "layerwise_target", False) for layer in model.layers):
        raise ValueError("Expected already projected shared-context Draft caches")


class DraftBackbone(nn.Module):
    """Actual checkpoint layers; ONNX lowers torch SDPA into standard math ops."""
    def __init__(self, model):
        super().__init__()
        check_draft_contract(model)
        self.layers = nn.ModuleList(model.layers)
        self.output_norm = model.output_norm
        self.output_projection = model.output_projection
        self.interface_output_norm = model.interface_output_norm
        self.lm_head = model.lm_head

    def forward(self, x, mask, *cache):
        if len(cache) != 2 * len(self.layers):
            raise ValueError("One historical K/V pair is required per Draft layer")
        hidden = x
        for index, layer in enumerate(self.layers):
            hidden = layer(hidden, context_k=cache[2 * index], context_v=cache[2 * index + 1],
                           context_mask=None, attention_mask=mask)
        hidden = self.interface_output_norm(self.output_projection(self.output_norm(hidden)))
        return hidden, self.lm_head(hidden)


class DraftContext(nn.Module):
    """Position-independent committed-hidden projection and all six K/V GEMMs.

    Native worker still decides which prefix positions commit. All dimensions,
    FP32 context projection/RMSNorm, and block K/V Q/DQ roles are preserved.
    """
    def __init__(self,model,*,gemm_2d=False,fp32_bf16_bias=False):
        super().__init__();check_draft_contract(model)
        if model.context_fusion_mode!='concat_projection' or model.include_target_final_hidden:
            raise ValueError('Context export requires five-layer concat projection only')
        self.model=model
        self.gemm_2d=bool(gemm_2d)
        self.fp32_bf16_bias=bool(fp32_bf16_bias)
        if self.fp32_bf16_bias and not self.gemm_2d:
            raise ValueError('Protected fused-bias expression requires rank2 context Gemm')

    def linear(self,op,context):
        if self.fp32_bf16_bias and getattr(op,'precision',None)=='bf16':
            # Preserve BF16 operands and the intended F.linear FP32 accumulator
            # through bias addition, then round once to the protected BF16 output.
            bias=op.bias.float() if op.bias is not None else None
            return F.linear(context.bfloat16().float(),op.weight.float(),bias).bfloat16().float()
        return op(context)

    def forward(self,hidden):
        context=self.model.project_context(self.model.prepare_context(hidden))
        keys,values=[],[]
        for layer in self.model.layers:
            if self.gemm_2d:
                # Rank2 F.linear exports as Gemm with its bias. The original
                # rank3 export emits MatMul+Add, exposing a BF16 rounding boundary.
                keys.append(self.linear(layer.k_proj,context).unflatten(-1,(layer.num_heads,layer.head_dim)))
                values.append(self.linear(layer.v_proj,context).unflatten(-1,(layer.num_heads,layer.head_dim)))
            else:
                key,value=layer.context_kv(context[None])
                keys.append(key[0].transpose(0,1));values.append(value[0].transpose(0,1))
        return context,torch.stack(keys,1).contiguous(),torch.stack(values,1).contiguous()


class DraftContextKV(nn.Module):
    """Only the two quantized K/V pairs; protected projection/first layer stay Torch."""
    def __init__(self,model):
        super().__init__();check_draft_contract(model)
        if len(model.layers)!=3 or any(not getattr(op,'quantized',False)
                for layer in model.layers[1:] for op in (layer.k_proj,layer.v_proj)):
            raise ValueError('Context K/V subset requires two quantized layers after the protected first layer')
        self.layers=nn.ModuleList(model.layers[1:])

    def forward(self,context):
        keys,values=[],[]
        for layer in self.layers:
            keys.append(layer.k_proj(context).unflatten(-1,(layer.num_heads,layer.head_dim)))
            values.append(layer.v_proj(context).unflatten(-1,(layer.num_heads,layer.head_dim)))
        return torch.stack(keys,1).contiguous(),torch.stack(values,1).contiguous()


def prepare_draft_roles(engine, artifact):
    """Wrap only canonical Draft block linears, retaining unfolded weight Q/DQ."""
    from inspark_infer.quantization.unified import iter_roles
    if getattr(engine, "deployment_state", "raw") != "raw":
        raise ValueError("Draft export requires a fresh raw checkpoint")
    model = engine.rt.engine.draft
    check_draft_contract(model)
    if any(getattr(module, "weight_quantization_folded", False) for module in model.modules()):
        raise ValueError("Export Draft before folding weight Q/DQ in a Torch reference")
    bindings = list(iter_roles(engine, artifact["scheme"], ("draft",)))
    expected = {role.path for role in bindings}
    declared = {path for path in artifact["role_specs"] if path.startswith("draft.")}
    if declared != expected:
        raise ValueError(f"Draft recipe role mismatch: missing={sorted(expected-declared)}, extra={sorted(declared-expected)}")
    roles = []
    for role in bindings:
        spec = deepcopy(artifact["role_specs"][role.path])
        if spec["precision"] != role.precision:
            raise ValueError(f"Draft protection/precision mismatch: {role.path}")
        if not isinstance(role.module, nn.Linear):
            raise ValueError(f"Unexpected Draft block operation: {role.path}")
        wrapped = ExportWeightOp(role.module, spec)
        role.parent.add_module(role.child_name, wrapped)
        roles.append(dict(canonical_path=role.path, precision=spec["precision"],
                          weight_shape=list(wrapped.weight.shape), explicit_weight_qdq=wrapped.quantized))
    return dict(component="draft", roles=roles, custom_domains=[], plugins=[],
                input_semantics="noise/anchor and absolute position embeddings already applied",
                attention="noncausal within seven current proposal slots; history selected by mask",
                fp32_boundaries=["RMSNorm", "attention", "KV", "output hidden", "lm_head/base logits"],
                rnn_included=False)


def input_names(layers=3):
    return ["x", "mask", *[name for index in range(layers) for name in (f"k_cache_{index}", f"v_cache_{index}")]]
