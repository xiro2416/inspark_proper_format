"""Canonical operator inventory independent of any one GPU schedule."""
from __future__ import annotations

from dataclasses import dataclass

from .model import HardwareProfile, OperatorSignature


@dataclass(frozen=True)
class PipelineModelSpec:
    target_hidden: int = 1280
    target_ffn: int = 5120
    target_qkv: int = 3840
    target_layers: int = 24
    target_protected: int = 6
    draft_hidden: int = 1280
    draft_ffn: int = 5120
    draft_qkv: int = 3840
    draft_layers: int = 3
    draft_protected: int = 1
    cfm_hidden: int = 512
    cfm_ffn: int = 1536
    cfm_qkv: int = 1536
    cfm_layers: int = 13
    cfm_protected: int = 4
    cfm_head_frames: int = 310


def _gemm(model: str, role: str, batch: int, m: int, n: int, k: int, dtype: str,
          *, calls: int, epilogue: tuple[str, ...] = (), metadata: dict[str, str] | None = None) -> OperatorSignature:
    return OperatorSignature(
        model=model, role=role, kind='gemm', m=m, n=n, k=k,
        input_dtype=dtype, weight_dtype=dtype, batch=batch, calls=calls,
        epilogue=epilogue, metadata=tuple(sorted((metadata or {}).items())),
    )


def canonical_inventory(profile: HardwareProfile, *, batches: tuple[int, ...] = tuple(range(1, 9)) + (16,),
                        spec: PipelineModelSpec = PipelineModelSpec(),
                        matrix_dtype: str | None = None) -> list[OperatorSignature]:
    matrix_dtype = profile.preferred_matrix_dtype if matrix_dtype is None else matrix_dtype
    if matrix_dtype not in ('bf16', 'fp8'):
        raise ValueError('matrix_dtype must be bf16 or fp8')
    result = []
    for batch in batches:
        for model, qlen, hidden, ffn, qkv, deep_layers in (
            ('target', 8, spec.target_hidden, spec.target_ffn, spec.target_qkv, spec.target_layers - spec.target_protected),
            ('draft', 7, spec.draft_hidden, spec.draft_ffn, spec.draft_qkv, spec.draft_layers - spec.draft_protected),
        ):
            m = batch * qlen
            common = {'precision_tier': 'deep', 'qlen': str(qlen)}
            result.extend((
                _gemm(model, 'qkv', batch, m, qkv, hidden, matrix_dtype, calls=deep_layers,
                      epilogue=('qk_block32', 'kv_slot'), metadata=common),
                _gemm(model, 'out', batch, m, hidden, hidden, matrix_dtype, calls=deep_layers,
                      epilogue=('residual',), metadata=common),
                _gemm(model, 'up', batch, m, ffn, hidden, matrix_dtype, calls=deep_layers, metadata=common),
                _gemm(model, 'down', batch, m, hidden, ffn, matrix_dtype, calls=deep_layers,
                      epilogue=('residual',), metadata=common),
            ))
        m = batch * spec.cfm_head_frames
        common = {'precision_tier': 'deep', 'frames': str(spec.cfm_head_frames), 'allowed_schedules': 'tiled'}
        deep = spec.cfm_layers - spec.cfm_protected
        result.extend((
            _gemm('cfm', 'qkv_up', batch, m, spec.cfm_qkv, spec.cfm_hidden, matrix_dtype,
                  calls=deep * 3, metadata=common),
            _gemm('cfm', 'out', batch, m, spec.cfm_hidden, spec.cfm_hidden, matrix_dtype,
                  calls=deep, metadata=common),
            _gemm('cfm', 'down', batch, m, spec.cfm_hidden, spec.cfm_ffn, matrix_dtype,
                  calls=deep, metadata=common),
            _gemm('cfm', 'skip', batch, m, spec.cfm_hidden, 2 * spec.cfm_hidden, matrix_dtype,
                  calls=deep, metadata=common),
        ))
        result.extend((
            OperatorSignature('target', 'kv_attention', 'attention', batch * 8, 64, 128,
                              matrix_dtype, matrix_dtype, batch=batch, calls=spec.target_layers,
                              input_layout='bhqd', weight_layout='bhsd', output_layout='bqhd',
                              metadata=(('softmax', 'fp32'), ('value', 'fp32'))),
            OperatorSignature('draft', 'kv_attention', 'attention', batch * 7, 64, 128,
                              matrix_dtype, matrix_dtype, batch=batch, calls=spec.draft_layers,
                              input_layout='bhqd', weight_layout='bhsd', output_layout='bqhd',
                              metadata=(('softmax', 'fp32'), ('value', 'fp32'))),
            OperatorSignature('draft', 'proposal_rnn', 'control', batch, 7, spec.draft_hidden,
                              matrix_dtype, matrix_dtype, batch=batch, calls=1,
                              epilogue=('sigmoid', 'tanh', 'softmax', 'rng', 'argmax')),
            OperatorSignature('target', 'accept_commit', 'control', batch, 8, 8194,
                              'fp32', 'fp32', batch=batch, calls=1,
                              epilogue=('acceptance', 'residual', 'commit')),
            OperatorSignature('draft', 'context_scatter', 'layout', batch * 8, spec.draft_hidden, spec.target_hidden,
                              'fp32', matrix_dtype, batch=batch, calls=spec.draft_layers,
                              output_layout='slot_hsd'),
            OperatorSignature('cfm', 'sdpa', 'attention', m, 64, spec.cfm_head_frames,
                              'fp32', 'fp32', batch=batch, calls=spec.cfm_layers * 2,
                              input_layout='bhqd', weight_layout='bhkd', output_layout='bqhd',
                              metadata=(('softmax', 'fp32'),)),
        ))
        for phase, length in (('prefill', 48), ('latent', 80)):
            phase_m = batch * length
            for role, n, k in (
                ('qkv', spec.target_qkv, spec.target_hidden), ('out', spec.target_hidden, spec.target_hidden),
                ('up', spec.target_ffn, spec.target_hidden), ('down', spec.target_hidden, spec.target_ffn),
            ):
                result.append(_gemm('target', f'{phase}_{role}', batch, phase_m, n, k, matrix_dtype,
                                    calls=spec.target_layers, metadata={'phase': phase, 'bucket': str(length)}))
        # Convolution signatures use implicit-GEMM M/N/K coordinates.
        for role, frames, ci, co, kw, calls, dtype in (
            ('wavenet_k5', 310, 512, 1024, 5, 8, matrix_dtype),
            ('wavenet_k1', 310, 512, 1024, 1, 7, matrix_dtype),
            ('residual_768_k3', 208, 768, 768, 3, 6, 'bf16'),
            ('residual_768_k7', 208, 768, 768, 7, 6, 'bf16'),
            ('residual_768_k11', 208, 768, 768, 11, 6, 'bf16'),
            ('residual_384_k3', 832, 384, 384, 3, 6, 'bf16'),
            ('residual_384_k7', 832, 384, 384, 7, 6, 'bf16'),
            ('residual_384_k11', 832, 384, 384, 11, 6, 'bf16'),
            ('residual_192_k3', 1664, 192, 192, 3, 6, matrix_dtype),
            ('residual_192_k7', 1664, 192, 192, 7, 6, matrix_dtype),
            ('residual_192_k11', 1664, 192, 192, 11, 6, matrix_dtype),
            ('residual_96_k3', 3328, 96, 96, 3, 6, matrix_dtype),
            ('residual_96_k7', 3328, 96, 96, 7, 6, matrix_dtype),
            ('residual_96_k11', 3328, 96, 96, 11, 6, matrix_dtype),
            ('residual_48_k3', 6656, 48, 48, 3, 6, matrix_dtype),
            ('residual_48_k7', 6656, 48, 48, 7, 6, matrix_dtype),
            ('residual_48_k11', 6656, 48, 48, 11, 6, matrix_dtype),
            ('residual_24_k3', 13312, 24, 24, 3, 6, matrix_dtype),
            ('residual_24_k7', 13312, 24, 24, 7, 6, matrix_dtype),
            ('residual_24_k11', 13312, 24, 24, 11, 6, matrix_dtype),
        ):
            result.append(OperatorSignature(
                model='cfm' if role.startswith('wavenet') else 'vocoder', role=role,
                kind='conv', m=batch * frames, n=co, k=ci * kw,
                input_dtype=dtype, weight_dtype=dtype, batch=batch, calls=calls,
                input_layout='ntc', weight_layout='kn', output_layout='nct',
                metadata=(('frames', str(frames)), ('kernel', str(kw))),
            ))
        result.extend((
            OperatorSignature('vocoder', 'conv_pre', 'conv', batch * 52, 1536, 80 * 7,
                              'bf16', 'bf16', batch=batch, calls=1,
                              input_layout='ntc', weight_layout='kn', output_layout='nct'),
            OperatorSignature('vocoder', 'conv_post', 'conv', batch * 13312, 1, 24 * 7,
                              'bf16', 'bf16', batch=batch, calls=1,
                              input_layout='ntc', weight_layout='kn', output_layout='nct'),
            OperatorSignature('vocoder', 'upsample', 'conv', batch * 52, 768, 1536 * 8,
                              'bf16', 'bf16', batch=batch, calls=6,
                              input_layout='nct', weight_layout='cok', output_layout='nct',
                              metadata=(('backend_class', 'cudnn'), ('transpose', 'true'))),
            OperatorSignature('vocoder', 'alias_free', 'pointwise', batch * 24 * 13312, 1, 1,
                              'fp32', 'fp32', batch=batch, calls=109,
                              input_layout='nct', output_layout='nct',
                              epilogue=('upsample_filter', 'snake', 'downsample_filter')),
            OperatorSignature('pipeline', 'first_chunk_graph', 'control', batch, 52, 1,
                              'fp32', 'fp32', batch=batch, calls=1,
                              metadata=(('cfm_frames', '52'), ('prefill_bucket', '48'), ('latent_bucket', '80'))),
        ))
    return result


def tunable(signature: OperatorSignature) -> bool:
    """Only matrix-like work is eligible for BM/BN/BK formula generation."""
    return signature.kind in ('gemm', 'conv') and signature.role != 'upsample'


def keyed(signatures: list[OperatorSignature]) -> dict[str, OperatorSignature]:
    result = {}
    for signature in signatures:
        key = f'{signature.role_key}:{signature.shape_key}'
        if key in result:
            raise ValueError(f'Duplicate operator signature {key}')
        result[key] = signature
    return result
