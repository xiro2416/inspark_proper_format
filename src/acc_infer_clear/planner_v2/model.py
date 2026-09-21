"""Canonical device, operator and schedule descriptions used by Planner V2."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal


DTYPE_BYTES = {'fp8': 1, 'bf16': 2, 'fp16': 2, 'tf32': 4, 'fp32': 4}


@dataclass(frozen=True)
class MeasuredRates:
    tensor_flops_s: float
    dram_bytes_s: float
    l2_bytes_s: float
    shared_bytes_s: float
    launch_s: float
    global_latency_s: float

    def __post_init__(self):
        if min(asdict(self).values()) <= 0:
            raise ValueError('All measured rates and latencies must be positive')


@dataclass(frozen=True)
class HardwareProfile:
    name: str
    sm: int
    sms: int
    warp_size: int
    shared_per_sm: int
    shared_per_cta: int
    registers_per_sm: int
    max_registers_per_thread: int
    threads_per_sm: int
    max_ctas_per_sm: int
    l2_bytes: int
    memory_bus_width: int = 0
    rates: MeasuredRates | None = None
    software: dict[str, str] = field(default_factory=dict)

    @property
    def native_fp8(self) -> bool:
        return self.sm in (89, 90, 100, 103, 110, 120, 121)

    @property
    def async_copy(self) -> bool:
        return self.sm >= 80

    @property
    def tma(self) -> bool:
        return self.sm >= 90

    @property
    def preferred_matrix_dtype(self) -> str:
        return 'fp8' if self.native_fp8 else 'bf16'

    @classmethod
    def current(cls) -> 'HardwareProfile':
        import torch
        p = torch.cuda.get_device_properties(0)
        return cls(
            name=p.name, sm=p.major * 10 + p.minor,
            sms=p.multi_processor_count, warp_size=p.warp_size,
            shared_per_sm=p.shared_memory_per_multiprocessor,
            shared_per_cta=p.shared_memory_per_block_optin,
            registers_per_sm=p.regs_per_multiprocessor,
            max_registers_per_thread=255,
            threads_per_sm=p.max_threads_per_multi_processor,
            max_ctas_per_sm=getattr(p, 'max_blocks_per_multi_processor', 32),
            l2_bytes=p.L2_cache_size,
            memory_bus_width=getattr(p, 'memory_bus_width', 0),
            software={
                'torch': torch.__version__, 'cuda': str(torch.version.cuda),
            },
        )

    @classmethod
    def synthetic(cls, sm: int, *, sms: int = 80) -> 'HardwareProfile':
        """Conservative profiles for formula/unit testing, never performance claims."""
        if sm not in (80, 86, 89, 90, 120):
            raise ValueError('Unsupported synthetic SM family')
        shared = 164 * 1024 if sm in (80, 90) else (100 * 1024 if sm in (86, 89) else 100 * 1024)
        return cls(
            name=f'synthetic-sm{sm}', sm=sm, sms=sms, warp_size=32,
            shared_per_sm=shared, shared_per_cta=min(shared, 99 * 1024),
            registers_per_sm=65536, max_registers_per_thread=255,
            threads_per_sm=2048 if sm in (80, 86, 89) else 1536,
            max_ctas_per_sm=32, l2_bytes=0,
            software={'status': 'synthetic-no-performance-claim'},
        )

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class OperatorSignature:
    model: str
    role: str
    kind: Literal['gemm', 'conv', 'attention', 'pointwise', 'layout', 'control']
    m: int
    n: int
    k: int
    input_dtype: str
    weight_dtype: str
    accumulator_dtype: str = 'fp32'
    input_layout: str = 'mk'
    weight_layout: str = 'nk'
    output_layout: str = 'mn'
    epilogue: tuple[str, ...] = ()
    batch: int = 1
    calls: int = 1
    critical_weight: float = 1.0
    graph_eligible: bool = True
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self):
        if min(self.m, self.n, self.k, self.batch, self.calls) <= 0:
            raise ValueError('Positive shapes, batch and calls required')
        for dtype in (self.input_dtype, self.weight_dtype, self.accumulator_dtype):
            if dtype not in DTYPE_BYTES:
                raise ValueError(f'Unsupported dtype {dtype}')
        if self.critical_weight <= 0:
            raise ValueError('critical_weight must be positive')

    @property
    def role_key(self) -> str:
        return f'{self.model}:{self.role}'

    @property
    def shape_key(self) -> str:
        return f'b{self.batch}_m{self.m}_n{self.n}_k{self.k}'

    def as_dict(self) -> dict:
        result = asdict(self)
        result['epilogue'] = list(self.epilogue)
        result['metadata'] = dict(self.metadata)
        return result


@dataclass(frozen=True)
class ScheduleSpec:
    backend: str
    schedule: Literal['tiled', 'full_m', 'persistent', 'split_k']
    bm: int
    bn: int
    bk: int
    warps: int
    stages: int
    split_k: int = 1
    warp_m: int = 1
    warp_n: int = 4
    global_layout: str = 'mk_nk'
    shared_layout_a: str = 'xor_k'
    shared_layout_b: str = 'xor_n'
    swizzle_bytes: int = 16
    activation_eviction: str = ''
    weight_eviction: str = ''
    fused_epilogue: tuple[str, ...] = ()

    def __post_init__(self):
        if min(self.bm, self.bn, self.bk, self.warps, self.stages, self.split_k) <= 0:
            raise ValueError('Positive schedule parameters required')
        if self.warps not in (1, 2, 4, 8):
            raise ValueError('Unsupported warp count')
        if self.warp_m * self.warp_n != self.warps:
            raise ValueError('warp_m*warp_n must equal warps')
        if self.schedule != 'split_k' and self.split_k != 1:
            raise ValueError('split_k>1 requires split_k schedule')

    def as_dict(self) -> dict:
        result = asdict(self)
        result['fused_epilogue'] = list(self.fused_epilogue)
        return result
