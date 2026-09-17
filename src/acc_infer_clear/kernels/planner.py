"""Explainable hardware-resource pruning, not a promise of analytic optimality."""
from dataclasses import dataclass,asdict
import hashlib,json,math

@dataclass(frozen=True)
class DeviceCaps:
    name:str
    sm:int
    sms:int
    shared_per_sm:int
    shared_per_cta:int
    registers_per_sm:int
    threads_per_sm:int
    max_ctas_per_sm:int=32
    @property
    def native_fp8(self):return self.sm in (89,90,100,103,110,120,121)
    @classmethod
    def current(cls):
        import torch
        p=torch.cuda.get_device_properties(0)
        return cls(p.name,p.major*10+p.minor,p.multi_processor_count,
                   p.shared_memory_per_multiprocessor,p.shared_memory_per_block_optin,
                   p.regs_per_multiprocessor,p.max_threads_per_multi_processor,
                   getattr(p,'max_blocks_per_multi_processor',32))

@dataclass(frozen=True)
class OpSignature:
    m:int
    n:int
    k:int
    dtype:str='fp8'
    phase:str='unknown'
    layout:str='row_major'
    def __post_init__(self):
        if min(self.m,self.n,self.k)<=0:raise ValueError('Positive M/N/K required')

@dataclass(frozen=True)
class Tile:
    bm:int
    bn:int
    bk:int
    warps:int
    stages:int
    split_k:int=1
    schedule:str='tiled'

def estimate(caps,op,tile,compiled_registers=None,compiled_shared=None):
    """All memory sizes in bytes, registers in32-bit words. Approximate until compiled."""
    a_bytes=b_bytes=1 if op.dtype=='fp8' else 2
    shared=compiled_shared if compiled_shared is not None else tile.stages*(tile.bm*tile.bk*a_bytes+tile.bk*tile.bn*b_bytes)
    # Accumulator lower bound + explicit conservative operand/index allowance.
    registers=compiled_registers if compiled_registers is not None else math.ceil(tile.bm*tile.bn/(32*tile.warps))+32
    reasons=[]
    if op.dtype=='fp8' and not caps.native_fp8:reasons.append('no_native_fp8')
    if shared>caps.shared_per_cta:reasons.append('shared_per_cta')
    if registers>255:reasons.append('registers_per_thread')
    if tile.split_k>math.ceil(op.k/tile.bk):reasons.append('empty_split_k_partition')
    if tile.bk%32:reasons.append('mma_k_alignment')
    resident=min(caps.registers_per_sm//max(1,registers*32*tile.warps),
                 caps.shared_per_sm//max(1,shared),caps.threads_per_sm//(32*tile.warps),caps.max_ctas_per_sm)
    if resident<1:reasons.append('no_resident_cta')
    jobs=math.ceil(op.m/tile.bm)*math.ceil(op.n/tile.bn)*tile.split_k
    capacity=caps.sms*max(1,resident)
    wave_efficiency=jobs/(math.ceil(jobs/capacity)*capacity)
    tile_efficiency=(op.m*op.n*op.k)/(math.ceil(op.m/tile.bm)*tile.bm*math.ceil(op.n/tile.bn)*tile.bn*math.ceil(op.k/tile.bk)*tile.bk)
    # Ranking proxy only. Resource allocation and latency must be checked after compilation.
    score=tile_efficiency*wave_efficiency/(1+.12*(tile.split_k-1))
    return dict(legal=not reasons,reasons=reasons,shared=shared,registers_per_thread=registers,
                resident_ctas=resident,jobs=jobs,wave_efficiency=wave_efficiency,
                tile_efficiency=tile_efficiency,ranking_proxy=score,compiled=compiled_registers is not None)

def candidates(caps,op,limit=8,split_k=False):
    rows=[]
    for bm in (16,32,64):
        for bn in (32,64,128):
            for bk in (32,64,128):
                for warps in (4,8):
                    for stages in (1,2,3):
                        for splits in ((1,2,4) if split_k else (1,)):
                            tile=Tile(bm,bn,bk,warps,stages,splits)
                            info=estimate(caps,op,tile)
                            if info['legal']:rows.append((tile,info))
    ordered=sorted(rows,key=lambda x:x[1]['ranking_proxy'],reverse=True)
    # Keep arithmetic-shape diversity: occupancy proxy alone overfavours tiny BM.
    selected=[]
    for bm in (16,32,64):
        match=next((row for row in ordered if row[0].bm==bm),None)
        if match is not None and len(selected)<limit:selected.append(match)
    if split_k:
        for splits in (1,2,4):
            match=next((row for row in ordered if row[0].split_k==splits),None)
            if match is not None and match not in selected and len(selected)<limit:selected.append(match)
    for row in ordered:
        if len(selected)>=limit:break
        if row not in selected:selected.append(row)
    return selected

@dataclass(frozen=True)
class Rates:
    tensor_flops_s:float
    dram_bytes_s:float
    l2_bytes_s:float
    launch_s:float

def latency_model(caps,op,tile,rates,l2_hit_rate,quant_s=0.,reduction_s=0.,shared_s=None,dependency_s=None):
    """Calibrated ranking estimate. Missing dependency/shared evidence is explicit."""
    if not 0<=l2_hit_rate<=1:raise ValueError('Invalid L2 hit rate')
    resources=estimate(caps,op,tile)
    if not resources['legal']:return dict(legal=False,reasons=resources['reasons'])
    size=1 if op.dtype=='fp8' else 2
    mt=math.ceil(op.m/tile.bm);nt=math.ceil(op.n/tile.bn);kt=math.ceil(op.k/tile.bk)
    useful=2*op.m*op.n*op.k
    executed=2*mt*tile.bm*nt*tile.bn*kt*tile.bk
    unique=(op.m*op.k+op.k*op.n)*size
    requested=mt*nt*(tile.bm*op.k+tile.bn*op.k)*size
    output=op.m*op.n*4;workspace=2*tile.split_k*output if tile.split_k>1 else 0
    dram=unique+(1-l2_hit_rate)*max(0,requested-unique)+output+workspace
    terms=dict(tensor=executed/(rates.tensor_flops_s*max(resources['wave_efficiency'],1e-9)),
               dram=dram/rates.dram_bytes_s,l2=(requested+output+workspace)/rates.l2_bytes_s,
               shared=shared_s,dependency=dependency_s)
    bound=max(v for v in terms.values() if v is not None)
    return dict(legal=True,estimate_s=rates.launch_s+quant_s+bound+reduction_s,terms_s=terms,
                useful_flops=useful,executed_flops=executed,estimated_dram_bytes=dram,
                missing_terms=[k for k,v in terms.items() if v is None],not_an_optimality_guarantee=True)

def cache_key(caps,op,source_hash,software):
    payload=dict(device=asdict(caps),op=asdict(op),source=source_hash,software=software)
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
