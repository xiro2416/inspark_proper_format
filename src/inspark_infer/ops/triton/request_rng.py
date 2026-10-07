"""Counter-based, request-local random draws for device speculative rounds.

Each draw is a pure function of request seed, committed round and draw purpose.
Rows that have reached the head boundary do not advance their round counter.
"""
import torch
import triton
import triton.language as tl


def seed32(value):
    """Fold a request's full 64-bit seed into the device PRNG key."""
    value=int(value)&0xffffffffffffffff
    mixed=((value&0xffffffff)^(((value>>32)*0x9E3779B9)&0xffffffff))&0xffffffff
    return mixed if mixed<0x80000000 else mixed-0x100000000


@triton.jit
def _uniform(SEEDS, ROUNDS, OUT, N: tl.constexpr, PURPOSE: tl.constexpr,
             BLOCK: tl.constexpr, EXPONENTIAL: tl.constexpr):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    col = tile * BLOCK + tl.arange(0, BLOCK)
    seed = tl.load(SEEDS + row).to(tl.uint32)
    round_number = tl.load(ROUNDS + row).to(tl.uint32)
    mixed = seed ^ (round_number * 0x9E3779B9) ^ PURPOSE
    value = tl.rand(mixed, col)
    if EXPONENTIAL:
        value = -tl.log(tl.maximum(1.0 - value, 1.0e-7))
    tl.store(OUT + row * N + col, value, col < N)


def draw(seeds, rounds, shape, purpose, exponential=False):
    """Return independent float32 [B,...] draws with no host RNG state access."""
    if seeds.ndim != 1 or rounds.shape != seeds.shape or shape[0] != seeds.numel():
        raise ValueError('Request RNG batch mismatch')
    count = 1
    for extent in shape[1:]:
        count *= extent
    output = torch.empty(shape, device=seeds.device, dtype=torch.float32)
    _uniform[(seeds.numel(), triton.cdiv(count, 1024))](
        seeds, rounds, output, count, int(purpose), 1024, bool(exponential), num_warps=4)
    return output


def categorical(probabilities, uniform):
    """Inverse-CDF categorical draw; supports one or many draws per row."""
    if probabilities.ndim != 2 or uniform.ndim not in (1, 2):
        raise ValueError('Categorical draw shape mismatch')
    squeezed = uniform.ndim == 1
    if squeezed:
        uniform = uniform[:, None]
    if uniform.shape[0] != probabilities.shape[0]:
        raise ValueError('Categorical row mismatch')
    cumulative = probabilities.float().cumsum(-1).contiguous()
    mass = cumulative[:, -1:].clamp_min(1e-12)
    threshold = (uniform * mass).contiguous()
    result = torch.searchsorted(cumulative, threshold).clamp_max(probabilities.shape[1] - 1)
    return result[:, 0] if squeezed else result


def prepare(device,batches):
    """Compile all first-head draw signatures before serving requests."""
    for batch in batches:
        seeds=torch.arange(batch,device=device,dtype=torch.int32)
        rounds=torch.zeros_like(seeds)
        draw(seeds,rounds,(batch,7,8194),0x1537A105,exponential=True)
        for purpose in (0x1537A106,0x1537A107):
            draw(seeds,rounds,(batch,7),purpose)
        for purpose in (0x1537A108,0x1537A109,0x1537A10A):
            draw(seeds,rounds,(batch,64),purpose)
        for purpose in (0x1537A10B,0x1537A10C,0x1537A10D,0x1537A10E):
            draw(seeds,rounds,(batch,),purpose)
    torch.cuda.current_stream().synchronize()
    return dict(batches=list(batches),online_compile=False,streams='request_seed_plus_committed_round')
