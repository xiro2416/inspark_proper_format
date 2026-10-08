"""Pack normal finite Float32 constants to their existing TF32 RNA operand."""
import numpy as np

def pack_weight(weight):
    assert weight.dtype == np.float32 and np.isfinite(weight).all()
    magnitude = np.abs(weight)
    assert np.all((magnitude == 0) | (magnitude >= np.finfo(np.float32).tiny))
    assert magnitude.max() < np.float32(2.0 ** 126)
    bits = weight.copy().view(np.uint32)
    return (bits + np.uint32(4096) & np.uint32(4294959104)).view(np.float32).copy()
