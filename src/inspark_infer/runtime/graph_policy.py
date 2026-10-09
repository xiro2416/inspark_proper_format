"""Small, explicit offline graph inventory for first-packet service."""

BATCHES=(1,2,4,8,16,24,32,40,48,56,64,128)
FIRST_KV_LIMITS=(64,128)

def batches(max_batch):
    """Supported capture sizes no larger than the configured admission batch."""
    # A large fixed-batch deployment captures only its requested size.
    # Capturing smaller sizes can exhaust GPU memory before the full batch is
    # admitted; nonmatching dynamic active sizes retain their eager fallback.
    return (max_batch,) if max_batch>24 else tuple(b for b in BATCHES if b<=max_batch)
