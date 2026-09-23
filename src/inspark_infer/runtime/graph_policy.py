"""Small, explicit offline graph inventory for first-packet service."""

BATCHES=(1,2,3,4,5,6,7,8,16,32)
FIRST_KV_LIMITS=(64,128)

def batches(max_batch):
    """Supported capture sizes no larger than the configured admission batch."""
    return tuple(b for b in BATCHES if b<=max_batch)
