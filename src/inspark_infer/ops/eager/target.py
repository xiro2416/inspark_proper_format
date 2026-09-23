"""Cache-format adapter for compiling the unchanged Transformers Target body."""
from transformers.cache_utils import Cache, DynamicCache


class TargetCacheAdapter:
    """Move GPT2's legacy-cache conversion ahead of its transformer call.

    Transformers 4.52.1 converts tuples to DynamicCache internally, but its
    deprecation logger is not fullgraph-traceable. Perform the same conversion
    without that logging side effect, call the original body, and restore its
    legacy output format. No model math, weights or attention masks are copied.

    Each tuple/None call gets a fresh cache container. Existing Cache inputs
    retain the original in-place Cache API semantics; tuple input tensors are
    not modified by DynamicCache's concatenating updates.
    """

    def __init__(self, eager):
        self.eager = eager

    def __call__(self, embeddings, past_key_values, attention_mask, cache_position):
        if hasattr(past_key_values, "to_heads_first"):
            past_key_values = past_key_values.to_heads_first()
        return_legacy = not isinstance(past_key_values, Cache)
        cache = (DynamicCache.from_legacy_cache(past_key_values)
                 if return_legacy else past_key_values)
        logits, cache, selected, final = self.eager(
            embeddings, cache, attention_mask, cache_position)
        if return_legacy:
            # Match GPT2Model.forward's self-attention-only legacy return.
            if hasattr(cache, "self_attention_cache"):
                cache = cache.self_attention_cache
            cache = cache.to_legacy_cache()
        return logits, cache, selected, final
