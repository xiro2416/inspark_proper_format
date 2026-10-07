"""Target math only: explicit hidden-state outputs; no backend selection."""
import torch
from transformers.cache_utils import Cache, DynamicCache

def sample_logits(logits,temperature=.8,generator=None):
    p=torch.softmax(logits.float()/temperature,dim=-1)
    token=torch.multinomial(p,num_samples=1,generator=generator).squeeze(-1)
    return token,p

def crop_legacy_cache(cache,length):
    if hasattr(cache,'crop'):cache.crop(length);return cache
    return tuple((k[:,:,:length],v[:,:,:length]) for k,v in cache)

class IndexTTS2TargetEngine:
    def __init__(self,gpt,target_layer_ids):
        self.gpt=gpt;self.model=gpt.inference_model;self.target_layer_ids=target_layer_ids
        assert target_layer_ids==[1,6,11,16,21]
        # HF5 GPT2 removed its legacy tuple adapter. Keep this boundary local:
        # the vendored GPT2 and HF4 retain their existing cache semantics.
        self._hf_cache_only = (
            not hasattr(DynamicCache, 'from_legacy_cache')
            and any(cls.__module__ == 'transformers.models.gpt2.modeling_gpt2'
                    for cls in type(self.model.transformer).__mro__)
        )
    def _take_hidden_states(self,hidden):
        return torch.cat([hidden[i+1] for i in self.target_layer_ids],dim=-1),hidden[-1]
    def _block_forward_with_hidden_states(self,embeddings,past_key_values,attention_mask,cache_position):
        if hasattr(past_key_values,'to_heads_first'):past_key_values=past_key_values.to_heads_first()
        return_legacy = self._hf_cache_only and not isinstance(past_key_values, Cache)
        if return_legacy:
            # A fresh container leaves request-owned input tensors untouched;
            # DynamicCache appends via cat, while RequestKV owns commit/crop.
            past_key_values = DynamicCache(ddp_cache_data=past_key_values,
                                           config=self.model.transformer.config)
        body=self.model.transformer(inputs_embeds=embeddings,past_key_values=past_key_values,attention_mask=attention_mask,
            cache_position=cache_position,use_cache=True,output_hidden_states=True,return_dict=True)
        selected,final=self._take_hidden_states(body.hidden_states)
        output_cache = body.past_key_values
        if return_legacy:
            if hasattr(output_cache, 'self_attention_cache'):
                output_cache = output_cache.self_attention_cache
            # HF5 iteration emits (key, value, sliding_window), whereas the
            # request store and BatchedTarget deliberately consume KV pairs.
            output_cache = tuple((layer[0], layer[1]) for layer in output_cache)
        return self.model.lm_head(body.last_hidden_state),output_cache,selected,final
