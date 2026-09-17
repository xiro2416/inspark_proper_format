"""Target math only: explicit hidden-state outputs; no backend selection."""
import torch

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
    def _take_hidden_states(self,hidden):
        return torch.cat([hidden[i+1] for i in self.target_layer_ids],dim=-1),hidden[-1]
    def _block_forward_with_hidden_states(self,embeddings,past_key_values,attention_mask,cache_position):
        if hasattr(past_key_values,'to_heads_first'):past_key_values=past_key_values.to_heads_first()
        body=self.model.transformer(inputs_embeds=embeddings,past_key_values=past_key_values,attention_mask=attention_mask,
            cache_position=cache_position,use_cache=True,output_hidden_states=True,return_dict=True)
        selected,final=self._take_hidden_states(body.hidden_states)
        return self.model.lm_head(body.last_hidden_state),body.past_key_values,selected,final
