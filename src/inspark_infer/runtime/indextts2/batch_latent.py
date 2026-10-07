"""Ragged batch for the original GPT latent recomputation before S2M.

Pack each complete [conditioning,text,mel] row before right-padding. Padding
text separately would shift the mel prefix and silently change conditioning.
"""
import torch

class BatchedLatent:

    def __init__(self, gpt, original, body=None):
        self.gpt = gpt
        self.original = original
        self.calls = 0
        self.max_error = 0.0
        self.body = body

    @torch.inference_mode()
    def from_cached_prefix(self, ops):
        """Reuse the exact first-head prefill embedding prefix for latent GPT.

        The generated speech-code suffix is embedded once for the whole batch.
        Invalid padded suffix positions are masked before the static Target
        Graph. The head's cached prefix includes the mel BOS at position zero.
        """
        gpt=self.gpt
        count=len(ops)
        prefixes=[op.cached_prefix for op in ops]
        codes=[op.args[3] for op in ops]
        if not count or any(prefix is None or prefix.shape[0]!=1 or code.shape[0]!=1
                            for prefix,code in zip(prefixes,codes)):
            raise ValueError('First-head latent requires one exact prefix and code row per request')
        token_lengths=[int(code.shape[1]) for code in codes]
        owner=getattr(self.body,'__self__',None)
        reuse=getattr(owner,'reused_latents',None)
        if callable(reuse):
            result=reuse(prefixes,codes,token_lengths)
            if result is not None:
                self.calls+=1
                return result
        lengths=[int(prefix.shape[1])+tokens+1 for prefix,tokens in zip(prefixes,token_lengths)]
        longest=max(lengths)
        device=codes[0].device
        token_matrix=torch.full((count,max(token_lengths)+1),gpt.stop_mel_token,
                                device=device,dtype=codes[0].dtype)
        owner=getattr(self.body,'__self__',None)
        vector_pack=owner is not None and getattr(getattr(owner,'engine',None),'latent_vector_pack',False)
        from inspark_infer.runtime.ragged_pack import shared_rows
        shared_codes=shared_rows(codes,max(token_lengths)) if vector_pack else None
        if shared_codes is not None:
            lens=torch.tensor(token_lengths,device=device)
            valid=torch.arange(shared_codes.shape[1],device=device)[None]<lens[:,None]
            token_matrix[:,:shared_codes.shape[1]].copy_(torch.where(valid,shared_codes,gpt.stop_mel_token))
        else:
            for index,code in enumerate(codes):
                token_matrix[index,:token_lengths[index]].copy_(code[0])
        positions=torch.arange(1,token_matrix.shape[1]+1,device=device)
        suffix=gpt.mel_embedding(token_matrix)+gpt.mel_pos_embedding.emb(positions)[None]
        owner=getattr(self.body,'__self__',None)
        backend=getattr(owner,'backends',{}).get('latent') if owner is not None else None
        direct=(backend is not None and callable(getattr(owner,'latent_cached_prefix',None))
                and count<=backend.batch and longest<=backend.extent)
        if direct:
            output=owner.latent_cached_prefix(prefixes,suffix,token_lengths)
        else:
            embeddings=prefixes[0].new_zeros(count,longest,prefixes[0].shape[-1])
            mask=torch.zeros(count,longest,device=device,dtype=torch.long)
            for index,(prefix,tokens,length) in enumerate(zip(prefixes,token_lengths,lengths)):
                end=prefix.shape[1]
                embeddings[index,:end].copy_(prefix[0])
                embeddings[index,end:length].copy_(suffix[index,:tokens+1])
                mask[index,:length]=1
            if self.body is None:
                output=gpt.gpt(inputs_embeds=embeddings,attention_mask=mask,return_dict=True,
                               output_attentions=False).last_hidden_state
                output=gpt.final_norm(output)
            else:
                output=self.body(embeddings,mask)
        result=[output[index:index+1,prefix.shape[1]-1:prefix.shape[1]-1+tokens].clone()
                for index,(prefix,tokens) in enumerate(zip(prefixes,token_lengths))]
        self.calls+=1
        return result

    @torch.inference_mode()
    def __call__(self, jobs):
        rows = []
        lengths = []
        offsets = []
        text_lengths = []
        mel_lengths = []
        for args, kwargs in jobs:
            names = ('speech_conditioning_inputs', 'first_inputs', 'first_head', 'second_inputs', 'second_head', 'get_attns', 'return_latent')
            j = dict(zip(names, args))
            j.update(kwargs)
            assert j.get('return_latent') and (not j.get('get_attns')) and (j['second_inputs'] is not None)
            parts = [j['speech_conditioning_inputs'], j['first_inputs'], j['second_inputs']]
            assert all((x.shape[0] == 1 for x in parts))
            row = torch.cat(parts, 1)
            rows.append(row)
            lengths.append(row.shape[1])
            offsets.append(parts[0].shape[1])
            text_lengths.append(parts[1].shape[1])
            mel_lengths.append(parts[2].shape[1])
        longest = max(lengths)
        embeddings = rows[0].new_zeros(len(rows), longest, rows[0].shape[-1])
        mask = torch.zeros(len(rows), longest, device=embeddings.device, dtype=torch.long)
        for i, row in enumerate(rows):
            embeddings[i, :lengths[i]].copy_(row[0])
            mask[i, :lengths[i]] = 1
        if self.body is None:
            output = self.gpt.gpt(inputs_embeds=embeddings, attention_mask=mask, return_dict=True, output_attentions=False).last_hidden_state
            output = self.gpt.final_norm(output)
        else:
            output = self.body(embeddings, mask)
        result = [(output[i:i + 1, o:o + t].clone(), output[i:i + 1, o + t:o + t + m].clone()) for i, (o, t, m) in enumerate(zip(offsets, text_lengths, mel_lengths))]
        self.calls += 1
        return result
