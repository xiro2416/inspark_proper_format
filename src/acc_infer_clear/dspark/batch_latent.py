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

