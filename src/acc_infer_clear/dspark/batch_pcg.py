"""Cross-request PCG work; preserve each request's independent RNG stream.

Sampling order per request is unchanged. Dense group-mass work and host decision
copies are batched. Rare enumeration fallback retains the original algorithm.
"""
import torch

def generator_for(task, device):
    gen = torch.Generator(device=device)
    gen.set_state(task.state['cuda_rng'])
    return gen

class HostDecisions:

    def __init__(self, rows):
        self.rows = rows

    def cpu(self):
        return self

    def tolist(self):
        return self.rows

class BatchedAcceptance:

    def __init__(self, groups, temperature=0.8):
        self.groups = groups
        self.temperature = temperature
        self.fused=False
    def prepare_fusion(self):
        from acc_infer_clear.kernels.acceptance import acceptance
        g=self.groups;v=g.token_group_counts.numel();device=g.token_groups.device
        args=(torch.zeros(1,7,v,device=device),torch.ones(1,7,v,device=device)/v,
              torch.zeros(1,7,device=device,dtype=torch.long),torch.full((1,7),.5,device=device),torch.full((1,7),.5,device=device))
        acceptance(g,self.temperature,*args);self.fused=True

    def tensor_body(self, logits, p, tokens, group_draws, accept_draws):
        g = self.groups
        b, k = tokens.shape
        q = torch.softmax(logits.float() / self.temperature, -1)
        q = q / q.sum(-1, keepdim=True).clamp_min(1e-12)
        p = p / p.sum(-1, keepdim=True).clamp_min(1e-12)
        counts = g.token_group_counts[tokens]
        choices = (group_draws * counts.float()).floor().long()
        ids = g.token_groups[tokens, choices]
        qmass = g.group_masses(q.flatten(0, 1), ids.flatten()).view(b, k)
        pmass = g.group_masses(p.flatten(0, 1), ids.flatten()).view(b, k)
        accept = (qmass / pmass.clamp_min(1e-12)).clamp(max=1.0)
        exact = (q.gather(2, tokens[:, :, None]).squeeze(-1) / p.gather(2, tokens[:, :, None]).squeeze(-1).clamp_min(1e-12)).clamp(max=1.0)
        flags = accept_draws < accept
        return (q, accept, torch.stack((flags.float(), tokens.float(), g.group_sizes[ids].float(), accept, exact), -1))

    @torch.inference_mode()
    def __call__(self, jobs, tasks):
        g = self.groups
        b = len(jobs)
        k = jobs[0][0].shape[0]
        logits = torch.stack([j[0] for j in jobs])
        p = torch.stack([j[1] for j in jobs]).float()
        tokens = torch.stack([j[2] for j in jobs])
        group_draws = []
        accept_draws = []
        for task in tasks:
            gen = generator_for(task, logits.device)
            group_draws.append(torch.rand(k, device=logits.device, generator=gen))
            accept_draws.append(torch.rand(k, device=logits.device, generator=gen))
            task.state['cuda_rng'] = gen.get_state()
        gd, ad = (torch.stack(group_draws), torch.stack(accept_draws))
        if self.fused:
            from acc_infer_clear.kernels.acceptance import acceptance
            fn=lambda *args:acceptance(self.groups,self.temperature,*args)
        else:fn = self.tensor_body
        q, accept, packed = fn(logits, p, tokens, gd, ad)
        packed = packed.cpu().tolist()
        return [(q[i].clone(), accept[i].clone(), HostDecisions(packed[i])) for i in range(b)]

class BatchedResidual:

    def __init__(self, groups, original):
        self.groups = groups
        self.original = original
        self.fallbacks = 0

    def tensor_body(self, q, p, ids, uniform):
        g = self.groups
        n = ids.shape[1]
        members = g.group_members[ids]
        weights = g.group_weights[ids]
        qr = q[:, None, :].expand(-1, n, -1).gather(2, members).mul(weights).sum(-1).clamp_min(1e-12)
        pr = p[:, None, :].expand(-1, n, -1).gather(2, members).mul(weights).sum(-1)
        success = uniform < (1 - pr / qr).clamp(0, 1)
        first = success.int().argmax(-1)
        chosen = ids.gather(1, first[:, None]).squeeze(1)
        selected_members = g.group_members[chosen]
        sizes = g.group_sizes[chosen]
        valid = torch.arange(selected_members.shape[1], device=q.device)[None] < sizes[:, None]
        conditional = q.gather(1, selected_members) / g.sparse.membership_count[selected_members].to(q.dtype)
        conditional = conditional.masked_fill(~valid, 0.0)
        conditional = torch.where((conditional.sum(-1) <= 1e-12)[:, None], valid.to(q.dtype), conditional)
        decisions = torch.stack((success.any(-1).long(), first, chosen, sizes), -1)
        return (conditional, selected_members, decisions)

    @torch.inference_mode()
    def __call__(self, jobs, tasks):
        g = self.groups
        attempts = [int(kw.get('max_thinning_attempts', 64)) for args, kw in jobs]
        assert len(set(attempts)) == 1
        n = attempts[0]
        b = len(jobs)
        q = torch.stack([a[0] for a, kw in jobs]).float()
        p = torch.stack([a[1] for a, kw in jobs]).float()
        q = q / q.sum(-1, keepdim=True).clamp_min(1e-12)
        p = p / p.sum(-1, keepdim=True).clamp_min(1e-12)
        gens = []
        group_ids = []
        uniform = []
        choices = []
        for i, task in enumerate(tasks):
            gen = generator_for(task, q.device)
            gens.append(gen)
            candidates = torch.multinomial(q[i], n, replacement=True, generator=gen)
            group_ids.append(g.sample_groups(candidates, generator=gen))
            uniform.append(torch.rand(n, device=q.device, generator=gen))
        ids = torch.stack(group_ids)
        uniform_tensor = torch.stack(uniform)
        fn = self.tensor_body
        conditional, selected_members, decisions = fn(q, p, ids, uniform_tensor)
        decisions = decisions.cpu().tolist()
        outputs = []
        for i, ((args, kw), task, gen, (ok, index, group, size)) in enumerate(zip(jobs, tasks, gens, decisions)):
            if ok:
                selected = torch.multinomial(conditional[i, :size], 1, generator=gen)
                token = selected_members[i, selected].squeeze(0)
                output = (token, group, False, index + 1)
            else:
                gen.set_state(task.state['cuda_rng'])
                output = self.original(*args, **dict(kw, generator=gen))
                self.fallbacks += 1
            task.state['cuda_rng'] = gen.get_state()
            outputs.append(output)
        return outputs
