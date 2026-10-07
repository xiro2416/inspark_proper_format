"""Cross-request PCG work; preserve each request's independent RNG stream.

Sampling order per request is unchanged. Dense group-mass work and host decision
copies are batched. Rare enumeration fallback retains the original algorithm.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _coarse_residual(Q, P, MEMBERS, WEIGHTS, NEED, OUT,
                     V: tl.constexpr, G: tl.constexpr, W: tl.constexpr,
                     GROUP_BLOCK: tl.constexpr, WIDTH_BLOCK: tl.constexpr):
    row = tl.program_id(0)
    group = tl.program_id(1) * GROUP_BLOCK + tl.arange(0, GROUP_BLOCK)
    offsets = tl.arange(0, WIDTH_BLOCK)
    valid = (group[:, None] < G) & (offsets[None, :] < W)
    if tl.load(NEED + row):
        member = tl.load(MEMBERS + group[:, None] * W + offsets[None, :], valid, 0)
        weight = tl.load(WEIGHTS + group[:, None] * W + offsets[None, :], valid, 0).to(tl.float32)
        q = tl.load(Q + row * V + member, valid, 0)
        p = tl.load(P + row * V + member, valid, 0)
        mass = tl.maximum(tl.sum((q - p) * weight, 1), 0.)
        tl.store(OUT + row * G + group, mass, group < G)
    else:
        tl.store(OUT + row * G + group, 0., group < G)

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

class DeviceAcceptance:
    """Acceptance tensors and compact prefix decisions remain device resident."""
    def __init__(self,q,accept,packed,plan):
        self.q,self.accept,self.packed,self.plan=q,accept,packed,plan
    def host_plan(self):
        # Transitional boundary: one compact [B,4] copy replaces [B,7,5].
        return torch.stack((self.plan[0],self.plan[1].int(),
                            self.plan[2].int(),self.plan[3].int()),1).cpu().tolist()

class BatchedAcceptance:

    def __init__(self, groups, temperature=0.8):
        self.groups = groups
        self.temperature = temperature
        self.fused=False
        self.device_plan=False
    def prepare_fusion(self):
        from inspark_infer.ops.triton.acceptance import acceptance
        g=self.groups;v=g.token_group_counts.numel();device=g.token_groups.device
        args=(torch.zeros(1,7,v,device=device),torch.ones(1,7,v,device=device)/v,
              torch.zeros(1,7,device=device,dtype=torch.long),torch.full((1,7),.5,device=device),torch.full((1,7),.5,device=device))
        acceptance(g,self.temperature,*args);self.fused=True
    def prepare_device_plan(self,eos,max_tokens):
        if not self.fused:raise RuntimeError('Device plan requires fused acceptance')
        from inspark_infer.ops.triton.acceptance import prefix_plan
        device=self.groups.token_groups.device
        packed=torch.zeros(1,7,5,device=device);remaining=torch.full((1,),7,device=device,dtype=torch.int32);current=torch.ones(1,device=device,dtype=torch.int32)
        prefix_plan(packed,remaining,current,eos,max_tokens)
        self.device_plan=True;self.device_eos=int(eos);self.device_max_tokens=int(max_tokens)
        return dict(k=7,compact_host_values_per_row=4,packed_host_values_before=35,
                    online_compile=False,semantics='accepted_prefix_exact')

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
            from inspark_infer.ops.triton.acceptance import acceptance
            fn=lambda *args:acceptance(self.groups,self.temperature,*args)
        else:fn = self.tensor_body
        q, accept, packed = fn(logits, p, tokens, gd, ad)
        if self.device_plan:
            from inspark_infer.ops.triton.acceptance import prefix_plan
            remaining=torch.tensor([min(k,self.device_max_tokens-len(task.codes)) for task in tasks],device=logits.device,dtype=torch.int32)
            current=torch.tensor([len(task.codes) for task in tasks],device=logits.device,dtype=torch.int32)
            return DeviceAcceptance(q,accept,packed,prefix_plan(
                packed,remaining,current,self.device_eos,self.device_max_tokens))
        packed = packed.cpu().tolist()
        return [(q[i].clone(), accept[i].clone(), HostDecisions(packed[i])) for i in range(b)]

class BatchedResidual:

    def __init__(self, groups, original):
        self.groups = groups
        self.original = original
        self.fallbacks = 0
        self.device_normal=False
        self.device_failures=None

    def prepare_device_normal(self):
        """Prepare request-owned GPU thinning with exact coarse fallback."""
        device=self.groups.group_members.device
        self.device_failures=torch.zeros((),device=device,dtype=torch.int32)
        self.device_enumerations=torch.zeros((),device=device,dtype=torch.int32)
        self.device_normal=True
        vocab=self.groups.sparse.vocab_size
        q=torch.full((1,7,vocab),1./vocab,device=device)
        gen=torch.Generator(device=device).manual_seed(0x7E517E51)
        self.device_batch(q,q,torch.zeros(1,device=device,dtype=torch.int32),
                          torch.zeros(1,device=device,dtype=torch.bool),[gen])
        torch.cuda.current_stream().synchronize()
        return dict(max_thinning_attempts=64,decision_d2h=False,
                    fallback_policy='exact coarse-residual enumeration on GPU',
                    final_group_sampling='explicit request-owned uniform',online_compile=False)

    def device_batch(self,q_block,p_block,indices,mask,generators):
        """Fixed-B request-owned PCG residual with no host decision copy."""
        b,k,v=q_block.shape;row=torch.arange(b,device=q_block.device)
        at=indices.long().clamp(0,k-1);q=q_block[row,at].float();p=p_block[row,at].float()
        q=q/q.sum(-1,keepdim=True).clamp_min(1e-12);p=p/p.sum(-1,keepdim=True).clamp_min(1e-12)
        n=64
        ids=torch.stack([self.groups.sample_groups(
            torch.multinomial(q[i],n,replacement=True,generator=gen),generator=gen)
            for i,gen in enumerate(generators)])
        uniform=torch.stack([torch.rand(n,device=q.device,generator=gen) for gen in generators])
        conditional,members,decisions=self.tensor_body(q,p,ids,uniform)
        need_exact=mask&(~decisions[:,0].bool())
        self.device_enumerations.add_(need_exact.sum())
        draw=torch.stack([torch.rand((),device=q.device,generator=gen) for gen in generators])
        mass=conditional.sum(-1).clamp_min(1e-12);chosen=(conditional.cumsum(-1)<(draw*mass)[:,None]).sum(-1).clamp_max(conditional.shape[1]-1)
        fast_token=members.gather(1,chosen[:,None]).squeeze(1)
        g=self.groups;group_count=g.group_members.shape[0];width=g.group_members.shape[1]
        residual=torch.empty((b,group_count),device=q.device,dtype=torch.float32)
        _coarse_residual[(b,triton.cdiv(group_count,16))](q,p,g.group_members,g.group_weights,
            need_exact,residual,v,group_count,width,16,triton.next_power_of_2(width),num_warps=4)
        group_mass=residual.sum(-1)
        group_draw=torch.stack([torch.rand((),device=q.device,generator=gen) for gen in generators])
        selected=(residual.cumsum(-1)<(group_draw*group_mass)[:,None]).sum(-1).clamp_max(group_count-1)
        exact_members=g.group_members[selected]
        exact_weights=g.group_weights[selected]
        exact_conditional=q.gather(1,exact_members)*exact_weights
        exact_mass=exact_conditional.sum(-1)
        member_draw=torch.stack([torch.rand((),device=q.device,generator=gen) for gen in generators])
        member_index=(exact_conditional.cumsum(-1)<(member_draw*exact_mass)[:,None]).sum(-1).clamp_max(width-1)
        exact_token=exact_members.gather(1,member_index[:,None]).squeeze(1)
        q_index=(q.cumsum(-1)<member_draw[:,None]).sum(-1).clamp_max(v-1)
        exact_token=torch.where(group_mass>1e-12,exact_token,q_index)
        invalid=need_exact&(~torch.isfinite(exact_mass))
        self.device_failures.add_(invalid.sum())
        return torch.where(need_exact,exact_token,fast_token),decisions

    def device_batch_draws(self,q_block,p_block,indices,mask,draws):
        """Request-local GPU residual sampling without per-row host generators."""
        from inspark_infer.ops.triton.request_rng import categorical
        b,k,v=q_block.shape;row=torch.arange(b,device=q_block.device)
        at=indices.long().clamp(0,k-1)
        q=q_block[row,at].float();p=p_block[row,at].float()
        q=q/q.sum(-1,keepdim=True).clamp_min(1e-12)
        p=p/p.sum(-1,keepdim=True).clamp_min(1e-12)
        candidates=categorical(q,draws['candidates'])
        counts=self.groups.token_group_counts[candidates]
        choices=(draws['groups']*counts.float()).long().clamp_min(0)
        ids=self.groups.token_groups[candidates,choices]
        conditional,members,decisions=self.tensor_body(q,p,ids,draws['thin'])
        need_exact=mask&(~decisions[:,0].bool())
        self.device_enumerations.add_(need_exact.sum())
        mass=conditional.sum(-1).clamp_min(1e-12)
        chosen=(conditional.cumsum(-1)<(draws['fast']*mass)[:,None]).sum(-1).clamp_max(conditional.shape[1]-1)
        fast_token=members.gather(1,chosen[:,None]).squeeze(1)
        g=self.groups;group_count=g.group_members.shape[0];width=g.group_members.shape[1]
        residual=torch.empty((b,group_count),device=q.device,dtype=torch.float32)
        _coarse_residual[(b,triton.cdiv(group_count,16))](q,p,g.group_members,g.group_weights,
            need_exact,residual,v,group_count,width,16,triton.next_power_of_2(width),num_warps=4)
        group_mass=residual.sum(-1)
        selected=(residual.cumsum(-1)<(draws['coarse']*group_mass)[:,None]).sum(-1).clamp_max(group_count-1)
        exact_members=g.group_members[selected];exact_weights=g.group_weights[selected]
        exact_conditional=q.gather(1,exact_members)*exact_weights
        exact_mass=exact_conditional.sum(-1)
        member_index=(exact_conditional.cumsum(-1)<(draws['member']*exact_mass)[:,None]).sum(-1).clamp_max(width-1)
        exact_token=exact_members.gather(1,member_index[:,None]).squeeze(1)
        q_index=categorical(q,draws['member'])
        exact_token=torch.where(group_mass>1e-12,exact_token,q_index)
        invalid=need_exact&(~torch.isfinite(exact_mass))
        self.device_failures.add_(invalid.sum())
        return torch.where(need_exact,exact_token,fast_token),decisions

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
        if self.device_normal:
            ok=decisions[:,0].bool();self.device_failures.add_((~ok).sum())
            # One explicit request-owned uniform per row; padded invalid members
            # already have zero probability. This is distribution-equivalent to
            # categorical sampling, not bitwise torch.multinomial equivalence.
            draws=[]
            for task,gen in zip(tasks,gens):
                draws.append(torch.rand((),device=q.device,generator=gen));task.state['cuda_rng']=gen.get_state()
            draw=torch.stack(draws);mass=conditional.sum(-1).clamp_min(1e-12)
            index=(conditional.cumsum(-1)<(draw*mass)[:,None]).sum(-1).clamp_max(conditional.shape[1]-1)
            tokens=selected_members.gather(1,index[:,None]).squeeze(1)
            return [(tokens[i],decisions[i,2],False,decisions[i,1]+1) for i in range(b)]
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
