"""Actual NVIDIA DSparkWorker driving fixed TensorRT Target/Draft graphs.

The NVIDIA worker owns and updates the Draft context arena. The Target cache
belongs to the TensorRT compute provider. PCG and linear_wide are model-level
extensions; this is not the high-level TRT-LLM LLM/Executor service.
"""
import torch

from .runtime import FrameworkRoundRuntime
from .native_bridge import NvidiaDraftInterface, NativePCGPolicy, native_pcg_worker_class


class NativeWorkerRoundRuntime(FrameworkRoundRuntime):
    backend = "native_dspark_worker_trt_compute"

    def __init__(self, **kwargs):
        from tensorrt_llm.llmapi.llm_args import DSparkDecodingConfig
        from tensorrt_llm.mapping import Mapping
        from tensorrt_llm._torch.speculative.dflash import DFlashSpecMetadata
        from tensorrt_llm._torch.speculative.interface import SpeculativeDecodingMode
        from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata

        super().__init__(**kwargs)
        self.provider = getattr(kwargs["draft_provider"], "__self__", None)
        if self.provider is None or not hasattr(self.provider, "draft_engine"):
            raise TypeError("Native runtime requires the fixed TensorRT compute provider")
        draft = self.provider.runtime.engine.draft
        if not draft.scratch_absolute_position:
            raise ValueError("This native provider mapping requires the audited absolute-position checkpoint")
        if any(type(module).__name__ == "MatrixLinear" and getattr(module, "precision", None) == "fp8"
               for module in draft.modules()):
            raise ValueError("Replace legacy custom FP8 math with the official/reference recipe before native capture")
        b = self.batch
        self.policy = NativePCGPolicy(
            proposal=self.proposal, groups=kwargs["groups"], seeds=[*kwargs["seeds"], 0],
            device=self.device, eos=self.eos, max_rounds=self.max_rounds, max_tokens=self.max_tokens,
            initial_token_lengths=torch.cat((self.token_lengths, self.token_lengths.new_ones(1))),
            initial_last_tokens=torch.cat((self.last, self.last.new_zeros(1))),
            ready_token_count=self.ready_token_count)
        self.pcg = self.policy.pcg
        self.draws = self.policy.draws
        self.token_lengths = self.policy.lengths[:b]
        self.rounds = self.policy.rounds[:b]
        self.last = self.policy.last[:b]
        self.ready = self.policy.ready[:b]
        self.committed = self.policy.last_committed[:b]
        self.failures = self.policy.failures
        self.native_model = NvidiaDraftInterface(
            draft, torch.cat((self.mel_lengths, self.mel_lengths.new_zeros(1))),
            forward_provider=self._draft_from_native_pool,
            logits_provider=lambda hidden: self.provider.draft_engine.outputs["base"].flatten(0, 1),
            context_projection_provider=(self.provider.project_native_context if hasattr(self.provider,'context_engine') and self.provider.context_engine.plan.get('kind')=='context' else None),
            context_kv_provider=(self.provider.native_context_kv if hasattr(self.provider,'context_engine') else None))
        self.mel_lengths = self.native_model.mel_lengths[:b]
        config = DSparkDecodingConfig(max_draft_len=7, block_size=7, markov_rank=256,
            markov_head_type="rnn", mask_token_id=draft.mask_token_id,
            target_layer_ids=draft.target_layer_ids, attention_backend="VANILLA")
        mapping = Mapping(world_size=1, rank=0, tp_size=1)
        self.worker = native_pcg_worker_class()(config, mapping, self.policy)
        self.worker.set_draft_model(self.native_model)
        self.attention = AttentionMetadata(max_num_requests=b, max_num_tokens=8*b,
            seq_lens=torch.full((b,), 8, dtype=torch.int32), num_contexts=0,
            seq_lens_kv=None, mapping=mapping)
        self.attention.max_seq_len = self.kv_capacity
        self.attention.kv_lens_cuda = self.past
        self.metadata = DFlashSpecMetadata(max_num_requests=b, max_draft_len=7,
            max_total_draft_tokens=7, num_generations=b, runtime_draft_len=7,
            runtime_tokens_per_gen_step=8, spec_dec_mode=SpeculativeDecodingMode.DSPARK,
            layers_to_capture=draft.target_layer_ids, hidden_size=draft.interface_size,
            max_num_tokens=8*b, dtype=torch.float32,
            request_ids=list(range(101, 101+b)), seq_lens=[8]*b)
        self.metadata.batch_indices_cuda.copy_(torch.arange(b, device=self.device, dtype=torch.int32))
        self.worker._lazy_init_ctx_buffers(self.native_model, self.metadata, self.attention, None)
        self.head_major_arena=getattr(self.provider,'head_major_arena',False)
        if self.head_major_arena:
            from .head_major_arena import install
            install(self.worker,b,self.kv_capacity)
        if hasattr(self.provider,'context_engine') and self.worker._ctx_k_buf.dtype!=torch.float32:
            raise ValueError('Context engine requires the existing FP32 native arena')
        for request_id in self.metadata.request_ids:
            self.worker._assign_slot(request_id, reset=True)
        self.worker._batch_to_slot[:b].copy_(torch.arange(b, device=self.device))
        self.slots = self.worker._batch_to_slot[:b]
        self.identity_slots = getattr(self.provider, 'identity_slots', False)
        if self.identity_slots and self.slots.cpu().tolist() != list(range(b)):
            raise ValueError('Identity-slot Draft route requires fixed row-to-slot ownership')
        self.draft_lengths = self.worker._ctx_len[:b]
        self.next_tokens = torch.zeros(b, 8, device=self.device, dtype=torch.int32)
        self.prime_graph = None
        self.verify_graph=None;self.late_verify_after=getattr(self.provider,'late_verify_after',0)
        self.reset(**{key: kwargs[key] for key in ("initial_tokens", "token_lengths", "past_lengths",
                                                 "draft_lengths", "mel_lengths", "seeds")})

    def _draft_from_native_pool(self, noise, absolute, context_positions, lengths, key_pool, value_pool, slots):
        draft = self.provider.runtime.engine.draft
        x = draft.input_projection(noise + draft.position_embedding(absolute.long()))
        valid = self.provider.positions[None] < lengths[:, None]
        mask = torch.cat((valid[:, None, None].expand(-1, 1, 7, -1),
                          torch.ones(self.batch, 1, 7, 7, device=self.device, dtype=torch.bool)), -1)
        # Native logical shape is [slot,L,K,H,D]; TRT consumes [B,H,K,D].
        # With head-major backing, each transposed layer already is contiguous
        # and .contiguous() aliases its storage. The original arena still copies.
        keys = (key_pool[:self.batch, :, :self.kv_capacity] if self.identity_slots else
                key_pool[:, :, :self.kv_capacity].index_select(0, slots.long()))
        values = (value_pool[:self.batch, :, :self.kv_capacity] if self.identity_slots else
                  value_pool[:, :, :self.kv_capacity].index_select(0, slots.long()))
        bindings = {"x": x.contiguous(), "mask": mask}
        for layer in range(3):
            bindings[f"k_cache_{layer}"] = keys[:, layer].transpose(1, 2).contiguous()
            bindings[f"v_cache_{layer}"] = values[:, layer].transpose(1, 2).contiguous()
        return self.provider.draft_engine(bindings)["hidden"]

    def _prime(self):
        model = self.provider.runtime.engine.draft
        noise = model.mask_embedding[None, None].expand(self.batch, 7, -1).clone()
        noise[:, 0] = model.token_embedding(self.last.long())
        query = self.draft_lengths[:, None] + self.step7
        hidden = self.native_model.dflash_forward(noise, query, self.draft_lengths,
            self.worker._ctx_k_buf, self.worker._ctx_v_buf, self.slots).view(self.batch, 7, -1)
        base = self.provider.draft_engine.outputs["base"]
        self.policy.propose(hidden, base, self.last, self.slots)
        self.next_tokens[:, 0].copy_(self.last)
        self.next_tokens[:, 1:].copy_(self.policy.tokens[:self.batch])

    def reset(self, *, initial_tokens, token_lengths, past_lengths, draft_lengths, mel_lengths, seeds):
        if initial_tokens.shape[0] != self.batch or token_lengths.shape != (self.batch,):
            raise ValueError("Native admission batch must match the captured profile")
        self.tokens.zero_(); self.tokens[:, :initial_tokens.shape[1]].copy_(initial_tokens)
        self.policy.lengths.fill_(1); self.token_lengths.copy_(token_lengths)
        self.past.copy_(past_lengths)
        self.worker._ctx_len.zero_(); self.draft_lengths.copy_(draft_lengths)
        host_lengths = getattr(self.provider, "host_draft_lengths", None)
        if host_lengths is None:
            host_lengths = draft_lengths.detach().cpu().tolist()
        self.worker._ctx_len_host[:] = [*map(int, host_lengths), 0]
        self.native_model.mel_lengths.zero_(); self.mel_lengths.copy_(mel_lengths)
        self.policy.last.zero_()
        self.last.copy_(self.tokens.gather(1, (self.token_lengths.long()-1)[:, None]).squeeze(1))
        self.done.copy_((self.last == self.eos) | (self.token_lengths >= self.max_tokens))
        self.policy.ready.fill_(True); self.ready.copy_(self.done | (self.token_lengths >= self.ready_token_count))
        self.policy.rounds.zero_(); self.policy.tokens.zero_(); self.policy.probabilities.zero_()
        self.policy.has_proposal.zero_(); self.policy.last_committed.zero_()
        self.accepted.fill_(-1); self.active.zero_(); self.failures.zero_(); self.capacity_failures.zero_(); self.status.zero_()
        self.draws.reset([*seeds, 0])
        self.worker._ctx_k_buf.zero_(); self.worker._ctx_v_buf.zero_()
        for layer in range(3):
            self.worker._ctx_k_buf[:self.batch, layer, :self.kv_capacity].copy_(self.provider.draft_cache[layer, 0].transpose(1, 2))
            self.worker._ctx_v_buf[:self.batch, layer, :self.kv_capacity].copy_(self.provider.draft_cache[layer, 1].transpose(1, 2))
        if self.prime_graph is None:
            self._prime()
        else:
            self.prime_graph.replay()

    def step(self):
        fits = (self.past+8 <= self.kv_capacity) & (self.draft_lengths+8 <= self.kv_capacity)
        self.capacity_failures.add_(((~self.ready) & (~fits)).sum().int())
        self.ready.logical_or_(~fits)
        self.active.copy_(~self.ready)
        old_lengths = self.token_lengths.clone()
        old_rounds = self.rounds.clone()
        absolute = self.past[:, None] + 1 - self.mel_lengths[:, None] + self.step8
        logits, selected, final = self.provider.target(self.next_tokens.long(), absolute,
                                                       self.past.clamp_max(self.kv_capacity-8))
        self.failures.add_(((~torch.isfinite(logits).flatten(1).all(-1)) & self.active).sum().int())
        self.metadata.captured_hidden_states.copy_(selected.reshape(self.batch*8, -1))
        output = self.worker(input_ids=self.next_tokens.flatten(),
            position_ids=(self.past[:, None]+self.step8).flatten(), hidden_states=final.flatten(0, 1),
            logits=logits.flatten(0, 1), attn_metadata=self.attention,
            spec_metadata=self.metadata, draft_model=self.native_model)
        produced = output["new_tokens_lens"].long()
        indices = old_lengths.long()[:, None]+self.step8
        valid = self.step8 < produced[:, None]
        value = torch.where(valid, output["new_tokens"].long(), self.tokens.gather(1, indices))
        self.tokens.scatter_(1, indices, value)
        history_index = old_rounds.long().clamp_max(self.max_rounds-1)[:, None]
        accepted = torch.where(self.active, self.committed-1, self.accepted.gather(1, history_index).squeeze(1))
        self.accepted.scatter_(1, history_index, accepted[:, None])
        self.next_tokens.copy_(output["next_new_tokens"])
        self.done.copy_((self.last == self.eos) | (self.token_lengths >= self.max_tokens))
        self.status.copy_(self.ready.all().int() | ((self.failures>0).int()<<1)
                          | ((self.capacity_failures>0).int()<<2))

    def capture(self, burst_rounds=2, warmups=2):
        saved_ready = self.ready.clone()
        self.ready.fill_(True)
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(warmups):
                self._prime()
        stream.synchronize()
        self.prime_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.prime_graph, stream=stream):
            self._prime()
        torch.cuda.current_stream(self.device).wait_stream(stream)
        self.ready.copy_(saved_ready)
        self.metadata.is_cuda_graph = True
        super().capture(burst_rounds, warmups)
        if self.late_verify_after:
            saved_ready=self.ready.clone();saved_done=self.done.clone()
            self.ready.fill_(True);self.done.fill_(True)
            self.native_model.skip_backbone=True;self.worker.skip_next_proposal=True
            try:
                with torch.cuda.stream(stream):self.step()
                stream.synchronize();self.verify_graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.verify_graph,stream=stream):self.step()
                torch.cuda.current_stream(self.device).wait_stream(stream)
            finally:
                self.native_model.skip_backbone=False;self.worker.skip_next_proposal=False
                self.ready.copy_(saved_ready);self.done.copy_(saved_done)
                self.status.zero_();self.failures.zero_();self.capacity_failures.zero_()

    def run(self):
        result = super().run()
        tail=result.get('compacted_tail')
        result['target_enqueues_by_batch']={str(self.batch):result['launched_rounds']-(tail['launched_rounds'] if tail else 0)}
        if tail:
            result['target_enqueues_by_batch'].update(tail.get('target_enqueues_by_batch',
                {str(tail['tail_batch']):tail['launched_rounds']}))
        result["target_enqueues"] = result["launched_rounds"]
        result["draft_enqueues"] = result["launched_rounds"] + 1-result.get('skipped_final_proposals',0)
        result["prime_enqueues"] = 1+(tail or {}).get('prime_enqueues',0)
        result["native_dspark_worker"] = True
        result["native_executor"] = False
        result["draft_cache_backend"] = ("NVIDIA DSparkWorker head-major backing with strided logical slots"
                                          if self.head_major_arena else
                                          "NVIDIA DSparkWorker contiguous context arena")
        result["draft_cache_layout_copies"] = not self.head_major_arena
        result['draft_cache_native_strides']=list(self.worker._ctx_k_buf.stride())
        result["draft_cache_slot_gather"] = not self.identity_slots
        result['context_compute']=('TensorRT K/V layers1/2; Torch projection and protected K/V0'
                                   if hasattr(self.provider,'context_engine') and self.provider.context_engine.plan.get('kind')=='context_kv'
                                   else 'TensorRT static context/KV engine' if hasattr(self.provider,'context_engine')
                                   else 'torch_same_recipe_reference')
        return result
