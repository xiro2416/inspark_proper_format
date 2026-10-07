"""Model and sampler adapters for NVIDIA's actual standalone DSpark worker.

The factory subclasses the installed DSparkWorker without replacing its
forward/metadata-cleanup/context-cache code. Only its sampler extension points
are overridden. Full executor integration is a separate, explicitly reported
validation step; constructing this class does not certify that integration.
"""
from types import SimpleNamespace

import torch
from torch import nn

from .pcg import FrameworkPCG, categorical
from .runtime import RequestDraws


class _MaskEmbedding(nn.Module):
    def __init__(self, draft):
        super().__init__()
        self.token_embedding = draft.token_embedding
        self.mask_embedding = draft.mask_embedding
        self.mask_token_id = draft.mask_token_id

    def forward(self, tokens):
        mask = tokens == self.mask_token_id
        safe = tokens.masked_fill(mask, 0)
        return torch.where(mask[..., None], self.mask_embedding, self.token_embedding(safe))


class NvidiaDraftInterface(nn.Module):
    """Preserve the IndexTTS model behind the native DFlash model protocol.

    ``mel_lengths`` is indexed by native context slot, including its padding
    slot. Native query positions are context positions; IndexTTS absolute
    speech positions are ``query_position + 1 - mel_length``. Existing RoPE,
    if any, continues to use the context positions independently.
    """
    def __init__(self, draft, mel_lengths, forward_provider=None,
                 logits_provider=None, context_projection_provider=None,
                 context_kv_provider=None):
        super().__init__()
        self.model = draft
        self.fc, self.hidden_norm, self.lm_head = draft.context_projection, draft.context_norm, draft.lm_head
        self.block_size, self.mask_token_id = draft.block_size, draft.mask_token_id
        self._dspark_shift_label = True
        self.has_markov_head = True
        self.has_candidate_selector = False
        self.dflash_attention_backend = "VANILLA"
        self.config = SimpleNamespace(hidden_size=draft.hidden_size,
                                      max_position_embeddings=draft.position_embedding.num_embeddings,
                                      vocab_size=draft.vocab_size)
        self.register_buffer("mel_lengths", mel_lengths)
        self.forward_provider = forward_provider
        self.logits_provider = logits_provider
        self.context_projection_provider = context_projection_provider
        self.context_kv_provider = context_kv_provider
        self.last_block_hidden = None
        self.draft_model_full = nn.Module()
        self.draft_model_full.model = nn.Module()
        self.draft_model_full.model.embed_tokens = _MaskEmbedding(draft)
        self._build_fused_kv_buffers()

    def _build_fused_kv_buffers(self):
        # The native worker calls this to obtain geometry. The checkpoint's
        # existing projections are retained; no Qwen/RoPE model is substituted.
        self._num_attn_layers = len(self.model.layers)
        self._num_heads = self.model.layers[0].num_heads
        self._num_kv_heads = getattr(self.model.layers[0], "num_kv_heads", self._num_heads)
        self._head_dim = self.model.layers[0].head_dim

    def project_target_hidden(self, hidden_states):
        if self.context_projection_provider is not None:
            return self.context_projection_provider(hidden_states)
        return self.model.project_context(self.model.prepare_context(hidden_states))

    def precompute_context_kv(self, projected_hidden, positions):
        if self.context_kv_provider is not None:
            return self.context_kv_provider(projected_hidden, positions)
        keys, values = [], []
        for layer in self.model.layers:
            key, value = layer.context_kv(projected_hidden[None], positions[None])
            keys.append(key[0].transpose(0, 1)); values.append(value[0].transpose(0, 1))
        return torch.stack(keys, 1).contiguous(), torch.stack(values, 1).contiguous()

    def dflash_forward(self, noise_embedding, query_positions, num_ctx_per_req,
                       ctx_k_cache, ctx_v_cache, ctx_cache_batch_idx,
                       ctx_kv_cache=None, ctx_page_table=None):
        if getattr(self,'skip_backbone',False):
            if self.last_block_hidden is None:raise RuntimeError('No captured Draft hidden for verify-only route')
            return self.last_block_hidden.flatten(0,1)
        if ctx_kv_cache is not None or ctx_page_table is not None:
            raise ValueError("IndexTTS bridge currently requires the contiguous native context pool")
        absolute = query_positions + 1 - self.mel_lengths[ctx_cache_batch_idx.long(), None]
        if self.forward_provider is not None:
            hidden = self.forward_provider(noise_embedding, absolute, query_positions, num_ctx_per_req,
                                           ctx_k_cache, ctx_v_cache, ctx_cache_batch_idx)
        else:
            hidden = noise_embedding + self.model.position_embedding(absolute.long())
            extent = ctx_k_cache.shape[2]
            keep = torch.arange(extent, device=hidden.device)[None] < num_ctx_per_req[:, None]
            keys = ctx_k_cache.index_select(0, ctx_cache_batch_idx.long())
            values = ctx_v_cache.index_select(0, ctx_cache_batch_idx.long())
            for index, layer in enumerate(self.model.layers):
                hidden = layer(hidden,
                    context_k=keys[:, index].transpose(1, 2),
                    context_v=values[:, index].transpose(1, 2),
                    context_mask=keep, position_ids=query_positions)
                hidden = self.model.apply_query_temporal(hidden, index)
            hidden = self.model.project_output(hidden)
        self.last_block_hidden = hidden
        return hidden.flatten(0, 1)

    def logits_processor(self, hidden_states, lm_head, attn_metadata, gather_output):
        if self.logits_provider is not None:
            return self.logits_provider(hidden_states)
        return self.model.base_logits(hidden_states)

    def load_weights_from_target_model(self, target_model):
        # The draft checkpoint owns its embeddings/head. Native base models
        # normally alias these from Target, which is invalid for this model.
        return None


class NativePCGPolicy:
    """GPU state used by the native worker's three sampling hooks."""
    def __init__(self, *, proposal, groups, seeds, device, eos, max_rounds=64,
                 max_tokens=1500, initial_token_lengths=None, initial_last_tokens=None,
                 ready_token_count=31):
        self.proposal, self.pcg = proposal, FrameworkPCG(groups)
        self.eos, self.max_tokens = int(eos), int(max_tokens)
        self.draws = RequestDraws(seeds, max_rounds, device)
        self.rounds = torch.zeros(len(seeds), device=device, dtype=torch.int32)
        self.lengths = (torch.ones(len(seeds), device=device, dtype=torch.int32)
                        if initial_token_lengths is None else initial_token_lengths.to(device=device, dtype=torch.int32).clone())
        vocab = groups.token_group_counts.numel()
        self.probabilities = torch.zeros(len(seeds), 7, vocab, device=device)
        self.tokens = torch.zeros(len(seeds), 7, device=device, dtype=torch.long)
        self.has_proposal = torch.zeros(len(seeds), device=device, dtype=torch.bool)
        self.failures = torch.zeros((), device=device, dtype=torch.int32)
        self.last_committed = torch.zeros(len(seeds), device=device, dtype=torch.int32)
        self.last = (torch.zeros(len(seeds), device=device, dtype=torch.long)
                     if initial_last_tokens is None else initial_last_tokens.to(device=device, dtype=torch.long).clone())
        self.ready_token_count = int(ready_token_count)
        self.ready = (self.lengths >= self.ready_token_count) | (self.last == self.eos)
        self.pending_tokens = None

    def _random(self, slots):
        return self.draws.values[slots.long(), self.rounds[slots.long()].long().clamp_max(self.draws.values.shape[1] - 1)]

    def propose(self, hidden, base, anchor, slots):
        random = self._random(slots)
        token, probability, logits = self.proposal.sample_uniform(hidden, base, random[:, :7], anchor)
        active = ~self.ready[slots.long()]
        token = torch.where(active[:, None], token, self.tokens[slots.long()])
        probability = torch.where(active[:, None, None], probability, self.probabilities[slots.long()])
        self.tokens.index_copy_(0, slots.long(), token)
        self.probabilities.index_copy_(0, slots.long(), probability)
        self.has_proposal.index_copy_(0, slots.long(), self.has_proposal[slots.long()] | active)
        self.pending_tokens = token
        return logits

    def accept(self, logits, slots):
        slots = slots.long()
        active = ~self.ready[slots]
        random = self._random(slots)
        proposed, p = self.tokens[slots], self.probabilities[slots]
        q, _, packed = self.pcg.acceptance(logits[:, :7], p, proposed, random[:, 7:14], random[:, 14:21])
        present = self.has_proposal[slots]
        packed[..., 0] = torch.where(present[:, None], packed[..., 0], torch.zeros_like(packed[..., 0]))
        count, end, correction, residual_mask = self.pcg.prefix_plan(
            packed, (self.max_tokens - self.lengths[slots]).clamp(0, 7),
            self.lengths[slots], self.eos, self.max_tokens)
        residual_mask = residual_mask & present & active
        draws = {"candidates": random[:, 21:85], "groups": random[:, 85:149], "thin": random[:, 149:213],
                 "fast": random[:, 213], "coarse": random[:, 214], "member": random[:, 215]}
        residual, _, invalid = self.pcg.residual(q, p, count, residual_mask, draws)
        self.failures.add_(invalid.sum().to(torch.int32))
        row = torch.arange(logits.shape[0], device=logits.device)
        target = torch.softmax(logits[row, count.long().clamp_max(7)].float() / .8, -1)
        bonus = categorical(target, random[:, 216])
        corrected = torch.where(residual_mask, residual, bonus)
        positions = torch.arange(8, device=logits.device)[None]
        padded = torch.cat((proposed, corrected[:, None]), 1)
        accepted_last = proposed.gather(1, (count.long() - 1).clamp(0, 6)[:, None]).squeeze(1)
        filler = torch.where(end, accepted_last, corrected)
        accepted = torch.where(positions < count[:, None], padded, filler[:, None])
        lengths = count + correction.int()
        lengths = torch.where(active, lengths, 0)
        # Target/Draft cache commit includes the input anchor. When an accepted
        # proposal is EOS, output count and KV commit count differ by one.
        self.last_committed.index_copy_(0, slots, torch.where(active, count + 1, 0))
        accepted = torch.where(active[:, None], accepted, self.last[slots, None])
        self.lengths.index_add_(0, slots, lengths)
        new_last = accepted.gather(1, (lengths.long() - 1).clamp_min(0)[:, None]).squeeze(1)
        self.last.index_copy_(0, slots, new_last.long())
        self.ready.index_copy_(0, slots, self.ready[slots] | (self.lengths[slots] >= self.ready_token_count)
                              | (new_last == self.eos) | (self.lengths[slots] >= self.max_tokens))
        # An unprimed first generation samples Target once, then produces its
        # first actual proposal. It must not consume a nonexistent RNN round.
        self.rounds.index_add_(0, slots, (present & active).int())
        return accepted.int(), lengths.int()


def native_pcg_worker_class():
    """Use native context/KV scheduling with semantic-preserving sampler hooks.

    Mixed prefill/generation batches are intentionally rejected by this first
    fixed-batch bridge. Call native prefill separately, seed its context pool,
    and prime the first proposal before the head's generation-only rounds.
    """
    from tensorrt_llm._torch.speculative.dspark import DSparkWorker

    class IndexTTSPCGWorker(DSparkWorker):
        def __init__(self, spec_config, mapping, policy):
            super().__init__(spec_config, mapping, use_separate_draft_kv_cache=False)
            self.pcg_policy = policy

        def _refine_block_logits(self, draft_model, gen_logits, inputs, spec_metadata):
            if getattr(self,'skip_next_proposal',False):return gen_logits
            if draft_model.last_block_hidden is None:
                raise RuntimeError("Native RNN hook requires the hidden states of this actual draft forward")
            slots = self._batch_to_slot[:gen_logits.shape[0]]
            return self.pcg_policy.propose(draft_model.last_block_hidden, gen_logits,
                                           inputs["first_prev_tokens"], slots)

        def sample_draft_tokens(self, logits, spec_metadata, batch_size, *, num_contexts=0,
                                draft_step=None, mapping_lm_head_tp=None):
            if getattr(self,'skip_next_proposal',False):return self.pcg_policy.tokens[:batch_size].int()
            if num_contexts or draft_step is not None or mapping_lm_head_tp is not None:
                raise ValueError("The fixed IndexTTS bridge requires generation-only, unsharded Q7")
            if self.pcg_policy.pending_tokens is None:
                raise RuntimeError("RNN proposal must run before sampling returns tokens")
            return self.pcg_policy.pending_tokens.int()

        def sample_and_accept_draft_tokens(self, logits, attn_metadata, spec_metadata):
            if attn_metadata.num_contexts:
                raise ValueError("Seed/prime native context before the generation-only PCG worker")
            batch = attn_metadata.num_seqs
            if spec_metadata.runtime_draft_len != 7:
                raise ValueError("The checkpoint's PCG block size is seven")
            return self.pcg_policy.accept(logits.reshape(batch, 8, -1), self._batch_to_slot[:batch])

        def _prepare_kv_for_draft_forward(self, attn_metadata, num_accepted_tokens, num_contexts, batch_size):
            if num_contexts:
                raise ValueError("The fixed IndexTTS bridge requires generation-only KV updates")
            commits = self.pcg_policy.last_committed[self._batch_to_slot[:batch_size]]
            return super()._prepare_kv_for_draft_forward(attn_metadata, commits, 0, batch_size)

        def prepare_1st_drafter_inputs(self, **kwargs):
            batch = kwargs["attn_metadata"].num_seqs
            # Padding in accepted_tokens repeats the last token, so an accepted
            # EOS can use its KV count while retaining EOS as the next anchor.
            kwargs["num_accepted_tokens"] = self.pcg_policy.last_committed[self._batch_to_slot[:batch]]
            return super().prepare_1st_drafter_inputs(**kwargs)

    return IndexTTSPCGWorker


@torch.inference_mode()
def probe_native_worker_forward(checkpoint, group_checkpoint, device="cuda:0", force_eos=False):
    """Run the real native worker's full forward against frozen Target I/O.

    This isolates KV/context/sampler integration. Target logits are synthetic;
    the report explicitly does not claim an end-to-end native Target engine.
    The Draft and RNN use actual checkpoint weights.
    """
    from tensorrt_llm.llmapi.llm_args import DSparkDecodingConfig
    from tensorrt_llm.mapping import Mapping
    from tensorrt_llm._torch.speculative.dflash import DFlashSpecMetadata
    from tensorrt_llm._torch.speculative.interface import SpeculativeDecodingMode
    from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
    from inspark_infer.models.indextts2.dspark.draft import IndexTTS2DSpark
    from inspark_infer.models.indextts2.pcg.asg import AcousticGroups
    from .adapter import OfficialRNNProposal
    from .official import native_rnn_factory

    draft = IndexTTS2DSpark.from_checkpoint(checkpoint, device).float().eval()
    groups = AcousticGroups.load(group_checkpoint).dense(device)
    proposal = OfficialRNNProposal(draft, native_rnn_factory(), provenance="installed_official_module")
    policy = NativePCGPolicy(proposal=proposal, groups=groups, seeds=[113, 0], device=device,
                             eos=draft.vocab_size - 1, initial_last_tokens=torch.tensor([5, 0], device=device))
    facade = NvidiaDraftInterface(draft, torch.tensor([2, 0], device=device))
    config = DSparkDecodingConfig(max_draft_len=7, block_size=7, markov_rank=256,
                                  markov_head_type="rnn", mask_token_id=draft.mask_token_id,
                                  target_layer_ids=draft.target_layer_ids, attention_backend="VANILLA")
    mapping = Mapping(world_size=1, rank=0, tp_size=1)
    worker = native_pcg_worker_class()(config, mapping, policy)
    worker.set_draft_model(facade)
    attention = AttentionMetadata(max_num_requests=1, max_num_tokens=8,
                                  seq_lens=torch.tensor([8], dtype=torch.int32),
                                  num_contexts=0, seq_lens_kv=None, mapping=mapping)
    attention.max_seq_len = 64
    attention.kv_lens_cuda = torch.tensor([4], device=device, dtype=torch.int32)
    metadata = DFlashSpecMetadata(max_num_requests=1, max_draft_len=7,
        max_total_draft_tokens=7, num_generations=1, runtime_draft_len=7,
        runtime_tokens_per_gen_step=8, spec_dec_mode=SpeculativeDecodingMode.DSPARK,
        layers_to_capture=draft.target_layer_ids, hidden_size=draft.interface_size,
        max_num_tokens=8, dtype=torch.float32, request_ids=[101], seq_lens=[8])
    metadata.batch_indices_cuda.zero_()
    worker._lazy_init_ctx_buffers(facade, metadata, attention, None)
    slot = worker._assign_slot(101, reset=True)
    worker._batch_to_slot[0] = slot
    worker._ctx_len[slot] = 4; worker._ctx_len_host[slot] = 4
    generator = torch.Generator(device=device).manual_seed(71)
    prefix = torch.randn((4, 5 * draft.interface_size), device=device, generator=generator)
    k, v = facade.precompute_context_kv(facade.project_target_hidden(prefix), torch.arange(4, device=device))
    worker._ctx_k_buf[slot, :, :4].copy_(k.transpose(0, 1))
    worker._ctx_v_buf[slot, :, :4].copy_(v.transpose(0, 1))
    before_k = worker._ctx_k_buf[slot, :, :4].clone()
    before_v = worker._ctx_v_buf[slot, :, :4].clone()
    anchors = torch.tensor([5], device=device)
    noise = draft.mask_embedding[None, None].expand(1, 7, -1).clone()
    noise[:, 0] = draft.token_embedding(anchors)
    slots = torch.tensor([slot], device=device)
    hidden = facade.dflash_forward(noise, torch.arange(4, 11, device=device)[None],
                                   torch.tensor([4], device=device), worker._ctx_k_buf,
                                   worker._ctx_v_buf, slots).view(1, 7, -1)
    policy.propose(hidden, draft.base_logits(hidden), anchors, slots)
    if force_eos:
        # Explicit distribution fixture for the accepted-EOS branch. The
        # report distinguishes this from natural model sampling.
        policy.tokens[slot].fill_(policy.eos)
        policy.probabilities[slot].zero_()
        policy.probabilities[slot, :, policy.eos] = 1
    verify = torch.cat((anchors[:, None], policy.tokens[slots]), 1)
    target_logits = torch.randn((8, draft.vocab_size), generator=generator, device=device)
    if force_eos:
        target_logits.fill_(-100)
        target_logits[:, policy.eos] = 0
    captured = torch.randn((8, 5 * draft.interface_size), generator=generator, device=device)
    metadata.captured_hidden_states.copy_(captured)
    output = worker(input_ids=verify.flatten(), position_ids=torch.arange(4, 12, device=device),
                    hidden_states=torch.zeros(8, draft.interface_size, device=device),
                    logits=target_logits, attn_metadata=attention, spec_metadata=metadata,
                    draft_model=facade)
    torch.cuda.synchronize()
    count = int(output["new_tokens_lens"][0])
    committed = int(policy.last_committed[slot])
    expected_k, expected_v = facade.precompute_context_kv(
        facade.project_target_hidden(captured[:committed]), torch.arange(4, 4 + committed, device=device))
    committed_k = worker._ctx_k_buf[slot, :, 4:4 + committed].transpose(0, 1)
    committed_v = worker._ctx_v_buf[slot, :, 4:4 + committed].transpose(0, 1)
    report = {"native_worker_forward": True, "native_executor_validated": False,
            "worker_type": type(worker).__name__, "target_input": "frozen_synthetic_logits_not_target_engine",
            "draft_weights": "actual_checkpoint", "accepted_count": count, "kv_commit_count": committed,
            "forced_eos_distribution_fixture": bool(force_eos),
            "context_length": int(worker._ctx_len[slot]),
            "existing_prefix_unchanged": torch.equal(before_k, worker._ctx_k_buf[slot, :, :4]) and torch.equal(before_v, worker._ctx_v_buf[slot, :, :4]),
            "committed_k_max_abs": float((expected_k - committed_k).abs().max()),
            "committed_v_max_abs": float((expected_v - committed_v).abs().max()),
            "context_only_committed_prefix": int(worker._ctx_len[slot]) == 4 + committed,
            "next_new_tokens_shape": list(output["next_new_tokens"].shape),
            "metadata_restored": not attention.has_spec_dec_saved_state,
            "pcg_failures": int(policy.failures), "custom_kernel_added": False}
    if force_eos:
        frozen_k = worker._ctx_k_buf[slot, :, :4+committed].clone()
        frozen_v = worker._ctx_v_buf[slot, :, :4+committed].clone()
        frozen_rounds, frozen_lengths = policy.rounds.clone(), policy.lengths.clone()
        metadata.is_cuda_graph = True
        call = lambda: worker(input_ids=verify.flatten(), position_ids=torch.arange(4, 12, device=device),
            hidden_states=torch.zeros(8, draft.interface_size, device=device), logits=target_logits,
            attn_metadata=attention, spec_metadata=metadata, draft_model=facade)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            call(); call()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            inactive = call()
        graph.replay(); graph.replay(); torch.cuda.synchronize()
        report["eos_graph_replay"] = {
            "output_length_zero": int(inactive["new_tokens_lens"][0]) == 0,
            "rng_counter_unchanged": torch.equal(frozen_rounds, policy.rounds),
            "token_lengths_unchanged": torch.equal(frozen_lengths, policy.lengths),
            "committed_prefix_unchanged": torch.equal(frozen_k, worker._ctx_k_buf[slot, :, :4+committed]) and torch.equal(frozen_v, worker._ctx_v_buf[slot, :, :4+committed]),
            "context_length_unchanged": int(worker._ctx_len[slot]) == 4 + committed,
            "metadata_restored": not attention.has_spec_dec_saved_state,
        }
    return report
