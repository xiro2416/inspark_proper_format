"""Checkpoint-faithful interfaces for a common DSpark runtime.

NVIDIA's RNNHead is used through its existing ``_rnn_step`` implementation.
The standalone worker in audited TensorRT-LLM currently rejects that head;
this module is an executable model-level integration boundary, not a claim
that the unmodified NVIDIA executor supports the IndexTTS checkpoint.
"""
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn


def padded_rnn_weights(model) -> dict[str, torch.Tensor]:
    """Map linear_wide into NVIDIA RNNHead without changing its equations.

    NVIDIA ties recurrent-state width to Markov rank. We pad the token and
    output ranks to state width, leaving added coordinates identically zero.
    This is algebraically equivalent, not a bitwise floating-point guarantee.
    """
    if model.markov_cell_type not in ("linear", "linear_wide"):
        raise ValueError("Only linear/linear_wide checkpoints map to NVIDIA RNNHead")
    if model.markov_in is None or model.markov_out is None:
        raise ValueError("Mapping requires free token embedding and output projection")
    if getattr(model, "persistent_markov_state", False):
        raise ValueError("Round-local RNN adapter cannot carry persistent state")
    state, rank, hidden = model.markov_state_size, model.markov_rank, model.interface_size
    if state < rank:
        raise ValueError("Native rank padding requires state_size >= markov_rank")
    weight = model.markov_rnn.weight.detach().float()
    expected = (2 * state + rank, state + rank + hidden)
    if tuple(weight.shape) != expected:
        raise ValueError(f"Unexpected RNN matrix {tuple(weight.shape)} != {expected}")
    joint = weight.new_zeros(3 * state, 2 * state + hidden)
    joint[:2 * state + rank, :state + rank] = weight[:, :state + rank]
    joint[:2 * state + rank, 2 * state:] = weight[:, state + rank:]
    bias = weight.new_zeros(3 * state)
    bias[:2 * state + rank] = model.markov_rnn.bias.detach().float()
    token = weight.new_zeros(model.vocab_size, state)
    output = weight.new_zeros(model.vocab_size, state)
    token[:, :rank] = model.markov_in.weight.detach().float()
    output[:, :rank] = model.markov_out.weight.detach().float()
    return {"joint_proj.weight": joint, "joint_proj.bias": bias,
            "markov_w1.weight": token, "markov_w2.weight": output}


class OfficialRNNProposal(nn.Module):
    """Reuse NVIDIA's RNN math while sampling the actual conditional chain.

    ``noise`` uses the existing exponential-race sampling interface. Random
    draws are request owned and supplied by the caller; this module neither
    reads nor mutates a global RNG. The returned probabilities correspond to
    the sampled previous tokens, never to a separate greedy chain.
    """
    def __init__(self, model, head_factory: Callable, *, provenance: str,
                 temperature: float = 0.8):
        super().__init__()
        if model.block_size != 7:
            raise ValueError("This checkpoint adapter requires seven proposal slots")
        self.block_size, self.state_size = model.block_size, model.markov_state_size
        self.original_rank = model.markov_rank
        self.temperature = float(temperature)
        self.provenance = str(provenance)
        self.head = head_factory(vocab_size=model.vocab_size,
                                 markov_rank=self.state_size,
                                 hidden_size=model.interface_size).float()
        self.head.load_state_dict(padded_rnn_weights(model), strict=True)
        self.head.to(device=model.markov_rnn.weight.device).eval()
        self.requires_grad_(False)

    def forward(self, hidden, base, noise, previous):
        if hidden.shape[:2] != base.shape[:2] or base.shape != noise.shape:
            raise ValueError("RNN hidden/logits/noise shapes disagree")
        if base.shape[1] != 7:
            raise ValueError("Expected Q7")
        state = hidden.new_zeros(hidden.shape[0], self.state_size, dtype=torch.float32)
        tokens, probabilities, logits = [], [], []
        for step in range(7):
            embedding = self.head.get_prev_embeddings(previous.long())
            state, delta = self.head._rnn_step(state, embedding, hidden[:, step].float())
            logit = base[:, step].float() + delta.float()
            probability = torch.softmax(logit / self.temperature, dim=-1)
            previous = (probability / noise[:, step]).argmax(-1)
            tokens.append(previous)
            probabilities.append(probability)
            logits.append(logit)
        return torch.stack(tokens, 1), torch.stack(probabilities, 1), torch.stack(logits, 1)

    def graph_rewrite(self, *, compile_graph=False, compiler_backend="inductor"):
        """Create an opt-in post-load graph rewrite; never mutate this baseline."""
        from .rnn_rewrite import FactorizedRNNProposal
        result = FactorizedRNNProposal(self)
        if compile_graph:
            result.enable_compilation(backend=compiler_backend)
        return result

    captured_math = forward

    def sample_uniform(self, hidden, base, uniform, previous):
        """Same conditional categorical law using seven scalar uniforms.

        This avoids preallocating a vocabulary-sized exponential tensor for
        every possible round. It changes the seeded trajectory relative to
        exponential-race sampling, not the categorical distribution.
        """
        from .pcg import categorical
        if uniform.shape != base.shape[:2] or base.shape[1] != 7:
            raise ValueError("Expected one uniform per request and proposal position")
        state = hidden.new_zeros(hidden.shape[0], self.state_size, dtype=torch.float32)
        tokens, probabilities, logits = [], [], []
        for step in range(7):
            embedding = self.head.get_prev_embeddings(previous.long())
            state, delta = self.head._rnn_step(state, embedding, hidden[:, step].float())
            logit = base[:, step].float() + delta.float()
            probability = torch.softmax(logit / self.temperature, dim=-1)
            previous = categorical(probability, uniform[:, step])
            tokens.append(previous); probabilities.append(probability); logits.append(logit)
        return torch.stack(tokens, 1), torch.stack(probabilities, 1), torch.stack(logits, 1)


@dataclass(frozen=True)
class ModelContract:
    target_layers: tuple[int, ...] = (1, 6, 11, 16, 21)
    hidden_size: int = 1280
    draft_layers: int = 3
    proposal_slots: int = 7
    verification_slots: int = 8


class DSparkModelAdapter(nn.Module):
    """Common model boundary for FP8 and INT8 execution providers.

    A provider performs model computation; request scheduling and PCG remain
    shared. Callable providers may be TensorRT subgraphs or framework modules.
    Context projection returns a commit mask which the provider must consume
    when storing K/V; rejected positions are not permitted to advance length.
    """
    def __init__(self, draft, proposal: OfficialRNNProposal,
                 draft_forward: Callable, target_forward: Callable,
                 *, precision: str, contract: ModelContract = ModelContract()):
        super().__init__()
        if precision not in ("fp8", "int8_smoothquant"):
            raise ValueError("Expected fp8 or int8_smoothquant compute provider")
        if tuple(draft.target_layer_ids) != contract.target_layers:
            raise ValueError("Target hidden capture indices differ from checkpoint contract")
        if draft.interface_size != contract.hidden_size or len(draft.layers) != contract.draft_layers:
            raise ValueError("Draft architecture differs from checkpoint contract")
        if draft.block_size != contract.proposal_slots:
            raise ValueError("Draft block size differs from checkpoint contract")
        self.draft, self.proposal = draft, proposal
        self.draft_forward, self.target_forward = draft_forward, target_forward
        self.precision, self.contract = precision, contract
        self.actual_rotary_layers = tuple(hasattr(layer, "rope_inv_freq") for layer in draft.layers)

    def propose(self, anchors, positions, slots, lengths, noise):
        hidden, base = self.draft_forward(anchors, positions, slots, lengths)
        return self.proposal(hidden, base, noise, anchors)

    def verify(self, token_embeddings, slots, lengths):
        if token_embeddings.shape[1] != self.contract.verification_slots:
            raise ValueError("Target verification must include anchor plus seven proposals")
        return self.target_forward(token_embeddings, slots, lengths)

    def project_committed_context(self, selected_hidden, committed_counts, active):
        if selected_hidden.shape[1:] != (8, 5 * self.contract.hidden_size):
            raise ValueError("Expected [B,8,6400] selected Target hidden states")
        valid = (torch.arange(8, device=selected_hidden.device)[None]
                 < committed_counts[:, None]) & active[:, None]
        # Mask before projection so even rejected NaNs cannot contaminate any
        # downstream reduction. The consumer still must mask its cache write.
        context = self.draft.project_context(selected_hidden.masked_fill(~valid[..., None], 0))
        return context.masked_fill(~valid[..., None], 0), valid
