"""Post-load graph rewrite of the NVIDIA RNN head using existing torch ops.

Eliminates the zero rank padding used to map linear_wide to NVIDIA RNNHead,
folds the constant token embedding projection, and evaluates the seven hidden
projections together. The recurrent state and actual sampled chain remain
unchanged. Optional torch.compile performs framework-generated fusion only.
No CUDA, Triton, CUTLASS or other custom math kernel is authored here.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .pcg import categorical


class FactorizedRNNProposal(nn.Module):
    optimization = "zero_rank_prune_token_fold_hidden_batch"

    def __init__(self, official):
        super().__init__()
        self.block_size = official.block_size
        self.state_size = int(official.state_size)
        self.original_rank = int(official.original_rank)
        self.temperature = float(official.temperature)
        self.provenance = official.provenance + "; graph_rewrite=" + self.optimization
        state, rank = self.state_size, self.original_rank
        head = official.head
        joint = head.joint_proj.weight.detach().float()
        bias = head.joint_proj.bias.detach().float()
        token = head.markov_w1.weight.detach().float()
        output = head.markov_w2.weight.detach().float()
        self.vocab_size = token.shape[0]
        self.hidden_size = joint.shape[1] - 2 * state
        if not 0 < rank <= state or joint.shape[0] != 3 * state:
            raise ValueError("Unexpected NVIDIA RNN rank/state geometry")
        if output.shape != token.shape or token.shape[1] != state:
            raise ValueError("Unexpected NVIDIA token/output projection geometry")
        if getattr(head.markov_w2, "bias", None) is not None:
            raise ValueError("Graph rewrite expects NVIDIA's bias-free Markov output")
        # Checked only at deployment. These exact zeros are the proof that the
        # removed coordinates cannot affect a gate, recurrent state or logit.
        eliminated = (joint[2*state+rank:], bias[2*state+rank:],
                      joint[:, state+rank:2*state], token[:, rank:], output[:, rank:])
        if any(bool(torch.count_nonzero(value)) for value in eliminated):
            raise ValueError("Cannot prune nonzero padded RNN coordinates")
        live = 2 * state + rank
        self.register_buffer("state_weight", joint[:live, :state].contiguous())
        self.register_buffer("hidden_weight", joint[:live, 2*state:].contiguous())
        self.register_buffer("hidden_bias", bias[:live].contiguous())
        self.register_buffer("output_weight", output[:, :rank].contiguous())
        # A one-off constant fold. Match the protected FP32 RNN policy even if
        # the caller temporarily enables TF32 for another model component.
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            folded = F.linear(token[:, :rank], joint[:live, state:state+rank]).detach()
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        self.register_buffer("token_table", folded.contiguous())
        self.requires_grad_(False)
        self.compilation = None
        self._compiled_uniform = None

    def _math(self, hidden, base, random, previous, uniform):
        state = hidden.new_zeros(hidden.shape[0], self.state_size, dtype=torch.float32)
        terms = F.linear(hidden.float(), self.hidden_weight, self.hidden_bias)
        tokens, probabilities, logits = [], [], []
        for step in range(self.block_size):
            # Preserve NVIDIA's out-of-vocabulary anchor masking for graph
            # padding. Valid requests use ordinary token IDs.
            valid = (previous >= 0) & (previous < self.vocab_size)
            safe = previous.long().masked_fill(~valid, 0)
            token_term = F.embedding(safe, self.token_table).masked_fill(~valid[:, None], 0)
            raw = F.linear(state, self.state_weight) + token_term + terms[:, step]
            gate, candidate, output = raw.split((self.state_size, self.state_size, self.original_rank), -1)
            gate = gate.sigmoid()
            state = gate * state + (1 - gate) * candidate.tanh()
            delta = F.linear(output.tanh(), self.output_weight)
            logit = base[:, step].float() + delta
            probability = torch.softmax(logit / self.temperature, -1)
            previous = categorical(probability, random[:, step]) if uniform else (probability / random[:, step]).argmax(-1)
            tokens.append(previous); probabilities.append(probability); logits.append(logit)
        return torch.stack(tokens, 1), torch.stack(probabilities, 1), torch.stack(logits, 1)

    def forward(self, hidden, base, noise, previous):
        if base.shape != noise.shape or base.shape[1] != self.block_size:
            raise ValueError("Expected Q7 exponential draws matching base logits")
        return self._math(hidden, base, noise, previous, False)

    captured_math = forward

    def _uniform_math(self, hidden, base, uniform, previous):
        return self._math(hidden, base, uniform, previous, True)

    def sample_uniform(self, hidden, base, uniform, previous):
        if base.shape[:2] != uniform.shape or base.shape[1] != self.block_size:
            raise ValueError("Expected Q7 scalar uniforms")
        implementation = self._compiled_uniform or self._uniform_math
        return implementation(hidden, base, uniform, previous)

    def enable_compilation(self, *, backend="inductor"):
        """Opt-in compilation of RNN only; first warmup must precede capture.

        TensorRT enqueue and PCG/cuSPARSE live in the surrounding runtime and
        cannot enter this graph. Inductor's own CUDA Graph feature is disabled
        because the deployment captures the complete parent round separately.
        ``aot_eager`` is supported for CPU graph/export semantic checks.
        """
        if self._compiled_uniform is not None:
            raise RuntimeError("RNN compilation was already selected")
        if backend not in ("inductor", "aot_eager"):
            raise ValueError("Supported compiler backends: inductor/aot_eager")
        if backend=='inductor':
            import os
            from pathlib import Path
            cache_root=Path(__file__).resolve().parents[4]/'.cache'
            os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR',str(cache_root/'inductor_baseline_completion'))
            os.environ.setdefault('TRITON_CACHE_DIR',str(cache_root/'triton_baseline_completion'))
        options = {"triton.cudagraphs": False, "max_autotune": True,
                   "max_autotune_gemm_backends":"ATEN"} if backend == "inductor" else None
        self._compiled_uniform = torch.compile(self._uniform_math, backend=backend,
                                              fullgraph=True, dynamic=False, options=options)
        self.compilation = backend
        self.compilation_options = options
        return self

    def stats(self):
        state, rank, hidden, vocab = self.state_size, self.original_rank, self.hidden_size, self.vocab_size
        before = self.block_size * ((2*state+hidden)*(3*state) + state*vocab)
        after = self.block_size * ((state+hidden)*(2*state+rank) + rank*vocab)
        return {"optimization": self.optimization, "compilation": self.compilation,
                "compilation_options":getattr(self,'compilation_options',None),
                "state_size": state, "token_output_rank": rank,
                "dtype": "float32", "runtime_gemm_macs_per_request_round_before": before,
                "runtime_gemm_macs_per_request_round_after": after,
                "gemm_mac_reduction": 1 - after / before,
                "offline_fold_macs": vocab * rank * (2*state+rank),
                "constant_buffer_bytes": sum(t.numel()*t.element_size() for t in self.buffers()),
                "native_existing_operators": True, "custom_math_kernel": False,
                "default_enabled": False, "e2e_gain_measured": False}
