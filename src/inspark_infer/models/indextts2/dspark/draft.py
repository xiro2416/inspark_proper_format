from __future__ import annotations
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn

class RMSNorm(nn.Module):
    """Qwen-style RMSNorm used by the official DSpark decoder."""

    def __init__(self, hidden_size: int, eps: float=1e-06):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = float(eps)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        dtype = hidden.dtype
        normalized = hidden.float() * torch.rsqrt(hidden.float().square().mean(dim=-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float()).to(dtype)

class MarkovResidualSwiGLUBlock(nn.Module):
    """Deep recurrent-cell block used only by the enlarged Markov head."""

    def __init__(self, width: int, expansion: int=2):
        super().__init__()
        inner = int(width * expansion)
        self.norm = nn.LayerNorm(width)
        self.gate = nn.Linear(width, inner, bias=False)
        self.up = nn.Linear(width, inner, bias=False)
        self.down = nn.Linear(inner, width, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(hidden)
        return hidden + self.down(F.silu(self.gate(normalized)) * self.up(normalized))

class OfficialQwenDSparkLayer(nn.Module):
    """One Qwen3-4B DSpark block, adapted to an external IndexTTS2 interface.

    The dimensions and primitives mirror DeepSpec's Qwen3 DSpark layer:
    32 query heads, 8 KV heads, explicit 128-wide heads, Q/K RMSNorm,
    bias-free projections, RoPE and a 9728-wide SwiGLU MLP.
    """

    def __init__(self, hidden_size: int, *, num_attention_heads: int, num_key_value_heads: int, head_dim: int, intermediate_size: int, rms_norm_eps: float, rope_theta: float, attention_dropout: float=0.0):
        super().__init__()
        if num_attention_heads % num_key_value_heads:
            raise ValueError('query heads must be divisible by KV heads')
        if head_dim % 2:
            raise ValueError('RoPE head_dim must be even')
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_attention_heads)
        self.num_kv_heads = int(num_key_value_heads)
        self.head_dim = int(head_dim)
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.attention_dropout_p = float(attention_dropout)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.input_norm = RMSNorm(self.hidden_size, eps=rms_norm_eps)
        self.post_norm = RMSNorm(self.hidden_size, eps=rms_norm_eps)
        self.gate_proj = nn.Linear(self.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, self.hidden_size, bias=False)
        inv_freq = 1.0 / float(rope_theta) ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    def _heads(self, value: torch.Tensor, num_heads: int) -> torch.Tensor:
        batch, length, _ = value.shape
        return value.view(batch, length, num_heads, self.head_dim).transpose(1, 2)

    def _rope(self, value: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        angles = torch.einsum('bl,d->bld', position_ids.float(), self.inv_freq.float())
        embedding = torch.cat((angles, angles), dim=-1).unsqueeze(1)
        cos = embedding.cos().to(value.dtype)
        sin = embedding.sin().to(value.dtype)
        return value * cos + _rotate_half(value) * sin

    def context_kv(self, context: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        key = self._heads(self.k_proj(context), self.num_kv_heads)
        value = self._heads(self.v_proj(context), self.num_kv_heads)
        key = self._rope(self.k_norm(key), position_ids)
        return (key, value)

    def forward(self, hidden: torch.Tensor, *, context_k: torch.Tensor, context_v: torch.Tensor, position_ids: torch.Tensor, attention_mask: torch.Tensor | None=None) -> torch.Tensor:
        residual = hidden
        normalized = self.input_norm(hidden)
        query = self._heads(self.q_proj(normalized), self.num_heads)
        noise_key = self._heads(self.k_proj(normalized), self.num_kv_heads)
        noise_value = self._heads(self.v_proj(normalized), self.num_kv_heads)
        query = self._rope(self.q_norm(query), position_ids)
        noise_key = self._rope(self.k_norm(noise_key), position_ids)
        key = torch.cat((context_k, noise_key), dim=2)
        value = torch.cat((context_v, noise_value), dim=2)
        if self.num_kv_groups > 1:
            key = key.repeat_interleave(self.num_kv_groups, dim=1)
            value = value.repeat_interleave(self.num_kv_groups, dim=1)
        attended = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=self.attention_dropout_p if self.training else 0.0, is_causal=False, scale=self.head_dim ** (-0.5))
        attended = attended.transpose(1, 2).flatten(2)
        hidden = residual + self.o_proj(attended)
        residual = hidden
        normalized = self.post_norm(hidden)
        mlp = self.down_proj(F.silu(self.gate_proj(normalized)) * self.up_proj(normalized))
        return residual + mlp

class DraftLayer(nn.Module):

    def __init__(self, hidden_size: int, num_heads: int, *, target_hidden_size: int | None=None, num_target_layers: int=0, target_fusion_type: str='softmax', dropout: float=0.0, attention_dropout: float | None=None, residual_dropout: float | None=None):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError('hidden_size must be divisible by num_heads')
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.attention_dropout_p = float(dropout if attention_dropout is None else attention_dropout)
        self.residual_dropout_p = float(dropout if residual_dropout is None else residual_dropout)
        if not 0.0 <= self.attention_dropout_p < 1.0:
            raise ValueError('attention_dropout must be in [0, 1)')
        if not 0.0 <= self.residual_dropout_p < 1.0:
            raise ValueError('residual_dropout must be in [0, 1)')
        self.residual_dropout = nn.Dropout(self.residual_dropout_p)
        self.input_norm = nn.LayerNorm(hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.post_norm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(nn.Linear(hidden_size, 4 * hidden_size), nn.GELU(approximate='tanh'), nn.Linear(4 * hidden_size, hidden_size))
        self.layerwise_target = target_hidden_size is not None
        if self.layerwise_target:
            self.target_hidden_size = int(target_hidden_size)
            self.num_target_layers = int(num_target_layers)
            self.target_fusion_type = str(target_fusion_type)
            if self.target_fusion_type not in ('softmax', 'concat'):
                raise ValueError(f'unsupported target_fusion_type: {self.target_fusion_type}')
            target_width = self.target_hidden_size
            if self.target_fusion_type == 'softmax':
                self.target_fusion_logits = nn.Parameter(torch.zeros(self.num_target_layers))
            else:
                target_width *= self.num_target_layers
            self.target_norm = nn.LayerNorm(target_width)
            self.target_k_proj = nn.Linear(target_width, hidden_size, bias=True)
            self.target_v_proj = nn.Linear(target_width, hidden_size, bias=True)

    def _heads(self, value: torch.Tensor) -> torch.Tensor:
        batch, length, _ = value.shape
        return value.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    def rotate(self, value, positions):
        if not hasattr(self, 'rope_inv_freq'):
            return value
        angles = positions.float().unsqueeze(-1) * self.rope_inv_freq.float()
        angles = torch.cat((angles, angles), -1).unsqueeze(1)
        half = value.shape[-1] // 2
        rotated = torch.cat((-value[..., half:], value[..., :half]), -1)
        return value * angles.cos().to(value.dtype) + rotated * angles.sin().to(value.dtype)

    def context_kv(self, target_context: torch.Tensor, positions=None) -> tuple[torch.Tensor, torch.Tensor]:
        if self.layerwise_target:
            expected = self.num_target_layers * self.target_hidden_size
            if target_context.shape[-1] != expected:
                raise ValueError(f'selected target width must be {expected}, got {target_context.shape[-1]}')
            if self.target_fusion_type == 'softmax':
                layers = target_context.unflatten(-1, (self.num_target_layers, self.target_hidden_size))
                weights = torch.softmax(self.target_fusion_logits.float(), dim=0).to(layers.dtype)
                fused = torch.einsum('bsth,t->bsh', layers, weights)
            else:
                fused = target_context
            fused = self.target_norm(fused)
            return (self._heads(self.target_k_proj(fused)), self._heads(self.target_v_proj(fused)))
        key = self._heads(self.k_proj(target_context))
        if hasattr(self, 'rope_inv_freq'):
            if positions is None:
                positions = torch.arange(target_context.shape[1], device=target_context.device)[None]
            key = self.rotate(key, positions)
        return (key, self._heads(self.v_proj(target_context)))

    def forward(self, hidden: torch.Tensor, *, context_k: torch.Tensor, context_v: torch.Tensor, context_mask: torch.Tensor | None, attention_mask: torch.Tensor | None=None, position_ids: torch.Tensor | None=None) -> torch.Tensor:
        residual = hidden
        normalized = self.input_norm(hidden)
        use_conv = hasattr(self, 'attention_conv')
        if use_conv:
            slot_count = self.conv_block_size if hidden.shape[1] >= self.conv_block_size else hidden.shape[1]
            normalized, attention_kernel = self.attention_conv.prepare(normalized.unflatten(1, (-1, slot_count)))
            normalized = normalized.flatten(1, 2)
        q = self._heads(self.q_proj(normalized))
        noise_k = self._heads(self.k_proj(normalized))
        noise_v = self._heads(self.v_proj(normalized))
        if hasattr(self, 'rope_inv_freq'):
            if position_ids is None:
                position_ids = torch.arange(hidden.shape[1], device=hidden.device)[None] + context_k.shape[2]
            q = self.rotate(q, position_ids)
            noise_k = self.rotate(noise_k, position_ids)
        key = torch.cat((context_k, noise_k), dim=2)
        value = torch.cat((context_v, noise_v), dim=2)
        if attention_mask is None and context_mask is not None:
            noise_allowed = torch.ones(context_mask.shape[0], hidden.shape[1], dtype=torch.bool, device=hidden.device)
            allowed = torch.cat((context_mask.bool(), noise_allowed), dim=1)
            attention_mask = allowed[:, None, None, :]
        attended = F.scaled_dot_product_attention(q, key, value, attn_mask=attention_mask, dropout_p=self.attention_dropout_p if self.training else 0.0, is_causal=False)
        attended = attended.transpose(1, 2).reshape_as(hidden)
        attended = self.o_proj(attended)
        if use_conv:
            attended = self.attention_conv.finish(attended.unflatten(1, (-1, slot_count)), attention_kernel).flatten(1, 2)
        hidden = residual + self.residual_dropout(attended)
        normalized = self.post_norm(hidden)
        if use_conv:
            normalized, mlp_kernel = self.mlp_conv.prepare(normalized.unflatten(1, (-1, slot_count)))
            normalized = normalized.flatten(1, 2)
        mlp = self.mlp(normalized)
        if use_conv:
            mlp = self.mlp_conv.finish(mlp.unflatten(1, (-1, slot_count)), mlp_kernel).flatten(1, 2)
        hidden = hidden + self.residual_dropout(mlp)
        return hidden

@dataclass
class DraftContextCache:
    keys: list[torch.Tensor]
    values: list[torch.Tensor]
    length: int
    recent_context: torch.Tensor
    recent_tokens: torch.Tensor
    persistent_markov_state: torch.Tensor
    proposal_markov_states: torch.Tensor
    recent_final_hidden: torch.Tensor

class CausalMarkovRefinementBlock(nn.Module):

    def __init__(self, rank: int, num_heads: int):
        super().__init__()
        self.rank = rank
        self.num_heads = num_heads
        self.head_dim = rank // num_heads
        self.norm = nn.LayerNorm(rank)
        self.qkv = nn.Linear(rank, 3 * rank, bias=False)
        self.output = nn.Linear(rank, rank, bias=False)
        self.mlp = MarkovResidualSwiGLUBlock(rank, expansion=2)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(hidden)
        q, k, v = self.qkv(normalized).chunk(3, dim=-1)
        length = hidden.shape[-2]
        leading = hidden.shape[:-2]
        q = q.view(*leading, length, self.num_heads, self.head_dim).transpose(-3, -2)
        k = k.view(*leading, length, self.num_heads, self.head_dim).transpose(-3, -2)
        v = v.view(*leading, length, self.num_heads, self.head_dim).transpose(-3, -2)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attended = attended.transpose(-3, -2).reshape(*leading, length, self.rank)
        return self.mlp(hidden + self.output(attended))

class CausalMarkovHead(nn.Module):
    """Causal full-prefix correction head for one speculative block.

    Row ``k`` receives the token immediately preceding target row ``k`` plus
    the parallel-backbone feature for row ``k``.  Causal attention lets it
    retain every earlier token/feature pair without exposing future rows.
    """

    def __init__(self, interface_size: int, rank: int, block_size: int, num_heads: int=8, depth: int=1) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError('causal Markov attention requires markov_rank > 0')
        if rank % num_heads:
            raise ValueError(f'markov_rank={rank} must be divisible by num_heads={num_heads}')
        self.rank = rank
        self.num_heads = num_heads
        self.head_dim = rank // num_heads
        self.hidden_projection = nn.Linear(interface_size, rank, bias=False)
        self.position_embedding = nn.Embedding(block_size, rank)
        self.input_norm = nn.LayerNorm(rank)
        self.qkv_projection = nn.Linear(rank, 3 * rank, bias=False)
        self.output_projection = nn.Linear(rank, rank, bias=False)
        self.output_norm = nn.LayerNorm(rank)
        self.refinement_blocks = nn.ModuleList([CausalMarkovRefinementBlock(rank, num_heads) for _ in range(max(0, int(depth) - 1))])

    def forward(self, previous_embeddings: torch.Tensor, hidden_states: torch.Tensor, initial_state: torch.Tensor | None=None) -> torch.Tensor:
        if previous_embeddings.shape[:-1] != hidden_states.shape[:-1]:
            raise ValueError(f'causal Markov token/hidden shapes disagree: {tuple(previous_embeddings.shape)} vs {tuple(hidden_states.shape)}')
        length = previous_embeddings.shape[-2]
        positions = torch.arange(length, device=previous_embeddings.device, dtype=torch.long)
        x = previous_embeddings + self.hidden_projection(hidden_states) + self.position_embedding(positions)
        if initial_state is not None:
            expected = previous_embeddings.shape[:-2] + (self.rank,)
            if initial_state.shape != expected:
                raise ValueError(f'causal initial state must have shape {expected}, got {tuple(initial_state.shape)}')
            x = x + initial_state.unsqueeze(-2).to(x.dtype)
        residual = x
        qkv = self.qkv_projection(self.input_norm(x))
        q, k, v = qkv.chunk(3, dim=-1)
        leading = q.shape[:-2]
        q = q.view(*leading, length, self.num_heads, self.head_dim).transpose(-3, -2)
        k = k.view(*leading, length, self.num_heads, self.head_dim).transpose(-3, -2)
        v = v.view(*leading, length, self.num_heads, self.head_dim).transpose(-3, -2)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attended = attended.transpose(-3, -2).reshape(*leading, length, self.rank)
        hidden = residual + self.output_projection(attended)
        for block in self.refinement_blocks:
            hidden = block(hidden)
        return self.output_norm(hidden)

    def incremental_step(self, previous_embedding: torch.Tensor, hidden_state: torch.Tensor, position: int, cache: list[tuple[torch.Tensor, torch.Tensor]] | None=None, initial_state: torch.Tensor | None=None) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Exact one-row causal decode with per-layer K/V caches."""
        if previous_embedding.ndim != 2 or hidden_state.ndim != 2:
            raise ValueError('incremental causal inputs must be [batch, width]')
        if previous_embedding.shape[0] != hidden_state.shape[0]:
            raise ValueError('incremental causal batch sizes disagree')
        x = previous_embedding + self.hidden_projection(hidden_state) + self.position_embedding.weight[int(position)].to(previous_embedding.dtype)
        if initial_state is not None:
            x = x + initial_state.to(x.dtype)
        old_cache = cache or []
        new_cache = []

        def attend_one(row, norm, qkv_projection, output_projection, cached):
            normalized = norm(row)
            q, k, v = qkv_projection(normalized).chunk(3, dim=-1)
            batch = q.shape[0]
            q = q.view(batch, self.num_heads, 1, self.head_dim)
            k = k.view(batch, self.num_heads, 1, self.head_dim)
            v = v.view(batch, self.num_heads, 1, self.head_dim)
            if cached is not None:
                k = torch.cat((cached[0], k), dim=-2)
                v = torch.cat((cached[1], v), dim=-2)
            attended = F.scaled_dot_product_attention(q, k, v)
            attended = attended.transpose(1, 2).reshape(batch, self.rank)
            return (row + output_projection(attended), (k, v))
        x, layer_cache = attend_one(x, self.input_norm, self.qkv_projection, self.output_projection, old_cache[0] if old_cache else None)
        new_cache.append(layer_cache)
        for index, block in enumerate(self.refinement_blocks, start=1):
            x, layer_cache = attend_one(x, block.norm, block.qkv, block.output, old_cache[index] if len(old_cache) > index else None)
            x = block.mlp(x)
            new_cache.append(layer_cache)
        return (self.output_norm(x), new_cache)

class IndexTTS2DSpark(nn.Module):

    def __init__(self, config: dict[str, Any], max_positions: int):
        super().__init__()
        self.config = dict(config)
        self.hidden_size = int(config['hidden_size'])
        self.architecture = str(config.get('architecture', 'shared_context'))
        if self.architecture not in ('shared_context', 'layerwise_fusion', 'official_qwen3'):
            raise ValueError(f'unsupported draft architecture: {self.architecture}')
        self.interface_size = int(config.get('interface_size', self.hidden_size))
        self.vocab_size = int(config['vocab_size'])
        self.block_size = int(config['block_size'])
        self.proposal_top_k = int(config.get('proposal_top_k', 0))
        if self.proposal_top_k < 0 or self.proposal_top_k > self.vocab_size:
            raise ValueError('proposal_top_k must be in [0, vocab_size]')
        self.markov_rank = int(config['markov_rank'])
        self.markov_state_size = int(config.get('markov_state_size', self.markov_rank))
        if self.markov_state_size <= 0:
            raise ValueError('markov_state_size must be positive')
        if self.markov_state_size < self.markov_rank:
            raise ValueError('markov_state_size must be >= markov_rank')
        self.optimized_rnn_runtime = bool(config.get('optimized_rnn_runtime', False))
        self.fused_markov_output = bool(config.get('fused_markov_output', False))
        self._folded_markov_token_table = None
        self.logit_residual_rank = int(config.get('logit_residual_rank', 0))
        self.markov_type = str(config.get('markov_type', 'vanilla')).lower()
        if self.markov_type not in ('vanilla', 'rnn', 'causal_attention', 'parallel_causal'):
            raise ValueError(f'unsupported markov_type: {self.markov_type}')
        self.markov_cell_type = str(config.get('markov_cell_type', 'linear')).lower()
        self.incremental_causal_runtime = bool(config.get('incremental_causal_runtime', False))
        self.markov_output_from_updated_state = bool(config.get('markov_output_from_updated_state', False))
        self.detach_hidden_from_markov = bool(config.get('detach_hidden_from_markov', False))
        if self.markov_cell_type not in ('linear', 'deep_residual', 'multiplicative', 'stacked_residual', 'codec_state', 'contextual_mealy', 'stacked_gru', 'stacked_gru_vq', 'vqvae_gru', 'position_aware', 'beta_position_aware', 'linear_wide'):
            raise ValueError('markov_cell_type must be linear, deep_residual, multiplicative, stacked_residual, codec_state, contextual_mealy, stacked_gru, stacked_gru_vq, vqvae_gru, position_aware, or beta_position_aware')
        if self.markov_output_from_updated_state and (self.markov_type != 'rnn' or self.markov_cell_type != 'contextual_mealy'):
            raise ValueError('markov_output_from_updated_state currently requires markov_type=rnn and markov_cell_type=contextual_mealy')
        if self.markov_cell_type not in ('codec_state', 'contextual_mealy', 'stacked_gru', 'stacked_gru_vq', 'vqvae_gru', 'position_aware', 'beta_position_aware', 'linear_wide') and self.markov_state_size != self.markov_rank:
            raise ValueError('markov_state_size may differ from markov_rank only for markov_cell_type=codec_state/contextual_mealy/stacked_gru/vqvae_gru/position_aware/beta_position_aware/linear_wide')
        self.markov_history_window = int(config.get('markov_history_window', 0))
        self.persistent_markov_state = bool(config.get('persistent_markov_state', False))
        self.markov_input_geometry = str(config.get('markov_input_geometry', 'free_embedding')).lower()
        if self.markov_input_geometry not in ('free_embedding', 'target_mel_embedding'):
            raise ValueError('markov_input_geometry must be free_embedding or target_mel_embedding')
        self.markov_output_geometry = str(config.get('markov_output_geometry', 'free_projection')).lower()
        if self.markov_output_geometry not in ('free_projection', 'target_mel_head'):
            raise ValueError('markov_output_geometry must be free_projection or target_mel_head')
        self.position_specific_markov_out = bool(config.get('position_specific_markov_out', False))
        if self.position_specific_markov_out and (self.markov_type != 'rnn' or self.markov_output_geometry != 'free_projection'):
            raise ValueError('position_specific_markov_out requires RNN + free_projection')
        if self.position_specific_markov_out and self.optimized_rnn_runtime:
            raise ValueError('optimized_rnn_runtime is not yet defined for position-specific output')
        self.latent_dynamics_head = bool(config.get('latent_dynamics_head', False))
        self.phase_conditioning_head = bool(config.get('phase_conditioning_head', False))
        self.fullband_joint_head = bool(config.get('fullband_joint_head', False))
        codec_head_count = sum((self.latent_dynamics_head, self.phase_conditioning_head, self.fullband_joint_head))
        if codec_head_count > 1:
            raise ValueError('codec-specific Markov heads are mutually exclusive')
        self.markov_final_history_window = int(config.get('markov_final_history_window', 0))
        if (self.latent_dynamics_head or self.phase_conditioning_head) and self.markov_final_history_window < 5:
            raise ValueError('latent/phase heads require at least 5 final-hidden history frames')
        if self.persistent_markov_state and self.markov_type != 'rnn':
            raise ValueError('persistent Markov state requires markov_type=rnn')
        if self.persistent_markov_state and self.markov_history_window > 0:
            raise ValueError('persistent state and boundary-history state are mutually exclusive')
        if self.markov_history_window < 0:
            raise ValueError('markov_history_window must be non-negative')
        if self.markov_history_window and self.markov_type not in ('rnn', 'parallel_causal'):
            raise ValueError('markov history is supported only with RNN or parallel causal Markov')
        self.enable_confidence_head = bool(config.get('enable_confidence_head', False))
        self.confidence_head_with_markov = bool(config.get('confidence_head_with_markov', True))
        if self.enable_confidence_head and self.confidence_head_with_markov:
            if self.markov_rank <= 0:
                raise ValueError('confidence_head_with_markov requires markov_rank > 0')
        self.target_layer_ids = [int(x) for x in config['target_layer_ids']]
        self.draft_initialization_layer_ids = [int(x) for x in config.get('draft_initialization_layer_ids', self.target_layer_ids)]
        self.include_target_final_hidden = bool(config.get('include_target_final_hidden', False))
        self.num_target_features = len(self.target_layer_ids) + int(self.include_target_final_hidden)
        self.context_fusion_mode = str(config.get('context_fusion_mode', 'concat_projection')).lower()
        if self.context_fusion_mode not in ('concat_projection', 'depth_aligned', 'global_softmax'):
            raise ValueError('unsupported context_fusion_mode')
        self.recent_context_layers = int(config.get('recent_context_layers', 0))
        self.recent_context_window = int(config.get('recent_context_window', 0))
        self.dropout = float(config.get('dropout', 0.0))
        self.attention_dropout = float(config.get('attention_dropout', self.dropout))
        self.residual_dropout = float(config.get('residual_dropout', self.dropout))
        self.context_feature_dropout = float(config.get('context_feature_dropout', 0.0))
        self.query_temporal_kernel = int(config.get('query_temporal_kernel', 0))
        self.midblock_refresh_at = int(config.get('midblock_refresh_at', 0))
        if self.midblock_refresh_at and (not 0 < self.midblock_refresh_at < self.block_size):
            raise ValueError('midblock_refresh_at must be 0 or in [1, block_size-1]')
        if self.query_temporal_kernel not in (0, 3, 5, 7):
            raise ValueError('query_temporal_kernel must be 0 or an odd value in {3,5,7}')
        if not 0.0 <= self.context_feature_dropout < 1.0:
            raise ValueError('context_feature_dropout must be in [0, 1)')
        self.mask_token_id = int(config['mask_token_id'])
        self.token_embedding = nn.Embedding(self.vocab_size, self.interface_size)
        self.position_embedding = nn.Embedding(max_positions, self.interface_size)
        self.mask_embedding = nn.Parameter(torch.empty(self.interface_size))
        if self.architecture == 'official_qwen3':
            self.input_projection = nn.Linear(self.interface_size, self.hidden_size, bias=False)
            self.context_projection = nn.Linear(self.num_target_features * self.interface_size, self.hidden_size, bias=False)
            rms_norm_eps = float(config.get('rms_norm_eps', 1e-06))
            self.context_norm = RMSNorm(self.hidden_size, eps=rms_norm_eps)
            self.layers = nn.ModuleList([OfficialQwenDSparkLayer(self.hidden_size, num_attention_heads=int(config.get('num_attention_heads', 32)), num_key_value_heads=int(config.get('num_key_value_heads', 8)), head_dim=int(config.get('head_dim', 128)), intermediate_size=int(config.get('intermediate_size', 9728)), rms_norm_eps=rms_norm_eps, rope_theta=float(config.get('rope_theta', 1000000.0)), attention_dropout=self.attention_dropout) for _ in range(int(config['num_layers']))])
            self.output_projection = nn.Linear(self.hidden_size, self.interface_size, bias=False)
        elif self.architecture == 'shared_context':
            if self.interface_size != self.hidden_size:
                raise ValueError('shared_context requires interface_size == hidden_size')
            self.input_projection = nn.Identity()
            if self.context_fusion_mode == 'depth_aligned':
                if self.num_target_features != int(config['num_layers']):
                    raise ValueError('depth-aligned context requires one target feature per Draft layer')
                self.context_projection = None
                self.context_projections = nn.ModuleList([nn.Linear(self.interface_size, self.hidden_size, bias=False) for _ in range(int(config['num_layers']))])
            elif self.context_fusion_mode == 'global_softmax':
                self.context_projection = nn.Linear(self.interface_size, self.hidden_size, bias=False)
                self.context_fusion_logits = nn.Parameter(torch.zeros(self.num_target_features))
                self.context_projections = None
            else:
                self.context_projection = nn.Linear(self.num_target_features * self.interface_size, self.hidden_size, bias=False)
                self.context_projections = None
            self.context_norm = nn.LayerNorm(self.hidden_size)
            self.layers = nn.ModuleList([DraftLayer(self.hidden_size, int(config['num_heads']), dropout=self.dropout, attention_dropout=self.attention_dropout, residual_dropout=self.residual_dropout) for _ in range(int(config['num_layers']))])
            self.output_projection = nn.Identity()
        else:
            self.input_projection = nn.Linear(self.interface_size, self.hidden_size, bias=False)
            self.context_projection = nn.Identity()
            self.context_norm = nn.Identity()
            self.layers = nn.ModuleList([DraftLayer(self.hidden_size, int(config['num_heads']), target_hidden_size=self.interface_size, num_target_layers=self.num_target_features, target_fusion_type=str(config.get('target_fusion_type', 'softmax')), dropout=self.dropout, attention_dropout=self.attention_dropout, residual_dropout=self.residual_dropout) for _ in range(int(config['num_layers']))])
            self.output_projection = nn.Linear(self.hidden_size, self.interface_size, bias=False)
        self.query_temporal_norms = nn.ModuleList()
        self.query_temporal_convs = nn.ModuleList()
        if self.query_temporal_kernel > 0:
            for _ in self.layers:
                self.query_temporal_norms.append(nn.LayerNorm(self.hidden_size))
                self.query_temporal_convs.append(nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=self.query_temporal_kernel, padding=self.query_temporal_kernel // 2, groups=self.hidden_size, bias=True))
        if self.architecture == 'official_qwen3':
            self.output_norm = RMSNorm(self.hidden_size, eps=float(config.get('rms_norm_eps', 1e-06)))
            self.interface_output_norm = nn.LayerNorm(self.interface_size)
        else:
            self.output_norm = nn.LayerNorm(self.hidden_size)
            self.interface_output_norm = nn.Identity() if self.architecture == 'shared_context' else nn.LayerNorm(self.interface_size)
        self.target_output_norm = nn.LayerNorm(self.interface_size)
        self.lm_head = nn.Linear(self.interface_size, self.vocab_size, bias=True)
        self.random_rope_draft = bool(config.get('random_rope_draft', False))
        self.scratch_absolute_position = bool(config.get('scratch_absolute_position', False))
        if self.scratch_absolute_position and (not self.random_rope_draft):
            raise ValueError('scratch absolute positions require the scratch RMSNorm branch')
        if self.random_rope_draft:
            if self.architecture != 'shared_context':
                raise ValueError('random_rope_draft requires shared_context')
            self.target_lm_head = nn.Linear(self.interface_size, self.vocab_size, bias=True)
            self.target_lm_head.requires_grad_(False)
            self.target_output_norm.requires_grad_(False)
            self.context_norm = RMSNorm(self.hidden_size)
            self.output_norm = RMSNorm(self.hidden_size)
            for layer in self.layers:
                layer.input_norm = RMSNorm(self.hidden_size)
                layer.post_norm = RMSNorm(self.hidden_size)
                if not self.scratch_absolute_position:
                    layer.register_buffer('rope_inv_freq', 1.0 / float(config.get('rope_theta', 10000.0)) ** (torch.arange(0, layer.head_dim, 2).float() / layer.head_dim), persistent=False)
            self.position_embedding = nn.Embedding(max_positions, self.interface_size if self.scratch_absolute_position else 0)
        if self.logit_residual_rank > 0:
            self.logit_residual_in = nn.Linear(self.interface_size, self.logit_residual_rank, bias=False)
            self.logit_residual_out = nn.ModuleList([nn.Linear(self.logit_residual_rank, self.vocab_size, bias=False) for _ in range(self.block_size)])
        if self.markov_input_geometry == 'target_mel_embedding':
            self.markov_in = None
            self.markov_input_projection = nn.Linear(self.interface_size, self.markov_rank, bias=False)
        else:
            self.markov_in = nn.Embedding(self.vocab_size, self.markov_rank)
            self.markov_input_projection = None
        if self.markov_output_geometry == 'target_mel_head':
            self.markov_out = None
            self.markov_out_by_position = None
            self.markov_output_projection = nn.Linear(self.markov_rank, self.interface_size, bias=False)
        else:
            self.markov_out = None if self.position_specific_markov_out else nn.Linear(self.markov_rank, self.vocab_size, bias=False)
            self.markov_out_by_position = nn.ModuleList([nn.Linear(self.markov_rank, self.vocab_size, bias=False) for _ in range(self.block_size)]) if self.position_specific_markov_out else None
            self.markov_output_projection = None
        self.register_buffer('markov_coefficient_scale', torch.ones(self.markov_rank), persistent=False)
        scale_initializer = config.get('markov_coefficient_scale_initializer')
        if scale_initializer:
            tensors = load_file(str(Path(scale_initializer).resolve()), device='cpu')
            if 'component_scale' not in tensors:
                raise ValueError('coefficient scale initializer lacks component_scale')
            scale = tensors['component_scale'].float()
            if scale.shape != self.markov_coefficient_scale.shape:
                raise ValueError('Markov coefficient scale shape mismatch')
            self.markov_coefficient_scale.copy_(scale)
        confidence_width = self.interface_size
        if self.confidence_head_with_markov:
            confidence_width += self.markov_rank
        self.confidence_head = nn.Linear(confidence_width, 1) if self.enable_confidence_head else None
        if self.markov_type == 'rnn':
            cell_input = self.markov_state_size + self.markov_rank + self.interface_size
            if self.markov_cell_type == 'linear':
                self.markov_rnn = nn.Linear(cell_input, 3 * self.markov_rank)
            elif self.markov_cell_type == 'linear_wide':
                self.markov_rnn = nn.Linear(cell_input, 2 * self.markov_state_size + self.markov_rank)
            elif self.markov_cell_type == 'beta_position_aware':
                self.markov_rnn = nn.Linear(cell_input, 2 * self.markov_state_size + self.markov_rank)
                self.beta_projection = nn.Linear(self.markov_rank, 64, bias=True)
                self.position_beta_weight = nn.Parameter(torch.empty(self.block_size, 64))
                self.position_beta_bias = nn.Parameter(torch.ones(self.block_size))
                nn.init.normal_(self.position_beta_weight, mean=0.0, std=0.02)
            elif self.markov_cell_type == 'position_aware':
                self.markov_rnn = nn.Linear(cell_input, 2 * self.markov_state_size + self.markov_rank)
                raw_width = 2 * self.markov_state_size + self.markov_rank
                self.position_raw_bias = nn.Parameter(torch.zeros(self.block_size, raw_width))
                with torch.no_grad():
                    self.position_raw_bias[:, :self.markov_state_size] = torch.linspace(-1.0, 1.0, self.block_size)[:, None]
                mix = torch.linspace(0.2, 0.8, self.block_size).clamp(0.0001, 1 - 0.0001)
                self.position_memory_mix = nn.Parameter(mix)
            elif self.markov_cell_type == 'deep_residual':
                width = int(config.get('markov_cell_width', 4 * self.markov_rank))
                self.markov_rnn = nn.Linear(cell_input, 3 * self.markov_rank)
                depth = int(config.get('markov_cell_depth', 1))
                if depth <= 1:
                    self.markov_deep_conditioner = nn.Sequential(nn.Linear(self.interface_size, width), nn.SiLU(), nn.Linear(width, 3 * self.markov_rank))
                else:
                    self.markov_deep_conditioner = nn.Sequential(nn.Linear(self.interface_size, width), nn.SiLU(), *[MarkovResidualSwiGLUBlock(width, expansion=2) for _ in range(depth)], nn.LayerNorm(width), nn.Linear(width, 3 * self.markov_rank))
            elif self.markov_cell_type == 'multiplicative':
                self.markov_rnn = nn.Linear(cell_input, 3 * self.markov_rank)
                self.markov_film_conditioner = nn.Linear(self.interface_size, 6 * self.markov_rank)
            elif self.markov_cell_type == 'stacked_residual':
                width = int(config.get('markov_cell_width', 4 * self.markov_rank))
                depth = int(config.get('markov_cell_depth', 4))
                if depth < 1:
                    raise ValueError('markov_cell_depth must be positive')
                self.markov_rnn = nn.Sequential(nn.Linear(cell_input, width), nn.SiLU(), *[MarkovResidualSwiGLUBlock(width, expansion=2) for _ in range(depth)], nn.LayerNorm(width), nn.Linear(width, 3 * self.markov_rank))
            elif self.markov_cell_type in ('stacked_gru', 'stacked_gru_vq', 'vqvae_gru'):
                self.markov_gru_layers = int(config.get('markov_gru_layers', 2))
                if self.markov_gru_layers < 2:
                    raise ValueError('stacked_gru requires at least two layers')
                expected_state = self.markov_gru_layers * self.markov_rank
                if self.markov_state_size != expected_state:
                    raise ValueError(f'stacked_gru requires markov_state_size == markov_gru_layers * markov_rank ({expected_state})')
                self.markov_gru_cells = nn.ModuleList([nn.GRUCell(self.markov_rank + self.interface_size, self.markov_rank) for _ in range(self.markov_gru_layers)])
                readout_width = int(config.get('markov_cell_width', 4 * self.markov_rank))
                self.markov_gru_readout = nn.Sequential(nn.RMSNorm(self.markov_state_size + self.interface_size), nn.Linear(self.markov_state_size + self.interface_size, readout_width, bias=False), nn.SiLU(), nn.Linear(readout_width, self.markov_rank, bias=False))
                if self.markov_cell_type == 'stacked_gru_vq':
                    self.markov_vq_groups = int(config.get('markov_vq_groups', 8))
                    self.markov_vq_codes = int(config.get('markov_vq_codes', 256))
                    if self.markov_rank % self.markov_vq_groups:
                        raise ValueError('markov_rank must be divisible by markov_vq_groups')
                    group_width = self.markov_rank // self.markov_vq_groups
                    self.markov_vq_codebook = nn.Parameter(torch.empty(self.markov_vq_groups, self.markov_vq_codes, group_width))
                    nn.init.normal_(self.markov_vq_codebook, 0.0, 0.02)
                elif self.markov_cell_type == 'vqvae_gru':
                    self.markov_vq_groups = int(config.get('markov_vq_groups', 8))
                    self.markov_vq_codes = int(config.get('markov_vq_codes', 256))
                    self.markov_vq_latent_size = int(config.get('markov_vq_latent_size', self.markov_rank))
                    self.markov_vqvae_width = int(config.get('markov_vqvae_width', 4 * self.markov_rank))
                    if self.markov_vq_latent_size % self.markov_vq_groups:
                        raise ValueError('markov_vq_latent_size must be divisible by markov_vq_groups')
                    self.markov_vq_encoder = nn.Sequential(nn.RMSNorm(self.markov_state_size), nn.Linear(self.markov_state_size, self.markov_vqvae_width, bias=False), nn.SiLU(), nn.Linear(self.markov_vqvae_width, self.markov_vq_latent_size, bias=False))
                    self.markov_vq_decoder = nn.Sequential(nn.RMSNorm(self.markov_vq_latent_size), nn.Linear(self.markov_vq_latent_size, self.markov_vqvae_width, bias=False), nn.SiLU(), nn.Linear(self.markov_vqvae_width, self.markov_state_size, bias=False))
                    group_width = self.markov_vq_latent_size // self.markov_vq_groups
                    self.markov_vq_codebook = nn.Parameter(torch.empty(self.markov_vq_groups, self.markov_vq_codes, group_width))
                    nn.init.normal_(self.markov_vq_codebook, 0.0, 0.02)
                    self._vqvae_aux_terms: list[dict[str, torch.Tensor]] = []
                    self._collect_vqvae_aux = False
                self.markov_rnn = None
            elif self.markov_cell_type == 'contextual_mealy':
                if self.markov_history_window or self.persistent_markov_state:
                    raise ValueError('contextual_mealy currently requires block-local state')
                self.contextual_codec_projection = nn.Linear(self.interface_size, self.markov_rank, bias=False)
                self.contextual_velocity = bool(config.get('contextual_velocity', False))
                self.contextual_velocity_projection = nn.Linear(self.interface_size, self.markov_rank, bias=False) if self.contextual_velocity else None
                contextual_cell_input = self.markov_state_size + 2 * self.markov_rank + 1 + self.interface_size
                contextual_cell_output = 2 * self.markov_state_size
                if not self.markov_output_from_updated_state:
                    contextual_cell_output += self.markov_rank
                self.markov_rnn = nn.Linear(contextual_cell_input, contextual_cell_output)
                self.markov_updated_state_norm = nn.RMSNorm(self.markov_state_size) if self.markov_output_from_updated_state else None
                self.markov_updated_state_to_output = nn.Linear(self.markov_state_size, self.markov_rank, bias=False) if self.markov_output_from_updated_state else None
            else:
                if self.markov_history_window or self.persistent_markov_state:
                    raise ValueError('codec_state RNN currently requires block-local state')
                self.codec_context_projection = nn.Linear(self.interface_size, self.markov_rank, bias=False)
                self.codec_recurrent_projection = nn.Linear(self.markov_state_size + self.markov_rank + 4, 2 * self.markov_state_size, bias=False)
                self.codec_hidden_to_state = nn.Linear(self.interface_size, 2 * self.markov_state_size, bias=True)
                self.codec_hidden_to_output = nn.Linear(self.interface_size, self.markov_rank, bias=False)
                self.markov_position_scale = nn.Parameter(torch.ones(self.block_size, self.markov_rank))
                self.markov_position_bias = nn.Parameter(torch.zeros(self.block_size, self.markov_rank))
                self.markov_rnn = None
        if self.latent_dynamics_head:
            self.latent_history_projection = nn.Linear(self.interface_size, self.markov_rank, bias=False)
            self.latent_history_gru = nn.GRU(self.markov_rank, self.markov_rank, batch_first=True)
            self.latent_query_projection = nn.Linear(self.interface_size, self.markov_rank, bias=False)
            self.latent_horizon_embedding = nn.Parameter(torch.empty(self.block_size, self.markov_rank))
            self.latent_mlp = nn.Sequential(nn.Linear(self.markov_rank, 4 * self.markov_rank), nn.SiLU(), nn.Linear(4 * self.markov_rank, self.markov_rank))
            self.latent_output_projection = nn.Linear(self.markov_rank, self.interface_size, bias=False)
        else:
            self.latent_history_projection = None
        if self.phase_conditioning_head:
            self.phase_projection = nn.Linear(4 * self.interface_size, self.markov_rank, bias=False)
            self.phase_horizon_embedding = nn.Parameter(torch.empty(self.block_size, self.markov_rank))
            self.phase_to_rnn = nn.Linear(self.markov_rank, 3 * self.markov_rank, bias=False)
        else:
            self.phase_projection = None
        if self.fullband_joint_head:
            self.fullband_input_projection = nn.Linear(self.interface_size, self.markov_rank, bias=False)
            self.fullband_position_embedding = nn.Parameter(torch.empty(self.block_size, self.markov_rank))
            self.fullband_layer = nn.TransformerEncoderLayer(d_model=self.markov_rank, nhead=8, dim_feedforward=4 * self.markov_rank, dropout=0.0, activation='gelu', batch_first=True, norm_first=True)
            self.fullband_output_projection = nn.Linear(self.markov_rank, self.interface_size, bias=False)
        else:
            self.fullband_input_projection = None
        if self.markov_type in ('rnn', 'parallel_causal'):
            self.markov_boundary_gru = nn.GRUCell(self.hidden_size + self.markov_rank, self.markov_rank) if self.markov_history_window > 0 else None
        if self.markov_type in ('causal_attention', 'parallel_causal'):
            self.markov_causal_attention = CausalMarkovHead(self.interface_size, self.markov_rank, self.block_size, num_heads=int(config.get('markov_attention_heads', 8)), depth=int(config.get('markov_attention_depth', 1)))
        self.apply(self._initialize_module)
        if bool(config.get('zero_initialize_markov_out', False)):
            outputs = list(self.markov_out_by_position) if self.markov_out_by_position is not None else [self.markov_out] if self.markov_out is not None else []
            if not outputs:
                raise ValueError('zero_initialize_markov_out requires free Markov output')
            for output in outputs:
                nn.init.zeros_(output.weight)
        if bool(config.get('zero_initialize_markov_output_coefficients', False)):
            if self.markov_type != 'rnn' or self.markov_cell_type not in ('linear', 'linear_wide'):
                raise ValueError('zero_initialize_markov_output_coefficients requires linear RNN cell')
            with torch.no_grad():
                output_begin = 2 * self.markov_state_size
                self.markov_rnn.weight[output_begin:].zero_()
                self.markov_rnn.bias[output_begin:].zero_()
        if self.markov_input_projection is not None:
            nn.init.orthogonal_(self.markov_input_projection.weight)
        if self.markov_output_projection is not None:
            nn.init.orthogonal_(self.markov_output_projection.weight)
        for convolution in self.query_temporal_convs:
            nn.init.zeros_(convolution.weight)
            nn.init.zeros_(convolution.bias)
        if self.architecture != 'official_qwen3':
            residual_std = 0.02 / math.sqrt(2.0 * len(self.layers))
            for layer in self.layers:
                nn.init.normal_(layer.o_proj.weight, mean=0.0, std=residual_std)
                nn.init.normal_(layer.mlp[-1].weight, mean=0.0, std=residual_std)
        nn.init.normal_(self.mask_embedding, mean=0.0, std=0.02)
        for name in ('latent_horizon_embedding', 'phase_horizon_embedding', 'fullband_position_embedding'):
            value = getattr(self, name, None)
            if value is not None:
                nn.init.normal_(value, mean=0.0, std=0.02)
        if self.latent_dynamics_head:
            nn.init.zeros_(self.latent_output_projection.weight)
        if self.phase_conditioning_head:
            nn.init.zeros_(self.phase_to_rnn.weight)
        if self.fullband_joint_head:
            nn.init.zeros_(self.fullband_output_projection.weight)
        if self.logit_residual_rank > 0:
            for output in self.logit_residual_out:
                nn.init.zeros_(output.weight)
        if bool(config.get('dflash_dynamic_local_conv', False)):
            if self.architecture != 'shared_context':
                raise ValueError('dynamic local conv requires shared_context')
            from inspark_infer.models.indextts2.dspark.dynamic_conv import GroupedDynamicCausalConv
            for layer in self.layers:
                layer.conv_block_size = self.block_size
                layer.attention_conv = GroupedDynamicCausalConv(self.hidden_size, kernel_size=int(config.get('dynamic_local_conv_kernel_size', 2)), mode=str(config.get('dynamic_local_conv_mode', 'full')))
                layer.mlp_conv = GroupedDynamicCausalConv(self.hidden_size, kernel_size=int(config.get('dynamic_local_conv_kernel_size', 2)), mode=str(config.get('dynamic_local_conv_mode', 'full')))

    def train(self, mode: bool=True):
        if mode:
            self._folded_markov_token_table = None
        return super().train(mode)

    @staticmethod
    def _initialize_module(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.GRUCell):
            nn.init.normal_(module.weight_ih, mean=0.0, std=0.02)
            nn.init.normal_(module.weight_hh, mean=0.0, std=0.02)
            nn.init.zeros_(module.bias_ih)
            nn.init.zeros_(module.bias_hh)

    def _noise_embeddings(self, anchor_tokens: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        batch = anchor_tokens.shape[0]
        hidden = self.mask_embedding.view(1, 1, -1).expand(batch, self.block_size, -1).clone()
        hidden[:, 0, :] = self.token_embedding(anchor_tokens.long())
        if self.architecture != 'official_qwen3' and (not self.random_rope_draft or self.scratch_absolute_position):
            hidden = hidden + self.position_embedding(position_ids.long())
        return self.input_projection(hidden)

    def project_context(self, selected_hidden: torch.Tensor, layer_index: int | None=None) -> torch.Tensor:
        if self.context_fusion_mode == 'depth_aligned':
            if layer_index is None:
                raise ValueError('depth-aligned context requires layer_index')
            layers = selected_hidden.unflatten(-1, (self.num_target_features, self.interface_size))
            projected = self.context_projections[layer_index](layers[..., layer_index, :])
        elif self.context_fusion_mode == 'global_softmax':
            layers = selected_hidden.unflatten(-1, (self.num_target_features, self.interface_size))
            weights = torch.softmax(self.context_fusion_logits.float(), dim=0).to(layers.dtype)
            fused = torch.einsum('...th,t->...h', layers, weights)
            projected = self.context_projection(fused)
        else:
            projected = self.context_projection(selected_hidden)
        return self.context_norm(projected)

    def prepare_context(self, selected_hidden: torch.Tensor, final_hidden: torch.Tensor | None=None) -> torch.Tensor:
        if self.include_target_final_hidden and final_hidden is not None:
            selected_hidden = torch.cat((selected_hidden, final_hidden), dim=-1)
        expected = self.num_target_features * self.interface_size
        if selected_hidden.shape[-1] != expected:
            raise ValueError(f'target context width must be {expected}, got {selected_hidden.shape[-1]}')
        if self.training and self.context_feature_dropout > 0.0:
            layers = selected_hidden.unflatten(-1, (self.num_target_features, self.interface_size))
            keep = torch.rand(layers.shape[0], 1, self.num_target_features, 1, device=layers.device) >= self.context_feature_dropout
            all_dropped = ~keep.any(dim=2, keepdim=True)
            first_kept = torch.zeros_like(keep)
            first_kept[:, :, 0, :] = True
            keep = torch.where(all_dropped, first_kept, keep)
            selected_hidden = (layers * keep.to(layers.dtype) / (1.0 - self.context_feature_dropout)).flatten(-2)
        return selected_hidden

    def project_output(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = self.output_norm(hidden)
        return self.interface_output_norm(self.output_projection(hidden))

    def apply_query_temporal(self, hidden: torch.Tensor, layer_index: int, *, grouped_anchors: int | None=None) -> torch.Tensor:
        if self.query_temporal_kernel <= 0:
            return hidden
        if grouped_anchors is None:
            blocks = hidden
            leading = hidden.shape[:-2]
        else:
            batch = hidden.shape[0]
            blocks = hidden.unflatten(1, (grouped_anchors, self.block_size)).flatten(0, 1)
            leading = (batch, grouped_anchors)
        residual = blocks
        filtered = self.query_temporal_convs[layer_index](self.query_temporal_norms[layer_index](blocks).transpose(-2, -1)).transpose(-2, -1)
        blocks = residual + filtered
        if grouped_anchors is None:
            return blocks
        return blocks.unflatten(0, leading).flatten(1, 2)

    def base_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(hidden)
        if self.logit_residual_rank <= 0:
            return logits
        latent = torch.tanh(self.logit_residual_in(hidden))
        residuals = [output(latent[..., position, :]) for position, output in enumerate(self.logit_residual_out)]
        return logits + torch.stack(residuals, dim=-2)

    def _prepare_optimized_linear_rnn(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Move all token/hidden-only linear work off the sampled critical path."""
        if self.markov_cell_type not in ('linear', 'linear_wide') or self.markov_in is None:
            raise ValueError('optimized_rnn_runtime currently requires linear RNN and free embedding')
        state_end = self.markov_state_size
        token_end = state_end + self.markov_rank
        weight = self.markov_rnn.weight
        state_weight = weight[:, :state_end]
        token_weight = weight[:, state_end:token_end]
        hidden_weight = weight[:, token_end:]
        if self._folded_markov_token_table is None:
            self._folded_markov_token_table = F.linear(self.markov_in.weight, token_weight).detach()
        hidden_term = F.linear(hidden, hidden_weight, self.markov_rnn.bias)
        return (state_weight, self._folded_markov_token_table, hidden_term)

    def _optimized_linear_rnn_step(self, state: torch.Tensor, previous_tokens: torch.Tensor, state_weight: torch.Tensor, token_table: torch.Tensor, hidden_term: torch.Tensor, output_weight: torch.Tensor, output_base: torch.Tensor | None=None) -> tuple[torch.Tensor, torch.Tensor]:
        raw = F.linear(state, state_weight) + F.embedding(previous_tokens.long(), token_table).to(state.dtype) + hidden_term
        if self.markov_cell_type == 'linear_wide':
            gate_raw, candidate_raw, output_raw = raw.split((self.markov_state_size, self.markov_state_size, self.markov_rank), dim=-1)
        else:
            gate_raw, candidate_raw, output_raw = raw.chunk(3, dim=-1)
        gate = torch.sigmoid(gate_raw)
        state = gate * state + (1.0 - gate) * torch.tanh(candidate_raw)
        bounded = torch.tanh(output_raw)
        if output_base is None:
            output = F.linear(bounded, output_weight)
        else:
            output = torch.addmm(output_base.reshape(-1, self.vocab_size), bounded.reshape(-1, self.markov_rank), output_weight.transpose(0, 1)).reshape(*bounded.shape[:-1], self.vocab_size)
        return (state, output)

    def _effective_markov_output_weight(self, position_index: int | None=None) -> torch.Tensor:
        if self.markov_output_projection is None:
            if self.markov_out_by_position is not None:
                if position_index is None:
                    raise ValueError('position-specific Markov output requires position_index')
                return self.markov_out_by_position[position_index].weight
            if self.markov_out is None:
                raise RuntimeError('free Markov output projection is missing')
            return self.markov_out.weight
        return self.lm_head.weight @ self.markov_output_projection.weight

    def empty_cache(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> DraftContextCache:
        keys = [torch.empty(batch_size, getattr(layer, 'num_kv_heads', layer.num_heads), 0, layer.head_dim, device=device, dtype=dtype) for layer in self.layers]
        values = [tensor.clone() for tensor in keys]
        return DraftContextCache(keys=keys, values=values, length=0, recent_context=torch.empty(batch_size, 0, self.hidden_size, device=device, dtype=dtype), recent_tokens=torch.empty(batch_size, 0, device=device, dtype=torch.long), persistent_markov_state=torch.zeros(batch_size, self.markov_state_size, device=device, dtype=dtype), proposal_markov_states=torch.empty(batch_size, 0, self.markov_state_size, device=device, dtype=dtype), recent_final_hidden=torch.empty(batch_size, 0, self.interface_size, device=device, dtype=dtype))

    @torch.inference_mode()
    def append_context(self, cache: DraftContextCache, selected_hidden: torch.Tensor, final_hidden: torch.Tensor | None=None, committed_tokens: torch.Tensor | None=None) -> None:
        if selected_hidden.shape[1] == 0:
            return
        prepared_context = self.prepare_context(selected_hidden, final_hidden)
        default_context = None if self.context_fusion_mode == 'depth_aligned' else self.project_context(prepared_context)
        context_positions = torch.arange(cache.length, cache.length + prepared_context.shape[1], device=prepared_context.device, dtype=torch.long).unsqueeze(0).expand(prepared_context.shape[0], -1)
        for index, layer in enumerate(self.layers):
            context = self.project_context(prepared_context, index) if self.context_fusion_mode == 'depth_aligned' else default_context
            if self.architecture == 'official_qwen3':
                key, value = layer.context_kv(context, context_positions)
            else:
                key, value = layer.context_kv(context, context_positions) if self.random_rope_draft else layer.context_kv(context)
            cache.keys[index] = torch.cat((cache.keys[index], key), dim=2)
            cache.values[index] = torch.cat((cache.values[index], value), dim=2)
        cache.length += int(selected_hidden.shape[1])
        if committed_tokens is not None and self.markov_history_window > 0:
            if committed_tokens.shape != selected_hidden.shape[:2]:
                raise ValueError(f'committed token/context shapes disagree: {tuple(committed_tokens.shape)} vs {tuple(selected_hidden.shape[:2])}')
            cache.recent_context = torch.cat((cache.recent_context, context), dim=1)[:, -self.markov_history_window:, :]
            cache.recent_tokens = torch.cat((cache.recent_tokens, committed_tokens.long()), dim=1)[:, -self.markov_history_window:]
        if committed_tokens is not None and self.markov_final_history_window > 0 and (final_hidden is not None):
            if final_hidden.shape[:2] != committed_tokens.shape:
                raise ValueError('final-hidden history and committed tokens disagree')
            cache.recent_final_hidden = torch.cat((cache.recent_final_hidden, final_hidden.to(cache.recent_final_hidden.dtype)), dim=1)[:, -self.markov_final_history_window:, :]

    @classmethod
    def from_checkpoint(cls, directory: Path, device: torch.device | str='cpu') -> 'IndexTTS2DSpark':
        config = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
        model = cls(config, max_positions=int(config['max_positions']))
        model.load_state_dict(load_file(str(directory / 'model.safetensors'), device=str(device)))
        return model.to(device)
