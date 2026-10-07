# Copyright 2020 The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Small removed helpers used by the vendored IndexTTS GPT implementation.

Adapted from Hugging Face Transformers v4.52.1 pytorch_utils.py and
utils/model_parallel_utils.py. The inference model keeps its original math;
this is import compatibility with the native TRT-LLM Transformers 5.5 stack.
No process-wide monkeypatching or custom GPU kernel is performed.
"""
from math import ceil

import torch
from transformers.pytorch_utils import Conv1D


def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
    mask = torch.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads
    for head in heads:
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    return heads, torch.arange(len(mask))[mask].long()


def prune_conv1d_layer(layer, index, dim=1):
    index = index.to(layer.weight.device)
    weight = layer.weight.index_select(dim, index).detach().clone()
    bias = layer.bias.detach().clone() if dim == 0 else layer.bias[index].detach().clone()
    size = list(layer.weight.size()); size[dim] = len(index)
    new_layer = Conv1D(size[1], size[0]).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(weight.contiguous())
    new_layer.weight.requires_grad = True
    new_layer.bias.requires_grad = False
    new_layer.bias.copy_(bias.contiguous())
    new_layer.bias.requires_grad = True
    return new_layer


def assert_device_map(device_map, num_blocks):
    assigned = [item for values in device_map.values() for item in values]
    if len(set(assigned)) != len(assigned):
        raise ValueError("An attention block is assigned to more than one device")
    if set(assigned) != set(range(num_blocks)):
        raise ValueError("device_map must assign every existing attention block exactly once")


def get_device_map(n_layers, devices):
    width = int(ceil(n_layers / len(devices)))
    layers = list(range(n_layers))
    return dict(zip(devices, [layers[i:i + width] for i in range(0, n_layers, width)]))


def get_head_mask(model, head_mask, num_hidden_layers, is_attention_chunked=False):
    """Transformers 4.52 ModuleUtilsMixin behavior, removed in 5.5."""
    if head_mask is None:
        return [None] * num_hidden_layers
    if head_mask.dim() == 1:
        head_mask = head_mask[None, None, :, None, None].expand(num_hidden_layers, -1, -1, -1, -1)
    elif head_mask.dim() == 2:
        head_mask = head_mask[:, None, :, None, None]
    if head_mask.dim() != 5:
        raise ValueError(f"Expected 5D head mask, got {head_mask.dim()}")
    head_mask = head_mask.to(dtype=model.dtype)
    return head_mask.unsqueeze(-1) if is_attention_chunked else head_mask
