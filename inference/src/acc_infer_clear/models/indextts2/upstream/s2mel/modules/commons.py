import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from munch import Munch
import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class AttrDict(dict):

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(mean, std)

def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)

def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape

def intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result

def kl_divergence(m_p, logs_p, m_q, logs_q):
    """KL(P||Q)"""
    kl = logs_q - logs_p - 0.5
    kl += 0.5 * (torch.exp(2.0 * logs_p) + (m_p - m_q) ** 2) * torch.exp(-2.0 * logs_q)
    return kl

def rand_gumbel(shape):
    """Sample from the Gumbel distribution, protect from overflows."""
    uniform_samples = torch.rand(shape) * 0.99998 + 1e-05
    return -torch.log(-torch.log(uniform_samples))

def rand_gumbel_like(x):
    g = rand_gumbel(x.size()).to(dtype=x.dtype, device=x.device)
    return g

def slice_segments(x, ids_str, segment_size=4):
    ret = torch.zeros_like(x[:, :, :segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        ret[i] = x[i, :, idx_str:idx_end]
    return ret

def slice_segments_audio(x, ids_str, segment_size=4):
    ret = torch.zeros_like(x[:, :segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        ret[i] = x[i, idx_str:idx_end]
    return ret

def rand_slice_segments(x, x_lengths=None, segment_size=4):
    b, d, t = x.size()
    if x_lengths is None:
        x_lengths = t
    ids_str_max = x_lengths - segment_size + 1
    ids_str = (torch.rand([b]).to(device=x.device) * ids_str_max).clip(0).to(dtype=torch.long)
    ret = slice_segments(x, ids_str, segment_size)
    return (ret, ids_str)

def get_timing_signal_1d(length, channels, min_timescale=1.0, max_timescale=10000.0):
    position = torch.arange(length, dtype=torch.float)
    num_timescales = channels // 2
    log_timescale_increment = math.log(float(max_timescale) / float(min_timescale)) / (num_timescales - 1)
    inv_timescales = min_timescale * torch.exp(torch.arange(num_timescales, dtype=torch.float) * -log_timescale_increment)
    scaled_time = position.unsqueeze(0) * inv_timescales.unsqueeze(1)
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], 0)
    signal = F.pad(signal, [0, 0, 0, channels % 2])
    signal = signal.view(1, channels, length)
    return signal

def add_timing_signal_1d(x, min_timescale=1.0, max_timescale=10000.0):
    b, channels, length = x.size()
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    return x + signal.to(dtype=x.dtype, device=x.device)

def cat_timing_signal_1d(x, min_timescale=1.0, max_timescale=10000.0, axis=1):
    b, channels, length = x.size()
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    return torch.cat([x, signal.to(dtype=x.dtype, device=x.device)], axis)

def subsequent_mask(length):
    mask = torch.tril(torch.ones(length, length)).unsqueeze(0).unsqueeze(0)
    return mask

def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act_part, s_act_part = torch.split(in_act, n_channels_int, dim=1)
    t_act = torch.tanh(t_act_part)
    s_act = torch.sigmoid(s_act_part)
    acts = t_act * s_act
    return acts

def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape

def shift_1d(x):
    x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [1, 0]]))[:, :, :-1]
    return x

def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)

def avg_with_mask(x, mask):
    assert mask.dtype == torch.float, 'Mask should be float'
    if mask.ndim == 2:
        mask = mask.unsqueeze(1)
    if mask.shape[1] == 1:
        mask = mask.expand_as(x)
    return (x * mask).sum() / mask.sum()

def generate_path(duration, mask):
    """
    duration: [b, 1, t_x]
    mask: [b, 1, t_y, t_x]
    """
    device = duration.device
    b, _, t_y, t_x = mask.shape
    cum_duration = torch.cumsum(duration, -1)
    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = path - F.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
    path = path.unsqueeze(1).transpose(2, 3) * mask
    return path

def log_norm(x, mean=-4, std=4, dim=2):
    """
    normalized log mel -> mel -> norm -> log(norm)
    """
    x = torch.log(torch.exp(x * std + mean).norm(dim=dim))
    return x
MATPLOTLIB_FLAG = False

def normalize_f0(f0_sequence):
    voiced_indices = np.where(f0_sequence > 0)[0]
    f0_voiced = f0_sequence[voiced_indices]
    log_f0 = np.log2(f0_voiced)
    mean_f0 = np.mean(log_f0)
    std_f0 = np.std(log_f0)
    normalized_f0 = (log_f0 - mean_f0) / std_f0
    normalized_sequence = np.zeros_like(f0_sequence)
    normalized_sequence[voiced_indices] = normalized_f0
    normalized_sequence[f0_sequence <= 0] = -1
    return normalized_sequence

class MyModel(nn.Module):

    def __init__(self, args, use_emovec=False, use_gpt_latent=False):
        super(MyModel, self).__init__()
        from acc_infer_clear.models.indextts2.upstream.s2mel.modules.flow_matching import CFM
        from acc_infer_clear.models.indextts2.upstream.s2mel.modules.length_regulator import InterpolateRegulator
        length_regulator = InterpolateRegulator(channels=args.length_regulator.channels, sampling_ratios=args.length_regulator.sampling_ratios, is_discrete=args.length_regulator.is_discrete, in_channels=args.length_regulator.in_channels if hasattr(args.length_regulator, 'in_channels') else None, vector_quantize=args.length_regulator.vector_quantize if hasattr(args.length_regulator, 'vector_quantize') else False, codebook_size=args.length_regulator.content_codebook_size, n_codebooks=args.length_regulator.n_codebooks if hasattr(args.length_regulator, 'n_codebooks') else 1, quantizer_dropout=args.length_regulator.quantizer_dropout if hasattr(args.length_regulator, 'quantizer_dropout') else 0.0, f0_condition=args.length_regulator.f0_condition if hasattr(args.length_regulator, 'f0_condition') else False, n_f0_bins=args.length_regulator.n_f0_bins if hasattr(args.length_regulator, 'n_f0_bins') else 512)
        if use_gpt_latent:
            self.models = nn.ModuleDict({'cfm': CFM(args), 'length_regulator': length_regulator, 'gpt_layer': torch.nn.Sequential(torch.nn.Linear(1280, 256), torch.nn.Linear(256, 128), torch.nn.Linear(128, 1024))})
        else:
            self.models = nn.ModuleDict({'cfm': CFM(args), 'length_regulator': length_regulator})

def load_checkpoint2(model, path):
    state = torch.load(path, map_location='cpu', weights_only=True)['net']
    for key, module in model.models.items():
        weights = {k.removeprefix('module.'): v for k, v in state[key].items()}
        module.load_state_dict(weights, strict=True)
    return model.eval()

