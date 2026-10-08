"""Equivalent mono 24-kHz pydub edge trim, without full byte/reverse copies.

Preserves the reference's int16 truncation, integer audioop RMS, millisecond
rounding/padding, 100-ms edge retention, and int16-to-float round trip.
Not an amplitude-threshold replacement for its 10-ms RMS detector.
"""
import audioop
import math
from functools import lru_cache
import numpy as np


def trim_edges_exact(wave):
    samples=trim_quantized_edges_exact(wave)
    return (samples.astype(np.float32) / 32768.0)[None, :]


def trim_quantized_edges_exact(wave):
    wave = np.asarray(wave, dtype=np.float32).reshape(-1)
    samples = (wave * 32768.0).clip(-32768, 32767).astype(np.int16)

    def duration(data):
        return round(1000 * len(data) / 24000)

    def slice_ms(data, start, end):
        stop = min(end, duration(data)) * 24
        begin = min(start, duration(data)) * 24
        out = data[begin:stop]
        missing = stop - begin - len(out)
        if missing:
            out = np.pad(out, (0, missing))
        return out

    def leading(data):
        trim = 0
        length = duration(data)
        while True:
            part = slice_ms(data, trim, trim + 10)
            rms = audioop.rms(part.tobytes(), 2)
            db = 20 * math.log10(rms / 32768) if rms else -math.inf
            if not (db < -50 and trim < length):
                return min(trim, length)
            trim += 10

    start = max(0, leading(samples) - 100)
    samples = slice_ms(samples, start, duration(samples))
    # RMS is order-independent; only scanned 10-ms chunks need byte materializing.
    reversed_view = samples[::-1]
    end_trim = max(0, leading(reversed_view) - 100)
    samples = slice_ms(reversed_view, end_trim, duration(reversed_view))[::-1]
    return samples


@lru_cache(maxsize=64)
def fade_curves_exact(samples):
    phase=np.linspace(0.,np.pi,samples,dtype=np.float32)
    curves=(.5*(1.-np.cos(phase)),.5*(1.+np.cos(phase)))
    for curve in curves:curve.setflags(write=False)
    return curves


def pcm_quantized_exact(wave,cache_fades=False):
    """Reuse edge-trim int16 state; preserve service fades/pause/final rounding."""
    samples=trim_quantized_edges_exact(wave)
    # abs(q / 32768) > 10**(-50/20) is exactly abs(q) >= 104.
    # Signed comparisons also handle -32768 without int16 abs overflow.
    active=(samples>103)|(samples<-103)
    first=int(active.argmax()) if active.size else 0
    if not active.size or not active[first]:return np.zeros(3600,dtype='<i2')
    last=active.size-1-int(active[::-1].argmax())
    audio=samples[first:].astype(np.float32)/32768.0
    last-=first
    trailing=len(audio)-last-1
    fade=min(480,max(1,(last+1)//2))
    if fade>=2:
        if cache_fades:fade_in,fade_out=fade_curves_exact(fade)
        else:
            phase=np.linspace(0.,np.pi,fade,dtype=np.float32)
            fade_in,fade_out=.5*(1.-np.cos(phase)),.5*(1.+np.cos(phase))
        audio[:fade]*=fade_in
        audio[last-fade+1:last+1]*=fade_out
    pcm=np.clip(np.rint(audio*32767.),-32768,32767).astype('<i2',copy=False)
    missing=max(0,3600-trailing)
    if missing:pcm=np.concatenate((pcm,np.zeros(missing,dtype='<i2')))
    return pcm


@lru_cache(maxsize=1)
def pcm_roundtrip_table_exact():
    """Every int16 payload mapped with the original Float32 round trip."""
    samples=np.arange(65536,dtype=np.uint16).view(np.int16)
    audio=samples.astype(np.float32)/32768.0
    table=np.clip(np.rint(audio*32767.),-32768,32767).astype('<i2')
    table.setflags(write=False)
    return table


def pcm_quantized_lut_exact(wave,cache_fades=False):
    """Avoid full-wave Float32 temporaries; keep original edge arithmetic."""
    samples=trim_quantized_edges_exact(wave)
    active=(samples>103)|(samples<-103)
    first=int(active.argmax()) if active.size else 0
    if not active.size or not active[first]:return np.zeros(3600,dtype='<i2')
    last=active.size-1-int(active[::-1].argmax())
    samples=samples[first:]
    last-=first
    trailing=len(samples)-last-1
    pcm=pcm_roundtrip_table_exact()[samples.view(np.uint16)]
    fade=min(480,max(1,(last+1)//2))
    if fade>=2:
        if cache_fades:fade_in,fade_out=fade_curves_exact(fade)
        else:
            phase=np.linspace(0.,np.pi,fade,dtype=np.float32)
            fade_in,fade_out=.5*(1.-np.cos(phase)),.5*(1.+np.cos(phase))
        for region,curve in ((slice(0,fade),fade_in),(slice(last-fade+1,last+1),fade_out)):
            audio=samples[region].astype(np.float32)/32768.0
            audio*=curve
            pcm[region]=np.clip(np.rint(audio*32767.),-32768,32767).astype('<i2')
    missing=max(0,3600-trailing)
    if missing:pcm=np.concatenate((pcm,np.zeros(missing,dtype='<i2')))
    return pcm


def ordered_pcm_chunks(waves,executor,chunk_size,cache_fades=False,start_index=0,use_lut=False):
    """Reduce CPU Future bookkeeping; retain exact samples and output order.

    Call only after publishing the original first PCM and synchronizing CPU wave
    storage. This groups CPU tasks, not model requests or GPU inference batches.
    """
    if chunk_size<1:raise ValueError('chunk_size must be positive')
    renderer=pcm_quantized_lut_exact if use_lut else pcm_quantized_exact
    def render(start):
        return [(start_index+k,renderer(waves[k],cache_fades))
                for k in range(start,min(start+chunk_size,len(waves)))]
    for chunk in executor.map(render,range(0,len(waves),chunk_size)):
        yield from chunk


def prepare_mono_exact(waveform, *, sample_rate, fade_seconds,
                       target_pause_seconds, silence_threshold_db):
    """Same service processing, but compute only required active endpoints."""
    audio = np.asarray(waveform, dtype=np.float32)
    if audio.ndim == 1: audio = audio[None, :]
    if audio.ndim != 2 or audio.shape[0] != 1:
        raise ValueError(f'ZipVoice waveform must be mono, got shape {audio.shape}')
    if sample_rate <= 0: raise ValueError('sample_rate must be positive')
    if fade_seconds < 0 or target_pause_seconds < 0:
        raise ValueError('fade and pause durations must be non-negative')
    threshold = 10.0 ** (float(silence_threshold_db) / 20.0)
    active = np.abs(audio[0]) > threshold
    pause = max(0, round(target_pause_seconds * sample_rate))
    first = int(active.argmax()) if active.size else 0
    if not active.size or not active[first]:
        return np.zeros((1, pause), dtype=np.float32)
    last = active.size - 1 - int(active[::-1].argmax())
    audio = audio[:, first:].copy()
    last -= first
    trailing = audio.shape[-1] - last - 1
    fade = min(max(0, round(fade_seconds * sample_rate)), max(1, (last + 1) // 2))
    if fade >= 2:
        phase = np.linspace(0., np.pi, fade, dtype=np.float32)
        audio[:, :fade] *= 0.5 * (1.0 - np.cos(phase))
        audio[:, last - fade + 1:last + 1] *= 0.5 * (1.0 + np.cos(phase))
    missing = max(0, pause - trailing)
    if missing:
        audio = np.concatenate((audio, np.zeros((1, missing), dtype=np.float32)), axis=-1)
    return audio
