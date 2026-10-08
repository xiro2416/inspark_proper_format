"""NVML board telemetry for matched, sustained one-GPU inference runs.

Read-only sampling. Board watts are not a per-stage or per-request energy metric.
"""
from __future__ import annotations

import statistics
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


class PowerSampler:
    def __init__(self, physical_gpu: int, interval_s: float = 0.05):
        if physical_gpu < 0 or interval_s <= 0:
            raise ValueError('physical_gpu and interval_s must be valid')
        import pynvml
        self.nvml = pynvml
        self.physical_gpu = physical_gpu
        self.interval_s = interval_s
        self.stop_event = threading.Event()
        self.thread = None
        self.samples: list[dict] = []
        self.errors: list[str] = []
        self.handle = None

    def start(self):
        self.nvml.nvmlInit()
        self.handle = self.nvml.nvmlDeviceGetHandleByIndex(self.physical_gpu)
        self.stop_event.clear()
        self.samples.clear()
        self.thread = threading.Thread(target=self._loop, name='nvml-power-sampler', daemon=True)
        self.thread.start()
        return self

    def _loop(self):
        nvml = self.nvml
        while not self.stop_event.is_set():
            stamp = time.perf_counter_ns()
            try:
                utilization = nvml.nvmlDeviceGetUtilizationRates(self.handle)
                memory = nvml.nvmlDeviceGetMemoryInfo(self.handle)
                self.samples.append({
                    'time_ns': stamp,
                    'power_w': nvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0,
                    'gpu_util_percent': float(utilization.gpu),
                    'memory_util_percent': float(utilization.memory),
                    'memory_used_mib': memory.used / (1024.0 ** 2),
                    'sm_clock_mhz': float(nvml.nvmlDeviceGetClockInfo(self.handle, nvml.NVML_CLOCK_SM)),
                })
            except Exception as exc:
                self.errors.append(f'{type(exc).__name__}: {exc}')
            self.stop_event.wait(self.interval_s)

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=3.0)
        self.nvml.nvmlShutdown()
        self.thread = None
        return self.summary()

    def summary(self):
        if not self.samples:
            return {'status': 'no_samples', 'errors': self.errors, 'physical_gpu': self.physical_gpu}
        powers = [sample['power_w'] for sample in self.samples]
        utils = [sample['gpu_util_percent'] for sample in self.samples]
        clocks = [sample['sm_clock_mhz'] for sample in self.samples]
        memories = [sample['memory_used_mib'] for sample in self.samples]
        duration = (self.samples[-1]['time_ns'] - self.samples[0]['time_ns']) / 1e9
        return {
            'status': 'sampled',
            'physical_gpu': self.physical_gpu,
            'interval_requested_s': self.interval_s,
            'sample_count': len(self.samples),
            'sample_span_s': duration,
            'board_power_w': {'mean': statistics.mean(powers), 'p50': _percentile(powers, .5), 'p95': _percentile(powers, .95), 'max': max(powers)},
            'gpu_util_percent': {'mean': statistics.mean(utils), 'p50': _percentile(utils, .5), 'p95': _percentile(utils, .95)},
            'sm_clock_mhz_mean': statistics.mean(clocks),
            'memory_used_mib_peak': max(memories),
            'errors': self.errors,
        }
