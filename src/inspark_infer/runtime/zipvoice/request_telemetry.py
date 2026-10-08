"""Select NVML observations inside the same ordered-PCM latency windows."""
import hashlib
import statistics
from pathlib import Path
from inspark_infer.runtime.zipvoice.telemetry import _percentile


def summarize_request_windows(samples, requests):
    windows = [(row['request_start_ns'], row['request_done_ns']) for row in requests]
    assert windows and all(begin < end for begin, end in windows)
    assert all(left[1] <= right[0] for left, right in zip(windows, windows[1:]))
    assert all(row['time_ns'] <= following['time_ns'] for row, following in zip(samples, samples[1:]))

    def summary(rows):
        if not rows:
            return dict(status='no_samples', sample_count=0)
        watts = [row['power_w'] for row in rows]
        return dict(status='sampled', sample_count=len(rows),
                    board_power_w=dict(mean=statistics.mean(watts), p50=_percentile(watts, .5),
                                       p95=_percentile(watts, .95), max=max(watts)))

    pooled, per_request = [], []
    for request, (begin, end) in zip(requests, windows):
        assert abs(request['all_pcm_s'] - (end - begin) / 1e9) < 1e-12
        selected = [row for row in samples if begin <= row['time_ns'] < end]
        pooled.extend(selected)
        per_request.append(dict(start_ns=begin, done_ns=end, **summary(selected)))
    root = Path(__file__).resolve().parents[3]
    result = dict(**summary(pooled), requests=per_request, raw_samples=samples,
                  excluded_sample_count=len(samples) - len(pooled),
                  helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  sampler_sha256=hashlib.sha256(Path(__file__).with_name('telemetry.py').read_bytes()).hexdigest(),
                  scope='Timestamped whole-board NVML observations from request shaping/H2D through all ordered PCM. Excludes post-PCM finite-value validation gaps. Sensor averaging and sampling remain; max is a sampled maximum, not an instantaneous spike or request-only board attribution.')
    return result
