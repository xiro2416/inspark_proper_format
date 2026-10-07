#!/usr/bin/env python3
"""Benchmark device-control heads in one prepared worker."""
import argparse
import json
import random
import statistics
import subprocess
import sys
import threading
import time


def percentile(values, q):
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q / 100.0
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def distribution(values):
    return {
        "n": len(values), "min": min(values), "median": statistics.median(values),
        "p90": percentile(values, 90), "p95": percentile(values, 95),
        "mean": statistics.fmean(values), "max": max(values),
    }


class BoardSampler:
    def __init__(self, gpu):
        self.gpu = gpu
        self.samples = []
        self.process = None
        self.thread = None

    def start(self):
        self.process = subprocess.Popen([
            "nvidia-smi", "-i", str(self.gpu),
            "--query-gpu=timestamp,memory.used,power.draw.instant,utilization.gpu",
            "--format=csv,noheader,nounits", "-lms", "20",
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

        def collect():
            for line in self.process.stdout:
                fields = [field.strip() for field in line.split(",")]
                if len(fields) != 4:
                    continue
                try:
                    self.samples.append((time.perf_counter(), float(fields[1]),
                                         float(fields[2]), float(fields[3])))
                except ValueError:
                    continue
        self.thread = threading.Thread(target=collect, daemon=True)
        self.thread.start()
        time.sleep(0.15)

    def stop(self):
        self.process.terminate()
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()
        self.thread.join(timeout=2)

    def window(self, start, end):
        rows = [row for row in self.samples if start <= row[0] <= end]
        if not rows:
            rows = sorted(self.samples, key=lambda row: min(abs(row[0] - start), abs(row[0] - end)))[:2]
        return {
            "sensor_samples": len(rows),
            "board_memory_mib_mean": statistics.fmean(row[1] for row in rows),
            "board_memory_mib_peak": max(row[1] for row in rows),
            "power_w_mean": statistics.fmean(row[2] for row in rows),
            "power_w_peak": max(row[2] for row in rows),
            "gpu_util_mean": statistics.fmean(row[3] for row in rows),
        }
