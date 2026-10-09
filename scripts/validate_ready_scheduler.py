#!/usr/bin/env python3
"""Small real-GPU readiness/ownership check, separate from formal measurements.

Same arguments as benchmark_ready_first_chunk; defaults to one wave, one warmup,
no power window and lifecycle counts 1/15/17/N-1/N. Synthetic EOS and cancellation
coverage belong in the separately labelled scheduler state suite.
"""
from benchmarks.benchmark_ready_first_chunk import parser, run


def main():
    p = parser()
    p.set_defaults(waves=1, warmups=1, power_seconds=0, lifecycle=True)
    run(p.parse_args())


if __name__ == '__main__':
    main()
