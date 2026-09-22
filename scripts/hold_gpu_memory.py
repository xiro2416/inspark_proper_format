#!/usr/bin/env python3
"""Reserve otherwise-free VRAM on one explicitly selected physical GPU."""
import argparse
import os
import signal
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--headroom-mib", type=int, default=512)
    parser.add_argument("--reserve-mib", type=int, help="Reserve a fixed amount instead of filling free VRAM")
    args = parser.parse_args()
    if args.reserve_mib is None and args.headroom_mib < 256:
        raise ValueError("At least 256 MiB headroom is required")
    if args.reserve_mib is not None and args.reserve_mib <= 0:
        raise ValueError("reserve-mib must be positive")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch

    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible GPU")
    torch.cuda.init()
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    headroom = args.headroom_mib << 20
    request = (args.reserve_mib << 20) if args.reserve_mib is not None else free_bytes - headroom
    if request > free_bytes:
        raise RuntimeError(f"Requested {request >> 20} MiB but only {free_bytes >> 20} MiB is free")
    if request <= 0:
        raise RuntimeError(f"Only {free_bytes >> 20} MiB is free")

    reservation = None
    step = 128 << 20
    while request > 0:
        try:
            reservation = torch.empty(request, dtype=torch.uint8, device="cuda")
            break
        except torch.OutOfMemoryError:
            request -= step
            torch.cuda.empty_cache()
    if reservation is None:
        raise RuntimeError("Unable to reserve GPU memory")

    running = True

    def stop(*_args) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    print(
        f"holding gpu={args.gpu} reserved_mib={request >> 20} "
        f"requested_headroom_mib={args.headroom_mib} fixed={args.reserve_mib is not None} "
        f"total_mib={total_bytes >> 20}",
        flush=True,
    )
    while running:
        time.sleep(30)


if __name__ == "__main__":
    main()
