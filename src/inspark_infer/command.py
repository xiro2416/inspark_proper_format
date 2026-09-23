"""Project management commands; inference keeps its existing acc-clear entry."""
from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) >= 2 and args[:2] == ["trt", "build"]:
        from inspark_infer.build.trt113 import main as build
        return build(args[2:])
    if len(args) >= 2 and args[:2] in (["trt", "publish"], ["trt", "fetch"]):
        from inspark_infer.build.hf_cache import main as cache
        return cache(args[1:])
    print("usage: inspark trt {build,publish,fetch} [options]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
