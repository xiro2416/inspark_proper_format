#!/usr/bin/env python3
"""Generate or compare the deterministic SM89 quality corpus on one GPU."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path


def load_corpus(path: Path,expected_cases: int=256) -> dict:
    corpus = json.loads(path.read_text(encoding="utf-8"))
    if corpus.get("schema") != 1 or len(corpus.get("cases", ())) != expected_cases:
        raise ValueError(f"Quality corpus must be schema 1 with exactly {expected_cases} cases")
    if any(len(case.get("emotion", ())) != 8 for case in corpus["cases"]):
        raise ValueError("Every quality case requires an 8-value emotion vector")
    return corpus


def generate(args, corpus: dict) -> None:
    import numpy as np
    import soundfile as sf
    from acc_infer_clear.runtime.config import load
    from acc_infer_clear.runtime.deployment import load as load_deployment
    from acc_infer_clear.runtime.device import GPULease
    from acc_infer_clear.runtime.pool import Pool

    config = load(args.config)
    config["max_batch"] = 1
    deployment = load_deployment(args.deployment)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output_dir / "generation.jsonl"
    completed = {path.stem for path in args.output_dir.glob("sm89-q*.wav")} if args.resume else set()
    references = sorted({case["reference_audio"] for case in corpus["cases"]})
    voice_ids = {path: f"voice-{index}" for index, path in enumerate(references)}
    generated = 0
    with GPULease(args.gpu), Pool(config, args.gpu, 1) as pool:
        for path, voice_id in voice_ids.items():
            pool.prepare_reference(voice_id, path)
        manifest = pool.prepare_deployment(deployment)[0]
        with metadata_path.open("a" if args.resume else "w", encoding="utf-8") as metadata:
            for position, case in enumerate(corpus["cases"]):
                if case["id"] in completed:
                    continue
                started = time.perf_counter()
                pool.create_session(case["id"], voice_ids[case["reference_audio"]],
                                    case["seed"], case["emotion"])
                pool.push_text(case["id"], case["text"])
                pool.finish_input(case["id"])
                while pool.states[0]["ready_heads"] or pool.states[0]["ready_tails"]:
                    pool.run_ready()
                result = pool.result(case["id"])
                if not result["complete"] or result["error"]:
                    raise RuntimeError(f"Incomplete quality case {case['id']}: {result['error']}")
                pcm = np.concatenate([chunk["pcm"] for chunk in result["chunks"]])
                output = args.output_dir / f"{case['id']}.wav"
                sf.write(output, pcm, 22050, subtype="PCM_16")
                elapsed = time.perf_counter() - started
                row = {"id": case["id"], "output": str(output.resolve()), "samples": int(len(pcm)),
                       "chunks": len(result["chunks"]), "elapsed_s": elapsed,
                       "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
                metadata.write(json.dumps(row, ensure_ascii=False) + "\n")
                metadata.flush()
                pool.release(case["id"])
                generated += 1
                print(json.dumps({"progress": position + 1, "total": len(corpus["cases"]), **row},
                                 ensure_ascii=False), flush=True)
    summary = {"deployment": manifest, "generated": generated, "resumed": len(completed),
               "corpus": str(args.corpus.resolve()), "output_dir": str(args.output_dir.resolve())}
    (args.output_dir / "generation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def compare(args, corpus: dict) -> None:
    import numpy as np
    import soundfile as sf

    rows = []
    for case in corpus["cases"]:
        baseline_path = args.baseline_dir / f"{case['id']}.wav"
        candidate_path = args.candidate_dir / f"{case['id']}.wav"
        if not baseline_path.is_file() or not candidate_path.is_file():
            raise FileNotFoundError(f"Missing pair for {case['id']}")
        baseline, sr_a = sf.read(baseline_path, dtype="int16", always_2d=True)
        candidate, sr_b = sf.read(candidate_path, dtype="int16", always_2d=True)
        same_shape = baseline.shape == candidate.shape and sr_a == sr_b
        diff = candidate.astype(np.int32) - baseline.astype(np.int32) if same_shape else None
        signal = baseline.astype(np.float64) if same_shape else None
        noise = diff.astype(np.float64) if same_shape else None
        signal_power = None if signal is None else float(np.mean(signal ** 2))
        noise_power = None if noise is None else float(np.mean(noise ** 2))
        snr_db = None if noise_power is None else (float("inf") if noise_power == 0 else
                 float(10 * np.log10(max(signal_power, 1e-30) / noise_power)))
        rows.append({"id": case["id"], "same_shape": same_shape, "sample_rate_baseline": sr_a,
                     "sample_rate_candidate": sr_b, "samples_baseline": len(baseline),
                     "samples_candidate": len(candidate),
                     "max_abs_pcm16": None if diff is None else int(np.max(np.abs(diff), initial=0)),
                     "rmse_pcm16": None if diff is None else float(np.sqrt(noise_power)),
                     "snr_db": snr_db})
    exact = sum(row["same_shape"] and row["max_abs_pcm16"] == 0 for row in rows)
    report = {"cases": len(rows), "exact_pcm16": exact, "all_exact": exact == len(rows), "rows": rows}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("cases", "exact_pcm16", "all_exact")}, indent=2))
    if not report["all_exact"] and not args.allow_nonexact:
        raise SystemExit(2)


def write_eval_list(args, corpus: dict) -> None:
    args.eval_list.parent.mkdir(parents=True, exist_ok=True)
    with args.eval_list.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=";", lineterminator="\n")
        for case in corpus["cases"]:
            writer.writerow((case["id"], "", case["reference_audio"], case["text"]))
    print(json.dumps({"eval_list": str(args.eval_list.resolve()), "cases": len(corpus["cases"])}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path("configs/sm89_quality_256.json"))
    parser.add_argument("--expected-cases",type=int,default=256)
    sub = parser.add_subparsers(dest="command", required=True)
    generate_parser = sub.add_parser("generate")
    generate_parser.add_argument("--gpu", type=int, required=True)
    generate_parser.add_argument("--config", default="configs/runtime.yaml")
    generate_parser.add_argument("--deployment", required=True)
    generate_parser.add_argument("--output-dir", type=Path, required=True)
    generate_parser.add_argument("--resume", action="store_true")
    compare_parser = sub.add_parser("compare")
    compare_parser.add_argument("--baseline-dir", type=Path, required=True)
    compare_parser.add_argument("--candidate-dir", type=Path, required=True)
    compare_parser.add_argument("--report", type=Path, required=True)
    compare_parser.add_argument("--allow-nonexact",action="store_true")
    list_parser = sub.add_parser("eval-list")
    list_parser.add_argument("--eval-list", type=Path, required=True)
    args = parser.parse_args()
    corpus = load_corpus(args.corpus,args.expected_cases)
    {"generate": generate, "compare": compare, "eval-list": write_eval_list}[args.command](args, corpus)


if __name__ == "__main__":
    main()
