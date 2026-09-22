#!/usr/bin/env python3
"""Actual first-PCM and complete-EOS baseline, including packing/copies/launches.

Compile warmup uses real model inputs, outside the measured request window.
Unknown shape fallback counts and compilation cost are reported separately.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import statistics
import time
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--batch", type=int, choices=(1, 4, 8, 16), default=1)
    parser.add_argument("--config", default="configs/runtime_reference.yaml")
    parser.add_argument("--deployment", default="configs/sm89_eager_bf16.json")
    parser.add_argument("--reference", default="/workspace/index-tts/data/audio/old/mingxiang_gao.wav")
    parser.add_argument("--text", default="他正在整理文件。")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--head-only", action="store_true")
    parser.add_argument("--operator-repeats", type=int, default=10)
    parser.add_argument("--json-out", required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmups < 0 or args.operator_repeats < 1:
        parser.error("Invalid repeat/warmup counts")
    from acc_infer_clear.runtime.device import GPULease, select_gpu
    with GPULease(args.gpu) as lease:
        select_gpu(args.gpu)
        report = run(args)
        report["external_memory_mib"] = lease.initial_memory_mib
    path = Path(args.json_out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    if not report["execution_pass"]:
        raise SystemExit(1)


def run(args):
    import torch
    from acc_infer_clear.runtime.config import load
    from acc_infer_clear.runtime.deployment import load as load_plan
    from acc_infer_clear.runtime.engine import Engine
    from acc_infer_clear.runtime.pool import _engine_stats
    from scripts.trt113_provenance import source_identity, capture_provenance, file_record
    config = load(args.config)
    config["max_batch"] = args.batch
    plan = load_plan(args.deployment)
    engine = None
    report = dict(schema=1, hardware_scope="sm89", batch=args.batch,
                  text=args.text, requested=plan, config=config, measured=[], warmups=[],
                  errors=[], execution_pass=False, numerical_pass=None,
                  torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                  sm=list(torch.cuda.get_device_capability()),
                  source=source_identity(), deployment_file=file_record(args.deployment,"deployment"),
                  runtime_file=file_record(args.config,"runtime_config"),
                  cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                  timing_scope="host admission through first PCM / full EOS; includes CPU/GPU copies, allocation and launch; excludes model/reference preparation")
    try:
        start = time.perf_counter()
        engine = Engine(config)
        engine.prepare_reference("voice", args.reference)
        report["model_provenance"] = {component:capture_provenance(component,config,args.config,model=engine)
                                      for component in ("target","draft","cfm","vocoder")}
        report["deployment"] = engine.prepare_deployment(plan)
        report["prepare_seconds"] = time.perf_counter()-start
        bank = getattr(engine, "compile_bank", None)
        if bank is not None:
            from torch.utils._pytree import tree_flatten
            from acc_infer_clear.guardrails.numerics import compare
            report["operator_audit"] = {}
            def audit(op, inputs, kwargs, actual):
                flat_actual, actual_spec = tree_flatten(actual)
                frozen = [v.detach().clone() if isinstance(v, torch.Tensor) else v for v in flat_actual]
                expected = op.eager(*inputs, **kwargs)
                flat_expected, expected_spec = tree_flatten(expected)
                comparisons = []
                if actual_spec != expected_spec:
                    raise ValueError("Compile/eager output structure mismatch")
                for index, (reference, candidate) in enumerate(zip(flat_expected, frozen)):
                    if isinstance(reference, torch.Tensor):
                        comparisons.append(dict(output=index, **compare(reference, candidate, plan["precision"])))
                    elif reference != candidate:
                        raise ValueError("Compile/eager metadata mismatch")
                timings = {}
                for label, fn in (("eager", op.eager), ("compiled", op.compiled)):
                    for _ in range(2):
                        fn(*inputs, **kwargs)
                    torch.cuda.synchronize()
                    start_host = time.perf_counter()
                    start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    for _ in range(args.operator_repeats):
                        fn(*inputs, **kwargs)
                    end_event.record()
                    end_event.synchronize()
                    timings[label] = dict(gpu_ms=start_event.elapsed_time(end_event)/args.operator_repeats,
                                         wall_ms=(time.perf_counter()-start_host)*1000/args.operator_repeats)
                report["operator_audit"][op.name] = dict(comparisons=comparisons, timings=timings,
                    pass_gate=bool(comparisons) and all(v["pass_gate"] for v in comparisons),
                    timing_scope="real-input module call including internal casts/launches; excludes surrounding runtime packing, separately measured end-to-end")
            for op in bank.operators.values():
                op.audit_hook = audit
        for wave in range(args.warmups+args.repeats):
            warmup = wave < args.warmups
            ids = [f"wave{wave}-row{i}" for i in range(args.batch)]
            first = {}
            def chunk(event):
                first.setdefault(event["request_id"], (time.perf_counter()-started)*1000)
            torch.cuda.synchronize()
            started = time.perf_counter()
            context = bank.offline_warmup() if bank is not None and warmup else nullcontext()
            with context:
                for i, rid in enumerate(ids):
                    # Warmup and measured input shapes/RNG agree. Request IDs
                    # are deliberately distinct; never part of random seeds.
                    engine.create_session(rid, "voice", seed=113+i)
                    engine.push_text(rid, args.text)
                    engine.finish_input(rid)
                turns = 0
                while engine.ready():
                    engine.run_ready(chunk)
                    turns += 1
                    if turns > 2000:
                        raise RuntimeError("Scheduler failed to reach terminal state")
                    if args.head_only and len(first) == args.batch:
                        break
                torch.cuda.synchronize()
            elapsed = (time.perf_counter()-started)*1000
            rows = []
            for rid in ids:
                session = engine.sessions[rid]
                pcm_samples = sum(len(c["pcm"]) for c in session["chunks"])
                rows.append(dict(id=rid, first_pcm_ms=first.get(rid), complete=session["complete"],
                    error=session["error"], chunks=len(session["chunks"]), samples=pcm_samples,
                    codes=len(session["codes"]), codes_sha256=hashlib.sha256(json.dumps(session["codes"]).encode()).hexdigest(),
                    audio_seconds=pcm_samples/22050, accepted=session.get("accepted")))
            item = dict(wave=wave, wall_ms=elapsed, all_first_pcm_ms=max(first.values()) if first else None,
                        rows=rows, memory_allocated=torch.cuda.memory_allocated(),
                        memory_reserved=torch.cuda.memory_reserved(), peak_allocated=torch.cuda.max_memory_allocated())
            report["warmups" if warmup else "measured"].append(item)
            print(json.dumps(dict(progress=wave+1,total=args.warmups+args.repeats,warmup=warmup,
                                  all_first_pcm_ms=item["all_first_pcm_ms"],wall_ms=elapsed)), flush=True)
            for rid in ids:
                if args.head_only:
                    engine.cancel(rid)
                else:
                    engine.release(rid)
        report["routes"] = _engine_stats(engine)
        if bank is not None:
            report["compile"] = bank.stats()
            report["numerical_pass"] = (len(report["operator_audit"]) == len(bank.operators)
                and all(row["pass_gate"] for row in report["operator_audit"].values()))
        report["execution_pass"] = all(r["first_pcm_ms"] is not None and not r["error"] and
            (args.head_only or r["complete"]) for w in report["measured"] for r in w["rows"])
        report["summary"] = dict(all_first_pcm_median_ms=statistics.median(w["all_first_pcm_ms"] for w in report["measured"]),
                                 complete_median_ms=None if args.head_only else statistics.median(w["wall_ms"] for w in report["measured"]),
                                 complete_eos=not args.head_only)
    except BaseException:
        report["errors"].append(traceback.format_exc())
        if engine is not None and getattr(engine, "compile_bank", None):
            report["compile"] = engine.compile_bank.stats()
    finally:
        if engine is not None:
            engine.close()
    return report


if __name__ == "__main__":
    main()
