#!/usr/bin/env python3
"""Reproducible B1/B8/B64 first-chunk comparison with two explicit latency clocks."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from benchmarks.unified_first_chunk import (
    VOICES, build_manifest, compare_reports, counter_delta, distribution, load_manifest,
    official_token_counter, read_texts, run_wave, save_json, sha256_file, summarize, wave_cases,
    first_segment_diversity,
)

ROOT = Path(__file__).resolve().parents[1]


def create_manifest(args):
    references = [dict(voice_id=name, path=str((args.voice_dir / name).resolve()),
                       sha256=sha256_file(args.voice_dir / name)) for name in VOICES]
    provenance = dict(dataset=str(args.dataset.resolve()), dataset_sha256=sha256_file(args.dataset),
                      tokenizer=str(args.tokenizer.resolve()), tokenizer_sha256=sha256_file(args.tokenizer))
    glossary = args.tokenizer.with_name("glossary.yaml")
    if glossary.exists():
        provenance["glossary_sha256"] = sha256_file(glossary)
    manifest = build_manifest(read_texts(args.dataset), references, official_token_counter(args.tokenizer),
                              requests=args.requests, seed=args.seed, provenance=provenance)
    save_json(args.output, manifest)
    print(json.dumps(dict(path=str(args.output), manifest_sha256=manifest["manifest_sha256"],
                          requests_per_split=args.requests, ranges=manifest["stratum_token_ranges"])))


def add_run_arguments(parser, *, profile=False):
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--batch", type=int, choices=(1, 8, 64, 128), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("calibration", "evaluation"), default="evaluation")
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "artifacts/current_release/runtime.yaml")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--waves", type=int, default=3 if profile else 100)
    parser.add_argument("--case-offset-wave", type=int, default=0,
                        help="Allows separately launched A/B blocks to use the same case sequence")
    parser.add_argument("--label", default="unlabelled")
    parser.add_argument("--quant-recipe", default="unspecified", help="Recipe ID; actual precision remains deployment-attested")
    parser.add_argument("--runtime-backend", default="existing_project_runtime",
                        help="Run label only; does not select or attest a backend")
    parser.add_argument("--admission-mode",choices=("auto","batch","serial"),default="auto",
                        help="Auto uses one admit_batch RPC per worker when available; serial retains the old API sequence")
    parser.add_argument("--admission-rpc-timing",action="store_true",
                        help="Diagnostic per-admission-API wall times; reports are explicitly labelled")
    parser.add_argument('--require-distinct-first-segments',action='store_true',
                        help='Additional diversity control: reject any wave with repeated first-chunk text')
    if not profile:
        parser.add_argument("--power-seconds", type=float, default=15)


def validate_run_arguments(args):
    if args.warmups < 0 or args.waves < 1 or args.case_offset_wave < 0:
        raise ValueError("Invalid warmup/wave/offset count")
    if hasattr(args, "power_seconds") and args.power_seconds < 0:
        raise ValueError("Power duration must be nonnegative")


def prepare_references(client, manifest):
    references = {}
    for reference in manifest["references"]:
        value = client.prepare_reference(reference["voice_id"], reference["path"])
        rows = value if isinstance(value, list) else [value]
        for row in rows:
            if row.get("shortfall") or abs(row["actual_seconds"] - 3.0) > 1e-6:
                raise RuntimeError(f"Reference did not supply three VAD seconds: {reference['voice_id']}")
        references[reference["voice_id"]] = rows
    return references


def power_window(pool, cases, args):
    """Independent sustained serving window. No result/quality IPC per wave."""
    if args.power_seconds == 0:
        return None
    from benchmarks.board_power import BoardSampler
    sampler = BoardSampler(args.gpu)
    sampler.start()
    count = 0
    started = time.perf_counter()
    try:
        while count == 0 or time.perf_counter() - started < args.power_seconds:
            run_wave(pool, wave_cases(cases, args.batch, count), f"power-{count}", details=False,
                     admission_mode=args.admission_mode)
            count += 1
        ended = time.perf_counter()
    finally:
        sampler.stop()
    samples = [row for row in sampler.samples if started <= row[0] <= ended]
    if not samples:
        raise RuntimeError("No board samples inside sustained power window")
    watts = distribution(row[2] for row in samples)
    elapsed = ended - started
    return dict(duration_seconds=elapsed, requested_seconds=args.power_seconds, waves=count,
                requests=count * args.batch, requests_per_second=count * args.batch / elapsed,
                power_w=watts, board_memory_mib=distribution(row[1] for row in samples),
                utilization_percent=distribution(row[3] for row in samples),
                board_joules_per_request=watts["mean"] * elapsed / (count * args.batch),
                timing_scope="continuous full waves including admission, delivery, cancellation; separate from latency",
                raw_samples=[dict(elapsed_seconds=row[0] - started, memory_mib=row[1],
                                  power_w=row[2], utilization_percent=row[3]) for row in samples])


def run(args):
    validate_run_arguments(args)
    manifest = load_manifest(args.manifest)
    cases = manifest["splits"][args.split]
    diversity=first_segment_diversity(cases,args.batch)
    if not diversity['all_first_segments_distinct'] and args.require_distinct_first_segments:
        raise ValueError('Distinct-prefix control requires different first-chunk texts in every wave')
    from inspark_infer.runtime.config import load
    from inspark_infer.runtime.deployment import load as load_deployment
    from inspark_infer.runtime.device import GPULease
    from inspark_infer.runtime.pool import Pool
    config = load(str(args.config))
    if config["reference_seconds"] != 3:
        raise ValueError("Runtime reference_seconds must be three")
    config["max_batch"] = args.batch
    config["precision_batches"] = [args.batch]
    plan = load_deployment(str(args.deployment))
    # Benchmark the ordinary path even if a deployment contains the optional
    # repeated-text optimization. Do not attribute content-reuse gains to it.
    plan=dict(plan,batch_text_dedup=False)
    report = dict(schema=1, kind="unified_first_chunk_benchmark", label=args.label, batch=args.batch,
                  gpu=args.gpu, quant_recipe=args.quant_recipe, runtime_backend_label=args.runtime_backend,
                  backend_label_is_attestation=False, manifest_sha256=manifest["manifest_sha256"],
                  manifest=str(args.manifest.resolve()), split=args.split,
                  deployment_path=str(args.deployment.resolve()), deployment_sha256=sha256_file(args.deployment),
                  config_path=str(args.config.resolve()), config_sha256=sha256_file(args.config),
                  allocator=os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
                  cfm_steps=4, warmups=args.warmups, cache_mode="four reference voices warmed before deployment",
                  numerical_pass=None, profiler_enabled=False,admission_mode_requested=args.admission_mode,
                  admission_rpc_timing=args.admission_rpc_timing,
                  workload_diversity=diversity,
                  repeated_text_reuse_gain_included=False,
                  benchmark_overrides={'batch_text_dedup':False},
                  timing=dict(admission="each create_session call entry to client PCM event.received",
                              historical="last finish_input returned to client PCM event.received",
                              stage_timing="collected only by separate profile command"))
    waves = []
    from benchmarks.board_power import BoardSampler
    lifecycle=BoardSampler(args.gpu)
    try:
        lifecycle.start()
        with GPULease(args.gpu) as lease, Pool(config, args.gpu, 1) as pool:
            report["initial_board_memory_mib"] = lease.initial_memory_mib
            report["references"] = prepare_references(pool, manifest)
            report["deployment"] = pool.prepare_deployment(plan)[0]
            for index in range(args.warmups):
                run_wave(pool, wave_cases(cases, args.batch, index), f"warmup-{index}", details=False,
                         admission_mode=args.admission_mode)
            before = pool.stats()[0]
            for index in range(args.waves):
                at = args.case_offset_wave + index
                result = run_wave(pool, wave_cases(cases, args.batch, at), f"measured-{index}",
                                  admission_mode=args.admission_mode,admission_rpc_timing=args.admission_rpc_timing)
                result["case_offset_wave"] = at
                waves.append(result)
                if (index + 1) % 10 == 0 or index == args.waves - 1:
                    print(json.dumps(dict(label=args.label, batch=args.batch, waves=index + 1,
                                          last_wave_postadmission_ms=result["group_postadmission_to_last_pcm_ms"])), flush=True)
            after = pool.stats()[0]
            report.update(before=before, after=after, measured_counters=counter_delta(before, after),
                          summary=summarize(waves), waves=waves)
            report['admission_mode']=waves[0]['admission_mode']
            if report['admission_mode']=='batch':
                report['timing']['admission']='each request timestamp before grouped admit_batch RPC to client PCM event.received'
                report['timing']['historical']='admit_batch acknowledgment to client PCM event.received'
            report["sustained_power"] = power_window(pool, cases, args)
            report["after_power"] = pool.stats()[0]
            report["execution_pass"] = True
    except Exception as error:
        report.update(execution_pass=False, error=f"{type(error).__name__}: {error}", waves=waves)
        save_json(args.out, report)
        raise
    finally:
        if lifecycle.process is not None:lifecycle.stop()
    report['board_lifecycle']=dict(samples=len(lifecycle.samples),sample_interval_ms=20,
        memory_mib=distribution(x[1] for x in lifecycle.samples),
        scope='sampled whole process: model/reference preparation, deployment/capture, warmup, measured waves, power and teardown',
        note='NVML sampled peak; may miss allocations shorter than20ms; Torch allocator peaks are reported separately')
    save_json(args.out, report)
    print(json.dumps(dict(out=str(args.out), summary=report["summary"],
                          counters=report["measured_counters"]), ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    manifest = commands.add_parser("manifest", help="CPU-only deterministic calibration/evaluation cases")
    manifest.add_argument("--dataset", type=Path, default=Path("/workspace/index-tts/data/A_0811.txt"))
    manifest.add_argument("--voice-dir", type=Path, default=Path("/workspace/index-tts/data/audio"))
    manifest.add_argument("--tokenizer", type=Path, default=ROOT / "models/index_tts2/bpe.model")
    manifest.add_argument("--requests", type=int, default=128, help="Number in EACH of two disjoint splits")
    manifest.add_argument("--seed", type=int, default=924)
    manifest.add_argument("--output", type=Path, required=True)
    benchmark = commands.add_parser("run")
    add_run_arguments(benchmark)
    compare = commands.add_parser("compare", help="CPU-only paired comparison of matching run JSONs")
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--out", type=Path, required=True)
    compare.add_argument("--allow-admission-change",action="store_true",
                         help="Explicitly compare serial versus batch admission with the same CPU-PCM end boundary")
    args = parser.parse_args()
    if args.command == "manifest":
        create_manifest(args)
    elif args.command == "compare":
        report = compare_reports(json.loads(args.baseline.read_text()), json.loads(args.candidate.read_text()),
                                 allow_admission_change=args.allow_admission_change)
        save_json(args.out, report)
        print(json.dumps(report, ensure_ascii=False))
    else:
        run(args)


if __name__ == "__main__":
    main()
