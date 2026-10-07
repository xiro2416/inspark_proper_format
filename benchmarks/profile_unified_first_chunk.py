#!/usr/bin/env python3
"""Production-Graph stage/round profiling on the benchmark's identical case manifest.

Use under nsys with --cuda-profiler-range. Timings are diagnostic, never accepted
as the no-profiler E2E result. All instrumentation is temporary Python wrapping;
no deployment choice or GPU math kernel is changed.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import functools
import json
from pathlib import Path
import time

from benchmarks.benchmark_unified_first_chunk import add_run_arguments, prepare_references, validate_run_arguments
from benchmarks.unified_first_chunk import counter_delta, load_manifest, run_wave, save_json, sha256_file, wave_cases


class StageObserver:
    def __init__(self, engine, *, kv_trajectory=True):
        self.engine, self.torch = engine, engine.torch
        self.kv_trajectory = kv_trajectory
        self.patches, self.spans, self.trajectories, self.prepared = [], [], [], []
        self.frontend = []
        self.framework_completed = []
        self.framework_admissions = []
        self.wave = None
        self.graph_labels = {}

    @contextmanager
    def scope(self, name, **metadata):
        torch = self.torch
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        started = time.perf_counter()
        with torch.profiler.record_function(name):
            torch.cuda.nvtx.range_push(name)
            try:
                yield
            finally:
                torch.cuda.nvtx.range_pop()
                ended = time.perf_counter()
                end.record()
                self.spans.append(dict(name=name, wave=self.wave, start=begin, end=end,
                                       host_ms=(ended - started) * 1000, metadata=metadata))

    def patch(self, owner, name, wrapper):
        old = getattr(owner, name)
        self.patches.append((owner, name, old))
        setattr(owner, name, wrapper(old))

    def wrap(self, name):
        def decorate(fn):
            @functools.wraps(fn)
            def wrapped(*args, **kwargs):
                with self.scope(name):
                    return fn(*args, **kwargs)
            return wrapped
        return decorate

    def enter(self):
        from inspark_infer.runtime.graphs import Captured
        rt = self.engine.rt
        for name, table in (("draft_graph", getattr(rt.backbone, "graphs", {})),
                            ("target_graph", getattr(rt.target, "graphs", {})),
                            ("rnn_graph", getattr(rt.proposal, "graphs", {})),
                            ("context_graph", getattr(rt.context, "graphs", {}))):
            for shape, graph in table.items():
                self.graph_labels[id(graph)] = (name, str(shape))
        for component, obj in (("target", rt.target), ("draft", rt.backbone)):
            bank = getattr(obj, "native_full_bank", None)
            for shape, graph in getattr(bank, "graphs", {}).items():
                self.graph_labels[id(graph)] = (component + "_trt_graph", str(shape))
        head = getattr(self.engine, "head_graphs", None)
        for component in ("cfm", "vocoder"):
            for shape, graph in getattr(head, component, {}).items():
                self.graph_labels[id(graph)] = (component + "_graph", str(shape))
        observer = self

        def graph_wrap(fn):
            @functools.wraps(fn)
            def call(graph, *args, **kwargs):
                label, shape = observer.graph_labels.get(id(graph), ("graph_replay", "unclassified"))
                with observer.scope("unified/" + label, shape=shape):
                    return fn(graph, *args, **kwargs)
            return call
        self.patch(Captured, "__call__", graph_wrap)

        def step_wrap(fn):
            @functools.wraps(fn)
            def call(runner, *args, **kwargs):
                record = None
                if observer.kv_trajectory:
                    record = dict(wave=observer.wave,
                                  request_ids=[row.request["id"] for row in runner.rows],
                                  active_batch=runner.actual_b, physical_batch=runner.b,
                                  target_before=runner.past.detach().clone(),
                                  draft_before=runner.draft_lengths.detach().clone(),
                                  ready_before=runner.ready.detach().clone())
                with observer.scope("unified/device_round_step", physical_batch=runner.b):
                    result = fn(runner, *args, **kwargs)
                if record is not None:
                    record.update(target_after=runner.past.detach().clone(),
                                  draft_after=runner.draft_lengths.detach().clone(),
                                  committed=runner.committed.detach().clone(),
                                  ready_after=runner.ready.detach().clone(),
                                  done_after=runner.done.detach().clone(),
                                  token_lengths=runner.token_lengths.detach().clone())
                    observer.trajectories.append(record)
                return result
            return call
        controller = getattr(self.engine, "unified_first_chunk", None)
        if controller is not None:
            self.attach_framework(controller)
        else:
            from inspark_infer.runtime.indextts2 import device_round
            self.patch(device_round.DeviceRoundHead, "step", step_wrap)
            for method, label in (("__init__", "device_round_setup"), ("run", "device_round_run"),
                                  ("finish", "device_round_finish")):
                self.patch(device_round.DeviceRoundHead, method, self.wrap("unified/" + label))
            for fn, label in (("acceptance", "pcg_acceptance"), ("prefix_plan", "pcg_prefix"),
                              ("commit", "pcg_commit"), ("draw", "request_rng"),
                              ("status", "status_kernel")):
                self.patch(device_round, fn, self.wrap("unified/" + label))
            self.patch(rt.residual, "device_batch_draws", self.wrap("unified/pcg_residual"))

        def rows_wrap(fn):
            @functools.wraps(fn)
            def call(sessions, index):
                result = fn(sessions, index)
                observer.frontend.append(dict(wave=observer.wave, chunk_index=index,
                                               **dict(getattr(rt.frontend, "last_stats", {}))))
                observer.prepared.extend(dict(wave=observer.wave, chunk_index=index,
                                              request_id=row.request["id"], prefix_length=row.prefix_length,
                                              target_past_length=row.past_length,
                                              draft_context_length=row.cache.length) for row in result)
                return result
            return call
        self.patch(self.engine, "prepare_rows", rows_wrap)
        def phase_wrap(fn):
            @functools.wraps(fn)
            def call(name, batch, body, **metadata):
                label = "cfm4" if name == "cfm2" else name
                with observer.scope("unified/stage/" + label, batch=batch, **metadata):
                    return fn(name, batch, body, **metadata)
            return call
        # Engine._advance clears its phase spans on every call. Keep all waves in
        # this observer rather than silently returning only the final wave.
        self.patch(self.engine, "phase", phase_wrap)
        self.patch(self.engine, "run_ready", self.wrap("unified/engine_work"))
        # Covers staging, PCM conversion and the output wait absent from phase().
        self.patch(self.engine, "acoustic_rows", self.wrap("unified/acoustic_and_output"))
        return self

    def attach_framework(self, controller):
        """Observe the new parent Graph without recapturing or decomposing it."""
        runtime, provider, observer = controller.runtime, controller.provider, self
        def controller_wrap(fn):
            @functools.wraps(fn)
            def call(rows):
                observer.framework_admissions = [dict(request_id=row.request["id"],
                    target_before=row.past_length, draft_before=row.cache.length) for row in rows]
                with observer.scope("unified/framework_controller", batch=len(rows)):
                    return fn(rows)
            return call
        self.patch(controller, "run", controller_wrap)
        self.patch(provider, "import_rows", self.wrap("unified/framework_import_kv"))
        for name in ("reset", "run"):
            self.patch(runtime, name, self.wrap("unified/framework_" + name))
        for name in ("draft_provider", "target_provider", "context_writer"):
            self.patch(runtime, name, self.wrap("unified/framework_" + name))
        self.patch(runtime.proposal, "sample_uniform", self.wrap("unified/framework_rnn"))
        for name in ("acceptance", "prefix_plan", "residual"):
            self.patch(runtime.pcg, name, self.wrap("unified/framework_pcg_" + name))

        def block_wrap(fn, graph):
            @functools.wraps(fn)
            def call(*args, **kwargs):
                record = None
                if observer.kv_trajectory:
                    record = dict(wave=observer.wave, trajectory_kind="graph_burst" if graph else "round",
                                  physical_rounds=runtime.graph_burst if graph else 1,
                                  request_ids=[r["request_id"] for r in observer.framework_admissions],
                                  physical_batch=runtime.batch,
                                  target_before=runtime.past.detach().clone(),
                                  draft_before=runtime.draft_lengths.detach().clone(),
                                  ready_before=runtime.ready.detach().clone(),
                                  rounds_before=runtime.rounds.detach().clone())
                label = "framework_parent_graph" if graph else "framework_round"
                with observer.scope("unified/" + label, batch=runtime.batch,
                                    physical_rounds=runtime.graph_burst if graph else 1):
                    result = fn(*args, **kwargs)
                if record is not None:
                    record.update(target_after=runtime.past.detach().clone(),
                                  draft_after=runtime.draft_lengths.detach().clone(),
                                  rounds_after=runtime.rounds.detach().clone(),
                                  ready_after=runtime.ready.detach().clone(),
                                  done_after=runtime.done.detach().clone(),
                                  token_lengths=runtime.token_lengths.detach().clone())
                    observer.trajectories.append(record)
                return result
            return call
        if runtime.graph is not None:
            self.patch(runtime.graph, "replay", lambda fn: block_wrap(fn, True))
        else:
            self.patch(runtime, "step", lambda fn: block_wrap(fn, False))

        def result_wrap(fn):
            @functools.wraps(fn)
            def call():
                with observer.scope("unified/framework_result_metadata"):
                    result = fn()
                if observer.kv_trajectory:
                    observer.framework_completed.append(dict(wave=observer.wave,
                        initial=[dict(row) for row in observer.framework_admissions],
                        metadata=[list(row) for row in result["metadata"]],
                        accepted=result["accepted"].detach().clone()))
                return result
            return call
        self.patch(runtime, "result", result_wrap)

    def resolve(self):
        self.torch.cuda.synchronize()
        spans = [dict(name=row["name"], wave=row["wave"], host_ms=row["host_ms"],
                      gpu_ms=row["start"].elapsed_time(row["end"]), metadata=row["metadata"])
                 for row in self.spans]
        # The only additional D2H from this observer occurs AFTER measured waves.
        trajectories = [{key: value.detach().cpu().tolist() if self.torch.is_tensor(value) else value
                         for key, value in row.items()} for row in self.trajectories]
        reconstructed = []
        for record in self.framework_completed:
            history = record["accepted"].detach().cpu().tolist()
            for initial, meta, accepted in zip(record["initial"], record["metadata"], history):
                target, draft = initial["target_before"], initial["draft_before"]
                for step, count in enumerate(accepted[:meta[3]]):
                    if not 0 <= count <= 7:
                        raise RuntimeError("Invalid accepted count in completed framework trajectory")
                    reconstructed.append(dict(wave=record["wave"], request_id=initial["request_id"],
                        round=step, target_before=target, draft_before=draft, committed=count + 1,
                        target_verification_extent=target + 8, draft_context_plus_query_extent=draft + 7,
                        target_after=target + count + 1, draft_after=draft + count + 1))
                    target += count + 1
                    draft += count + 1
                if target != meta[1] or draft != meta[2]:
                    raise RuntimeError("Accepted-history reconstruction disagrees with final GPU KV metadata")
        return dict(spans=spans, prepared_requests=self.prepared, frontend=self.frontend, kv_trajectory=trajectories,
                    framework_logical_rounds=reconstructed,
                    framework_graph_note="Parent Graph remains intact: provider/RNN/PCG Python scopes do not execute on replay. Use engine inspector/Nsight for internal kernels; graph_burst rows are not individual-round measurements.",
                    framework_round_note="Logical per-request KV rounds reconstructed from actual accepted history and checked against final GPU lengths; not extra per-round D2H",
                    kv_trajectory_note="Diagnostic GPU clones outside Graph; batched D2H only after waves",
                    span_note="Nested/overlapping CUDA spans must not be added as exclusive stage time")

    def close(self):
        for owner, name, old in reversed(self.patches):
            setattr(owner, name, old)
        self.patches.clear()


def run(args):
    validate_run_arguments(args)
    manifest = load_manifest(args.manifest)
    cases = manifest["splits"][args.split]
    from inspark_infer.runtime.device import GPULease, select_gpu
    from inspark_infer.runtime.config import load
    from inspark_infer.runtime.deployment import load as load_deployment
    from inspark_infer.runtime.engine import Engine
    from inspark_infer.runtime.pool import _engine_stats
    config = load(str(args.config))
    config["max_batch"] = args.batch
    config["precision_batches"] = [args.batch]
    plan = load_deployment(str(args.deployment))
    plan=dict(plan,batch_text_dedup=False)
    report = dict(schema=1, kind="unified_first_chunk_profile", label=args.label, gpu=args.gpu,
                  batch=args.batch, quant_recipe=args.quant_recipe, runtime_backend_label=args.runtime_backend,
                  manifest_sha256=manifest["manifest_sha256"], split=args.split,
                  deployment_sha256=sha256_file(args.deployment), graphs="production",
                  benchmark_overrides={'batch_text_dedup':False},
                  numerical_pass=None, performance_acceptance_result=False,
                  timing_scope="in-process diagnostic; use separate Pool benchmark for client E2E",
                  cfm_steps=4)
    engine, observer = None, None
    with GPULease(args.gpu):
        select_gpu(args.gpu)
        import torch
        if torch.cuda.device_count() != 1:
            raise RuntimeError("Exactly one visible GPU is required")
        try:
            engine = Engine(config)
            report["references"] = prepare_references(engine, manifest)
            report["deployment"] = engine.prepare_deployment(plan)
            for index in range(args.warmups):
                run_wave(engine, wave_cases(cases, args.batch, index), f"warmup-{index}", details=False,
                         admission_mode=args.admission_mode)
            engine.configure_profiling(enabled=True, trace_ranges=True)
            observer = StageObserver(engine, kv_trajectory=not args.no_kv_trajectory).enter()
            before = _engine_stats(engine)
            trace = (torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                       torch.profiler.ProfilerActivity.CUDA],
                                            record_shapes=True, profile_memory=False)
                     if args.torch_trace else nullcontext())
            if args.cuda_profiler_range:
                torch.cuda.cudart().cudaProfilerStart()
            waves = []
            try:
                with trace as profiler:
                    for index in range(args.waves):
                        observer.wave = index
                        with torch.cuda.stream(engine.model.stream), observer.scope("unified/wave_lifecycle", batch=args.batch):
                            waves.append(run_wave(engine, wave_cases(cases, args.batch, args.case_offset_wave + index),
                                                  f"profile-{index}",admission_mode=args.admission_mode,
                                                  admission_rpc_timing=args.admission_rpc_timing))
                    torch.cuda.synchronize()
            finally:
                if args.cuda_profiler_range:
                    torch.cuda.cudart().cudaProfilerStop()
            after = _engine_stats(engine)
            stage_profile = engine.take_profile()
            for row in stage_profile["stages"]:
                if row["name"] == "cfm2":
                    row["original_range_name"] = row["name"]
                    row["name"] = "cfm4"
            report.update(before=before, after=after, counters=counter_delta(before, after), waves=waves,
                          last_engine_call_profile=stage_profile, observer=observer.resolve(), execution_pass=True)
            if args.torch_trace:
                args.torch_trace.parent.mkdir(parents=True, exist_ok=True)
                profiler.export_chrome_trace(str(args.torch_trace))
                report["torch_trace"] = str(args.torch_trace)
            if not report["observer"]["kv_trajectory"]:
                report["kv_trajectory_missing_reason"] = ("disabled" if args.no_kv_trajectory else
                                                           "no DeviceRoundHead.step calls; inspect actual route/fallback")
        except Exception as error:
            report.update(execution_pass=False, error=f"{type(error).__name__}: {error}")
            save_json(args.out, report)
            raise
        finally:
            if observer is not None:
                observer.close()
            if engine is not None:
                engine.close()
    save_json(args.out, report)
    print(json.dumps(dict(out=str(args.out), spans=len(report["observer"]["spans"]),
                          kv_rounds=len(report["observer"]["kv_trajectory"]))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_arguments(parser, profile=True)
    parser.add_argument("--cuda-profiler-range", action="store_true",
                        help="Bound nsys capture to warmed diagnostic waves")
    parser.add_argument("--torch-trace", type=Path, help="Optional torch trace; do not combine with nsys")
    parser.add_argument("--no-kv-trajectory", action="store_true")
    args = parser.parse_args()
    if args.torch_trace and args.cuda_profiler_range:
        parser.error("Choose one profiler owner, torch or nsys")
    run(args)


if __name__ == "__main__":
    main()
