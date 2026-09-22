#!/usr/bin/env python3
"""Layer-3 module power and energy, whiteboard PROFILE_SM89_WHITEBOARD.md section 10.

The existing batch benchmark reports whole-request instantaneous board power, which
the whiteboard explicitly says is not enough: a short kernel cannot be attributed
from nvidia-smi's instantaneous sensor. This script isolates one module at a time,
repeats it until the measurement window is at least --seconds long, samples NVML
board power continuously, and reports baseline-subtracted energy plus J/call.

Throttle policy: if the SM clock drops below --min-clock-fraction of the tier's own
median, or the driver reports a thermal/power throttle reason, the tier is marked
invalid (whiteboard section 10, item 6) and reported separately rather than averaged.

  bash scripts/run.sh scripts/profile_sm89_power.py --gpu 6 --batch 8 \
      --modules idle,ar_round,cfm,vocoder,first_chunk \
      --seconds 2.0 --ref-audio outputs/profile_sm89/reference.wav \
      --json-out outputs/profile_sm89/power_b8.json
"""
import argparse
import json
import os
import statistics
import subprocess
import threading
import time
from pathlib import Path


class NvmlSampler:
    """Continuous board-power / clock / throttle sampling for one physical GPU."""

    THROTTLE_BITS = {
        "gpu_idle": 0x1,
        "applications_clocks": 0x2,
        "sw_power_cap": 0x4,
        "hw_slowdown": 0x8,
        "sync_boost": 0x10,
        "sw_thermal_slowdown": 0x20,
        "hw_thermal_slowdown": 0x40,
        "hw_power_brake_slowdown": 0x80,
        "display_clock_setting": 0x100,
    }

    def __init__(self, gpu):
        self.gpu = str(gpu)
        self.samples = []
        self.process = None
        self.thread = None

    def start(self):
        command = ["nvidia-smi", "-i", self.gpu,
                   "--query-gpu=power.draw,clocks.sm,temperature.gpu,utilization.gpu,clocks_throttle_reasons.active",
                   "--format=csv,noheader,nounits", "-lms", "10"]
        self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        text=True, bufsize=1)

        def collect():
            for line in self.process.stdout:
                fields = [f.strip() for f in line.split(",")]
                if len(fields) < 5:
                    continue
                try:
                    throttle_raw = fields[4]
                    throttle = float(int(throttle_raw, 16)) if throttle_raw.lower().startswith("0x") \
                        else float(throttle_raw)
                    self.samples.append((time.perf_counter(), float(fields[0]), float(fields[1]),
                                         float(fields[2]), float(fields[3]), throttle))
                except (ValueError, TypeError):
                    continue

        self.thread = threading.Thread(target=collect, daemon=True)
        self.thread.start()
        time.sleep(0.2)

    def stop(self):
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.thread.join(timeout=2)

    def window(self, start, end):
        rows = [r for r in self.samples if start <= r[0] <= end]
        return rows

    @staticmethod
    def summarise(rows, baseline_w, seconds, calls, min_clock_fraction):
        if not rows:
            return {"samples": 0, "valid": False, "reason": "no NVML samples in window"}
        power = [r[1] for r in rows]
        clocks = [r[2] for r in rows]
        temps = [r[3] for r in rows]
        throttle = [int(r[5]) for r in rows]
        median_clock = statistics.median(clocks)
        throttled = any(t != 0 for t in throttle)
        low_clock = median_clock < min_clock_fraction * max(clocks)
        mean_w = statistics.fmean(power)
        energy_j = mean_w * seconds
        net_energy_j = max(0.0, (mean_w - baseline_w)) * seconds
        return {
            "samples": len(rows),
            "seconds": seconds,
            "power_w_mean": mean_w,
            "power_w_peak": max(power),
            "power_w_min": min(power),
            "baseline_w": baseline_w,
            "energy_j": energy_j,
            "net_energy_j": net_energy_j,
            "j_per_call": net_energy_j / calls if calls else None,
            "calls": calls,
            "sm_clock_mhz_mean": statistics.fmean(clocks),
            "sm_clock_mhz_median": median_clock,
            "sm_clock_mhz_min": min(clocks),
            "sm_clock_mhz_max": max(clocks),
            "temp_c_max": max(temps),
            "throttle_reasons": sorted({t for t in throttle if t}),
            "throttled": throttled,
            "valid": not throttled and not low_clock,
            "reason": ("driver reported throttle reason(s) " + str(sorted({t for t in throttle if t})))
                      if throttled else ("sm clock collapsed within window" if low_clock else None),
        }


def build_modules(engine, torch, batch, text):
    """Return callables that run exactly one module, using the real runtime objects."""
    captured = {}
    original_student = engine.student
    original_vocoder = engine.vocoder

    class Recorder(torch.nn.Module):
        """Wraps a callable, remembers the first real argument tuple, proxies attributes.

        The attribute proxy matters: callers reach through engine.student to things like
        .model and .identity, and those must keep working.
        """

        def __init__(self, fn, key):
            super().__init__()
            self.fn = fn
            self.key = key

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(object.__getattribute__(self, "fn"), name)

        def forward(self, *args, **kwargs):
            captured.setdefault(self.key, (tuple(args), dict(kwargs)))
            return self.fn(*args, **kwargs)

    engine.student = Recorder(original_student, "cfm")
    engine.vocoder = Recorder(original_vocoder, "vocoder")

    def admit_and_run_to_first_chunk(prefix):
        identifiers = [f"{prefix}-{i}" for i in range(batch)]
        for ident in identifiers:
            engine.create_session(ident, "reference", 0)
            engine.push_text(ident, text)
            engine.finish_input(ident)
        pending = set(identifiers)
        while pending:
            events = engine.run_ready()
            if not events:
                raise RuntimeError("Scheduler returned no events")
            for event in events:
                pending.discard(event["request_id"])
        return identifiers

    def cfm_call():
        args, kwargs = captured["cfm"]
        with torch.cuda.stream(engine.model.stream), torch.inference_mode():
            return engine.student(*args, **kwargs)

    def vocoder_call():
        args, kwargs = captured["vocoder"]
        with torch.cuda.stream(engine.model.stream), torch.inference_mode():
            return engine.vocoder(*args, **kwargs)

    def prepare(sessions, index=0):
        # The engine's own path always runs prepare_rows under inference_mode and on
        # the model stream; calling it bare trips the inference-tensor inplace guard.
        with torch.cuda.stream(engine.model.stream), torch.inference_mode():
            return engine.prepare_rows(sessions, index)

    def admit(prefix):
        identifiers = [f"{prefix}-{i}" for i in range(batch)]
        for ident in identifiers:
            engine.create_session(ident, "reference", 0)
            engine.push_text(ident, text)
            engine.finish_input(ident)
        return identifiers

    def ar_round_factory():
        """Prepare a fresh row set, then return a stepper that re-runs one AR round."""
        state = {"rows": [], "identifiers": [], "steps": 0}

        def refresh():
            # The rows must be parked on the session as _row, otherwise engine.cancel
            # cannot find them and the Target KV slots leak until exhaustion.
            identifiers = admit("ar")
            sessions = [engine.sessions[i] for i in identifiers]
            rows = prepare(sessions)
            for session, row in zip(sessions, rows):
                session["_row"] = row
            state["rows"] = rows
            state["identifiers"] = identifiers

        refresh()

        def step():
            with torch.cuda.stream(engine.model.stream), torch.inference_mode():
                engine.rt._step(state["rows"], engine.config["max_speech_tokens"])
            state["steps"] += 1
            # Rebuild well before the speech-token cap, otherwise accepted_prefix()
            # receives a negative remaining budget and raises.
            if state["steps"] % 150 == 0:
                for ident in state["identifiers"]:
                    engine.cancel(ident)
                refresh()

        return step, lambda: state["identifiers"]

    def device_child_factory():
        """Build stable B8 child-graph inputs for isolated steady-state power."""
        if not getattr(engine, "device_round_b8", False) or batch not in engine.device_round_batches:
            raise RuntimeError("Device child profiling requires an enabled exact-batch device round")
        identifiers = admit("device-child")
        sessions = [engine.sessions[ident] for ident in identifiers]
        rows = prepare(sessions)
        for session, row in zip(sessions, rows):
            session["_row"] = row
        from acc_infer_clear.dspark.device_round import DeviceRoundHead
        from acc_infer_clear.kernels.device_commit import mark_keep
        runner = DeviceRoundHead(engine.rt, rows, engine.config["max_speech_tokens"])
        first = runner.past + 1 - runner.mel
        draft_args = (runner.last, first[:, None] + runner.step7,
                      runner.draft_slots, runner.draft_lengths)
        with torch.cuda.stream(engine.model.stream), torch.inference_mode():
            hidden, base = engine.rt.backbone.graphs[batch, 128](*draft_args)
            noise = torch.empty_like(base).exponential_(generator=engine.rt.proposal.batch_generator)
            proposed, _, _ = engine.rt.proposal.graphs[batch](hidden, base, noise, runner.last)
            tokens = torch.cat((runner.last[:, None], proposed), 1)
            positions = first[:, None] + runner.step8
            target_model = engine.rt.engine.target.model
            target_x = target_model.embeddings(tokens) + target_model.text_pos_embedding.emb(positions)
            mark_keep(engine.rt.target.keep, runner.target_slots, runner.past)
        target_args = (target_x, runner.target_slots, runner.past)
        native_bank = getattr(engine.rt.target, "native_full_bank", None)
        native_target = (native_bank if native_bank is not None and
                         native_bank.eligible(batch, runner.target_slots.cpu().tolist(),
                                              int(runner.past.max().item())) else None)

        def draft_backbone():
            with torch.cuda.stream(engine.model.stream), torch.inference_mode():
                return engine.rt.backbone.graphs[batch, 128](*draft_args)

        def draft_full():
            with torch.cuda.stream(engine.model.stream), torch.inference_mode():
                current_hidden, current_base = engine.rt.backbone.graphs[batch, 128](*draft_args)
                return engine.rt.proposal.graphs[batch](current_hidden, current_base, noise, runner.last)

        def target_verify():
            with torch.cuda.stream(engine.model.stream), torch.inference_mode():
                if native_target is not None:
                    return native_target.graphs[batch, 128](*target_args)
                return engine.rt.target.graphs[batch, 128](*target_args)

        def cleanup():
            torch.cuda.synchronize()
            for ident in identifiers:
                engine.cancel(ident)

        return {"draft_backbone": draft_backbone, "draft_full": draft_full,
                "target_verify": target_verify, "cleanup": cleanup}

    return {
        "first_chunk": admit_and_run_to_first_chunk,
        "cfm": cfm_call,
        "vocoder": vocoder_call,
        "ar_round_factory": ar_round_factory,
        "device_child_factory": device_child_factory,
        "captured": captured,
        "restore": lambda: (setattr(engine, "student", original_student),
                            setattr(engine, "vocoder", original_vocoder)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--modules", default="idle,ar_round,cfm,vocoder,first_chunk")
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--baseline-seconds", type=float, default=3.0)
    parser.add_argument("--min-clock-fraction", type=float, default=0.9)
    parser.add_argument("--config", default="configs/runtime.yaml")
    parser.add_argument("--deployment", default="configs/sm89_bf16_triton.json")
    parser.add_argument("--ref-audio", default="outputs/profile_sm89/reference.wav")
    parser.add_argument("--text", default="他正在整理文件。")
    parser.add_argument("--json-out")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    from acc_infer_clear.config import load as load_config
    from acc_infer_clear.runtime.deployment import load as load_deployment
    from acc_infer_clear.runtime.device import GPULease
    from acc_infer_clear.streaming.engine import Engine

    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible GPU")

    config = load_config(args.config)
    config["max_batch"] = args.batch
    deployment = load_deployment(args.deployment)
    modules = [m.strip() for m in args.modules.split(",") if m.strip()]
    # With head graphs enabled the acoustic route is graph replay, so the eager
    # engine.student / engine.vocoder call sites the recorder hooks are never reached.
    # CFM and vocoder module tiers therefore require the graph-free diagnostic plan.
    if deployment.get("head_graphs") and ({"cfm", "vocoder"} & set(modules)):
        raise SystemExit(
            "cfm/vocoder power tiers need a plan with head_graphs=false "
            "(use outputs/profile_sm89/deployment_nographs_diagnostic.json)")
    report = {"gpu": args.gpu, "batch": args.batch, "deployment": deployment["status"],
              "seconds_per_tier": args.seconds, "modules": {}, "online_tuning": False}

    def host_state():
        output = subprocess.check_output(
            ["nvidia-smi", "-i", str(args.gpu),
             "--query-gpu=power.draw,clocks.sm,temperature.gpu,persistence_mode,power.limit,clocks_throttle_reasons.active",
             "--format=csv,noheader,nounits"], text=True).strip()
        return output

    with GPULease(args.gpu):
        engine = Engine(config)
        try:
            engine.prepare_reference("reference", args.ref_audio)
            engine.prepare_deployment(deployment)
            torch.cuda.synchronize()
            report["state_after_deployment"] = host_state()

            sampler = NvmlSampler(args.gpu)
            sampler.start()
            # Idle baseline: model resident, nothing executing.
            torch.cuda.synchronize()
            time.sleep(args.baseline_seconds)
            baseline_rows = sampler.window(time.perf_counter() - args.baseline_seconds, time.perf_counter())
            idle_w = statistics.fmean(r[1] for r in baseline_rows) if baseline_rows else 0.0
            report["idle"] = {
                "power_w_mean": idle_w,
                "samples": len(baseline_rows),
                "sm_clock_mhz_mean": statistics.fmean(r[2] for r in baseline_rows) if baseline_rows else None,
                "temp_c_mean": statistics.fmean(r[3] for r in baseline_rows) if baseline_rows else None,
            }

            if "idle" in modules:
                report["modules"]["idle"] = dict(report["idle"], valid=True)

            helpers = build_modules(engine, torch, args.batch, args.text)

            # One real pass first: fills the recorder and leaves the engine warm.
            warm = helpers["first_chunk"]("capture")
            torch.cuda.synchronize()
            for ident in warm:
                engine.cancel(ident)

            def timed(label, fn):
                # Warm the exact closure, then measure.
                fn()
                torch.cuda.synchronize()
                started = time.perf_counter()
                calls = 0
                while time.perf_counter() - started < args.seconds:
                    fn()
                    calls += 1
                torch.cuda.synchronize()
                completed = time.perf_counter()
                rows = sampler.window(started, completed)
                summary = NvmlSampler.summarise(rows, idle_w, completed - started, calls,
                                                args.min_clock_fraction)
                summary["calls_per_second"] = calls / (completed - started)
                report["modules"][label] = summary
                return summary

            if "ar_round" in modules:
                step, current_identifiers = helpers["ar_round_factory"]()
                timed("ar_round", step)
                for ident in current_identifiers():
                    engine.cancel(ident)

            device_labels = {"draft_backbone", "draft_full", "target_verify"} & set(modules)
            if device_labels:
                child = helpers["device_child_factory"]()
                try:
                    for label in ("draft_backbone", "draft_full", "target_verify"):
                        if label in device_labels:
                            timed(label, child[label])
                finally:
                    child["cleanup"]()

            if "cfm" in modules:
                timed("cfm", helpers["cfm"])

            if "vocoder" in modules:
                timed("vocoder", helpers["vocoder"])

            if "first_chunk" in modules:
                state = {"calls": 0}

                def one_first_chunk():
                    identifiers = helpers["first_chunk"](f"p{state['calls']}")
                    state["calls"] += 1
                    torch.cuda.synchronize()
                    for ident in identifiers:
                        engine.cancel(ident)

                summary = timed("first_chunk", one_first_chunk)
                summary["j_per_first_chunk"] = summary["j_per_call"]
                summary["j_per_request_at_batch"] = (summary["j_per_call"] / args.batch
                                                     if summary["j_per_call"] else None)

            helpers["restore"]()
            report["state_after"] = host_state()
            sampler.stop()
        finally:
            engine.close()

    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.json_out:
        Path(args.json_out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
