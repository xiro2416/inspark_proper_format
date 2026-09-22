#!/usr/bin/env python3
"""Bounded real native Target/Draft graph snapshots plus acoustic trajectory.

This supplements, not replaces, all-shape concurrency/fallback validation. Four
real calls per native component are frozen by default; graph dictionary identity
is preserved in both the runtime and native bank. No runtime source edits.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

from acc_infer_clear.guardrails.numerics import TOLERANCES, compare
from acc_infer_clear.guardrails.snapshots import cpu_copy, load_bundle, save_bundle, write_json
from trt113_provenance import capture_provenance, file_record, source_identity


def new_target_kv(cache, lengths):
    import torch
    return torch.stack([cache[:, :, row, :, int(length):int(length) + 8]
                        for row, length in enumerate(lengths)], dim=2)


def compare_outputs(reference, candidate):
    if not reference or set(reference) != set(candidate):
        raise ValueError("Missing/mismatched output names")
    checks = {name: compare(reference[name], candidate[name], "bf16") for name in reference}
    return {"checks": checks, "pass_gate": all(value["pass_gate"] for value in checks.values())}


class GraphProxy:
    def __init__(self, graph, recorder, component, batch, backend, bank):
        self.wrapped, self.recorder, self.component = graph, recorder, component
        self.batch, self.backend, self.bank = batch, backend, bank

    def __getattr__(self, name):
        return getattr(self.wrapped, name)

    def __call__(self, *args):
        if self.component == "draft":
            self.recorder.draft_positions[self.batch] = cpu_copy(args[1])
        if self.recorder.counts[self.component] >= self.recorder.limit:
            return self.wrapped(*args)
        return self.recorder.call(self, args)


class ProposalProxy:
    """Observe existing proposals, including callers using .graph.replay()."""
    def __init__(self, graph, recorder):
        self.wrapped, self.recorder = graph, recorder
        proxy = self
        class Replay:
            def __getattr__(self, name):
                return getattr(graph.graph, name)
            def replay(self):
                result = graph.graph.replay()
                proxy.record()
                return result
        self.graph = Replay()

    def __getattr__(self, name):
        return getattr(self.wrapped, name)

    def record(self):
        import torch
        previous = self.wrapped.inputs[3]
        self.recorder.verify_tokens[previous.shape[0]] = cpu_copy(
            torch.cat((previous[:, None], self.wrapped.outputs[0]), dim=1))

    def __call__(self, *args):
        result = self.wrapped(*args)
        self.record()
        return result


class ProposalCallProxy:
    """Observe generic request-owned sampling without changing draws/results."""
    def __init__(self, proposal, recorder):
        self.wrapped, self.recorder = proposal, recorder

    def __getattr__(self, name):
        return getattr(self.wrapped, name)

    def __call__(self, jobs, tasks):
        import torch
        result = self.wrapped(jobs, tasks)
        batch = len(jobs)
        self.recorder.verify_tokens[batch] = cpu_copy(torch.cat([
            torch.cat((job["anchor_token"].reshape(1, 1), output[0]), dim=1)
            for job, output in zip(jobs, result)]))
        first = torch.tensor([job["first_position"] for job in jobs])
        self.recorder.draft_positions[batch] = first[:, None] + torch.arange(7)[None]
        return result


def ar_coverage(calls):
    counts = {component: sum(row["component"] == component for row in calls)
              for component in ("target", "draft")}
    batches = {component: sorted({row["batch"] for row in calls if row["component"] == component})
               for component in counts}
    return {"required_components": ["target", "draft"], "calls": counts, "batches": batches,
            "pass_gate": all(counts.values()),
            "scope": "at least one actual native call per component; installed engines alone are not coverage"}


def bitwise_equal(left, right):
    import torch
    return (left.shape == right.shape and left.dtype == right.dtype and
            torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8)))


def target_untouched_cache_equal(before, after, lengths):
    return all(bitwise_equal(before[:, :, row, :, :length], after[:, :, row, :, :length]) and
               bitwise_equal(before[:, :, row, :, length+8:], after[:, :, row, :, length+8:])
               for row, length in enumerate(lengths))


def replay_ar_engine_evidence(manifest, model_provenance):
    """Recheck frozen engine/metadata hashes and actual current loader files."""
    from validate_trt113_ar import engine_evidence
    result = {}
    for row in manifest["ar_calls"]:
        component, batch = row["component"], row["batch"]
        if str(batch) in result.get(component, {}):
            continue
        frozen = manifest["ar_engines"][component][str(batch)]
        if row["route"]["artifact"]["sha256"] != frozen["sha256"]:
            raise ValueError("Captured AR route does not match its engine evidence")
        paths = {item["role"]: item["path"] for item in model_provenance[component]["model_sources"]}
        current = engine_evidence(frozen["path"], batch, component, paths)
        if current["sha256"] != frozen["sha256"]:
            raise ValueError("AR engine SHA256 changed after capture")
        if current["build_metadata_sha256"] != frozen["build_metadata_sha256"]:
            raise ValueError("AR build metadata SHA256 changed after capture")
        if current["weight_identity_verified"] and not frozen["weight_identity_verified"]:
            raise ValueError("Reference replay cannot upgrade unverified AR capture provenance")
        result.setdefault(component, {})[str(batch)] = current
    return result


class BoundedARRecorder:
    def __init__(self, engine, directory, manifest, limit):
        self.engine, self.directory, self.manifest = engine, Path(directory), manifest
        self.limit, self.counts = limit, {"target": 0, "draft": 0}
        self.verify_tokens, self.draft_positions = {}, {}
        manifest["ar_calls"] = []
        manifest["ar_capture_coverage"] = ar_coverage([])
        manifest["ar_scope"] = {"max_calls_per_component": limit,
            "reference": "same real graph inputs/cache, not independent regenerated trajectory",
            "not_covered": ["generic fallback AR calls", "proposal RNN/acceptance/residual tensors",
                            "prefix prefill numerical comparison", "context projection/scatter replay"]}

    def install(self):
        from validate_trt113_ar import engine_evidence
        self.manifest["ar_engines"] = {}
        for batch, graph in list(self.engine.rt.proposal.graphs.items()):
            self.engine.rt.proposal.graphs[batch] = ProposalProxy(graph, self)
        self.engine.rt.proposal = ProposalCallProxy(self.engine.rt.proposal, self)
        for component, owner in (("target", self.engine.rt.target), ("draft", self.engine.rt.backbone)):
            bank = getattr(owner, "native_full_bank", None)
            if bank is None:
                continue
            self.manifest["ar_engines"][component] = {}
            for key, graph in list(bank.graphs.items()):
                actual_paths = {row["role"]: row["path"] for row in
                                self.manifest["model_provenance"][component]["model_sources"]}
                evidence = engine_evidence(
                    bank.artifacts[key[0]]["path"], key[0], component, actual_paths)
                if evidence["sha256"] != bank.artifacts[key[0]]["sha256"]:
                    raise ValueError("AR engine file differs from the already loaded engine")
                self.manifest["ar_engines"][component][str(key[0])] = evidence
                proxy = GraphProxy(graph, self, component, key[0], bank.backends[key[0]], bank)
                bank.graphs[key] = proxy
                # DeviceRoundHead requires this identity, not merely equal keys.
                if owner.graphs.get(key) is graph:
                    owner.graphs[key] = proxy

    def call(self, proxy, inputs):
        import torch
        component, backend, batch = proxy.component, proxy.backend, proxy.batch
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("AR recorder cannot run during graph capture")
        slots, lengths = inputs[-2:]
        if slots.tolist() != list(range(batch)):
            raise ValueError("Native actual-trace slot identity invariant failed")
        before = cpu_copy(backend.cache[:, :, :batch])
        bundle = {"inputs": cpu_copy(inputs), "cache_before": before}
        if component == "target":
            bundle["keep"] = cpu_copy(self.engine.rt.target.keep[:batch])
            if batch not in self.verify_tokens or batch not in self.draft_positions:
                raise ValueError("No actual proposal/position evidence for native verification")
            bundle["verify_tokens"] = self.verify_tokens[batch].clone()
            bundle["target_positions"] = self.draft_positions[batch][:, :1] + torch.arange(8)[None]
            target = self.engine.rt.engine.target.model
            reproduced_x = target.embeddings(bundle["verify_tokens"].to(inputs[0].device)) + target.text_pos_embedding.emb(
                bundle["target_positions"].to(inputs[0].device))
            if not torch.equal(reproduced_x, inputs[0]):
                raise ValueError("Captured proposal tokens do not reproduce actual Target embeddings")
        else:
            self.draft_positions[batch] = cpu_copy(inputs[1])
        members = []
        for slot in slots.tolist():
            matching = [s for s in self.engine.sessions.values() if "_row" in s and
                        (getattr(s["_row"].kv, "slot", None) if component == "target"
                         else getattr(s["_row"].cache, "pool_slot", None)) == slot]
            if len(matching) != 1:
                raise ValueError("Cannot establish native AR slot-to-request mapping")
            members.append({"id": matching[0]["case"]["id"], "slot": slot})
        outputs = proxy.wrapped(*inputs)
        names = ("logits", "selected", "final") if component == "target" else ("hidden", "base")
        frozen = cpu_copy(dict(zip(names, outputs)))
        after = cpu_copy(backend.cache[:, :, :batch])
        length_values = lengths.tolist()
        state_checks = {"graph_untouched_cache": (target_untouched_cache_equal(before, after, length_values)
                         if component == "target" else bitwise_equal(before, after))}
        if component == "target":
            frozen["new_kv"] = new_target_kv(after, lengths.tolist())
        # Save frozen graph results before the direct validation call can reuse
        # outputs. Restore all Target cache bytes afterward to preserve state.
        device = inputs[0].device
        try:
            if component == "target":
                backend.cache[:, :, :batch].copy_(before.to(device))
                direct = backend.run(inputs[0], self.engine.rt.target.keep, slots, lengths)
            else:
                x = self.engine.rt.engine.draft._noise_embeddings(inputs[0], inputs[1])
                direct = backend.run(x, slots, lengths)
            direct_frozen = cpu_copy(dict(zip(names, direct)))
            direct_after = cpu_copy(backend.cache[:, :, :batch])
            state_checks["direct_untouched_cache"] = (
                target_untouched_cache_equal(before, direct_after, length_values)
                if component == "target" else bitwise_equal(before, direct_after))
            if component == "target":
                direct_frozen["new_kv"] = new_target_kv(direct_after, length_values)
        finally:
            # Draft's declared cache input is read-only. Restoring its original
            # bytes also prevents a failed audit invocation polluting the run.
            restored = after if component == "target" else before
            backend.cache[:, :, :batch].copy_(restored.to(device))
            state_checks["audit_cache_restored"] = bitwise_equal(restored, cpu_copy(backend.cache[:, :, :batch]))
        bundle.update(output=frozen, direct_output=direct_frozen, cache_after=after)
        exact = {name: bool(torch.equal(frozen[name], direct_frozen[name])) for name in frozen}
        index = len(self.manifest["ar_calls"])
        evidence = save_bundle(self.directory / "ar_bundles", index, bundle)
        self.manifest["ar_calls"].append({"index": index, "component": component,
            "batch": batch, "lengths": lengths.tolist(), "members": members,
            "route": {"backend": "tensorrt113", "execution": "existing_graph", "graph_key": [batch, 128],
                      "artifact": proxy.bank.artifacts[batch]},
            "evidence": evidence, "graph_direct_exact": exact,
            "graph_direct_exact_gate": all(exact.values()), "cache_state_checks": state_checks,
            "cache_state_gate": all(state_checks.values())})
        self.counts[component] += 1
        self.manifest["ar_capture_coverage"] = ar_coverage(self.manifest["ar_calls"])
        write_json(self.directory / "capture.json", self.manifest)
        return tuple(frozen[name].to(device) for name in names)


def actual_target_reference(engine, bundle):
    """Each ragged row consumes its exact BF16-stored cache promoted to FP32."""
    import torch
    x, slots, lengths = bundle["inputs"]
    cache = bundle["cache_before"]
    results = {name: [] for name in ("logits", "selected", "final", "new_kv")}
    for row, length in enumerate(lengths.tolist()):
        past = tuple((cache[layer, 0, row:row+1, :, :length].float(),
                      cache[layer, 1, row:row+1, :, :length].float()) for layer in range(cache.shape[0]))
        keep = bundle["keep"][row:row+1, :length+8]
        logits, present, selected, final = engine.rt.engine.target._block_forward_with_hidden_states(
            x[row:row+1], past, keep, None)
        results["logits"].append(logits); results["selected"].append(selected); results["final"].append(final)
        results["new_kv"].append(torch.stack([torch.stack((k[:, :, -8:], v[:, :, -8:])) for k, v in present]))
    return {name: torch.cat(values, dim=2 if name == "new_kv" else 0) for name, values in results.items()}


def actual_draft_reference(engine, bundle):
    import torch
    from acc_infer_clear.runtime.indextts2.batch_draft import BatchedDraftBackbone
    anchors, positions, slots, lengths = bundle["inputs"]
    cache = bundle["cache_before"]
    if cache.dtype != torch.float32:
        raise ValueError("Native Draft context must remain FP32 storage")
    backbone = BatchedDraftBackbone(engine.rt.engine.draft)
    keep = torch.arange(cache.shape[-2], device=anchors.device)[None] < lengths[:, None]
    context_positions = lengths[:, None] + torch.arange(7, device=anchors.device)[None]
    hidden, base = backbone.forward(anchors, positions,
        tuple(cache[layer, 0] for layer in range(cache.shape[0])),
        tuple(cache[layer, 1] for layer in range(cache.shape[0])), keep,
        context_positions if backbone.context_uses_positions else None)
    return {"hidden": hidden, "base": base}


def capture(args):
    import audit_real_acoustics as acoustic
    original = acoustic.AcousticRecorder
    class CombinedRecorder(original):
        def install(self):
            super().install()
            BoundedARRecorder(self.engine, self.directory, self.manifest, args.ar_max_calls).install()
    original_provenance = acoustic.reference_provenance
    try:
        acoustic.AcousticRecorder = CombinedRecorder
        acoustic.reference_provenance = lambda engine, options: {
            component: capture_provenance(component, engine.config, options.config, model=engine)
            for component in ("target", "draft", "cfm", "vocoder")}
        acoustic.capture_run(args)
    finally:
        acoustic.AcousticRecorder = original
        acoustic.reference_provenance = original_provenance


def reference(args):
    import torch
    from acc_infer_clear.runtime.config import load
    from acc_infer_clear.runtime.engine import Engine
    manifest_path = args.capture_dir / "capture.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "captured" or not manifest.get("ar_calls"):
        raise ValueError("Completed actual native AR capture is required")
    output = args.capture_dir / f"ar_reference_{args.precision}"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Preserve immutable reference replay evidence")
    output.mkdir(parents=True, exist_ok=True)
    report = {"schema": 1, "status": "running", "scope": "real_native_AR_same_input",
              "capture": file_record(manifest_path, "capture"), "calls": [], "pass_gate": False,
              "source": source_identity(), "precision": args.precision, "tolerances": TOLERANCES,
              "reference": "raw Engine; original Target method and BatchedDraftBackbone.forward; no graphs/custom ops",
              "cache_boundary": "Target exact captured BF16 KV promoted to FP32, NOT raw unrounded FP32 trajectory; Draft captured FP32 cache unchanged",
              "target_batching": "ragged row-wise eager; same input semantics, not a batched performance baseline",
              "arithmetic": "BF16 Linear wrappers may output FP32; native BF16 attention/cache arithmetic is not identical",
              "performance_claim": False}
    engine = None
    try:
        config = load(args.config); config["max_batch"] = manifest["config"]["max_batch"]
        config["target_tf32"] = False
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        engine = Engine(config)
        report["model_provenance"] = {component: capture_provenance(component, config, args.config, model=engine)
                                       for component in ("target", "draft")}
        for component, provenance in report["model_provenance"].items():
            before = {r["role"]: r["sha256"] for r in manifest["model_provenance"][component]["model_sources"]}
            current = {r["role"]: r["sha256"] for r in provenance["model_sources"]}
            if before != current:
                raise ValueError(f"Actual {component} loader checkpoint/config hash changed")
        report["coverage"] = ar_coverage(manifest["ar_calls"])
        report["engine_identity"] = replay_ar_engine_evidence(manifest, report["model_provenance"])
        identities = [evidence for engines in report["engine_identity"].values() for evidence in engines.values()]
        report["weight_identity_verified"] = bool(identities) and all(evidence["weight_identity_verified"] for evidence in identities)
        if args.precision == "bf16":
            engine.prepare_precision("bf16", ["target", "draft"], False)
        with torch.cuda.stream(engine.model.stream), torch.inference_mode():
            for row in manifest["ar_calls"]:
                bundle = load_bundle(args.capture_dir / "ar_bundles", row["evidence"])
                device_bundle = {"inputs": tuple(x.to("cuda:0") for x in bundle["inputs"]),
                                 "cache_before": bundle["cache_before"].to("cuda:0")}
                if "keep" in bundle:
                    device_bundle["keep"] = bundle["keep"].to("cuda:0")
                fn = actual_target_reference if row["component"] == "target" else actual_draft_reference
                expected = cpu_copy(fn(engine, device_bundle))
                evidence = save_bundle(output / "bundles", row["index"], expected)
                checks = {"native_graph_vs_reference": compare_outputs(expected, bundle["output"]),
                          "native_direct_vs_reference": compare_outputs(expected, bundle["direct_output"])}
                if args.precision == "bf16":
                    fp32_report_path = args.capture_dir / "ar_reference_fp32" / "report.json"
                    if fp32_report_path.exists():
                        fp32_report = json.loads(fp32_report_path.read_text())
                        if fp32_report["capture"]["sha256"] != report["capture"]["sha256"]:
                            raise ValueError("FP32/BF16 references belong to different captures")
                        fp32_row = next(item for item in fp32_report["calls"] if item["index"] == row["index"])
                        fp32_output = load_bundle(args.capture_dir / "ar_reference_fp32" / "bundles", fp32_row["evidence"])
                        checks["bf16_reference_vs_fp32_reference"] = compare_outputs(fp32_output, expected)
                report["calls"].append({"index": row["index"], "component": row["component"],
                    "evidence": evidence, "checks": checks, "graph_direct_exact_gate": row["graph_direct_exact_gate"],
                    "cache_state_gate": row.get("cache_state_gate", False),
                    "pass_gate": row["graph_direct_exact_gate"] and row.get("cache_state_gate", False)
                                 and all(v["pass_gate"] for v in checks.values())})
                write_json(output / "report.json", report)
        report["status"] = "completed"
        report["pass_gate"] = report["coverage"]["pass_gate"] and all(row["pass_gate"] for row in report["calls"])
    except Exception as error:
        report["status"] = "error"
        report["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        raise
    finally:
        if engine is not None:
            engine.close()
        write_json(output / "report.json", report)
    if not report["pass_gate"]:
        raise SystemExit(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--config", default="configs/runtime_reference.yaml")
    commands = parser.add_subparsers(dest="command", required=True)
    cap = commands.add_parser("capture")
    cap.add_argument("--deployment", required=True)
    cap.add_argument("--corpus", type=Path, default=Path("configs/sm89_quality_256.json"))
    cap.add_argument("--start", type=int, default=0)
    cap.add_argument("--cases", type=int, default=1)
    cap.add_argument("--batch", type=int, choices=(1, 4, 8, 16), default=1)
    cap.add_argument("--ar-max-calls", type=int, default=4)
    cap.add_argument("--output-dir", type=Path, required=True)
    ref = commands.add_parser("reference")
    ref.add_argument("--capture-dir", type=Path, required=True)
    ref.add_argument("--precision", choices=("fp32", "bf16"), required=True)
    args = parser.parse_args()
    if args.command == "capture" and (args.ar_max_calls < 1 or args.cases < 1 or args.start < 0):
        parser.error("Invalid capture count/range")
    from acc_infer_clear.runtime.device import GPULease, select_gpu
    with GPULease(args.gpu):
        select_gpu(args.gpu)
        {"capture": capture, "reference": reference}[args.command](args)


if __name__ == "__main__":
    main()
