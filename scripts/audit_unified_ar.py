#!/usr/bin/env python3
"""Report first-round static TRT Draft/RNN/Target error on real frozen KV.

References share the actual deployed prefix caches and sampled proposal chain.
The FP32 reference changes weights/compute precision, not the frozen history.
This audit has no numerical acceptance thresholds and makes no CER claim.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from inspark_infer.runtime.bundle_paths import read_json

import torch
from torch.nn import functional as F

from scripts.audit_unified_acoustics import cases_for_capture, digest, metrics


def target_reference(target, values, bf16_attention):
    """Mirror Q8 append verification, including explicit BF16 Q/K/V boundaries."""
    tm = target.model
    hidden, cache = values["target_x"], values["target_cache"]
    batch, query, _ = hidden.shape
    capacity = cache.shape[-2]
    position = torch.arange(capacity, device=hidden.device)
    valid = (position[None] < values["target_lengths"][:, None]) & values["target_keep"]
    current = torch.arange(query, device=hidden.device)
    causal = (current[:, None] >= current[None, :])[None, None].expand(batch, 1, -1, -1)
    mask = torch.cat((valid[:, None, None, :].expand(-1, 1, query, -1), causal), -1)
    dtype = torch.bfloat16 if bf16_attention else torch.float32
    selected, appended = [], []
    if cache.shape[0] != len(tm.transformer.h):
        raise ValueError("Target cache layer count differs from checkpoint")
    for index, block in enumerate(tm.transformer.h):
        attention = block.attn
        q, k, v = attention.c_attn(block.ln_1(hidden)).split(attention.split_size, dim=-1)
        reshape = lambda value: value.to(dtype).view(batch, query, attention.num_heads, attention.head_dim).transpose(1, 2)
        q, k, v = reshape(q), reshape(k), reshape(v)
        appended.append(torch.stack((k, v)))
        keys = torch.cat((cache[index, 0].to(dtype), k), 2)
        vals = torch.cat((cache[index, 1].to(dtype), v), 2)
        # The engine scales Q before attention, in Q's BF16 dtype. Supplying
        # SDPA's default scale instead would move that rounding boundary.
        context = F.scaled_dot_product_attention(q * (attention.head_dim ** -0.5), keys, vals,
                                                 attn_mask=mask, scale=1.0).float()
        context = context.transpose(1, 2).contiguous().view_as(hidden)
        hidden = hidden + attention.c_proj(context)
        hidden = hidden + block.mlp(block.ln_2(hidden))
        if index in target.target_layer_ids:
            selected.append(hidden)
    final = tm.transformer.ln_f(hidden)
    return {"target_logits": tm.lm_head(final), "target_selected": torch.cat(selected, -1),
            "target_final": final, "target_kv_append": torch.stack(appended)}


def draft_reference(backbone, values):
    cache = values["draft_cache"]
    keep = torch.arange(cache.shape[-2], device=cache.device)[None] < values["draft_lengths"][:, None]
    context_positions = values["draft_lengths"][:, None] + torch.arange(7, device=cache.device)[None]
    hidden, base = backbone.forward(values["anchors"], values["draft_positions"],
                                    tuple(cache[i, 0] for i in range(cache.shape[0])),
                                    tuple(cache[i, 1] for i in range(cache.shape[0])), keep,
                                    context_positions if backbone.context_uses_positions else None)
    return {"draft_hidden": hidden, "draft_base": base}


def conditional_rnn_reference(model, hidden, base, anchors, sampled_tokens):
    """Original checkpoint equations, teacher-forced on the ACTUAL sample chain."""
    if model.markov_cell_type not in ("linear", "linear_wide") or model.persistent_markov_state:
        raise ValueError("This audit requires the checkpoint's round-local linear RNN")
    state = hidden.new_zeros(hidden.shape[0], model.markov_state_size, dtype=torch.float32)
    previous, logits, probabilities = anchors.long(), [], []
    for step in range(7):
        embedding = model.markov_in(previous).float()
        raw = model.markov_rnn(torch.cat((state, embedding, hidden[:, step].float()), -1))
        gate, candidate, output = raw.split((model.markov_state_size, model.markov_state_size, model.markov_rank), -1)
        gate = torch.sigmoid(gate)
        state = gate * state + (1 - gate) * torch.tanh(candidate)
        delta = F.linear(torch.tanh(output), model._effective_markov_output_weight(step))
        logit = base[:, step].float() + delta
        logits.append(logit)
        probabilities.append(torch.softmax(logit / 0.8, -1))
        previous = sampled_tokens[:, step].long()
    return {"proposal_logits": torch.stack(logits, 1), "proposal_q": torch.stack(probabilities, 1)}


def _cpu(values):
    return {name: value.detach().cpu().clone() for name, value in values.items()}


def plan_model_sources(plan, batch):
    provenance = plan["provenance"]
    if "model_sources" not in provenance:
        provenance = provenance[str(batch)]
    return provenance["model_sources"]


def capture(args):
    from inspark_infer.runtime.config import load
    from inspark_infer.runtime.device import GPULease, select_gpu
    from inspark_infer.runtime.deployment import load as load_deployment
    from inspark_infer.runtime.engine import Engine
    from inspark_infer.ops.trtllm.runtime import RequestDraws
    from inspark_infer.ops.trtllm.pcg import categorical

    deployment = load_deployment(args.deployment)
    if deployment.get("schema") != 9:
        raise ValueError("Expected a unified schema-9 deployment")
    batch = deployment["batch"]
    cases = cases_for_capture(json.loads(args.manifest.read_text()), batch, 1, args.start)
    config = load(args.config)
    config["max_batch"] = batch
    plans = {name: read_json(Path(deployment[name + "_plan"])) for name in ("target", "draft")}
    with GPULease(args.gpu):
        select_gpu(args.gpu)
        engine = Engine(config)
        try:
            voices = {p: f"ar-audit-voice-{i}" for i, p in enumerate(dict.fromkeys(c["reference_audio"] for c in cases))}
            for source, voice in voices.items():
                engine.prepare_reference(voice, source)
            engine.prepare_deployment(deployment)
            identifiers = [f"ar-audit-{i}" for i in range(batch)]
            for ident, case in zip(identifiers, cases):
                engine.create_session(ident, voices[case["reference_audio"]], case["seed"], case.get("emotion"))
                engine.push_text(ident, case["text"])
                engine.finish_input(ident)
            with torch.cuda.stream(engine.model.stream), torch.inference_mode():
                rows = engine.prepare_rows([engine.sessions[ident] for ident in identifiers], 0)
                for ident, row in zip(identifiers, rows):
                    engine.sessions[ident]["_row"] = row
                controller = engine.unified_first_chunk
                provider, proposal = controller.provider, controller.runtime.proposal
                provider.import_rows(rows)
                provider.active.copy_(torch.tensor([not row.done for row in rows], device="cuda"))
                anchors = torch.cat([row.codes[-1] for row in rows]).long()
                lengths = torch.tensor([row.past_length for row in rows], device="cuda", dtype=torch.int32)
                draft_lengths = torch.tensor([row.cache.length for row in rows], device="cuda", dtype=torch.int32)
                mel_lengths = torch.tensor([row.mel_length for row in rows], device="cuda", dtype=torch.int32)
                first = lengths + 1 - mel_lengths
                draft_positions = first[:, None] + torch.arange(7, device="cuda")[None]
                target_positions = first[:, None] + torch.arange(8, device="cuda")[None]
                draws = RequestDraws([case["seed"] for case in cases], controller.runtime.max_rounds, provider.device if hasattr(provider, "device") else anchors.device)
                uniform = draws.values[:, 0, :7].clone()
                values = _cpu({"target_cache": provider.target_cache, "target_keep": provider.target_keep,
                               "draft_cache": provider.draft_cache, "anchors": anchors, "target_lengths": lengths,
                               "draft_lengths": draft_lengths, "mel_lengths": mel_lengths,
                               "draft_positions": draft_positions, "target_positions": target_positions,
                               "proposal_uniform": uniform})
                before = (provider.draft_engine.calls, provider.target_engine.calls)
                hidden, base = provider.draft(anchors, draft_positions, draft_lengths)
                proposed, probability, proposal_logits = proposal.sample_uniform(hidden, base, uniform, anchors)
                reproduced = torch.stack([categorical(probability[:, j], uniform[:, j]) for j in range(7)], 1)
                if not torch.equal(reproduced, proposed):
                    raise RuntimeError("Actual conditional q/uniforms do not reproduce the sampled proposal")
                verify_tokens = torch.cat((anchors[:, None], proposed), 1)
                tm = engine.rt.engine.target.model
                values.update(_cpu({"verify_tokens": verify_tokens,
                                    "target_x": tm.embeddings(verify_tokens) + tm.text_pos_embedding.emb(target_positions)}))
                logits, selected, final = provider.target(verify_tokens, target_positions, lengths)
                appended = torch.stack([torch.stack((provider.target_engine.outputs[f"k_append_{i}"],
                                                     provider.target_engine.outputs[f"v_append_{i}"])) for i in range(24)])
                actual = _cpu({"draft_hidden": hidden, "draft_base": base, "proposal_tokens": proposed,
                               "proposal_q": probability, "proposal_logits": proposal_logits,
                               "target_logits": logits, "target_selected": selected,
                               "target_final": final, "target_kv_append": appended})
                if (provider.draft_engine.calls - before[0], provider.target_engine.calls - before[1]) != (1, 1):
                    raise RuntimeError("Expected one actual Draft and Target TRT enqueue")
            payload = {"schema": 1, "kind": "unified_frozen_first_ar_round", "scheme": deployment["precision"],
                       "batch": batch, "kv_capacity": provider.capacity, "cases": cases,
                       "calibration": {"path": deployment["calibration"], "sha256": digest(deployment["calibration"])},
                       "deployment": {"path": str(args.deployment.resolve()), "sha256": digest(args.deployment), "resolved": deployment},
                       "plans": plans, "inputs": values, "actual": actual,
                       "execution": "direct first-round calls through the deployed StaticARProvider and official RNN adapter",
                       "rng": "actual request-owned framework DSpark round-0 uniforms",
                       "rnn_initial_state": "zero_per_round", "conditional_q_chain": "actual_sampled_previous_tokens",
                       "cache_source": "real deployed same-static-recipe prefill; Target BF16, Draft FP32; masked per-row lengths",
                       "initial_eos_rows": [bool(row.done) for row in rows]}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, args.output)
            for ident in identifiers:
                engine.cancel(ident)
            print(json.dumps({"capture": str(args.output), "scheme": payload["scheme"], "batch": batch,
                              "target_lengths": values["target_lengths"].tolist(), "kv_capacity": provider.capacity}), flush=True)
        finally:
            engine.close()


def replay_reference(engine, inputs, actual, bf16_attention):
    draft = draft_reference(engine.rt.backbone, inputs)
    target = target_reference(engine.rt.engine.target, inputs, bf16_attention)
    proposal = conditional_rnn_reference(engine.rt.engine.draft, draft["draft_hidden"], draft["draft_base"],
                                         inputs["anchors"], actual["proposal_tokens"])
    return _cpu(dict(draft, **target, **proposal))


def audit(args):
    from inspark_infer.runtime.config import load
    from inspark_infer.runtime.device import GPULease, select_gpu
    from inspark_infer.runtime.engine import Engine
    from inspark_infer.runtime.unified_deployment import install_reference_recipe
    from inspark_infer.quantization.unified import load_artifact
    from trt113_provenance import capture_provenance

    frozen = torch.load(args.capture, map_location="cpu", weights_only=True)
    if frozen.get("kind") != "unified_frozen_first_ar_round" or frozen.get("schema") != 1:
        raise ValueError("Expected a frozen unified first AR round")
    calibration = args.calibration or Path(frozen["calibration"]["path"])
    if digest(calibration) != frozen["calibration"]["sha256"]:
        raise ValueError("Calibration changed after the capture")
    artifact = load_artifact(calibration, frozen["scheme"])
    config = load(args.config)
    config.update(max_batch=frozen["batch"], target_tf32=False)
    report = {"schema": 1, "audit_kind": "reporting_only_no_numerical_gate", "numerical_thresholds": None,
              "scheme": frozen["scheme"], "batch": frozen["batch"], "kv_capacity": frozen["kv_capacity"],
              "capture_sha256": digest(args.capture), "calibration_sha256": digest(calibration),
              "same_recipe_reference": "static calibrated weighted ops; Target BF16 QKV/attention boundary; FP32 norm/residual/logits/RNN",
              "high_precision_reference": "original FP32 weights/attention on the SAME frozen cache values; no counterfactual FP32 prefill",
              "scope": "one real prefilled speculative round, fixed actual sampled proposal chain; no acceptance-distribution or speech-quality claim",
              "target_selected_layers": [1, 6, 11, 16, 21], "cases": frozen["cases"]}
    with GPULease(args.gpu):
        select_gpu(args.gpu)
        engine = Engine(config)
        try:
            report["weight_identity"] = {}
            for component in ("target", "draft"):
                current = capture_provenance(component, config, args.config, model=engine)
                expected = plan_model_sources(frozen["plans"][component], frozen["batch"])
                lhs = {row["role"]: row["sha256"] for row in current["model_sources"]}
                rhs = {row["role"]: row["sha256"] for row in expected}
                if lhs != rhs:
                    raise ValueError(f"Reference checkpoint differs from deployed {component} engine")
                report["weight_identity"][component] = {"verified": True, "sha256_by_role": lhs}
            with torch.cuda.stream(engine.model.stream), torch.inference_mode():
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                inputs = {key: value.cuda() for key, value in frozen["inputs"].items()}
                actual = {key: value.cuda() for key, value in frozen["actual"].items()}
                high = replay_reference(engine, inputs, actual, False)
                mapping = _cpu(conditional_rnn_reference(engine.rt.engine.draft, actual["draft_hidden"], actual["draft_base"],
                                                         inputs["anchors"], actual["proposal_tokens"]))
                reference_roles = install_reference_recipe(engine, artifact)
                same = replay_reference(engine, inputs, actual, True)
            outputs = frozen["actual"]
            report["reference_roles"] = [role for role in reference_roles if role["path"].startswith(("target.", "draft."))]
            report["outputs"] = {name: {"trt_vs_same_recipe": metrics(same[name], value),
                                         "trt_vs_original_fp32_on_frozen_cache": metrics(high[name], value),
                                         "same_recipe_vs_original_fp32": metrics(high[name], same[name])}
                                 for name, value in outputs.items() if name != "proposal_tokens"}
            report["rnn_adapter_on_identical_trt_hidden"] = {name: metrics(reference, outputs[name]) for name, reference in mapping.items()}
            audited_values = [value for comparisons in report["outputs"].values() for value in comparisons.values()]
            audited_values.extend(report["rnn_adapter_on_identical_trt_hidden"].values())
            if any(not value["finite"] for value in audited_values):
                raise RuntimeError("AR audit contains a non-finite selected output or reference")
            report["conditional_q"] = {"conditioning": "actual anchors + actual preceding sampled proposals", "rnn_initial_state": "zero",
                                         "temperature": 0.8, "tokens": outputs["proposal_tokens"].tolist(),
                                         "actual_probability_sums": outputs["proposal_q"].sum(-1).tolist()}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
            print(json.dumps({"output": str(args.output), "scheme": frozen["scheme"], "batch": frozen["batch"],
                              "outputs": list(report["outputs"]), "numerical_gate": False}), flush=True)
        finally:
            engine.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("capture", "audit"):
        command = commands.add_parser(name)
        command.add_argument("--gpu", type=int, default=7)
        command.add_argument("--config", default="artifacts/current_release/runtime.yaml")
        command.add_argument("--output", type=Path, required=True)
        if name == "capture":
            command.add_argument("--manifest", type=Path, required=True)
            command.add_argument("--deployment", type=Path, required=True)
            command.add_argument("--start", type=int, default=0)
        else:
            command.add_argument("--capture", type=Path, required=True)
            command.add_argument("--calibration", type=Path)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    capture(args) if args.command == "capture" else audit(args)


if __name__ == "__main__":
    main()
