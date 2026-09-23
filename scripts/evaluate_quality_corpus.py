#!/usr/bin/env python3
"""Offline CPU Paraformer character-CER + UTMOS over paired full-EOS outputs.

Uses the independently smoke-tested local evaluator models, never downloads.
The defaults require exactly 256 complete pairs. An explicit smaller count is a
diagnostic only and cannot report the release quality gate as passed.

Paired reference-audio identity compares generation-time recorded paths/hashes/
sizes; original references need not remain available for offline evaluation.
The corpus file itself is rehashed, but its embedded original-text source hash
is a declaration, not independently verified by this evaluator.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import string
import sys
import traceback

from inspark_infer.guardrails.quality import paired_quality
from inspark_infer.guardrails.snapshots import file_sha256, write_json
from trt113_provenance import file_record, source_identity


def configure_cpu(cache):
    if not cache.resolve().is_relative_to("/workspace"):
        raise ValueError("Evaluator cache must stay within /workspace")
    os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_DATASETS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                      OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                      NUMBA_NUM_THREADS="4", NUMBA_DISABLE_JIT="0")
    for key, name in {"HF_HOME": "hf", "TMPDIR": "tmp", "XDG_CACHE_HOME": "xdg-cache",
                      "XDG_CONFIG_HOME": "xdg-config", "XDG_DATA_HOME": "xdg-data",
                      "XDG_STATE_HOME": "xdg-state", "TORCH_HOME": "torch",
                      "MODELSCOPE_CACHE": "modelscope", "NUMBA_CACHE_DIR": "numba",
                      "MPLCONFIGDIR": "matplotlib"}.items():
        directory = cache / name; directory.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(directory.resolve())
    def deny_network(event, _args):
        if event in {"socket.connect", "socket.getaddrinfo"}:
            raise RuntimeError("Offline quality evaluator forbids network access")
    sys.addaudithook(deny_network)


def load_generation(directory, cases, corpus_sha256):
    import soundfile as sf
    metadata = json.loads((directory / "generation_summary.json").read_text())
    if metadata.get("status") != "completed" or metadata.get("corpus", {}).get("sha256") != corpus_sha256:
        raise ValueError("Generation is incomplete or belongs to another corpus")
    rows = [json.loads(line) for line in (directory / "generation.jsonl").read_text().splitlines()]
    by_id = {row["id"]: row for row in rows}
    if len(by_id) != len(rows) or set(by_id) != {case["id"] for case in cases}:
        raise ValueError("Generation must cover every requested case exactly once")
    for case in cases:
        row = by_id[case["id"]]
        path = directory / f"{case['id']}.wav"
        if not row.get("complete") or not row.get("eos") or row.get("samples", 0) <= 0:
            raise ValueError(f"Missing full-EOS generation evidence: {case['id']}")
        for field in ("seed", "text", "reference_audio"):
            if field not in row or type(row[field]) is not type(case[field]) or row[field] != case[field]:
                raise ValueError(f"Generation {field} differs from the hash-bound corpus: {case['id']}")
        if "emotion" in row and row["emotion"] != case["emotion"]:
            raise ValueError(f"Generation emotion differs from the hash-bound corpus: {case['id']}")
        if row.get("sha256") != file_sha256(path):
            raise ValueError(f"Generated wave hash differs from generation metadata: {path}")
        info = sf.info(path)
        if (type(row["samples"]) is not int or row["samples"] != info.frames
                or row.get("sample_rate") != info.samplerate or info.samplerate != 22050
                or info.channels != 1 or info.subtype != "PCM_16"):
            raise ValueError(f"Generation sample count/rate/channels/encoding mismatch: {case['id']}")
    metadata["validation"] = {"case_coverage": "exactly once", "audio_hashes_verified": True,
        "row_fields_match_corpus": ["seed", "text", "reference_audio"],
        "wave_metadata_verified": ["samples", "sample_rate=22050", "channels=1", "subtype=PCM_16"],
        "emotion_evidence": "row checked when present; otherwise bound only by the exact corpus hash, not independently observed"}
    return metadata


def _recorded_absolute_path(value):
    """Normalize recorded paths lexically only; never resolve live symlinks."""
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise ValueError("Generation reference paths must be recorded absolute paths")
    return os.path.normpath(value)


def _valid_hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_paired_generation_identity(baseline, candidate, cases):
    """Compare frozen generation identities without reopening historical assets.

    Each reference path must occur exactly once in each manifest and cover the
    exact requested case inventory (nine references for the full fixed corpus).
    Matching path strings alone are insufficient: SHA256 and byte count must
    match. This checks recorded generation-time bindings, not today's files.
    """
    required = {_recorded_absolute_path(case["reference_audio"]) for case in cases}
    if not required:
        raise ValueError("Paired identity validation requires nonempty cases")
    inventories = {}
    for arm, metadata in (("baseline", baseline), ("candidate", candidate)):
        rows = metadata.get("references")
        if not isinstance(rows, list):
            raise ValueError(f"{arm} has no recorded reference-audio inventory")
        by_path = {}
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"Invalid {arm} reference-audio record")
            path = _recorded_absolute_path(row.get("path"))
            if path in by_path:
                raise ValueError(f"Duplicate {arm} reference-audio path: {path}")
            if not _valid_hash(row.get("sha256")) or type(row.get("bytes")) is not int or row["bytes"] <= 0:
                raise ValueError(f"Invalid {arm} reference-audio hash/bytes: {path}")
            by_path[path] = row
        if set(by_path) != required:
            raise ValueError(f"{arm} reference-audio coverage differs from requested cases")
        inventories[arm] = by_path
    reference_checks = []
    for path in sorted(required):
        a, b = inventories["baseline"][path], inventories["candidate"][path]
        if (a["sha256"], a["bytes"]) != (b["sha256"], b["bytes"]):
            raise ValueError(f"Quality arms use different generation-time reference audio: {path}")
        reference_checks.append({"path": path, "sha256": a["sha256"], "bytes": a["bytes"],
                                 "sha256_and_bytes_match": True})
    expected_roles = {"target": {"target_checkpoint", "target_config"},
                      "draft": {"draft_checkpoint", "draft_config"},
                      "cfm": {"s2mel_checkpoint", "student_checkpoint", "s2mel_config"},
                      "vocoder": {"vocoder_checkpoint", "vocoder_config"}}
    for component, required_roles in expected_roles.items():
        source_hashes = {}
        for arm, metadata in (("baseline", baseline), ("candidate", candidate)):
            sources = metadata.get("model_provenance", {}).get(component, {}).get("model_sources")
            if not isinstance(sources, list) or any(not isinstance(row, dict) for row in sources):
                raise ValueError(f"Invalid {arm} {component} model-source inventory")
            by_role = {row.get("role"): row for row in sources}
            if len(by_role) != len(sources) or set(by_role) != required_roles:
                raise ValueError(f"Invalid {arm} {component} model-source roles")
            if any(not _valid_hash(row.get("sha256")) for row in sources):
                raise ValueError(f"Invalid {arm} {component} model-source hash")
            source_hashes[arm] = {role: row["sha256"] for role, row in by_role.items()}
        if source_hashes["baseline"] != source_hashes["candidate"]:
            raise ValueError(f"Quality arms use different actual {component} checkpoint/config hashes")
    return {"same_reference_audio_identity": True, "same_loader_checkpoint_identity": True,
            "reference_audio_count": len(required), "reference_audio_checks": reference_checks,
            "reference_audio_files_rehashed_by_evaluator": False,
            "reference_audio_identity_scope": "generation-time recorded canonical paths, SHA256 and bytes; original reference files are not reopened or rehashed",
            "loader_identity_scope": "generation-time recorded actual loader checkpoint/config hashes; no independent engine constant extraction"}


def corpus_source_declaration(corpus):
    return {"path": corpus.get("source"), "sha256": corpus.get("source_sha256"),
            "independently_verified": False,
            "scope": "declaration embedded in the hash-verified corpus file; original source text is not read or rehashed by this evaluator"}


def evaluate(args):
    configure_cpu(args.cache)
    sys.path.insert(0, str(args.zipvoice.resolve()))
    report = {"schema": 1, "scope": "paired_full_eos_quality", "status": "running",
              "device": "cpu", "rows": {"baseline": [], "candidate": []},
              "pass_gate": False, "required_release_cases": 256}
    try:
        import torch
        import jiwer
        import numpy as np
        from rapidfuzz.distance import Levenshtein
        import soundfile as sf
        import zhconv
        from zhon.hanzi import punctuation
        from funasr import AutoModel
        from zipvoice.eval.mos.utmos import UTMOSScore
        from zipvoice.eval.utils import load_waveform

        torch.set_num_threads(4); torch.set_num_interop_threads(1)
        if torch.cuda.is_initialized():
            raise RuntimeError("CPU evaluator must not initialize CUDA")
        corpus = json.loads(args.corpus.read_text())
        cases = corpus["cases"][:args.expected_cases]
        if len(cases) != args.expected_cases or len({case["id"] for case in cases}) != len(cases):
            raise ValueError("Insufficient or duplicate corpus cases")
        report["corpus"] = file_record(args.corpus, "corpus")
        report["corpus_original_text_source"] = corpus_source_declaration(corpus)
        report["source"] = source_identity()
        report["software"] = {name: importlib.metadata.version(name) for name in (
            "torch", "torchaudio", "funasr", "jiwer", "zhconv", "zhon", "soundfile",
            "numpy", "librosa", "soxr", "rapidfuzz", "transformers")}
        report["evaluator_sources"] = {name: file_record(importlib.import_module(name).__file__, "evaluator_source")
            for name in ("funasr.auto.auto_model", "funasr.models.seaco_paraformer.model",
                         "zipvoice.eval.models.utmos", "zipvoice.eval.mos.utmos", "zipvoice.eval.utils",
                         "jiwer.process", "librosa.core.audio")}
        report["models"] = [file_record(args.models / path, "evaluator_model") for path in (
            "mos/utmos22_strong_step7459_v1.pt", "wer/paraformer-zh/model.pt",
            "wer/paraformer-zh/config.yaml", "wer/paraformer-zh/configuration.json",
            "wer/paraformer-zh/tokens.json", "wer/paraformer-zh/am.mvn", "wer/paraformer-zh/seg_dict")]
        report["generation"] = {name: load_generation(directory, cases, report["corpus"]["sha256"])
                                 for name, directory in (("baseline", args.baseline_dir), ("candidate", args.candidate_dir))}
        report["same_loader_checkpoint_identity"] = report["same_reference_audio_identity"] = False
        report["paired_generation_identity"] = validate_paired_generation_identity(
            report["generation"]["baseline"], report["generation"]["candidate"], cases)
        report["same_loader_checkpoint_identity"] = report["same_reference_audio_identity"] = True
        report["engine_identity_note"] = "Equal loader checkpoint hashes do not independently verify legacy TensorRT engine weight provenance"
        report["preprocessing"] = {"function": "zipvoice.eval.utils.load_waveform", "dtype": "float32",
            "sample_rate": 16000, "mono": "channel_mean", "resampler": "librosa/soxr_hq",
            "clip": None, "loudness_normalization": False,
            "same_preprocessed_samples_for_asr_and_utmos": True}
        report["normalization"] = "zh_char_v1: zh-cn; remove zhon + ASCII punctuation except apostrophe; remove whitespace; no number expansion"
        remove = set(punctuation + string.punctuation) - {"'"}
        normalize = lambda text: "".join(c for c in zhconv.convert(text, "zh-cn") if not c.isspace() and c not in remove)
        asr = AutoModel(model=str(args.models / "wer/paraformer-zh"), device="cpu", ncpu=4,
                        disable_update=True, disable_pbar=True, trust_remote_code=False)
        utmos = UTMOSScore(str(args.models / "mos/utmos22_strong_step7459_v1.pt"))
        if utmos.device.type != "cpu" or next(asr.model.parameters()).device.type != "cpu":
            raise RuntimeError("Evaluator model is not on CPU")
        if utmos.model.training:
            raise RuntimeError("UTMOS must be in evaluation mode")
        for case in cases:
            reference = normalize(case["text"])
            if not reference:
                raise ValueError("Normalized reference is empty")
            for arm, directory in (("baseline", args.baseline_dir), ("candidate", args.candidate_dir)):
                path = directory / f"{case['id']}.wav"
                info = sf.info(path)
                if info.frames <= 0 or info.samplerate != 22050 or info.channels != 1:
                    raise ValueError("Expected nonempty mono 22050 Hz generated audio")
                speech = load_waveform(str(path), 16000, device=torch.device("cpu"))
                if not speech.numel() or not bool(torch.isfinite(speech).all()):
                    raise ValueError("Invalid preprocessed audio")
                with torch.no_grad():
                    decoded = asr.generate(input=speech.numpy(), fs=16000, batch_size_s=300, disable_pbar=True)
                    mos = utmos.model(speech.unsqueeze(0), 16000)
                if len(decoded) != 1 or not isinstance(decoded[0].get("text"), str):
                    raise ValueError("ASR must return exactly one transcript")
                if mos.numel() != 1 or not bool(torch.isfinite(mos).all()):
                    raise ValueError("Non-finite or nonscalar UTMOS score")
                hypothesis = normalize(decoded[0]["text"])
                edits = jiwer.process_characters(reference, hypothesis)
                count = edits.substitutions + edits.deletions + edits.insertions
                if count != Levenshtein.distance(reference, hypothesis):
                    raise ValueError("Independent character distance implementations disagree")
                row = {"id": case["id"], "audio": file_record(path, "generated_audio"),
                       "duration_s": info.frames / info.samplerate,
                       "transcription_raw": decoded[0]["text"], "reference_normalized": reference,
                       "hypothesis_normalized": hypothesis, "reference_characters": len(reference),
                       "substitutions": edits.substitutions, "deletions": edits.deletions,
                       "insertions": edits.insertions, "hits": edits.hits, "cer": count / len(reference),
                       "utmos": float(mos.item()), "preprocessed_samples": speech.numel(),
                       "preprocessed_sha256": __import__("hashlib").sha256(speech.numpy().tobytes()).hexdigest()}
                report["rows"][arm].append(row)
            write_json(args.report, report)
            print(json.dumps({"completed_pairs": len(report["rows"]["candidate"]), "cases": len(cases)}), flush=True)
        report["quality"] = paired_quality(report["rows"]["baseline"], report["rows"]["candidate"],
                                             [case["id"] for case in cases])
        report["release_coverage"] = len(cases) == 256
        report["pass_gate"] = report["release_coverage"] and report["quality"]["pass_gate"]
        report["status"] = "completed"
        report["cuda_initialized"] = torch.cuda.is_initialized()
    except Exception as error:
        report["status"] = "error"
        report["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        raise
    finally:
        write_json(args.report, report)
    if not report["pass_gate"]:
        raise SystemExit(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("configs/hardware/sm89/sm89_quality_256.json"))
    parser.add_argument("--expected-cases", type=int, default=256)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--zipvoice", type=Path, default=Path("/workspace/ZipVoice"))
    parser.add_argument("--models", type=Path, default=Path("/workspace/models/TTS_eval_models"))
    parser.add_argument("--cache", type=Path, default=Path("/workspace/A_inspark_marlin/.work/quality_eval_cache"))
    args = parser.parse_args()
    if not 1 <= args.expected_cases <= 256:
        parser.error("--expected-cases must be in [1,256]")
    evaluate(args)


if __name__ == "__main__":
    main()
