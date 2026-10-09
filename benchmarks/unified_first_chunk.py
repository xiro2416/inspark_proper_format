"""Shared case and measurement contract; importing this module never imports CUDA."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import sys
import time

VOICES = ("mingxiang_gao.wav", "paimeng_5s_calm_vocal.wav", "positive.wav", "xiaoyuan_gao.wav")
STRATA = ("short", "medium", "long")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def read_texts(path):
    rows, seen = [], set()
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        value = line.strip()
        match = re.search(r'文本：["“](.*?)["”]\s*\|\s*情绪：', value)
        if match:
            value = match.group(1)
        if value and value not in seen:
            seen.add(value)
            rows.append({"source_line": line_number, "text": value})
    if not rows:
        raise ValueError("Dataset has no nonempty unique text")
    return rows


def official_token_counter(tokenizer_path):
    """CPU-only tokenizer, matching the model normalizer and glossary."""
    os.environ.setdefault("ACC_CLEAR_CACHE", str(Path(__file__).resolve().parents[1] / ".cache"))
    from inspark_infer.models.indextts2.upstream.utils.front import TextNormalizer, TextTokenizer
    normalizer = TextNormalizer(enable_glossary=True)
    tokenizer = TextTokenizer(str(tokenizer_path), normalizer)
    glossary = Path(tokenizer_path).with_name("glossary.yaml")
    if glossary.exists():
        normalizer.load_glossary_from_yaml(str(glossary))
    return lambda text: len(tokenizer.tokenize(text))


def build_manifest(text_rows, references, token_count, *, requests=128, seed=924,
                   provenance=None):
    """Build disjoint calibration/evaluation sets, each with ``requests`` cases.

    Strata use full-text tokenizer lengths. The first streaming segment is recorded
    separately: long full text must never be described as a long first-chunk prefix.
    """
    from inspark_infer.runtime.splitter import split
    if requests < 4 or requests % 4:
        raise ValueError("Requests per split must be positive and divisible by four")
    if len(references) != 4 or len({row['voice_id'] for row in references}) != 4:
        raise ValueError("Exactly four distinct voice references are required")
    prepared, seen = [], set()
    for original in text_rows:
        value = original["text"]
        if value in seen:
            continue
        seen.add(value)
        parts = split(value)
        if not parts:
            continue
        prepared.append(dict(original, text_tokens=int(token_count(value)),
                             first_segment=parts[0], first_segment_tokens=int(token_count(parts[0])),
                             segments=len(parts)))
    prepared.sort(key=lambda row: (row["text_tokens"], row["source_line"]))
    rng = random.Random(seed)
    pools = {name: prepared[len(prepared) * i // 3:len(prepared) * (i + 1) // 3]
             for i, name in enumerate(STRATA)}
    quotas = {name: requests // 3 + (i < requests % 3) for i, name in enumerate(STRATA)}
    for name, rows in pools.items():
        if len(rows) < quotas[name] * 2:
            raise ValueError(f"Insufficient unique {name} texts for disjoint splits")
        rng.shuffle(rows)
    splits = {}
    for split_index, split_name in enumerate(("calibration", "evaluation")):
        chosen = {name: pools[name][split_index * quotas[name]:(split_index + 1) * quotas[name]]
                  for name in STRATA}
        # Interleave strata so a short B1/B8 smoke test exercises all length groups.
        ordered = [(name, chosen[name][i]) for i in range(max(quotas.values()))
                   for name in STRATA if i < len(chosen[name])]
        cases = []
        for index, (name, row) in enumerate(ordered):
            case_seed = seed + split_index * requests + index
            emotion_rng = random.Random(case_seed ^ 0x454D4F)
            weights = [emotion_rng.random() for _ in range(8)]
            scale = emotion_rng.random() / sum(weights)
            cases.append(dict(row, case_id=f"{split_name}-{index:04d}", stratum=name,
                              voice_id=references[index % 4]["voice_id"], seed=case_seed,
                              emotion=[value * scale for value in weights],
                              text_sha256=hashlib.sha256(row["text"].encode()).hexdigest()))
        splits[split_name] = cases
    result = dict(schema=1, kind="unified_first_chunk_cases", seed=seed,
                  requests_per_split=requests, reference_seconds=3, references=references,
                  length_basis="full-text official tokenizer; first_segment_tokens recorded separately",
                  cfm_steps=4, provenance=provenance or {},
                  stratum_token_ranges={name: [min(r["text_tokens"] for r in rows),
                                               max(r["text_tokens"] for r in rows)]
                                        for name, rows in pools.items()}, splits=splits)
    result["manifest_sha256"] = canonical_hash(result)
    return result


def load_manifest(path, *, verify_references=True):
    manifest = json.loads(Path(path).read_text())
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if manifest.get("manifest_sha256") != canonical_hash(payload):
        raise ValueError("Manifest content hash mismatch")
    if manifest.get("cfm_steps") != 4 or manifest.get("reference_seconds") != 3:
        raise ValueError("Expected four CFM steps and three VAD seconds")
    if verify_references:
        for reference in manifest["references"]:
            if sha256_file(reference["path"]) != reference["sha256"]:
                raise ValueError(f"Reference hash mismatch: {reference['voice_id']}")
    return manifest


def distribution(values):
    ordered = sorted(float(value) for value in values)
    if any(not math.isfinite(value) for value in ordered):
        raise ValueError("Nonfinite measurement")
    if not ordered:
        return dict(n=0, p50=None, p95=None, p99=None, mean=None, min=None, max=None)
    def percentile(q):
        pos = (len(ordered) - 1) * q
        lo = int(pos)
        return ordered[lo] + (ordered[min(lo + 1, len(ordered) - 1)] - ordered[lo]) * (pos - lo)
    return dict(n=len(ordered), p50=percentile(.5), p95=percentile(.95), p99=percentile(.99),
                mean=statistics.fmean(ordered), min=ordered[0], max=ordered[-1])


def wave_cases(cases, batch, wave):
    if batch not in (1, 4, 8, 16, 32, 64, 128) or not cases:
        raise ValueError("Expected B1/B4/B8/B16/B32/B64/B128 and nonempty cases")
    return [cases[(wave * batch + i) % len(cases)] for i in range(batch)]


def first_segment_diversity(cases,batch):
    """Check every cyclic batch window using the serving splitter, not full text."""
    from inspark_infer.runtime.splitter import split
    if not cases:raise ValueError('No benchmark cases')
    windows=len(cases)//math.gcd(len(cases),batch)
    counts=[]
    for wave in range(windows):
        parts=[split(row['text']) for row in wave_cases(cases,batch,wave)]
        if any(not p for p in parts):raise ValueError('Empty first segment')
        counts.append(len({p[0] for p in parts}))
    return dict(batch=batch,cyclic_windows=windows,unique_first_segments_per_wave=counts,
                all_first_segments_distinct=all(n==batch for n in counts),
                minimum_unique_first_segments=min(counts))


def run_wave(client, cases, wave, *, details=True, clock=time.perf_counter,
             admission_mode='auto', admission_rpc_timing=False):
    """Measure a Pool or in-process Engine without adding CUDA synchronizations.

    The two clocks share the same PCM event. Metadata inspection and cleanup are
    outside both latency windows. Warmups/power can skip expensive result IPC.
    """
    if admission_mode not in ('auto','batch','serial'):raise ValueError('Unknown admission mode')
    bulk=callable(getattr(client,'admit_batch',None))
    if admission_mode=='batch' and not bulk:raise ValueError('Client has no batch admission API')
    mode='batch' if bulk and admission_mode!='serial' else 'serial'
    assigned, first, admitted, rpc_samples = [], {}, {}, []
    acknowledgment=None
    def invoke(name,*args,**kwargs):
        if not admission_rpc_timing:return getattr(client,name)(*args,**kwargs)
        began=clock()
        result=getattr(client,name)(*args,**kwargs)
        rpc_samples.append(dict(operation=name,wall_ms=(clock()-began)*1000))
        return result
    group_start = clock()
    try:
        if mode=='batch':
            payload=[dict(request_id=f'unified-{wave}-{i}',voice_id=case['voice_id'],
                          text=case['text'],seed=case['seed'],emotion=case['emotion'],finish=True)
                     for i,case in enumerate(cases)]
            arrived=clock()
            for record,case in zip(payload,cases):
                record['arrival']=arrived;admitted[record['request_id']]=arrived
            acknowledgment=invoke('admit_batch',payload)
            assigned=[(record['request_id'],case) for record,case in zip(payload,cases)]
        else:
            for index, case in enumerate(cases):
                identifier = f"unified-{wave}-{index}"
                admitted[identifier] = clock()
                invoke('create_session',identifier, case["voice_id"], seed=case["seed"],
                       emotion=case["emotion"], arrival=admitted[identifier])
                assigned.append((identifier, case))
                invoke('push_text',identifier, case["text"])
                invoke('finish_input',identifier)
        group_ready = clock()
        def received(event):
            identifier = event["request_id"]
            if identifier not in admitted or identifier in first:
                return
            timestamp = event.get("received")
            timestamp = clock() if timestamp is None else timestamp
            chunk = event["chunk"]
            if chunk["index"] != 0 or len(chunk["pcm"]) == 0:
                raise RuntimeError("Expected a nonempty first PCM chunk")
            first[identifier] = dict(received=timestamp, host_ready=chunk.get("ready"),
                                     pcm=chunk['pcm'] if details else None,
                                     samples=chunk["sample_end"] - chunk["sample_start"],
                                     cfm_batch=chunk.get("cfm_batch"), vocoder_batch=chunk.get("vocoder_batch"))
        turns = 0
        while len(first) < len(cases):
            events = client.run_ready(on_chunk=received)
            for event in events or ():
                received(event)
            turns += 1
            if turns >= 2000:
                raise RuntimeError("No complete first-chunk wave after 2000 scheduler calls")
        completed = max(row["received"] for row in first.values())
        rows = []
        for identifier, case in assigned:
            event = first[identifier]
            record = dict(case_id=case["case_id"], voice_id=case["voice_id"], stratum=case["stratum"],
                          seed=case["seed"], text_tokens=case["text_tokens"],
                          first_segment_tokens=case["first_segment_tokens"],
                          admission_to_pcm_ms=(event["received"] - admitted[identifier]) * 1000,
                          postadmission_to_pcm_ms=(event["received"] - group_ready) * 1000,
                          predispatch_ms=(group_ready - admitted[identifier]) * 1000,
                          delivery_ms=((event["received"] - event["host_ready"]) * 1000
                                       if event["host_ready"] is not None else None),
                          audio_seconds=event["samples"] / 22050,
                          cfm_batch=event["cfm_batch"], vocoder_batch=event["vocoder_batch"])
            if details:
                state = client.result(identifier) if hasattr(client, "result") else client.sessions[identifier]
                if state.get("error"):
                    raise RuntimeError(state["error"])
                counts = list(state.get("accepted") or [])
                record.update(rounds=state.get("rounds"), accepted=counts, accepted_tokens=sum(counts),
                              code_sha256=canonical_hash(state.get('codes') or []),
                              pcm_sha256=hashlib.sha256(event['pcm']).hexdigest(),
                              proposed_tokens=7 * len(counts), logical_target_verified_positions=8 * len(counts),
                              kv_at_head=state.get("kv_head_lengths"), generated_codes=len(state.get("codes") or []),
                              eos=state.get("eos"))
            rows.append(record)
        return dict(wave=wave, batch=len(cases), scheduler_turns=turns, rows=rows,
                    admission_mode=mode,admission_ack=acknowledgment,
                    admission_rpc_samples=rpc_samples if admission_rpc_timing else None,
                    admission_phase_ms=(group_ready - group_start) * 1000,
                    group_admission_to_last_pcm_ms=(completed - group_start) * 1000,
                    group_postadmission_to_last_pcm_ms=(completed - group_ready) * 1000,
                    measurement_start=group_start, measurement_end=completed)
    finally:
        primary = sys.exc_info()[0] is not None
        cleanup_errors = []
        for identifier, _ in assigned:
            try:
                client.cancel(identifier)
            except Exception as error:
                cleanup_errors.append(error)
        if cleanup_errors and not primary:
            raise RuntimeError("Wave cancellation failed") from cleanup_errors[0]


def summarize(waves):
    rows = [row for wave in waves for row in wave["rows"]]
    accepted = sum(row.get("accepted_tokens", 0) for row in rows)
    proposed = sum(row.get("proposed_tokens", 0) for row in rows)
    return dict(waves=len(waves), requests=len(rows),
                statistics_unit="wave for batch latency; requests within a wave are correlated",
                request_admission_to_pcm_ms=distribution(row["admission_to_pcm_ms"] for row in rows),
                request_postadmission_to_pcm_ms=distribution(row["postadmission_to_pcm_ms"] for row in rows),
                wave_admission_to_last_pcm_ms=distribution(w["group_admission_to_last_pcm_ms"] for w in waves),
                wave_postadmission_to_last_pcm_ms=distribution(w["group_postadmission_to_last_pcm_ms"] for w in waves),
                admission_phase_ms=distribution(w["admission_phase_ms"] for w in waves),
                delivery_ms=distribution(r["delivery_ms"] for r in rows if r["delivery_ms"] is not None),
                rounds=distribution(r["rounds"] for r in rows if r.get("rounds") is not None),
                kv_at_head=distribution(r["kv_at_head"] for r in rows if r.get("kv_at_head") is not None),
                postadmission_requests_per_second=distribution(
                    w["batch"] * 1000 / w["group_postadmission_to_last_pcm_ms"] for w in waves),
                admission_requests_per_second=distribution(
                    w["batch"] * 1000 / w["group_admission_to_last_pcm_ms"] for w in waves),
                accepted_tokens=accepted, proposed_tokens=proposed,
                acceptance_rate=accepted / proposed if proposed else None,
                logical_target_verified_positions=sum(r.get("logical_target_verified_positions", 0) for r in rows),
                work_count_note="Logical active-request rounds exclude extra physical work on ready/padded rows; inspect device step counters and profile trajectory",
                voices=dict(Counter(r["voice_id"] for r in rows)),
                strata=dict(Counter(r["stratum"] for r in rows)))


def counter_delta(before, after):
    """Keep raw snapshots too: graph replays do not increment wrapper enqueues."""
    keys = ("device_round_attempts", "device_round_successes", "device_round_fallbacks",
            "device_round_status_reads", "device_round_status_wait_ms", "device_round_launched_rounds",
            "device_target_steps", "device_draft_steps",
            "native_target_steps", "native_draft_steps", "native_cfm_fallbacks", "native_vocoder_fallbacks",
            "device_code_direct_rows", "speech_codes_host_rows", "output_d2h_bytes",
            "output_d2h_wait_ms", "output_d2h_transfer_ms")
    result = {key: after.get(key, 0) - before.get(key, 0) for key in keys}
    result["device_round_fallback_reasons"] = {
        key: value - before.get("device_round_fallback_reasons", {}).get(key, 0)
        for key, value in after.get("device_round_fallback_reasons", {}).items()}
    result["head_graph_hits"] = {
        key: (after.get("head_graphs") or {}).get("hits", {}).get(key, 0)
        - (before.get("head_graphs") or {}).get("hits", {}).get(key, 0) for key in ("cfm", "vocoder")}
    old_unified, new_unified = before.get("unified_dspark") or {}, after.get("unified_dspark") or {}
    if new_unified:
        result["unified_dspark"] = {key:new_unified.get(key,0)-old_unified.get(key,0)
                                    for key in ("calls","failures","graph_replays","launched_rounds",
                                                "status_reads","status_wait_ms","draft_enqueues","target_enqueues","prime_enqueues")}
    old_prefix, new_prefix = before.get('unified_prefix') or {}, after.get('unified_prefix') or {}
    if new_prefix:
        result['unified_prefix'] = {field: {kind: new_prefix.get(field,{}).get(kind,0)-old_prefix.get(field,{}).get(kind,0)
                                         for kind in ('prefill','latent')}
                                    for field in ('hits','tail_eager','graph_hits','direct_hits')}
    return result


def compare_reports(baseline, candidate, *, allow_admission_change=False):
    """Paired wave comparison; rejects mismatched inputs or timing contracts."""
    for report in (baseline, candidate):
        if report.get("kind") != "unified_first_chunk_benchmark" or not report.get("execution_pass"):
            raise ValueError("Both inputs must be successful unified benchmarks")
    for key in ("batch", "manifest_sha256", "split", "allocator", "config_sha256", "gpu"):
        if baseline.get(key) != candidate.get(key):
            raise ValueError(f"Unmatched comparison field: {key}")
    old_mode=baseline.get('admission_mode','serial');new_mode=candidate.get('admission_mode','serial')
    if old_mode!=new_mode and not allow_admission_change:
        raise ValueError('Admission API changed; explicitly allow this scheduling comparison')
    if baseline.get('timing')!=candidate.get('timing'):
        if not allow_admission_change:raise ValueError('Unmatched comparison field: timing')
        admitted={"each create_session call entry to client PCM event.received",
                  "each request timestamp before grouped admit_batch RPC to client PCM event.received"}
        ready={"last finish_input returned to client PCM event.received",
               "admit_batch acknowledgment to client PCM event.received"}
        for report in (baseline,candidate):
            timing=report.get('timing',{})
            if timing.get('admission') not in admitted or timing.get('historical') not in ready:
                raise ValueError('Admission change cannot waive a different PCM timing boundary')
    old, new = baseline["waves"], candidate["waves"]
    if len(old) != len(new) or not old:
        raise ValueError("Paired comparison needs equal nonempty wave counts")
    for left, right in zip(old, new):
        identity = lambda row: [(r["case_id"], r["voice_id"], r["seed"]) for r in row["rows"]]
        if identity(left) != identity(right):
            raise ValueError("Paired waves use different cases, voices or seeds")
    metrics = {}
    rng = random.Random(924)
    for field in ("group_postadmission_to_last_pcm_ms", "group_admission_to_last_pcm_ms"):
        differences = [a[field] - b[field] for a, b in zip(old, new)]
        boot = sorted(statistics.fmean(rng.choices(differences, k=len(differences))) for _ in range(1000))
        metrics[field] = dict(baseline=distribution(w[field] for w in old),
                             candidate=distribution(w[field] for w in new),
                             paired_gain_ms=distribution(differences),
                             paired_mean_gain_95pct_bootstrap_ci_ms=[boot[24], boot[974]])
    return dict(schema=1, kind="unified_first_chunk_paired_comparison", batch=baseline["batch"],
                manifest_sha256=baseline["manifest_sha256"], baseline_label=baseline.get("label"),
                candidate_label=candidate.get("label"), waves=len(old), metrics=metrics,
                admission_api_changed=old_mode!=new_mode,
                admission_comparison=dict(baseline=old_mode,candidate=new_mode,explicitly_allowed=allow_admission_change),
                note="Positive gain means candidate faster; resampling unit is whole wave. This report does not certify numerical correctness or automatically select deployment.")
