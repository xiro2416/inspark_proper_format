#!/usr/bin/env python3
"""Build the deterministic, balanced 256-case SM89 quality gate."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path


LINE = re.compile(
    r'^文本：[“"](?P<text>.*?)[”"]\s*\|\s*情绪：[“"](?P<label>.*?)[”"]\s*\|\s*'
    r'emo_vector：[\[](?P<vector>.*?)[\]］]\s*$'
)
EMOTIONS = ("happy", "angry", "sad", "fear", "disgust", "melancholy", "surprise", "calm")


def parse(path: Path) -> list[dict]:
    rows = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        match = LINE.match(raw.strip())
        if not match:
            raise ValueError(f"Cannot parse {path}:{line_number}")
        vector = [float(value.strip()) for value in match.group("vector").split(",")]
        if len(vector) != 8 or any(value < 0 for value in vector):
            raise ValueError(f"Invalid emotion vector at {path}:{line_number}")
        dominant = max(range(8), key=vector.__getitem__)
        rows.append({
            "source_line": line_number,
            "text": match.group("text"),
            "emotion_label": match.group("label"),
            "emotion": vector,
            "dominant_emotion": EMOTIONS[dominant],
            "dominant_index": dominant,
        })
    return rows


def balanced_targets(rows: list[dict], total: int = 256) -> list[int]:
    """Max-min allocation without duplicating undersupplied emotion classes."""
    capacities = [sum(row["dominant_index"] == index for row in rows) for index in range(8)]
    targets = [0] * 8
    for _ in range(total):
        eligible = [index for index in range(8) if targets[index] < capacities[index]]
        if not eligible:
            raise ValueError(f"Source has fewer than {total} usable rows")
        index = min(eligible, key=lambda value: (targets[value], value))
        targets[index] += 1
    return targets


def stratified(rows: list[dict], rng: random.Random) -> tuple[list[dict], list[int]]:
    selected = []
    targets = balanced_targets(rows)
    for emotion_index, emotion_name in enumerate(EMOTIONS):
        group = sorted((row for row in rows if row["dominant_index"] == emotion_index),
                       key=lambda row: (len(row["text"]), row["source_line"]))
        target = targets[emotion_index]
        # Spread each class allocation across its four length quartiles. Sampling
        # is fixed but not tied to file order.
        buckets = []
        for quartile in range(4):
            begin = len(group) * quartile // 4
            end = len(group) * (quartile + 1) // 4
            buckets.append(group[begin:end])
        takes = [target * len(bucket) // len(group) for bucket in buckets]
        remaining = target - sum(takes)
        order = sorted(range(4), key=lambda q: (-(target * len(buckets[q]) % len(group)), q))
        for quartile in order:
            if remaining and takes[quartile] < len(buckets[quartile]):
                takes[quartile] += 1
                remaining -= 1
        if remaining:
            raise AssertionError(f"Could not allocate {emotion_name} across length quartiles")
        for bucket, take in zip(buckets, takes):
            if take:
                selected.extend(rng.sample(bucket, take))
    rng.shuffle(selected)
    return selected, targets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path,
                        default=Path("/workspace/index-tts/data/emotion_data_curated_emotext_clean.txt"))
    parser.add_argument("--reference-dir", type=Path,
                        default=Path("/workspace/index-tts/data/audio/old"))
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--output", type=Path, default=Path("configs/sm89_quality_256.json"))
    args = parser.parse_args()

    references = sorted(args.reference_dir.glob("*.wav"), key=lambda path: path.name)
    if not references:
        raise FileNotFoundError(f"No WAV references in {args.reference_dir}")
    rng = random.Random(args.seed)
    chosen, targets = stratified(parse(args.source), rng)
    reference_order = [references[index % len(references)] for index in range(len(chosen))]
    rng.shuffle(reference_order)
    cases = []
    for index, (row, reference) in enumerate(zip(chosen, reference_order)):
        case = dict(row)
        case.update(id=f"sm89-q{index:03d}", seed=rng.randrange(2**31),
                    reference_audio=str(reference.resolve()))
        cases.append(case)

    counts = {name: sum(case["dominant_emotion"] == name for case in cases) for name in EMOTIONS}
    reference_counts = {path.name: sum(case["reference_audio"] == str(path.resolve()) for case in cases)
                        for path in references}
    payload = {
        "schema": 1,
        "purpose": "SM89 paired quality gate; identical cases for baseline and candidate",
        "seed": args.seed,
        "source": str(args.source.resolve()),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "selection": "max-min unique allocation by dominant emotion; stratified across four length quartiles",
        "target_emotion_counts": dict(zip(EMOTIONS, targets)),
        "emotion_counts": counts,
        "reference_counts": reference_counts,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "cases": len(cases),
                      "emotion_counts": counts, "reference_counts": reference_counts},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
