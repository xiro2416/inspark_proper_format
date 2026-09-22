#!/usr/bin/env python3
"""Build a paired SM89 candidate vs official IndexTTS2 eager listening set."""
from __future__ import annotations

import argparse
import html
import json
import random
import re
import shutil
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "sm89_vs_indextts2_eager_20260921"
EMOTION_NAMES = ("高兴", "愤怒", "悲伤", "恐惧", "反感", "低落", "惊讶", "自然")
LINE = re.compile(r'^文本：[“"](?P<text>.*?)[”"]\s*\|')


def read_cases(output: Path) -> list[dict]:
    data = json.loads((output / "cases.json").read_text(encoding="utf-8"))
    cases = data.get("cases", ())
    if len(cases) != 18:
        raise ValueError("Expected exactly 18 paired cases")
    return cases


def command_build(args) -> None:
    source = Path("/workspace/index-tts/data/emotion_data_curated_emotext_clean.txt")
    references = sorted(Path("/workspace/index-tts/data/audio/old").glob("*.wav"), key=lambda p: p.name)
    texts = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        match = LINE.match(line.strip())
        if match:
            texts.append((line_number, match.group("text")))
    if len(references) != 9:
        raise ValueError(f"Expected 9 reference voices, found {len(references)}")
    rng = random.Random(args.seed)
    chosen_texts = rng.sample(texts, len(references) * 2)
    cases = []
    for voice_index, reference in enumerate(references):
        for take in range(2):
            line_number, text = chosen_texts[voice_index * 2 + take]
            raw = [rng.random() for _ in range(8)]
            intensity = rng.uniform(0.45, 0.8)
            emotion = [round(intensity * value / sum(raw), 6) for value in raw]
            dominant = max(range(8), key=emotion.__getitem__)
            cases.append({
                "id": f"voice{voice_index + 1:02d}_take{take + 1}",
                "voice": reference.stem,
                "reference_audio": str(reference.resolve()),
                "text": text,
                "source_line": line_number,
                "seed": rng.randrange(2**31),
                "emotion": emotion,
                "dominant_emotion": EMOTION_NAMES[dominant],
                "emotion_intensity": round(sum(emotion), 6),
            })
    args.output.mkdir(parents=True, exist_ok=True)
    reference_dir = args.output / "references"
    reference_dir.mkdir(exist_ok=True)
    for reference in references:
        shutil.copy2(reference, reference_dir / reference.name)
    payload = {
        "schema": 1,
        "seed": args.seed,
        "description": "Paired SM89 candidate and official IndexTTS2 eager listening cases",
        "emotion_order": list(EMOTION_NAMES),
        "cases": cases,
    }
    (args.output / "cases.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "voices": len(references),
                      "cases": len(cases), "seed": args.seed}, ensure_ascii=False, indent=2))


def command_optimized(args) -> None:
    import numpy as np
    import soundfile as sf
    from acc_infer_clear.config import load
    from acc_infer_clear.runtime.deployment import load as load_deployment
    from acc_infer_clear.runtime.device import GPULease
    from acc_infer_clear.runtime.pool import Pool

    cases = read_cases(args.output)
    destination = args.output / "optimized"
    destination.mkdir(parents=True, exist_ok=True)
    config = load(args.config)
    config["max_batch"] = 1
    references = sorted({case["reference_audio"] for case in cases})
    voices = {path: f"voice-{index}" for index, path in enumerate(references)}
    rows = []
    with GPULease(args.gpu), Pool(config, args.gpu, 1) as pool:
        for path, voice_id in voices.items():
            pool.prepare_reference(voice_id, path)
        deployment = pool.prepare_deployment(load_deployment(args.deployment))[0]
        for index, case in enumerate(cases, 1):
            started = time.perf_counter()
            pool.create_session(case["id"], voices[case["reference_audio"]],
                                case["seed"], case["emotion"])
            pool.push_text(case["id"], case["text"])
            pool.finish_input(case["id"])
            while pool.states[0]["ready_heads"] or pool.states[0]["ready_tails"]:
                pool.run_ready()
            result = pool.result(case["id"])
            if not result["complete"] or result["error"]:
                raise RuntimeError(f"Optimized generation failed: {case['id']}: {result['error']}")
            pcm = np.concatenate([chunk["pcm"] for chunk in result["chunks"]])
            path = destination / f"{case['id']}.wav"
            sf.write(path, pcm, 22050, subtype="PCM_16")
            elapsed = time.perf_counter() - started
            rows.append({"id": case["id"], "audio": path.name, "seconds": len(pcm) / 22050,
                         "elapsed_s": elapsed, "chunks": len(result["chunks"])})
            pool.release(case["id"])
            print(f"optimized {index:02d}/{len(cases)} {case['id']} {elapsed:.3f}s", flush=True)
    (destination / "results.json").write_text(
        json.dumps({"engine": "inspark_marlin_sm89", "deployment": deployment,
                    "cases": rows}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def command_eager(args) -> None:
    import numpy as np
    import soundfile as sf
    import torch
    from indextts.infer_v2 import IndexTTS2

    if torch.cuda.device_count() != 1:
        raise RuntimeError("Official eager run requires exactly one visible GPU")
    cases = read_cases(args.output)
    destination = args.output / "eager"
    destination.mkdir(parents=True, exist_ok=True)
    model_root = Path("/workspace/index-tts/checkpoints")
    tts = IndexTTS2(cfg_path=str(model_root / "config.yaml"), model_dir=str(model_root),
                    device="cuda:0", use_fp16=False, use_cuda_kernel=False,
                    use_deepspeed=False, use_accel=False, use_torch_compile=False)
    rows = []
    for index, case in enumerate(cases, 1):
        random.seed(case["seed"])
        np.random.seed(case["seed"] % (2**32))
        torch.manual_seed(case["seed"])
        torch.cuda.manual_seed_all(case["seed"])
        path = destination / f"{case['id']}.wav"
        started = time.perf_counter()
        result = tts.infer(spk_audio_prompt=case["reference_audio"], text=case["text"],
                           output_path=str(path), emo_vector=case["emotion"], emo_alpha=1.0,
                           verbose=False, num_beams=3)
        elapsed = time.perf_counter() - started
        if result is None or not path.is_file():
            raise RuntimeError(f"Official eager generation failed: {case['id']}")
        audio, sample_rate = sf.read(path)
        rows.append({"id": case["id"], "audio": path.name, "sample_rate": sample_rate,
                     "seconds": len(audio) / sample_rate, "elapsed_s": elapsed})
        print(f"eager {index:02d}/{len(cases)} {case['id']} {elapsed:.3f}s", flush=True)
    (destination / "results.json").write_text(
        json.dumps({"engine": "official_indextts2_eager", "settings": {
            "fp16": False, "cuda_kernel": False, "torch_compile": False,
            "deepspeed": False, "accel": False, "num_beams": 3,
        }, "cases": rows}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def command_html(args) -> None:
    cases = read_cases(args.output)
    optimized = {row["id"]: row for row in json.loads(
        (args.output / "optimized" / "results.json").read_text(encoding="utf-8"))["cases"]}
    eager = {row["id"]: row for row in json.loads(
        (args.output / "eager" / "results.json").read_text(encoding="utf-8"))["cases"]}
    cards = []
    for case in cases:
        vector = ", ".join(f"{name} {value:.3f}" for name, value in zip(EMOTION_NAMES, case["emotion"]))
        reference_name = Path(case["reference_audio"]).name
        cards.append(f"""
        <article class="card">
          <h2>{html.escape(case['voice'])} · 第 {case['id'][-1]} 条</h2>
          <p class="text">{html.escape(case['text'])}</p>
          <p class="meta">主情绪：{case['dominant_emotion']}　总强度：{case['emotion_intensity']:.3f}<br>{html.escape(vector)}</p>
          <div class="players">
            <section><h3>参考音色</h3><audio controls preload="none" src="references/{html.escape(reference_name)}"></audio></section>
            <section><h3>SM89 优化版</h3><audio controls preload="none" src="optimized/{case['id']}.wav"></audio><small>{optimized[case['id']]['seconds']:.2f}s / 生成 {optimized[case['id']]['elapsed_s']:.2f}s</small></section>
            <section><h3>IndexTTS2 eager</h3><audio controls preload="none" src="eager/{case['id']}.wav"></audio><small>{eager[case['id']]['seconds']:.2f}s / 生成 {eager[case['id']]['elapsed_s']:.2f}s</small></section>
          </div>
        </article>""")
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SM89 优化版 vs IndexTTS2 eager</title>
<style>
body{{font-family:system-ui,sans-serif;margin:0;background:#f3f5f8;color:#18202a}}main{{max-width:1180px;margin:auto;padding:28px}}.intro,.card{{background:white;border:1px solid #dfe4ea;border-radius:14px;padding:20px;margin-bottom:18px;box-shadow:0 4px 16px #18202a0c}}h1{{margin-top:0}}h2{{font-size:1.1rem;margin:0 0 10px}}h3{{font-size:.92rem;margin:0 0 8px}}.text{{font-size:1.08rem}}.meta,small{{color:#5d6875;font-size:.85rem}}.players{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}}section{{background:#f7f9fb;border-radius:10px;padding:12px}}audio{{width:100%}}small{{display:block;margin-top:6px}}@media(max-width:760px){{.players{{grid-template-columns:1fr}}}}
</style></head><body><main><div class="intro"><h1>SM89 优化版 vs IndexTTS2 eager</h1><p>9 个音色 × 2 条随机文本/情绪，共 18 组严格配对输入。两列使用相同文本、情绪向量和 seed；eager 明确关闭 FP16、CUDA 自定义激活、torch.compile、DeepSpeed 与加速引擎。</p><p>随机种子：{args.seed}。情绪顺序：{' / '.join(EMOTION_NAMES)}。</p></div>{''.join(cards)}</main></body></html>"""
    (args.output / "index.html").write_text(document, encoding="utf-8")
    archive = shutil.make_archive(str(args.output), "zip", root_dir=args.output.parent,
                                  base_dir=args.output.name)
    print(json.dumps({"html": str((args.output / 'index.html').resolve()),
                      "zip": str(Path(archive).resolve()), "cases": len(cases)},
                     indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260921)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build")
    optimized = sub.add_parser("optimized")
    optimized.add_argument("--gpu", type=int, required=True)
    optimized.add_argument("--config", default="configs/runtime.yaml")
    optimized.add_argument("--deployment", default="configs/sm89_bf16_triton_device_control_alias_candidate.json")
    sub.add_parser("eager")
    sub.add_parser("html")
    args = parser.parse_args()
    {"build": command_build, "optimized": command_optimized,
     "eager": command_eager, "html": command_html}[args.command](args)


if __name__ == "__main__":
    main()
