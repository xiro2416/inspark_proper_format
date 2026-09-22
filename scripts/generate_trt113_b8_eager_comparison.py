#!/usr/bin/env python3
"""Generate a paired TRT113 B8 vs official IndexTTS2 eager listening set."""
from __future__ import annotations

import argparse
import html
import json
import random
import shutil
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EMOTION_NAMES = ("高兴", "愤怒", "悲伤", "恐惧", "反感", "低落", "惊讶", "自然")
SELECTED_EMOTIONS = ("愤怒", "悲伤", "高兴", "低落")
TEXTS = (
    "会议将在上午九点开始，请提前准备好需要讨论的材料。",
    "窗边摆着一盆绿色植物，旁边放着几本装订整齐的书。",
    "工作人员按照清单逐项检查设备，并记录了当前运行状态。",
    "这趟列车将在下一站停靠五分钟，随后继续向南行驶。",
    "桌上的文件已经按照日期分类，电子副本也完成了备份。",
    "社区图书馆本周调整了开放时间，入口处张贴了新的通知。",
    "送达的包裹共有三个，外包装完整，编号与订单信息一致。",
    "研究小组收集了本月的数据，计划在周五完成初步整理。",
    "厨房里的水已经烧开，可以依次放入准备好的食材。",
    "道路两侧安装了新的指示牌，夜间经过时也能够清楚看见。",
    "办公室下午进行网络维护，部分内部服务可能短暂停止。",
    "展厅按照时间顺序陈列作品，参观路线从左侧入口开始。",
    "天气预报显示明天多云，白天温度会比今天略高一些。",
    "维修人员更换了旧零件，测试结果显示机器已经正常工作。",
    "课程资料已上传到共享目录，同学们可以自行下载查看。",
    "河岸步道全长约三公里，中途设有座椅和饮水设施。",
)


def read_cases(output: Path) -> list[dict]:
    cases = json.loads((output / "cases.json").read_text(encoding="utf-8"))["cases"]
    if len(cases) != 16:
        raise ValueError(f"Expected 16 cases, found {len(cases)}")
    return cases


def build(args) -> None:
    source = Path("/workspace/index-tts/data/audio/babckup")
    references = sorted((path for path in source.glob("*.wav") if path.is_file()), key=lambda p: p.name)
    if not references:
        raise RuntimeError(f"No WAV references found in {source}")
    rng = random.Random(args.seed)
    labels = list(SELECTED_EMOTIONS) * 4
    rng.shuffle(labels)
    cases = []
    for index, (text, label) in enumerate(zip(TEXTS, labels), 1):
        reference = rng.choice(references)
        emotion = [0.0] * len(EMOTION_NAMES)
        emotion[EMOTION_NAMES.index(label)] = 0.5
        cases.append({
            "id": f"case{index:02d}",
            "text": text,
            "voice": reference.stem,
            "reference_audio": str(reference.resolve()),
            "emotion_label": label,
            "emotion": emotion,
            "emotion_intensity": 0.5,
            "seed": rng.randrange(2**31),
        })
    args.output.mkdir(parents=True, exist_ok=True)
    reference_dir = args.output / "references"
    reference_dir.mkdir(exist_ok=True)
    for reference in sorted({Path(case["reference_audio"]) for case in cases}):
        shutil.copy2(reference, reference_dir / reference.name)
    payload = {
        "schema": 1,
        "seed": args.seed,
        "batch": 8,
        "description": "TRT113 fixed Draft B8 vs official IndexTTS2 eager paired listening set",
        "emotion_order": list(EMOTION_NAMES),
        "selection": "one-hot emotion selected from angry/sad/happy/melancholic; fixed intensity 0.5",
        "cases": cases,
    }
    (args.output / "cases.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "references": len(references),
                      "used_voices": sorted({case['voice'] for case in cases}),
                      "cases": len(cases)}, ensure_ascii=False, indent=2))


def optimized(args) -> None:
    import numpy as np
    import soundfile as sf
    from acc_infer_clear.config import load
    from acc_infer_clear.runtime.deployment import load as load_deployment
    from acc_infer_clear.runtime.device import GPULease
    from acc_infer_clear.runtime.pool import Pool

    cases = read_cases(args.output)
    destination = args.output / "trt113_b8"
    destination.mkdir(parents=True, exist_ok=True)
    config = load(args.config)
    config["max_batch"] = 8
    references = sorted({case["reference_audio"] for case in cases})
    voices = {path: f"voice-{index:02d}" for index, path in enumerate(references)}
    rows = []
    with GPULease(args.gpu), Pool(config, args.gpu, 1) as pool:
        for path, voice_id in voices.items():
            pool.prepare_reference(voice_id, path)
        deployment = pool.prepare_deployment(load_deployment(args.deployment))[0]
        for group_index in range(0, len(cases), 8):
            group = cases[group_index:group_index + 8]
            if len(group) != 8:
                raise ValueError("TRT candidate requires exact B8 groups")
            started = time.perf_counter()
            for case in group:
                pool.create_session(case["id"], voices[case["reference_audio"]],
                                    case["seed"], case["emotion"])
                pool.push_text(case["id"], case["text"])
                pool.finish_input(case["id"])
            while pool.states[0]["ready_heads"] or pool.states[0]["ready_tails"]:
                pool.run_ready()
            elapsed = time.perf_counter() - started
            for case in group:
                result = pool.result(case["id"])
                if not result["complete"] or result["error"]:
                    raise RuntimeError(f"TRT generation failed: {case['id']}: {result['error']}")
                pcm = np.concatenate([chunk["pcm"] for chunk in result["chunks"]])
                path = destination / f"{case['id']}.wav"
                sf.write(path, pcm, 22050, subtype="PCM_16")
                rows.append({"id": case["id"], "audio": path.name,
                             "seconds": len(pcm) / 22050, "batch_elapsed_s": elapsed,
                             "chunks": len(result["chunks"])})
                pool.release(case["id"])
            print(f"trt B8 group {group_index // 8 + 1}/2 {elapsed:.3f}s", flush=True)
    (destination / "results.json").write_text(json.dumps({
        "engine": "tensorrt_11.3_target_and_fixed_draft_b8",
        "deployment": deployment,
        "cases": rows,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def eager(args) -> None:
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
        print(f"eager {index:02d}/16 {case['id']} {elapsed:.3f}s", flush=True)
    (destination / "results.json").write_text(json.dumps({
        "engine": "official_indextts2_eager",
        "settings": {"fp16": False, "cuda_kernel": False, "torch_compile": False,
                     "deepspeed": False, "accel": False, "num_beams": 3},
        "cases": rows,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def make_html(args) -> None:
    cases = read_cases(args.output)
    trt = {row["id"]: row for row in json.loads(
        (args.output / "trt113_b8" / "results.json").read_text(encoding="utf-8"))["cases"]}
    eager_rows = {row["id"]: row for row in json.loads(
        (args.output / "eager" / "results.json").read_text(encoding="utf-8"))["cases"]}
    cards = []
    for index, case in enumerate(cases, 1):
        reference_name = Path(case["reference_audio"]).name
        cards.append(f"""
        <article class="card">
          <h2>#{index:02d} · {html.escape(case['voice'])} · {html.escape(case['emotion_label'])} 0.5</h2>
          <p class="text">{html.escape(case['text'])}</p>
          <div class="players">
            <section><h3>参考音色</h3><audio controls preload="none" src="references/{html.escape(reference_name)}"></audio></section>
            <section><h3>TensorRT 11.3 修复版 · B8</h3><audio controls preload="none" src="trt113_b8/{case['id']}.wav"></audio><small>{trt[case['id']]['seconds']:.2f}s</small></section>
            <section><h3>IndexTTS2 eager</h3><audio controls preload="none" src="eager/{case['id']}.wav"></audio><small>{eager_rows[case['id']]['seconds']:.2f}s · 生成 {eager_rows[case['id']]['elapsed_s']:.2f}s</small></section>
          </div>
        </article>""")
    document = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TRT113 修复版 B8 vs IndexTTS2 eager</title><style>
body{{font-family:system-ui,sans-serif;margin:0;background:#f3f5f8;color:#18202a}}main{{max-width:1180px;margin:auto;padding:28px}}.intro,.card{{background:#fff;border:1px solid #dfe4ea;border-radius:14px;padding:20px;margin-bottom:18px;box-shadow:0 4px 16px #18202a0c}}h1{{margin-top:0}}h2{{font-size:1.08rem;margin:0 0 10px}}h3{{font-size:.92rem;margin:0 0 8px}}.text{{font-size:1.06rem}}.meta,small{{color:#5d6875;font-size:.85rem}}.players{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}}section{{background:#f7f9fb;border-radius:10px;padding:12px}}audio{{width:100%}}small{{display:block;margin-top:6px}}@media(max-width:760px){{.players{{grid-template-columns:1fr}}}}
</style></head><body><main><div class="intro"><h1>TensorRT 11.3 修复版 B8 vs IndexTTS2 eager</h1><p>16 条中性文本；每条从 backup 音色目录随机取样。情绪从愤怒、悲伤、高兴、低落中选择，单一维度强度固定为 0.5。左右两种实现使用相同文本、音色、情绪向量与 seed。</p><p class="meta">随机种子：{args.seed}。TRT 侧严格分成两个 Batch=8 生成批次；eager 关闭 FP16、自定义 CUDA kernel、torch.compile、DeepSpeed 和加速引擎。</p></div>{''.join(cards)}</main></body></html>"""
    (args.output / "index.html").write_text(document, encoding="utf-8")
    archive = shutil.make_archive(str(args.output), "zip", root_dir=args.output.parent,
                                  base_dir=args.output.name)
    print(json.dumps({"html": str((args.output / 'index.html').resolve()),
                      "zip": str(Path(archive).resolve()), "cases": len(cases)},
                     ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs" / "trt113_b8_vs_eager_16_20260922")
    parser.add_argument("--seed", type=int, default=20260922)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build")
    candidate = sub.add_parser("optimized")
    candidate.add_argument("--gpu", type=int, required=True)
    candidate.add_argument("--config", default="configs/runtime.yaml")
    candidate.add_argument("--deployment", default="configs/sm89_bf16_trt113_target_draft_b8.json")
    sub.add_parser("eager")
    sub.add_parser("html")
    args = parser.parse_args()
    {"build": build, "optimized": optimized, "eager": eager, "html": make_html}[args.command](args)


if __name__ == "__main__":
    main()
