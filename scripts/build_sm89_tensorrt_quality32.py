#!/usr/bin/env python3
"""Build the exact random32 B1 performance corpus as a quality corpus."""
import argparse
import json
import random
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--texts',type=Path,default=Path('configs/hardware/sm89/sm89_benchmark_32_texts.json'))
    parser.add_argument('--reference',default='/workspace/index-tts/data/audio/old/mingxiang_gao.wav')
    parser.add_argument('--emotion-seed',type=int,default=20260920)
    parser.add_argument('--output',type=Path,default=Path('configs/hardware/sm89/sm89_tensorrt_quality32.json'))
    args=parser.parse_args();texts=json.loads(args.texts.read_text());rng=random.Random(args.emotion_seed);cases=[]
    for index,text in enumerate(texts):
        raw=[rng.random() for _ in range(8)];intensity=rng.random();total=sum(raw)
        cases.append(dict(id=f'trt-q{index:03d}',text=text,reference_audio=str(Path(args.reference).resolve()),
                          seed=index,emotion=[intensity*value/total for value in raw]))
    args.output.write_text(json.dumps(dict(schema=1,cases=cases),indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'output':str(args.output.resolve()),'cases':len(cases)},ensure_ascii=False))


if __name__=='__main__':main()
