"""Inspect local INT8 deployment state without loading models or using a GPU."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    results = []
    for batch in (1,2,4,8,16):
        assets = ROOT / f'artifacts/sm89/int8_smoothquant/b{batch}'
        components = {}
        for name in ('target','draft','prefill','latent','cfm-estimator','vocoder-gemm'):
            path = assets / name / 'model.plan.json'
            engine = path.with_name('model.engine')
            if path.is_file() and engine.is_file():
                plan = json.loads(path.read_text())
                components[name] = dict(state='built',sm=plan['sm'],batch=plan['batch'],
                                        bytes=engine.stat().st_size,sha256=plan['sha256'],
                                        build_seconds=plan['build_seconds'])
            else:
                components[name] = dict(state='pending')
        path = ROOT / f'configs/hardware/sm89/indextts/int8_b{batch}_selected.json'
        validated = path.is_file() and json.loads(path.read_text())['status'] == 'validated_local_sm89_int8'
        results.append(dict(batch=batch,validated=validated,components=components))
    print(json.dumps(results,indent=2))


if __name__ == '__main__':
    main()
