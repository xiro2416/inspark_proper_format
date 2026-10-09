"""Final matched original-versus-best 30-wave E2E and separate power windows."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]


def main():
    from deployment.validate_matrix import benchmark
    from benchmarks.unified_first_chunk import compare_reports
    h=ROOT/'deployment/b32/history'
    original=benchmark(32,h/'migration-selected.json','delivery-migration-control-b32',final=True)
    retained=benchmark(32,h/'current-best-selected.json','delivery-current-best-b32',final=True)
    c=compare_reports(original,retained);(h/'compare-delivery-b32.json').write_text(json.dumps(c,indent=2)+'\n')
    print(json.dumps(dict(original_p50=original['summary']['wave_admission_to_last_pcm_ms']['p50'],best_p50=retained['summary']['wave_admission_to_last_pcm_ms']['p50'],gain_ci=c['metrics']['group_admission_to_last_pcm_ms']['paired_mean_gain_95pct_bootstrap_ci_ms'])),flush=True)


if __name__=='__main__':main()
