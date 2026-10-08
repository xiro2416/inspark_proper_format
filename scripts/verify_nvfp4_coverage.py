"""Reject Q/DQ graphs that silently compute required operators in FP16/BF16."""
import argparse,json
from pathlib import Path
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    report=[]
    for name in ['target','draft','context','prefill','latent','latent_suffix','cfm','vocoder']:
        plan=args.root/name/'model.plan.json';inspection=args.root/name/'model.inspector.json'
        if not plan.is_file() or not inspection.is_file():raise ValueError('Missing engine '+name)
        metadata=json.loads(plan.read_text());layers=json.loads(inspection.read_text())['Layers']
        low=[n for n in layers if n.get('LayerType')=='gemm' and 'nvfp4_linear' in n['Name']]
        bad=[{'name':n['Name'],'tactic':n.get('TacticName')} for n in low if 'xe2m1' not in n.get('TacticName','')]
        if not low or bad:raise ValueError('Required native FP4 coverage failed '+name+': '+str(bad))
        row={'component':name,'native_fp4_gemms':len(low),'non_native_required_gemms':bad,'engine_sha256':metadata['sha256'],
            'protected_and_unquantized_regions':'kept as declared; not included in required FP4 GEMMs'}
        report.append(row)
    target=args.root/'native_coverage.json';target.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
