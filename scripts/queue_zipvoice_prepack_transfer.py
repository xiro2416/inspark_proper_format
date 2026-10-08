"""Verify/apply the screened simple F32 operand cache serially to B8/16/32."""
import json,subprocess
from run_zipvoice_validation import ROOT,environment,sha


def main():
    report=ROOT/'reports/sm89/zipvoice/a1007/prepack-transfer-execution.json';state={'status':'running','batches':{}}
    def save():report.write_text(json.dumps(state,indent=2)+'\n')
    def run(batch,command,label):
        state['batches'][str(batch)]={'status':label};save();print(f'B{batch} {label}',flush=True)
        with (ROOT/f'outputs/build-logs/b{batch}-{label}.log').open('w') as log:subprocess.run(command,cwd=ROOT,env=environment(),stdout=log,stderr=subprocess.STDOUT,check=True)
    save()
    for batch in (8,16,32):
        history=ROOT/f'reports/sm89/zipvoice/a1007/b{batch}/history';base=json.loads((ROOT/'reports/sm89/zipvoice/a1007/model-graph-review.json').read_text())['batches'][str(batch)];app=base['application'];policy=base['pcm_policy']
        review=ROOT/f'reports/sm89/zipvoice/a1007/b{batch}-prepack-current-best.json';review.write_text(json.dumps({'batches':{str(batch):{'route':'inherited','application':app,'pcm_policy':policy}}},indent=2)+'\n')
        run(batch,[str(ROOT/'.venv-builder/bin/python'),str(ROOT/'scripts/validate_zipvoice_prepacked_weights.py'),'--batch',str(batch)],'prepack-actual-weights')
        run(batch,[str(ROOT/'.venv-zipvoice/bin/python'),str(ROOT/'scripts/build_zipvoice_variants.py'),'--batches',str(batch),'--suffix','wp'],'prepack-build-minimal')
        proof={x['module']:x for x in json.loads((history/'041-actual-weight-prepack.json').read_text())['rows']};build=json.loads((ROOT/f'artifacts/zipvoice/a1007/b{batch}/fm-wp/build.json').read_text())
        for x in build['f32_replacements']:assert x['source_weight_sha256']==proof[x['module']]['source_weight_sha256'] and x['weight_sha256']==proof[x['module']]['packed_weight_sha256']
        run(batch,[str(ROOT/'.venv-zipvoice/bin/python'),str(ROOT/'scripts/validate_zipvoice_prepack_application.py'),'--batch',str(batch)],'prepack-full-model-identity')
        run(batch,[str(ROOT/'.venv-zipvoice/bin/python'),str(ROOT/'scripts/measure_zipvoice_variant.py'),'--batch',str(batch),'--route','wp','--control','inherited','--application',app,'--control-application',app,'--pcm-chunk',str(policy['chunk']),'--pcm-workers',str(policy['workers']),'--current-best-review',str(review)],'prepack-current-best-formal')
        state['batches'][str(batch)]={'status':'complete_evidence_review_pending'};save()
    state['status']='three_prepack_transfers_complete_review_pending';save()


if __name__=='__main__':main()
