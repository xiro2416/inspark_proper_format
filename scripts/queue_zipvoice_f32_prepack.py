"""Build/validate simple exact existing TF32 operand prepacking on GPU1."""
import json,subprocess
from pathlib import Path
from run_zipvoice_validation import ROOT,environment,sha


def main():
    report_root=ROOT/'reports/sm89/zipvoice/a1007';history=report_root/'b64/history';evidence=report_root/'f32-prepack-execution.json'
    proof=json.loads((history/'038-actual-weight-prepack.json').read_text());assert proof['status']=='all12_actual_float32_ffn_weight_operands_gpu_rna_exact'
    retention=json.loads((history/'036-normal-k16-retention.json').read_text());assert retention['status']=='normal_k16_retained_remaining_operator_review_pending'
    review={'status':'current_best_for_focused_prepack_comparison','batches':{'64':{'route':'normtf32k16','application':'a1007_delivery_graph','pcm_policy':{'chunk':16,'workers':4},'retention_sha256':sha(history/'036-normal-k16-retention.json')}}}
    selected=report_root/'prepack-current-best.json';selected.write_text(json.dumps(review,indent=2)+'\n')
    record={'status':'prepared','current_best_sha256':sha(selected),'actual_operand_proof_sha256':sha(history/'038-actual-weight-prepack.json'),'commands':[]}
    def save():evidence.write_text(json.dumps(record,indent=2)+'\n')
    def run(command,label):
        record.update(status=label);record['commands'].append(command);save();print(label,flush=True)
        with (ROOT/f'outputs/build-logs/{label}.log').open('w') as log:
            subprocess.run(command,cwd=ROOT,env=environment(),stdout=log,stderr=subprocess.STDOUT,check=True)
    try:
        run([str(ROOT/'.venv-zipvoice/bin/python'),str(ROOT/'scripts/build_zipvoice_variants.py'),'--batches','64','--suffix','normtf32k16wp'],'f32-prepack-build-minimal')
        build=json.loads((ROOT/'artifacts/zipvoice/a1007/b64/fm-normtf32k16wp/build.json').read_text())
        maps={r['module']:r for r in proof['rows']}
        assert len(build['f32_replacements'])==12
        for row in build['f32_replacements']:
            assert row['source_weight_sha256']==maps[row['module']]['source_weight_sha256'] and row['weight_sha256']==maps[row['module']]['packed_weight_sha256']
        run([str(ROOT/'.venv-zipvoice/bin/python'),str(ROOT/'scripts/validate_zipvoice_prepack_application.py')],'f32-prepack-full-model-identity')
        run([str(ROOT/'.venv-zipvoice/bin/python'),str(ROOT/'scripts/measure_zipvoice_variant.py'),'--batch','64','--route','normtf32k16wp','--control','normtf32k16','--application','a1007_delivery_graph','--control-application','a1007_delivery_graph','--pcm-chunk','16','--pcm-workers','4','--current-best-review',str(selected)],'f32-prepack-current-best-formal')
        record['status']='prepack_complete_application_identity_e2e_power_review_pending';save()
    except Exception as exc:
        record.update(status='stopped_requires_evidence_review',error=repr(exc));save();raise


if __name__=='__main__':main()
