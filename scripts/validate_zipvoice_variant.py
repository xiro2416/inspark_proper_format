"""Independent full application/mapping/quality evidence for an actual FM candidate."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess

from run_zipvoice_validation import ROOT,contract_cases,environment,invoke,sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',type=int,required=True)
    parser.add_argument('--route',required=True)
    parser.add_argument('--control',choices=('native','inherited','normtf32k16'),default='native')
    parser.add_argument('--application',choices=('a1007','a1007_delivery','a1007_graph','a1007_delivery_graph'))
    parser.add_argument('--pcm-chunk',type=int)
    parser.add_argument('--pcm-workers',type=int)
    args=parser.parse_args();batch=args.batch;route=args.route
    extra=[]
    if args.pcm_chunk is not None:extra+=['--pcm-chunk',str(args.pcm_chunk)]
    if args.pcm_workers is not None:extra+=['--pcm-workers',str(args.pcm_workers)]
    def call(selected_route,case,dest,more=()):
        return invoke(batch,selected_route,case,dest,application=args.application,extra=[*extra,*more])
    lock=Path('/workspace/.cache/inspark/gpu-locks/1.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    history=ROOT/f'reports/sm89/zipvoice/a1007/b{batch}/history'
    minimal=json.loads((history/f'016-{route}-minimal.json').read_text())
    assert minimal['status']=='minimal_compute_passed_audio_quality_pending'
    metadata=json.loads((ROOT/f'artifacts/zipvoice/a1007/b{batch}/fm-{route}/build.json').read_text())
    assert minimal['engine_sha256']['fm-inherited']==metadata['engine_sha256']
    root=ROOT/f'outputs/zipvoice-validation/b{batch}/{route}-validation'
    report_path=history/f'018-{route}-application.json'
    manifest=json.loads((ROOT/'outputs/zipvoice-validation/cases/manifest.json').read_text())
    report={'status':'running','batch':batch,'route':route,'engine_sha256':metadata['engine_sha256'],
            'control':args.control,'application':args.application,'pcm_policy':{'chunk':args.pcm_chunk,'workers':args.pcm_workers},'cases':[],'quality_inputs':[],'matched_conditions':[],'contract_cases':[]}
    for index,case in enumerate(manifest['quality_cases']):
        for selected_route in (args.control,route):
            dest=root/selected_route/f'real-{index:03d}'
            result,rows=call(selected_route,case,dest)
            report['cases'].append({'route':selected_route,'case':index,'frames':case['total_frames'],
                                    'report':str(dest/'report.json'),'all_pcm_items':batch,'graph_direct_exact':True})
            for row in rows:
                path=dest/f'{row:04d}.wav'
                report['quality_inputs'].append({'batch':batch,'route':selected_route,'case':index,'row':row,
                    'path':str(path),'wav_sha256':sha(path),'target_text':case['text'],
                    'reference_wav':case['reference_wav'],'reference_sha256':case['reference_sha256']})
        import torch
        from safetensors.torch import load_file
        controls=load_file(str(root/args.control/f'real-{index:03d}'/'selected-state.safetensors'))
        candidate=load_file(str(root/route/f'real-{index:03d}'/'selected-state.safetensors'))
        for key in ('initial_state','text_condition','speech_condition','padding_mask','time_grid'):
            assert torch.equal(controls[key],candidate[key]),(batch,index,key,'candidate changed input mapping')
        expected=controls['final_state'];actual=candidate['final_state']
        report['matched_conditions'].append({'case':index,'original_noise_conditions_mask_grid_exact':True,
            'selected_state_relative_l2':float((actual-expected).norm()/expected.norm().clamp_min(1e-20))})
        report_path.write_text(json.dumps(report,indent=2)+'\n')
        print(f'B{batch} {route} real{index} paired mapping passed',flush=True)
    for case in contract_cases(batch,manifest):
        dest=root/route/case['name'];result,_=call(route,case,dest)
        if case['kind']=='natural_same_length_mixed':assert not result['text_reuse']
        report['contract_cases'].append({'name':case['name'],'kind':case['kind'],'report':str(dest/'report.json'),'passed':True})
        report_path.write_text(json.dumps(report,indent=2)+'\n')
    fulltext=root/route/'fulltext-control'
    result,_=call(route,manifest['primary_760'],fulltext,more=['--disable-text-reuse'])
    assert not result['text_reuse']
    report.update(status='variant_full_application_mapping_passed_quality_pending',fulltext_control=str(fulltext/'report.json'))
    report_path.write_text(json.dumps(report,indent=2)+'\n')
    inputs=root/'quality-inputs.json';inputs.write_text(json.dumps(report['quality_inputs'],indent=2)+'\n')
    output=history/f'019-{route}-quality.json'
    if output.exists() and json.loads(output.read_text()).get('input_inventory_sha256')==sha(inputs):return
    command=[str(ROOT/'.venv-evaluation/bin/python'),str(ROOT/'scripts/evaluate_zipvoice_audio.py'),
             '--device','cpu','--inputs',str(inputs),'--output',str(output)]
    with (ROOT/f'outputs/build-logs/b{batch}-{route}-quality.log').open('w') as log:
        process=subprocess.Popen(command,cwd=ROOT,env={**environment(),'CUDA_VISIBLE_DEVICES':''},
                                 stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    (history/f'019-{route}-quality-job.json').write_text(json.dumps({'status':'started_not_accepted','pid':process.pid,
        'command':command,'input_inventory_sha256':sha(inputs)},indent=2)+'\n')
    print(f'B{batch} {route} CPU quality started {process.pid}',flush=True)


if __name__=='__main__':main()
