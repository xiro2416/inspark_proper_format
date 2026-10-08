"""Complete the explicit fulltext fallback after the current formal run finishes."""
import argparse,subprocess,time,json
from run_zipvoice_validation import sha
from run_zipvoice_validation import ROOT,environment


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--after-pid',type=int,required=True);args=p.parse_args()
    while True:
        process=subprocess.run(['ps','-p',str(args.after_pid),'-o','stat='],capture_output=True,text=True)
        if process.returncode!=0 or not process.stdout.strip() or process.stdout.strip().startswith('Z'):break
        time.sleep(10)
    history=ROOT/'reports/sm89/zipvoice/a1007/b64/history'
    application=history/'018-normtf32k16wp-application.json'
    original_application_sha256=sha(application)
    with (ROOT/'outputs/build-logs/f32-prepack-final-fulltext.log').open('w') as log:
        subprocess.run([str(ROOT/'.venv-zipvoice/bin/python'),str(ROOT/'scripts/validate_zipvoice_prepack_application.py')],cwd=ROOT,env=environment(),stdout=log,stderr=subprocess.STDOUT,check=True)
    performance=history/'020-normtf32k16wp-performance.json'
    record=json.loads(performance.read_text())
    assert record['status']=='formal_candidate_e2e_quality_power_complete_review_pending'
    assert record['application_evidence_sha256']==original_application_sha256
    record['application_evidence_before_fulltext_extension_sha256']=original_application_sha256
    record['application_evidence_sha256']=sha(application)
    record['quality_evidence_sha256']=sha(history/'019-normtf32k16wp-quality.json')
    record['validation_extension']='Original formal run retained; cached realcase state/PCM/engine checks repeated and explicit fulltext fallback added. Runtime/engine/math/benchmark samples unchanged.'
    performance.write_text(json.dumps(record,indent=2)+'\n')
    print('Cached actual evidence rechecked; explicit new packed-engine fulltext fallback completed',flush=True)


if __name__=='__main__':main()
