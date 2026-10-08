"""Select complete natural-duration sample texts and prepare CPU conditions."""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('INSPARK_REPO_ROOT',str(ROOT))

def main():
    import torch
    from inspark_infer.models.zipvoice.frontend import prepare
    from inspark_infer.models.zipvoice.tokenizer import EmiliaTokenizer
    torch.set_num_threads(2)
    folder=ROOT/'outputs/zipvoice-validation/cases'
    refs=json.loads((folder/'reference-vad4s.json').read_text())
    samples=json.loads((folder/'sample70.json').read_text())
    tokenizer=EmiliaTokenizer(str(ROOT/'models/zipvoice/config/tokens.txt'))
    text_ids=tokenizer.texts_to_token_ids([x['text'] for x in samples['samples']])
    manifest={'batch':1,'engines':{}}
    for key,kind in [('fm','fm-inherited'),('text','text'),('unique','text'),('vocos','vocos')]:
        p=json.loads((ROOT/f'.work/zipvoice/b1/graphs/{kind}-profile.json').read_text())
        manifest['engines'][key]={'shape_profile':{n:[v[x] for x in ('min','opt','max')] for n,v in p.items()}}
    accepted=[];rejected=[]
    for ri,ref in enumerate(refs):
        prompt=tokenizer.texts_to_token_ids([ref['reference_text']])[0]
        for ti,(sample,tokens) in enumerate(zip(samples['samples'],text_ids)):
            target_frames=int(torch.ceil(torch.tensor(375,dtype=torch.float32)*len(tokens)/len(prompt)).item())
            total=375+target_frames
            padded=len(prompt)+len(tokens)+1
            record={'reference_index':ri,'sample_index':ti,'source_line':sample['line'],'text':sample['text'],
                    'reference_wav':ref['path'],'reference_sha256':ref['sha256'],'reference_text':ref['reference_text'],
                    'prompt_tokens':len(prompt),'target_tokens':len(tokens),'total_frames':total,'padded_tokens':padded}
            if not (600<=total<=920 and 52<=padded<=141):
                rejected.append(record);continue
            output=folder/f'condition-r{ri}-s{ti}.safetensors'
            workload=prepare(ROOT/'models/zipvoice',manifest,ref['path'],ref['reference_text'],sample['text'],output)
            record.update(condition=str(output),condition_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),workload=workload)
            accepted.append(record)
    # Representative real-audio quality cases: per-reference shortest/nearest760/longest,
    # plus every exact760 case. All legal candidates remain recorded for extension.
    selected={}
    for ri in range(len(refs)):
        subset=[c for c in accepted if c['reference_index']==ri]
        if not subset:continue
        for c in (min(subset,key=lambda c:c['total_frames']),min(subset,key=lambda c:abs(c['total_frames']-760)),max(subset,key=lambda c:c['total_frames'])):
            selected[(ri,c['sample_index'])]=c
    for c in accepted:
        if c['total_frames']==760:selected[(c['reference_index'],c['sample_index'])]=c
    result={'status':'prepared','selection':'complete sample70 texts crossed with five freshly VAD/ASR references; natural duration unchanged',
            'legal_cases':accepted,'excluded_cases':rejected,'quality_cases':list(selected.values()),
            'primary_760':next((c for c in accepted if c['total_frames']==760),None)}
    (folder/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'legal':len(accepted),'quality':len(selected),'exact760':sum(c['total_frames']==760 for c in accepted),'covered_references':sorted({c['reference_index'] for c in accepted})}))
    if not result['primary_760']:raise RuntimeError('No natural exact760 sample; do not substitute forced-duration audio')

if __name__=='__main__':main()
