"""Paired held-out CER/WER, UTMOS and SIM-o reports; no invented numeric gate."""
import argparse
import json
import os
from pathlib import Path
import re
import sys
import unicodedata
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from inspark_infer.runtime.zipvoice_fp8.common import BATCHES,sha,write,private_report


def distance(a,b):
    previous=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        current=[i]
        for j,y in enumerate(b,1):current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(x!=y)))
        previous=current
    return previous[-1]


def normalized(text,lang):
    text=unicodedata.normalize('NFKC',text).lower()
    return list(''.join(c for c in text if c.isalnum())) if lang=='zh' else re.findall(r"[a-z0-9]+(?:'[a-z]+)?",text)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batches',type=int,nargs='+',choices=BATCHES,default=list(BATCHES))
    p.add_argument('--variant',choices=['native','attention','attentiongeo'],default='native')
    args=p.parse_args();suffix='' if args.variant=='native' else '-'+args.variant
    if os.getenv('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('Quality evaluation is CPU-only')
    os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    from funasr import AutoModel
    from transformers import WhisperProcessor,WhisperForConditionalGeneration
    from inspark_infer.models.zipvoice.evaluation.utmos import UTMOS22Strong
    from inspark_infer.models.zipvoice.evaluation.ecapa_tdnn_wavlm import ECAPA_TDNN_WAVLM
    torch.set_num_threads(8)
    assets=ROOT/'models/evaluation'
    mos=UTMOS22Strong();mos.load_state_dict(torch.load(assets/'mos/utmos22_strong_step7459_v1.pt',map_location='cpu',weights_only=False),strict=True);mos.eval()
    ssl=assets/'speaker_similarity/wavlm_large'
    original_load=torch.hub.load
    def loader(repo,entry,*positional,**kw):
        assert Path(repo)==ssl.parent and entry=='wavlm_local' and kw.get('source')=='local'
        return original_load(str(ssl),entry,*positional,**kw)
    with patch('torch.hub.load',side_effect=loader):
        sim=ECAPA_TDNN_WAVLM(feat_dim=1024,channels=512,emb_dim=256,sr=16000,ssl_model_path=str(ssl))
    missing=sim.load_state_dict(torch.load(assets/'speaker_similarity/wavlm_large_finetune.pth',map_location='cpu',weights_only=False)['model'],strict=False)
    if missing.missing_keys:raise ValueError('SIM-o missing model weights')
    sim.eval()
    zh=AutoModel(model=str(assets/'wer/paraformer-zh'),device='cpu',disable_update=True,ncpu=8,disable_pbar=True)
    processor=WhisperProcessor.from_pretrained(assets/'wer/whisper-large-v3',local_files_only=True)
    en=WhisperForConditionalGeneration.from_pretrained(assets/'wer/whisper-large-v3',local_files_only=True).eval()
    refs={};cache={}
    def load(path):
        w,sr=sf.read(path,dtype='float32',always_2d=True);w=torch.from_numpy(w.mean(axis=1))
        if not len(w) or not torch.isfinite(w).all():raise ValueError('Invalid complete waveform')
        return torchaudio.functional.resample(w,sr,16000) if sr!=16000 else w
    def evaluate(path,reference,lang,target):
        key=(sha(path),sha(reference),lang,target)
        if key in cache:return {**cache[key],'reused_identical_wav_hash':True}
        wave=load(path)
        if lang=='zh':hypothesis=zh.generate(input=wave.numpy(),use_itn=True,disable_pbar=True)[0]['text']
        else:
            inp=processor(wave.numpy(),sampling_rate=16000,return_tensors='pt',return_attention_mask=True)
            with torch.inference_mode():pred=en.generate(inp.input_features,attention_mask=inp.attention_mask,language='en',task='transcribe')
            hypothesis=processor.batch_decode(pred,skip_special_tokens=True)[0]
        truth=normalized(target,lang);predicted=normalized(hypothesis,lang)
        if not truth:raise ValueError('Empty normalized target')
        with torch.inference_mode():
            mos_score=float(mos(wave[None],16000).item())
            rh=sha(reference)
            if rh not in refs:refs[rh]=sim([load(reference)])
            similarity=float(torch.nn.functional.cosine_similarity(sim([wave]),refs[rh],dim=-1).item())
        if not np.isfinite([mos_score,similarity]).all():raise ValueError('Nonfinite quality metric')
        result=dict(transcription=hypothesis,errors=distance(truth,predicted),reference_units=len(truth),
                    error_rate=distance(truth,predicted)/len(truth),metric='CER' if lang=='zh' else 'WER',
                    utmos=mos_score,sim_o=similarity,wav_sha256=key[0])
        cache[key]=result
        return result
    for batch in args.batches:
        source=private_report(batch,'004-fp32-reference'+suffix)
        audit=json.loads(source.read_text());rows=[]
        report=dict(status='running',batch=batch,source_audit_sha256=sha(source),fixed_quality_threshold=False,results=rows)
        for case in audit['cases']:
            for row in case['rows']:
                metrics={kind:evaluate(row[f'{kind}_wav'],case['reference_wav'],case['language'],case['text']) for kind in ['fp32','fp8']}
                rows.append(dict(case=case['case'],row=row['row'],language=case['language'],**metrics))
                write(private_report(batch,'005-quality'+suffix),report)
                print(json.dumps({'event':'quality_pair','batch':batch,'case':case['case'],'row':row['row']}),flush=True)
        summary={}
        for lang in ('zh','en'):
            subset=[r for r in rows if r['language']==lang]
            if not subset:raise ValueError('Missing held-out language')
            summary[lang]={}
            for kind in ('fp32','fp8'):
                summary[lang][kind]=dict(error_rate=sum(r[kind]['errors'] for r in subset)/sum(r[kind]['reference_units'] for r in subset),
                                        utmos=float(np.mean([r[kind]['utmos'] for r in subset])),sim_o=float(np.mean([r[kind]['sim_o'] for r in subset])))
        report.update(status='paired_quality_metrics_complete',summary=summary,count=len(rows),
                      limits='Finite held-out corpus and report-only quality; no perceptual equivalence or production certification')
        write(private_report(batch,'005-quality'+suffix),report)


if __name__=='__main__':main()
