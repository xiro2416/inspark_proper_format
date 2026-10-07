"""Download pinned releases and prepare relocatable current runtime configuration."""
import json,hashlib,os,shutil
from pathlib import Path

def sha256(path):
    with Path(path).open('rb') as s:return hashlib.file_digest(s,'sha256').hexdigest()

def registry():
    return json.loads((Path(__file__).resolve().parents[3]/'configs/current/release.json').read_text())

def fetch(destination,precision='fp8',batch=1):
    from huggingface_hub import snapshot_download
    info=registry();label=f'{precision}_b{batch}'
    if label not in info['deployments']:raise ValueError('No published deployment for '+label)
    destination=Path(destination).resolve();destination.mkdir(parents=True,exist_ok=True)
    # The published bundle contains only the seven selected dependency graphs.
    cache=snapshot_download(info['repo'],revision=info['engine_revision'],
        allow_patterns=[info['engine_prefix']+'/**'],local_dir=destination/'download')
    bundle=Path(cache)/info['engine_prefix']
    snapshot_download(info['repo'],revision=info['weights_revision'],
        allow_patterns=['unquantized/**','shared/**','training_provenance.json','manifest.json'],local_dir=destination/'weights')
    verify_bundle(bundle)
    verify_weights(destination/'weights')
    cfg_path,deploy_path=materialize(bundle,destination/'weights',destination/'runtime',precision,batch)
    return cfg_path,deploy_path

def verify_bundle(bundle):
    bundle=Path(bundle).resolve()
    if not (bundle/'.bundle_root').is_file():raise ValueError('Missing bundle marker')
    for record in json.loads((bundle/'manifest.json').read_text()):
        path=(bundle/record['path']).resolve()
        if not path.is_relative_to(bundle) or not path.is_file():raise ValueError('Missing/escaping release asset')
        if path.stat().st_size!=record['bytes'] or sha256(path)!=record['sha256']:
            raise ValueError('Release asset identity mismatch: '+record['path'])

def verify_weights(weights):
    weights=Path(weights).resolve()
    for record in json.loads((weights/'manifest.json').read_text()):
        if not record['path'].startswith(('unquantized/','shared/')):continue
        path=(weights/record['path']).resolve()
        if not path.is_relative_to(weights) or not path.is_file() or path.stat().st_size!=record['bytes'] or sha256(path)!=record['sha256']:
            raise ValueError('Downloaded model identity mismatch: '+record['path'])


def materialize(bundle,weights,output,precision='fp8',batch=1):
    """Reconstruct loader file containers without changing checkpoint tensor values."""
    import torch,yaml
    from safetensors.torch import load_file
    bundle=Path(bundle).resolve();weights=Path(weights).resolve();output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=True);models=output/'models';models.mkdir(exist_ok=True)
    shared=weights/'shared'
    base=models/'index_tts2';base.mkdir(exist_ok=True)
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists() or sha256(target)!=sha256(source):shutil.copyfile(source,target)
    for name in ['config.yaml','bpe.model']:copy(shared/'index_tts2'/name,base/name)
    for name in ['feat1.pt','feat2.pt','wav2vec2bert_stats.pt']:copy(shared/name,base/name)
    for name in ['campplus_cn_common.bin','semantic_codec_model.safetensors']:copy(shared/name,base/'hf_cache'/name)
    shutil.copytree(shared/'index_tts2/hf_cache/w2v-bert-2.0',base/'hf_cache/w2v-bert-2.0',dirs_exist_ok=True)
    shutil.copytree(shared/'asg',models/'asg',dirs_exist_ok=True)
    copy(shared/'index_tts2/hf_cache/bigvgan/config.json',base/'hf_cache/bigvgan/config.json')
    copy(bundle/'raw_sources/bigvgan_generator.pt',base/'hf_cache/bigvgan/bigvgan_generator.pt')
    copy(bundle/'raw_sources/s2mel.pth',base/'s2mel.pth')
    copy(shared/'draft_onpolicy100/config.json',models/'draft_onpolicy100/config.json')
    copy(weights/'unquantized/draft.safetensors',models/'draft_onpolicy100/model.safetensors')
    if not (base/'gpt.pth').exists():torch.save(load_file(str(weights/'unquantized/target.safetensors')),base/'gpt.pth')
    student=models/'cfm_bilingual40k_step800.pt'
    if not student.exists():torch.save({'format_version':1,'step':800,'student':load_file(str(weights/'unquantized/cfm.safetensors')),
        'source':{'designation':'bilingual40k stage1 step800; source hashes in training_provenance.json'}},student)
    cfg={'weights':str(models),'student':str(student),'student_sha256':sha256(student),'cache':str(output/'cache'),
        'max_batch':batch,'cpu_threads':8,'max_speech_tokens':1500,'max_text_tokens':120,'reference_seconds':3,
        'max_cached_voices':16,'target_tf32':False,'rnn_tf32':False}
    path=output/f'runtime_{precision}_b{batch}.yaml';path.write_text(yaml.safe_dump(cfg))
    os.environ['INSPARK_ASSET_ROOT']=str(bundle);os.environ['INSPARK_MODEL_ROOT']=str(models)
    return path,bundle/'artifacts/current_release/deployments'/f'{precision}_b{batch}.json'
