from pathlib import Path
import json
import yaml
from acc_infer_clear.runtime.graph_policy import BATCHES

def atomic_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2));tmp.replace(path)

def load(path):
    path=Path(path).resolve();cfg=yaml.safe_load(path.read_text())
    expected={'weights','student','student_sha256','cache','max_batch','cpu_threads','max_speech_tokens','max_text_tokens','reference_seconds','max_cached_voices','target_tf32','rnn_tf32'}
    if set(cfg)!=expected:raise ValueError('Unknown or missing configuration keys: '+str(set(cfg)^expected))
    for key in ('weights','student','cache'):
        value=Path(cfg[key]);cfg[key]=str(value if value.is_absolute() else (path.parent/value).resolve())
    if cfg['max_batch'] not in BATCHES:raise ValueError('Supported max_batch is one of '+str(BATCHES))
    if not 1<=cfg['max_speech_tokens']<=1500:raise ValueError('Supported speech-token cap is1..1500')
    if not 1<=cfg['max_text_tokens']<=120:raise ValueError('Supported cumulative text-token cap is1..120')
    if cfg['rnn_tf32'] is not False:raise ValueError('Baseline RNN uses full FP32 multiplication')
    return cfg
