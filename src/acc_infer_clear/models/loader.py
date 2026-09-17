"""Load only universal inference models and fixed student weights."""
from pathlib import Path
from types import SimpleNamespace
import threading,hashlib,os
import torch
from .upstream.infer_v2 import IndexTTS2
from .reference import VoiceBank,vad_crop
from acc_infer_clear.dspark.draft import IndexTTS2DSpark
from acc_infer_clear.dspark.target import IndexTTS2TargetEngine
from acc_infer_clear.pcg.asg import AcousticGroups
from acc_infer_clear.cfm.solver import Solver

class Model:
    def __init__(self,config):
        self.config=config;self.lock=threading.Lock();self.status='loading'
        os.environ['ACC_CLEAR_CACHE']=config['cache']
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('Select exactly one visible GPU before constructing Engine')
        torch.set_num_threads(config['cpu_threads']);torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32=config['target_tf32']
        self.stream=torch.cuda.Stream();root=Path(config['weights']);base=root/'index_tts2';aux=base/'hf_cache'
        paths=dict(w2v_bert=aux/'w2v-bert-2.0',semantic_codec=aux/'semantic_codec_model.safetensors',campplus=aux/'campplus_cn_common.bin',bigvgan=aux/'bigvgan')
        for path in [base/'config.yaml',root/'draft_onpolicy100/model.safetensors',root/'asg/target_asg_threshold_0p49.safetensors',*paths.values()]:
            if not path.exists():raise FileNotFoundError(path)
        with torch.cuda.stream(self.stream),torch.inference_mode():
            self.tts=IndexTTS2(cfg_path=str(base/'config.yaml'),model_dir=str(base),device='cuda:0',aux_paths={k:str(v) for k,v in paths.items()})
            draft=IndexTTS2DSpark.from_checkpoint(root/'draft_onpolicy100','cuda:0').float().eval()
            assert draft.block_size==7 and not draft.persistent_markov_state and draft.midblock_refresh_at==0
            groups=AcousticGroups.load(root/'asg/target_asg_threshold_0p49.safetensors')
            expected=groups.metadata.get('embedding_sha256_float32')
            actual=hashlib.sha256(self.tts.gpt.mel_embedding.weight.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()
            if expected and actual!=expected:raise ValueError('ASG/Target embedding fingerprint mismatch')
            self.engine=SimpleNamespace(target=IndexTTS2TargetEngine(self.tts.gpt,[1,6,11,16,21]),draft=draft,dense_groups=groups.dense('cuda:0'),max_thinning_attempts=64)
            self.student=Solver(self.tts.s2mel.models['cfm'].estimator,config['student'],config['student_sha256'],config['max_batch'])
            self.bank=VoiceBank(self.tts,config['max_cached_voices']);self.stream.synchronize()
        self.cfg={'data':{'max_text_tokens_per_segment':config['max_text_tokens']}}
        self.status='ready'
    def _acquire(self,operation):
        if self.status=='closed':raise RuntimeError('Model closed')
        if not self.lock.acquire(blocking=False):raise RuntimeError('One model execution owner; use scheduler or independent workers')
        self.status=operation
    def _release(self):self.status='ready';self.lock.release()
    def prepare_reference(self,voice_id,path):
        self._acquire('reference')
        try:
            wav,metadata=vad_crop(path,self.config['cache'],self.config['reference_seconds'])
            with torch.cuda.stream(self.stream),torch.inference_mode():
                self.bank.build(voice_id,wav,metadata);self.stream.synchronize()
            return metadata
        finally:self._release()
    def close(self):
        if self.status=='closed':return
        del self.tts.gpt.get_conditioning  # Restore class method and break the voice-cache closure cycle.
        self.bank.entries.clear();self.engine=None;self.student=None;self.tts=None;self.bank=None;self.status='closed'
