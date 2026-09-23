import os
from contextlib import nullcontext
import librosa
import torch
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
from omegaconf import OmegaConf
from inspark_infer.models.indextts2.upstream.gpt.model_v2 import UnifiedVoice
from inspark_infer.models.indextts2.upstream.codec.maskgct_codec import build_semantic_codec
from inspark_infer.models.indextts2.upstream.utils.checkpoint import load_checkpoint
from inspark_infer.models.indextts2.upstream.utils.front import TextNormalizer, TextTokenizer
from inspark_infer.models.indextts2.upstream.s2mel.modules.commons import load_checkpoint2, MyModel
from inspark_infer.models.indextts2.upstream.s2mel.modules.bigvgan import bigvgan
from inspark_infer.models.indextts2.upstream.s2mel.modules.campplus.DTDNN import CAMPPlus
from inspark_infer.models.indextts2.upstream.s2mel.modules.audio import mel_spectrogram
from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertModel
import safetensors
import torch.nn.functional as F

def _latency_span(profiler, name):
    """Return an optional benchmark span without affecting normal inference."""
    if profiler is None:
        return nullcontext()
    return profiler.span(name)

class IndexTTS2:

    def __init__(self, cfg_path='checkpoints/config.yaml', model_dir='checkpoints', use_fp16=False, device=None, aux_paths=None, latency_profiler=None):
        self.latency_profiler = latency_profiler
        with _latency_span(self.latency_profiler, 'init.aux_models'):
            if aux_paths is None:
                raise ValueError('Explicit local auxiliary model paths required; downloads disabled')
        if device is not None:
            self.device = device
            self.use_fp16 = False if device == 'cpu' else use_fp16
        elif torch.cuda.is_available():
            self.device = 'cuda:0'
            self.use_fp16 = use_fp16
        elif hasattr(torch, 'xpu') and torch.xpu.is_available():
            self.device = 'xpu'
            self.use_fp16 = use_fp16
        elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
            self.device = 'mps'
            self.use_fp16 = False
        else:
            self.device = 'cpu'
            self.use_fp16 = False
            print('>> Be patient, it may take a while to run in CPU mode.')
        self.cfg = OmegaConf.load(cfg_path)
        self.model_dir = model_dir
        self.dtype = torch.float16 if self.use_fp16 else None
        self.stop_mel_token = self.cfg.gpt.stop_mel_token
        with _latency_span(self.latency_profiler, 'init.qwen_emotion'):
            self.qwen_emo = None
            print('>> Using explicit emotion vectors')
        with _latency_span(self.latency_profiler, 'init.gpt'):
            self.gpt = UnifiedVoice(**self.cfg.gpt)
            self.gpt_path = os.path.join(self.model_dir, self.cfg.gpt_checkpoint)
            load_checkpoint(self.gpt, self.gpt_path)
            self.gpt = self.gpt.to(self.device)
            if self.use_fp16:
                self.gpt.eval().half()
            else:
                self.gpt.eval()
            print('>> GPT weights restored from:', self.gpt_path)
        with _latency_span(self.latency_profiler, 'init.gpt_runtime'):
            self.gpt.post_init_gpt2_config(kv_cache=True, half=self.use_fp16)
        with _latency_span(self.latency_profiler, 'init.w2v_bert'):
            w2v_bert_dir = aux_paths['w2v_bert']
            self.extract_features = SeamlessM4TFeatureExtractor.from_pretrained(w2v_bert_dir, local_files_only=True)
            self.semantic_model = Wav2Vec2BertModel.from_pretrained(w2v_bert_dir, local_files_only=True)
            self.semantic_model = self.semantic_model.to(self.device)
            self.semantic_model.eval()
            stat_mean_var = torch.load(os.path.join(self.model_dir, self.cfg.w2v_stat))
            self.semantic_mean = stat_mean_var['mean'].to(self.device)
            self.semantic_std = torch.sqrt(stat_mean_var['var']).to(self.device)
        with _latency_span(self.latency_profiler, 'init.semantic_codec'):
            semantic_codec = build_semantic_codec(self.cfg.semantic_codec)
            semantic_code_ckpt = aux_paths['semantic_codec']
            safetensors.torch.load_model(semantic_codec, semantic_code_ckpt)
            self.semantic_codec = semantic_codec.to(self.device)
            self.semantic_codec.eval()
            print('>> semantic_codec weights restored from: {}'.format(semantic_code_ckpt))
        with _latency_span(self.latency_profiler, 'init.s2mel'):
            s2mel_path = os.path.join(self.model_dir, self.cfg.s2mel_checkpoint)
            s2mel = MyModel(self.cfg.s2mel, use_gpt_latent=True)
            s2mel = load_checkpoint2(s2mel, s2mel_path)
            self.s2mel = s2mel.to(self.device)
            self.s2mel.models['cfm'].estimator.setup_caches(max_batch_size=1, max_seq_length=8192)
        self.s2mel.eval()
        print('>> s2mel weights restored from:', s2mel_path)
        with _latency_span(self.latency_profiler, 'init.campplus'):
            campplus_ckpt_path = aux_paths['campplus']
            campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
            campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location='cpu'))
            self.campplus_model = campplus_model.to(self.device)
            self.campplus_model.eval()
            print('>> campplus_model weights restored from:', campplus_ckpt_path)
        with _latency_span(self.latency_profiler, 'init.bigvgan'):
            bigvgan_dir = aux_paths['bigvgan']
            self.bigvgan = bigvgan.BigVGAN.from_pretrained(bigvgan_dir)
            self.bigvgan = self.bigvgan.to(self.device)
            self.bigvgan.remove_weight_norm()
            self.bigvgan.eval()
            print('>> bigvgan weights restored from:', bigvgan_dir)
        with _latency_span(self.latency_profiler, 'init.tokenizer_normalizer'):
            self.bpe_path = os.path.join(self.model_dir, self.cfg.dataset['bpe_model'])
            self.normalizer = TextNormalizer(enable_glossary=True)
            self.normalizer.load()
            print('>> TextNormalizer loaded')
            self.tokenizer = TextTokenizer(self.bpe_path, self.normalizer)
            print('>> bpe model loaded from:', self.bpe_path)
        self.glossary_path = os.path.join(self.model_dir, 'glossary.yaml')
        if os.path.exists(self.glossary_path):
            self.normalizer.load_glossary_from_yaml(self.glossary_path)
            print('>> Glossary loaded from:', self.glossary_path)
        with _latency_span(self.latency_profiler, 'init.emotion_matrices'):
            emo_matrix = torch.load(os.path.join(self.model_dir, self.cfg.emo_matrix))
            self.emo_matrix = emo_matrix.to(self.device)
            self.emo_num = list(self.cfg.emo_num)
            spk_matrix = torch.load(os.path.join(self.model_dir, self.cfg.spk_matrix))
            self.spk_matrix = spk_matrix.to(self.device)
            self.emo_matrix = torch.split(self.emo_matrix, self.emo_num)
            self.spk_matrix = torch.split(self.spk_matrix, self.emo_num)
        mel_fn_args = {'n_fft': self.cfg.s2mel['preprocess_params']['spect_params']['n_fft'], 'win_size': self.cfg.s2mel['preprocess_params']['spect_params']['win_length'], 'hop_size': self.cfg.s2mel['preprocess_params']['spect_params']['hop_length'], 'num_mels': self.cfg.s2mel['preprocess_params']['spect_params']['n_mels'], 'sampling_rate': self.cfg.s2mel['preprocess_params']['sr'], 'fmin': self.cfg.s2mel['preprocess_params']['spect_params'].get('fmin', 0), 'fmax': None if self.cfg.s2mel['preprocess_params']['spect_params'].get('fmax', 'None') == 'None' else 8000, 'center': False}
        self.mel_fn = lambda x: mel_spectrogram(x, **mel_fn_args)
        self.cache_spk_cond = None
        self.cache_s2mel_style = None
        self.cache_s2mel_prompt = None
        self.cache_spk_audio_prompt = None
        self.cache_emo_cond = None
        self.cache_emo_audio_prompt = None
        self.cache_mel = None
        self.gr_progress = None
        self.model_version = self.cfg.version if hasattr(self.cfg, 'version') else None

    @torch.no_grad()
    def get_emb(self, input_features, attention_mask):
        vq_emb = self.semantic_model(input_features=input_features, attention_mask=attention_mask, output_hidden_states=True)
        feat = vq_emb.hidden_states[17]
        feat = (feat - self.semantic_mean) / self.semantic_std
        return feat

    def _load_and_cut_audio(self, audio_path, max_audio_length_seconds, verbose=False, sr=None):
        if not sr:
            audio, sr = librosa.load(audio_path)
        else:
            audio, _ = librosa.load(audio_path, sr=sr)
        audio = torch.tensor(audio).unsqueeze(0)
        max_audio_samples = int(max_audio_length_seconds * sr)
        if audio.shape[1] > max_audio_samples:
            if verbose:
                print(f'Audio too long ({audio.shape[1]} samples), truncating to {max_audio_samples} samples')
            audio = audio[:, :max_audio_samples]
        return (audio, sr)

def find_most_similar_cosine(query_vector, matrix):
    query_vector = query_vector.float()
    matrix = matrix.float()
    similarities = F.cosine_similarity(query_vector, matrix, dim=1)
    most_similar_index = torch.argmax(similarities)
    return most_similar_index
