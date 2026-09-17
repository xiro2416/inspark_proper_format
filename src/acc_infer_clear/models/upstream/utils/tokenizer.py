import base64
import os
from functools import lru_cache
from typing import Optional
import torch
from transformers import AutoTokenizer
from whisper.tokenizer import Tokenizer
import tiktoken
LANGUAGES = {'en': 'english', 'zh': 'chinese', 'de': 'german', 'es': 'spanish', 'ru': 'russian', 'ko': 'korean', 'fr': 'french', 'ja': 'japanese', 'pt': 'portuguese', 'tr': 'turkish', 'pl': 'polish', 'ca': 'catalan', 'nl': 'dutch', 'ar': 'arabic', 'sv': 'swedish', 'it': 'italian', 'id': 'indonesian', 'hi': 'hindi', 'fi': 'finnish', 'vi': 'vietnamese', 'he': 'hebrew', 'uk': 'ukrainian', 'el': 'greek', 'ms': 'malay', 'cs': 'czech', 'ro': 'romanian', 'da': 'danish', 'hu': 'hungarian', 'ta': 'tamil', 'no': 'norwegian', 'th': 'thai', 'ur': 'urdu', 'hr': 'croatian', 'bg': 'bulgarian', 'lt': 'lithuanian', 'la': 'latin', 'mi': 'maori', 'ml': 'malayalam', 'cy': 'welsh', 'sk': 'slovak', 'te': 'telugu', 'fa': 'persian', 'lv': 'latvian', 'bn': 'bengali', 'sr': 'serbian', 'az': 'azerbaijani', 'sl': 'slovenian', 'kn': 'kannada', 'et': 'estonian', 'mk': 'macedonian', 'br': 'breton', 'eu': 'basque', 'is': 'icelandic', 'hy': 'armenian', 'ne': 'nepali', 'mn': 'mongolian', 'bs': 'bosnian', 'kk': 'kazakh', 'sq': 'albanian', 'sw': 'swahili', 'gl': 'galician', 'mr': 'marathi', 'pa': 'punjabi', 'si': 'sinhala', 'km': 'khmer', 'sn': 'shona', 'yo': 'yoruba', 'so': 'somali', 'af': 'afrikaans', 'oc': 'occitan', 'ka': 'georgian', 'be': 'belarusian', 'tg': 'tajik', 'sd': 'sindhi', 'gu': 'gujarati', 'am': 'amharic', 'yi': 'yiddish', 'lo': 'lao', 'uz': 'uzbek', 'fo': 'faroese', 'ht': 'haitian creole', 'ps': 'pashto', 'tk': 'turkmen', 'nn': 'nynorsk', 'mt': 'maltese', 'sa': 'sanskrit', 'lb': 'luxembourgish', 'my': 'myanmar', 'bo': 'tibetan', 'tl': 'tagalog', 'mg': 'malagasy', 'as': 'assamese', 'tt': 'tatar', 'haw': 'hawaiian', 'ln': 'lingala', 'ha': 'hausa', 'ba': 'bashkir', 'jw': 'javanese', 'su': 'sundanese', 'yue': 'cantonese', 'minnan': 'minnan', 'wuyu': 'wuyu', 'dialect': 'dialect', 'zh/en': 'zh/en', 'en/zh': 'en/zh', 'common': 'common'}
LANGUAGE_DICT = {lang: index for index, lang in enumerate(LANGUAGES.keys())}
TO_LANGUAGE_CODE = {**{language: code for code, language in LANGUAGES.items()}, 'burmese': 'my', 'valencian': 'ca', 'flemish': 'nl', 'haitian': 'ht', 'letzeburgesch': 'lb', 'pushto': 'ps', 'panjabi': 'pa', 'moldavian': 'ro', 'moldovan': 'ro', 'sinhalese': 'si', 'castilian': 'es', 'mandarin': 'zh'}
AUDIO_EVENT = {'ASR': 'ASR', 'AED': 'AED', 'SER': 'SER', 'Speech': 'Speech', '/Speech': '/Speech', 'BGM': 'BGM', '/BGM': '/BGM', 'Laughter': 'Laughter', '/Laughter': '/Laughter', 'Applause': 'Applause', '/Applause': '/Applause'}
EMOTION = {'HAPPY': 'HAPPY', 'SAD': 'SAD', 'ANGRY': 'ANGRY', 'NEUTRAL': 'NEUTRAL'}
TTS_Vocal_Token = {'TTS/B': 'TTS/B', 'TTS/O': 'TTS/O', 'TTS/Q': 'TTS/Q', 'TTS/A': 'TTS/A', 'TTS/CO': 'TTS/CO', 'TTS/CL': 'TTS/CL', 'TTS/H': 'TTS/H', **{f'TTS/SP{i:02d}': f'TTS/SP{i:02d}' for i in range(1, 14)}}

def lang_to_token(lang):
    lang = lang.lower()
    if lang not in LANGUAGE_DICT:
        lang = 'common'
    return LANGUAGE_DICT[lang]

@lru_cache(maxsize=None)
def get_encoding(name: str='gpt2', num_languages: int=99, model_dir: str='checkpoints'):
    vocab_path = os.path.join(model_dir, f'{name}.tiktoken')
    ranks = {base64.b64decode(token): int(rank) for token, rank in (line.split() for line in open(vocab_path) if line)}
    n_vocab = len(ranks)
    special_tokens = {}
    specials = ['<|endoftext|>', '<|startoftranscript|>', *[f'<|{lang}|>' for lang in list(LANGUAGES.keys())[:num_languages]], *[f'<|{audio_event}|>' for audio_event in list(AUDIO_EVENT.keys())], *[f'<|{emotion}|>' for emotion in list(EMOTION.keys())], '<|translate|>', '<|transcribe|>', '<|startoflm|>', '<|startofprev|>', '<|nospeech|>', '<|notimestamps|>', *[f'<|SPECIAL_TOKEN_{i}|>' for i in range(1, 31)], *[f'<|{tts}|>' for tts in list(TTS_Vocal_Token.keys())], *[f'<|{i * 0.02:.2f}|>' for i in range(1501)]]
    for token in specials:
        special_tokens[token] = n_vocab
        n_vocab += 1
    return tiktoken.Encoding(name=os.path.basename(vocab_path), explicit_n_vocab=n_vocab, pat_str="'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+", mergeable_ranks=ranks, special_tokens=special_tokens)

class WhisperTokenizer(Tokenizer):
    """
    Whisper tokenizer 没有提供 tokenize, convert_tokens_to_ids, convert_ids_to_tokens 函数
    如果使用 encode 将 str 转为 list[int] 再单独 decode 每个 token 会丢失上下文信息，对于像日语单个字符可能需要多个token来表示
    所以无法单纯使用 token_list = [tokenizer.decode([token_id]) for token_id in token_ids] 去做 token->index 的转换
    因此这里添加了 3 个函数来做这件事
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def tokenize(self, text, max_token_comb=4):
        """
        将输入文本根据token切分开转为list[str]
        通过智能组合token避免出现不完整的乱码字符
        """
        token_ids = self.encode(text, allowed_special='all')
        tokens = []
        i = 0
        while i < len(token_ids):
            best_token = None
            best_length = 0
            for length in range(1, min(max_token_comb + 1, len(token_ids) - i + 1)):
                try:
                    candidate_ids = token_ids[i:i + length]
                    candidate_token = self.decode(candidate_ids)
                    if '�' not in candidate_token and candidate_token.strip():
                        best_token = candidate_token
                        best_length = length
                        break
                except:
                    continue
            if best_token is not None:
                tokens.append(best_token)
                i += best_length
            else:
                try:
                    single_token = self.decode([token_ids[i]])
                    tokens.append(single_token)
                except:
                    tokens.append('<UNK>')
                i += 1
        return tokens

@lru_cache(maxsize=None)
def get_tokenizer(multilingual: bool, *, num_languages: int=99, language: Optional[str]=None, task: Optional[str]=None, model_dir: str='checkpoints') -> Tokenizer:
    if language is not None:
        language = language.lower()
        if language not in LANGUAGES:
            if language in TO_LANGUAGE_CODE:
                language = TO_LANGUAGE_CODE[language]
            else:
                raise ValueError(f'Unsupported language: {language}')
    if multilingual:
        encoding_name = 'multilingual_zh_ja_yue_char_del'
        language = language or 'en'
        task = task or 'transcribe'
    else:
        encoding_name = 'gpt2'
        language = None
        task = None
    encoding = get_encoding(name=encoding_name, num_languages=num_languages, model_dir=model_dir)
    return WhisperTokenizer(encoding=encoding, num_languages=num_languages, language=language, task=task)

class QwenTokenizer:

    def __init__(self, token_path, skip_special_tokens=True):
        super().__init__()
        special_tokens = {'eos_token': '<|endoftext|>', 'pad_token': '<|endoftext|>', 'additional_special_tokens': ['<|im_start|>', '<|im_end|>', '<|endofprompt|>', '[breath]', '<strong>', '</strong>', '[noise]', '[laughter]', '[cough]', '[clucking]', '[accent]', '[quick_breath]', '<laughter>', '</laughter>', '[hissing]', '[sigh]', '[vocalized-noise]', '[lipsmack]', '[mn]'], 'nonverbalspeech38k_speech_tokens': ['[snore]', '[throatclearing]', '[crying]', '[sniff]', '[laughing]', '[coughing]', '[gasp]', '[yawn]', '<B>', '</B>']}
        self.special_tokens = special_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(token_path)
        self.tokenizer.add_special_tokens(special_tokens)
        self.skip_special_tokens = skip_special_tokens

    def encode(self, text, **kwargs):
        tokens = self.tokenizer([text], return_tensors='pt')
        tokens = tokens['input_ids'][0].cpu().tolist()
        return tokens

    def decode(self, tokens):
        tokens = torch.tensor(tokens, dtype=torch.int64)
        text = self.tokenizer.batch_decode([tokens], skip_special_tokens=self.skip_special_tokens)[0]
        return text

@lru_cache(maxsize=None)
def get_qwen_tokenizer(token_path: str, skip_special_tokens: bool) -> QwenTokenizer:
    return QwenTokenizer(token_path=token_path, skip_special_tokens=skip_special_tokens)

