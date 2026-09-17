# -*- coding: utf-8 -*-
from functools import lru_cache
import os
import traceback
import re
from typing import List, Union, overload
import warnings
from acc_infer_clear.models.upstream.utils.common import tokenize_by_CJK_char, de_tokenized_by_CJK_char
from sentencepiece import SentencePieceProcessor

class TextNormalizer:

    def __init__(self, enable_glossary=False):
        self.zh_normalizer = None
        self.en_normalizer = None
        self.char_rep_map = {'：': ',', '；': ',', ';': ',', '，': ',', '。': '.', '！': '!', '？': '?', '\n': ' ', '·': '-', '、': ',', '...': '…', ',,,': '…', '，，，': '…', '……': '…', '“': "'", '”': "'", '"': "'", '‘': "'", '’': "'", '（': "'", '）': "'", '(': "'", ')': "'", '《': "'", '》': "'", '【': "'", '】': "'", '[': "'", ']': "'", '—': '-', '～': '-', '~': '-', '「': "'", '」': "'", ':': ','}
        self.zh_char_rep_map = {'$': '.', **self.char_rep_map}
        self.clean_pattern = re.compile('|'.join((re.escape(p) for p in self.char_rep_map.keys())))
        self.enable_glossary = enable_glossary
        self.term_glossary = dict()

    def match_email(self, email):
        pattern = '^[a-zA-Z0-9]+@[a-zA-Z0-9]+\\.[a-zA-Z]+$'
        return re.match(pattern, email) is not None
    PINYIN_TONE_PATTERN = '(?<![a-z])((?:[bpmfdtnlgkhjqxzcsryw]|[zcs]h)?(?:[aeiouüv]|[ae]i|u[aio]|ao|ou|i[aue]|[uüv]e|[uvü]ang?|uai|[aeiuv]n|[aeio]ng|ia[no]|i[ao]ng)|ng|er)([1-5])'
    '\n    匹配拼音声调格式：pinyin+数字，声调1-5，5表示轻声\n    例如：xuan4, jve2, ying1, zhong4, shang5\n    不匹配：beta1, voice2\n    '
    NAME_PATTERN = '[\\u4e00-\\u9fff]+(?:[-·—][\\u4e00-\\u9fff]+){1,2}'
    '\n    匹配人名，格式：中文·中文，中文·中文-中文\n    例如：克里斯托弗·诺兰，约瑟夫·高登-莱维特\n    '
    TECH_TERM_PATTERN = '[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+'
    '\n    匹配技术术语，格式：字母开头+(字母或数字)*+(-字母或数字)+\n    例如：GPT-5-nano, F5-TTS, Fish-Speech, GPT-5, CosyVoice-2\n    必须以字母开头，避免匹配纯数字（如电话号码 135-4567-8900）\n    用于保护连字符结构，防止中文normalizer将连字符解析为减号（如"负五减"）\n    '
    ENGLISH_CONTRACTION_PATTERN = "(what|where|who|which|how|t?here|it|s?he|that|this)'s"

    def use_chinese(self, s):
        has_chinese = bool(re.search('[\\u4e00-\\u9fff]', s))
        has_alpha = bool(re.search('[a-zA-Z]', s))
        is_email = self.match_email(s)
        if has_chinese or not has_alpha or is_email:
            return True
        has_pinyin = bool(re.search(TextNormalizer.PINYIN_TONE_PATTERN, s, re.IGNORECASE))
        return has_pinyin

    def load(self):
        import platform
        if self.zh_normalizer is not None and self.en_normalizer is not None:
            return
        if platform.system() != 'Linux':
            from wetext import Normalizer
            self.zh_normalizer = Normalizer(remove_erhua=False, lang='zh', operator='tn')
            self.en_normalizer = Normalizer(lang='en', operator='tn')
        else:
            from tn.chinese.normalizer import Normalizer as NormalizerZh
            from tn.english.normalizer import Normalizer as NormalizerEn
            from pathlib import Path
            import shutil
            cache_dir = os.path.join(os.environ['ACC_CLEAR_CACHE'], 'text_normalizer')
            os.makedirs(cache_dir, exist_ok=True)
            supplied = Path(__file__).parent / 'tagger_cache'
            for fst in supplied.glob('*.fst'):
                destination = Path(cache_dir) / fst.name
                if not destination.exists():
                    shutil.copyfile(fst, destination)
            self.zh_normalizer = NormalizerZh(cache_dir=cache_dir, remove_interjections=False, remove_erhua=False, overwrite_cache=False)
            self.en_normalizer = NormalizerEn(overwrite_cache=False)
    G2P_PRONUNCIATION_ANNOTATION_PATTERN = re.compile('<([^|>\\n]+)\\|([^>\\n]+)>')

    def _protect_pronunciation_annotations(self, text: str):
        """
        在 normalize 之前调用：将 <字|读音> 标注替换为纯字母占位符，
        防止 normalizer 把标注内的数字/符号展开（如 XING2 -> XING二）。
        返回 (替换后文本, 占位符字典)。
        """
        placeholders = {}

        def _idx_to_alpha(n):
            s = ''
            while True:
                s = chr(ord('a') + n % 26) + s
                n = n // 26 - 1
                if n < 0:
                    break
            return s

        def _replacer(m):
            tag = _idx_to_alpha(len(placeholders))
            key = f'PRONPLACEHOLDER{tag}PRONPLACEHOLDER'
            placeholders[key] = m.group(0)
            return key
        text = self.G2P_PRONUNCIATION_ANNOTATION_PATTERN.sub(_replacer, text)
        return (text, placeholders)

    @staticmethod
    def _restore_pronunciation_annotations(text: str, placeholders: dict) -> str:
        """在 normalize 之后调用：将占位符还原为原始 <字|读音> 标注。"""
        for key, val in placeholders.items():
            text = text.replace(key, val)
        return text

    def normalize(self, text: str) -> str:
        if not self.zh_normalizer or not self.en_normalizer:
            print('Error, text normalizer is not initialized !!!')
            return ''
        text, _pron_placeholders = self._protect_pronunciation_annotations(text)
        if self.use_chinese(text):
            text = re.sub(TextNormalizer.ENGLISH_CONTRACTION_PATTERN, '\\1 is', text, flags=re.IGNORECASE)
            if self.enable_glossary:
                text = self.apply_glossary_terms(text, lang='zh')
            replaced_text, tech_list = self.save_tech_terms(text.rstrip())
            replaced_text, pinyin_list = self.save_pinyin_tones(replaced_text)
            replaced_text, original_name_list = self.save_names(replaced_text)
            try:
                result = self.zh_normalizer.normalize(replaced_text)
            except Exception:
                result = ''
                print(traceback.format_exc())
            result = self.restore_names(result, original_name_list)
            result = self.restore_pinyin_tones(result, pinyin_list)
            result = self.restore_tech_terms(result, tech_list)
            pattern = re.compile('|'.join((re.escape(p) for p in self.zh_char_rep_map.keys())))
            result = pattern.sub(lambda x: self.zh_char_rep_map[x.group()], result)
        else:
            try:
                text = re.sub(TextNormalizer.ENGLISH_CONTRACTION_PATTERN, '\\1 is', text, flags=re.IGNORECASE)
                if self.enable_glossary:
                    text = self.apply_glossary_terms(text, lang='en')
                replaced_text, tech_list = self.save_tech_terms(text)
                result = self.en_normalizer.normalize(replaced_text)
                result = self.restore_tech_terms(result, tech_list)
            except Exception:
                result = text
                print(traceback.format_exc())
            pattern = re.compile('|'.join((re.escape(p) for p in self.char_rep_map.keys())))
            result = pattern.sub(lambda x: self.char_rep_map[x.group()], result)
        result = self._restore_pronunciation_annotations(result, _pron_placeholders)
        return result

    def correct_pinyin(self, pinyin: str):
        """
        将 jqx 的韵母为 u/ü 的拼音转换为 v
        如：ju -> jv , que -> qve, xün -> xvn
        """
        if pinyin[0] not in 'jqxJQX':
            return pinyin
        pattern = '([jqx])[uü](n|e|an)*(\\d)'
        repl = '\\g<1>v\\g<2>\\g<3>'
        pinyin = re.sub(pattern, repl, pinyin, flags=re.IGNORECASE)
        return pinyin.upper()

    def save_names(self, original_text):
        """
        替换人名为占位符 <n_a>、 <n_b>, ...
        例如：克里斯托弗·诺兰 -> <n_a>
        """
        name_pattern = re.compile(TextNormalizer.NAME_PATTERN, re.IGNORECASE)
        original_name_list = re.findall(name_pattern, original_text)
        if len(original_name_list) == 0:
            return (original_text, None)
        original_name_list = list(set((''.join(n) for n in original_name_list)))
        transformed_text = original_text
        for i, name in enumerate(original_name_list):
            number = chr(ord('a') + i)
            transformed_text = transformed_text.replace(name, f'<n_{number}>')
        return (transformed_text, original_name_list)

    def restore_names(self, normalized_text, original_name_list):
        """
        恢复人名为原来的文字
        例如：<n_a> -> original_name_list[0]
        """
        if not original_name_list or len(original_name_list) == 0:
            return normalized_text
        transformed_text = normalized_text
        for i, name in enumerate(original_name_list):
            number = chr(ord('a') + i)
            transformed_text = transformed_text.replace(f'<n_{number}>', name)
        return transformed_text

    def save_tech_terms(self, original_text):
        """
        保护技术术语中的连字符，防止被中文normalizer解析为减号
        策略：将术语中的连字符替换为特殊占位符<H>，数字仍可被正常处理
        例如：GPT-5-nano -> GPT<H>5<H>nano，然后 5 被转换为 五
        最终恢复为：GPT-五-nano
        """
        tech_pattern = re.compile(TextNormalizer.TECH_TERM_PATTERN)
        original_tech_list = tech_pattern.findall(original_text)
        if len(original_tech_list) == 0:
            return (original_text, None)
        original_tech_list = sorted(set(original_tech_list), key=len, reverse=True)
        transformed_text = original_text
        for term in original_tech_list:
            protected_term = term.replace('-', '<H>')
            transformed_text = transformed_text.replace(term, protected_term)
        return (transformed_text, original_tech_list)

    def restore_tech_terms(self, normalized_text, original_tech_list):
        """
        恢复技术术语中的连字符
        将占位符 <H> 恢复为连字符 -
        同时清理 normalizer 可能在占位符周围添加的多余空格
        """
        if not original_tech_list or len(original_tech_list) == 0:
            return normalized_text
        transformed_text = re.sub('\\s*<H>\\s*', '-', normalized_text)
        return transformed_text

    def apply_glossary_terms(self, text, lang='zh'):
        """
        应用术语词汇表，将专业术语替换为对应语言的读法

        Args:
            text: 待处理文本
            lang: 语言类型 "zh" 或 "en"

        Returns:
            处理后的文本

        Example:
            "M.2 NVMe SSD" -> (zh) "M 二 NVMe SSD"
            "M.2 NVMe SSD" -> (en) "M dot two NVMe SSD"
        """
        if not self.term_glossary:
            return text
        sorted_terms = sorted(self.term_glossary.keys(), key=len, reverse=True)

        @lru_cache(maxsize=42)
        def get_term_pattern(term: str):
            return re.compile(re.escape(term), re.IGNORECASE)
        transformed_text = text
        for term in sorted_terms:
            term_value = self.term_glossary[term]
            if isinstance(term_value, dict):
                replacement = term_value.get(lang, term_value.get(lang, term))
            else:
                replacement = term_value
            pattern = get_term_pattern(term)
            transformed_text = pattern.sub(replacement, transformed_text)
        return transformed_text

    def load_glossary(self, glossary_dict):
        """
        加载外部术语词汇表

        Args:
            glossary_dict: 术语词典，格式为 {"术语": {"en": "英文读法", "zh": "中文读法"}}

        Example:
            normalizer.load_glossary({
                "M.2": {"en": "M dot two", "zh": "M 二"},
                "PCIe": {"en": "PCIE", "zh": "PCIE"}
            })
        """
        if glossary_dict and isinstance(glossary_dict, dict):
            self.term_glossary.update(glossary_dict)

    def load_glossary_from_yaml(self, glossary_path):
        """
        从 YAML 文件加载术语词汇表

        Args:
            glossary_path: YAML 文件路径

        Example:
            normalizer.load_glossary_from_yaml("checkpoints/glossary.yaml")

        YAML 文件格式:
            M.2:
              en: M dot two
              zh: M 二
            NVMe: N-V-M-E  # 中英文相同读法
        """
        if glossary_path and os.path.exists(glossary_path):
            import yaml
            with open(glossary_path, 'r', encoding='utf-8') as f:
                external_glossary = yaml.safe_load(f)
                if external_glossary and isinstance(external_glossary, dict):
                    self.term_glossary = external_glossary
                    return True
        return False

    def save_glossary_to_yaml(self, glossary_path):
        """
        保存术语词汇表到 YAML 文件

        Args:
            glossary_path: YAML 文件路径
        """
        import yaml
        with open(glossary_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.term_glossary, f, allow_unicode=True, default_flow_style=False)

    def save_pinyin_tones(self, original_text):
        """
        替换拼音声调为占位符 <pinyin_a>, <pinyin_b>, ...
        例如：xuan4 -> <pinyin_a>
        """
        origin_pinyin_pattern = re.compile(TextNormalizer.PINYIN_TONE_PATTERN, re.IGNORECASE)
        original_pinyin_list = re.findall(origin_pinyin_pattern, original_text)
        if len(original_pinyin_list) == 0:
            return (original_text, None)
        original_pinyin_list = list(set((''.join(p) for p in original_pinyin_list)))
        transformed_text = original_text
        for i, pinyin in enumerate(original_pinyin_list):
            number = chr(ord('a') + i)
            transformed_text = transformed_text.replace(pinyin, f'<pinyin_{number}>')
        return (transformed_text, original_pinyin_list)

    def restore_pinyin_tones(self, normalized_text, original_pinyin_list):
        """
        恢复拼音中的音调数字（1-5）为原来的拼音
        例如：<pinyin_a> -> original_pinyin_list[0]
        """
        if not original_pinyin_list or len(original_pinyin_list) == 0:
            return normalized_text
        transformed_text = normalized_text
        for i, pinyin in enumerate(original_pinyin_list):
            number = chr(ord('a') + i)
            pinyin = self.correct_pinyin(pinyin)
            transformed_text = transformed_text.replace(f'<pinyin_{number}>', pinyin)
        return transformed_text

class TextTokenizer:

    def __init__(self, vocab_file: str, normalizer: TextNormalizer=None):
        self.vocab_file = vocab_file
        self.normalizer = normalizer
        if self.vocab_file is None:
            raise ValueError('vocab_file is None')
        if not os.path.exists(self.vocab_file):
            raise ValueError(f'vocab_file {self.vocab_file} does not exist')
        if self.normalizer:
            self.normalizer.load()
        self.sp_model = SentencePieceProcessor(model_file=self.vocab_file)
        self.pre_tokenizers = [tokenize_by_CJK_char]

    @property
    def vocab_size(self):
        return self.sp_model.GetPieceSize()

    @property
    def unk_token(self):
        return '<unk>'

    @property
    def pad_token(self):
        return None

    @property
    def bos_token(self):
        return '<s>'

    @property
    def eos_token(self):
        return '</s>'

    @property
    def pad_token_id(self):
        return -1

    @property
    def bos_token_id(self):
        return 0

    @property
    def eos_token_id(self):
        return 1

    @property
    def unk_token_id(self):
        return self.sp_model.unk_id()

    @property
    def special_tokens_map(self):
        return {'unk_token': self.unk_token, 'pad_token': self.pad_token, 'bos_token': self.bos_token, 'eos_token': self.eos_token}

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        return vocab

    @overload
    def convert_ids_to_tokens(self, ids: int) -> str:
        ...

    @overload
    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        ...

    def convert_ids_to_tokens(self, ids: Union[List[int], int]):
        return self.sp_model.IdToPiece(ids)

    def convert_tokens_to_ids(self, tokens: Union[List[str], str]) -> List[int]:
        if isinstance(tokens, str):
            tokens = [tokens]
        return [self.sp_model.PieceToId(token) for token in tokens]

    def tokenize(self, text: str) -> List[str]:
        return self.encode(text, out_type=str)

    def encode(self, text: str, **kwargs):
        if len(text) == 0:
            return []
        if len(text.strip()) == 1:
            return self.sp_model.Encode(text, out_type=kwargs.pop('out_type', int), **kwargs)
        if self.normalizer:
            text = self.normalizer.normalize(text)
        if len(self.pre_tokenizers) > 0:
            for pre_tokenizer in self.pre_tokenizers:
                text = pre_tokenizer(text)
        return self.sp_model.Encode(text, out_type=kwargs.pop('out_type', int), **kwargs)

    def batch_encode(self, texts: List[str], **kwargs):
        if self.normalizer:
            texts = [self.normalizer.normalize(text) for text in texts]
        if len(self.pre_tokenizers) > 0:
            for pre_tokenizer in self.pre_tokenizers:
                texts = [pre_tokenizer(text) for text in texts]
        return self.sp_model.Encode(texts, out_type=kwargs.pop('out_type', int), **kwargs)

    def decode(self, ids: Union[List[int], int], do_lower_case=False, **kwargs):
        if isinstance(ids, int):
            ids = [ids]
        decoded = self.sp_model.Decode(ids, out_type=kwargs.pop('out_type', str), **kwargs)
        return de_tokenized_by_CJK_char(decoded, do_lower_case=do_lower_case)

    @staticmethod
    def split_segments_by_token(tokenized_str: List[str], split_tokens: List[str], max_text_tokens_per_segment: int, quick_streaming_tokens: int=0) -> List[List[str]]:
        """
        将tokenize后的结果按特定token进一步分割
        """
        if len(tokenized_str) == 0:
            return []
        segments: List[List[str]] = []
        current_segment = []
        current_segment_tokens_len = 0
        for i in range(len(tokenized_str)):
            token = tokenized_str[i]
            current_segment.append(token)
            current_segment_tokens_len += 1
            if not (',' in split_tokens or '▁,' in split_tokens) and (',' in current_segment or '▁,' in current_segment):
                sub_segments = TextTokenizer.split_segments_by_token(current_segment, [',', '▁,'], max_text_tokens_per_segment=max_text_tokens_per_segment, quick_streaming_tokens=quick_streaming_tokens)
            elif '-' not in split_tokens and '-' in current_segment:
                sub_segments = TextTokenizer.split_segments_by_token(current_segment, ['-'], max_text_tokens_per_segment=max_text_tokens_per_segment, quick_streaming_tokens=quick_streaming_tokens)
            elif current_segment_tokens_len <= max_text_tokens_per_segment:
                if token in split_tokens and current_segment_tokens_len > 2:
                    if i < len(tokenized_str) - 1:
                        if tokenized_str[i + 1] in ["'", "▁'"]:
                            current_segment.append(tokenized_str[i + 1])
                            i += 1
                    segments.append(current_segment)
                    current_segment = []
                    current_segment_tokens_len = 0
                continue
            else:
                sub_segments = []
                for j in range(0, len(current_segment), max_text_tokens_per_segment):
                    if j + max_text_tokens_per_segment < len(current_segment):
                        sub_segments.append(current_segment[j:j + max_text_tokens_per_segment])
                    else:
                        sub_segments.append(current_segment[j:])
                warnings.warn(f'The tokens length of segment exceeds limit: {max_text_tokens_per_segment}, Tokens in segment: {current_segment}.Maybe unexpected behavior', RuntimeWarning)
            segments.extend(sub_segments)
            current_segment = []
            current_segment_tokens_len = 0
        if current_segment_tokens_len > 0:
            assert current_segment_tokens_len <= max_text_tokens_per_segment
            segments.append(current_segment)
        merged_segments = []
        total_token = 0
        for segment in segments:
            total_token += len(segment)
            if len(segment) == 0:
                continue
            if len(merged_segments) == 0:
                merged_segments.append(segment)
            elif len(merged_segments[-1]) + len(segment) <= max_text_tokens_per_segment and total_token > quick_streaming_tokens:
                merged_segments[-1] = merged_segments[-1] + segment
            elif len(merged_segments[-1]) + len(segment) <= max_text_tokens_per_segment / 2:
                merged_segments[-1] = merged_segments[-1] + segment
            else:
                merged_segments.append(segment)
        return merged_segments
    punctuation_marks_tokens = ['.', '!', '?', '▁.', '▁?', '▁...']

    def split_segments(self, tokenized: List[str], max_text_tokens_per_segment=120, quick_streaming_tokens=0) -> List[List[str]]:
        return TextTokenizer.split_segments_by_token(tokenized, self.punctuation_marks_tokens, max_text_tokens_per_segment=max_text_tokens_per_segment, quick_streaming_tokens=quick_streaming_tokens)
