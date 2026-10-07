"""Spawned CPU-only text workers; no model/GPU state crosses the boundary."""
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from copy import deepcopy
from types import SimpleNamespace

_frontend=None


def _initialize(spec):
    global _frontend
    from inspark_infer.models.indextts2.upstream.utils.front import TextNormalizer,TextTokenizer
    from inspark_infer.models.indextts2.batch_frontend import BatchFrontend
    normalizer=TextNormalizer();normalizer.load()
    normalizer.__dict__.update(spec['normalizer'])
    tokenizer=TextTokenizer(spec['vocab_file'],normalizer)
    # _text uses only tokenizer and maximum; never instantiate a GPU model.
    _frontend=BatchFrontend(SimpleNamespace(tts=SimpleNamespace(tokenizer=tokenizer),
        cfg={'data':{'max_text_tokens_per_segment':spec['maximum']}}),workers=1)


def _text(text):
    return _frontend._text({'text':text})


class TextProcessPool:
    def __init__(self,frontend,workers):
        from inspark_infer.models.indextts2.upstream.utils.front import TextTokenizer,TextNormalizer
        from inspark_infer.models.indextts2.upstream.utils.common import tokenize_by_CJK_char
        t=frontend.model.tts.tokenizer
        if (type(t) is not TextTokenizer or type(t.normalizer) is not TextNormalizer
                or t.pre_tokenizers!=[tokenize_by_CJK_char]):
            raise ValueError('CPU process pool requires the original tokenizer/normalizer')
        self.frontend=frontend
        settings=deepcopy({k:v for k,v in t.normalizer.__dict__.items()
                           if k not in ('zh_normalizer','en_normalizer')})
        spec=dict(vocab_file=t.vocab_file,normalizer=settings,
                  maximum=frontend.model.cfg['data']['max_text_tokens_per_segment'])
        self.pool=ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn'),
                                     initializer=_initialize,initargs=(spec,))

    def submit(self,fn,request):
        if getattr(fn,'__self__',None) is not self.frontend or getattr(fn,'__name__',None)!='_text':
            raise ValueError('Only frontend text work may enter CPU processes')
        return self.pool.submit(_text,request['text'])

    def shutdown(self,wait=True,**kwargs):
        self.pool.shutdown(wait=wait,**kwargs)
