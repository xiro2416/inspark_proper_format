"""Incremental raw-text boundaries. No model/GPU dependency."""
import re
import unicodedata

def spoken(c):
    return not c.isspace() and not unicodedata.category(c).startswith('P')

class Splitter:
    def __init__(self):
        self.pending = ''
        self.head_done = False
        self.closed = False

    def feed(self, delta='', final=False):
        if self.closed: raise ValueError('Input already ended')
        self.pending += delta
        out = []
        if not self.head_done:
            s = self.pending
            # Numeric prefix: complete number followed by ONE spoken character.
            match = re.match(r'^\s*[+-]?\d+(?:\.\d+)?', s)
            cut = None
            if match:
                for i in range(match.end(), len(s)):
                    # A trailing decimal point alone is not evidence of completion.
                    if s[i] == '.' and i == match.end(): break
                    if spoken(s[i]): cut = i+1; break
            elif re.match(r'^\s*[A-Za-z]', s):
                match = re.match(r'^\s*[A-Za-z]+(?=[^A-Za-z])', s)
                if match: cut = match.end()
            else:
                positions = [i for i,c in enumerate(s) if spoken(c)]
                if len(positions) >= 2: cut = positions[1]+1
            if cut is not None:
                out.append(s[:cut]); self.pending=s[cut:]; self.head_done=True
        if self.head_done:
            start = 0; counted = 0; has_speech = False
            for i,c in enumerate(self.pending):
                counted += not c.isspace(); has_speech |= spoken(c)
                if unicodedata.category(c).startswith('P') and counted >= 6 and has_speech:
                    out.append(self.pending[start:i+1]); start=i+1; counted=0; has_speech=False
            self.pending=self.pending[start:]
        if final:
            if any(spoken(c) for c in self.pending): out.append(self.pending)
            elif self.pending and out: out[-1] += self.pending
            self.pending=''; self.closed=True
        return out

def split(text):
    return Splitter().feed(text, final=True)

