"""Exact existing44-core/8-right-frame policy; audio hop256, sample rate22050."""
CORE=44
RIGHT=8
HOP=256
RATE=22050

def head_ready(code_count,eos):
    return eos or int(code_count*1.72)>=CORE+RIGHT

def head_window(total_frames):
    if total_frames<=0:raise ValueError('No speech frames')
    return min(CORE,total_frames),min(CORE+RIGHT,total_frames),CORE+RIGHT

