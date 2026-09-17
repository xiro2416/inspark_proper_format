"""Device-independent speculative commit rules, shared with CPU tests."""
def accepted_prefix(decisions, remaining, eos):
    if remaining < 0:raise ValueError('negative remaining codec budget')
    count=0
    for flag,token,*_ in decisions[:remaining]:
        if not flag:return count,False
        count+=1
        if int(token)==eos:return count,True
    return count,False

