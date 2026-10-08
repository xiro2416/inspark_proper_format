"""Operator-specific low precision, with the existing protected roles intact."""
from copy import deepcopy
SCHEME='nvfp4_fp8'

def make_recipe(nvfp4,fp8,shapes,fp8_component_overrides=None):
    recipe=deepcopy(nvfp4);recipe['scheme']=SCHEME
    recipe['algorithm']='NVFP4 max PTQ for Linear; FP8 calibrated Q/DQ for Conv'
    recipe['precision_policy']='current protected BF16 roles; native GEMM NVFP4; direct convolution FP8'
    mapping={}
    for name,spec in recipe['role_specs'].items():
        if spec['precision']=='bf16':mapping[name]={'kind':'protected','precision':'bf16'};continue
        component=name.split('.')[0]
        source=(fp8_component_overrides or {}).get(component,fp8)
        if len(shapes[name])==3:
            recipe['role_specs'][name]=deepcopy(source['role_specs'][name]);mapping[name]={'kind':'convolution','precision':'fp8'}
        else:mapping[name]={'kind':'gemm','precision':'nvfp4'}
    recipe['operator_precision_map']=mapping
    return recipe
