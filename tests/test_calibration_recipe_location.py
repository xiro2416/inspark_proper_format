from copy import deepcopy
from inspark_infer.runtime.asset_identity import same_quantization_recipe

def test_relocated_recipe_retains_all_content_and_role_checks():
    a={'scheme':'nvfp4_fp8','calibration':{'path':'/source/cal.json','sha256':'abc','bytes':123},'role_specs_sha256':'roles','role_manifest':{'roles':[{'path':'draft.q','precision':'nvfp4'}]}}
    b=deepcopy(a);b['calibration']['path']='/target/cal.json'
    assert same_quantization_recipe(a,b)
    for key,value in [('sha256','different'),('bytes',124)]:
        bad=deepcopy(b);bad['calibration'][key]=value
        assert not same_quantization_recipe(a,bad)
    bad=deepcopy(b);bad['role_manifest']['roles'][0]['precision']='fp8'
    assert not same_quantization_recipe(a,bad)
    bad=deepcopy(b);bad['role_specs_sha256']='different'
    assert not same_quantization_recipe(a,bad)
    assert not same_quantization_recipe({}, {})
