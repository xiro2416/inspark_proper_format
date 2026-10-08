import json
import pytest
from inspark_infer.quantization.mixed import make_recipe
from inspark_infer.runtime.asset_identity import validate_component_roles

def test_operator_mapping_preserves_protected_roles_and_weight_ranks():
    nv={'scheme':'nvfp4','role_specs':{'target.linear':{'precision':'nvfp4','activation_amax':1},'cfm.conv':{'precision':'nvfp4','activation_amax':2},'cfm.protected':{'precision':'bf16'}}}
    fp={'scheme':'fp8','role_specs':{'cfm.conv':{'precision':'fp8','input_scale':.1}}}
    mixed=make_recipe(nv,fp,{'target.linear':[8,8],'cfm.conv':[8,8,5]})
    assert mixed['role_specs']['target.linear']==nv['role_specs']['target.linear']
    assert mixed['role_specs']['cfm.conv']==fp['role_specs']['cfm.conv']
    assert mixed['role_specs']['cfm.protected']==nv['role_specs']['cfm.protected']
    assert nv['scheme']=='nvfp4'

def test_reused_component_requires_exact_specs_and_declared_precision(tmp_path):
    mixed={'scheme':'nvfp4_fp8','role_specs':{'target.linear':{'precision':'nvfp4'}}};old={'scheme':'nvfp4','role_specs':mixed['role_specs']}
    root=tmp_path/'mixed.json';root.write_text(json.dumps(mixed));source=tmp_path/'target.json';source.write_text(json.dumps(old))
    plan={'precision':'nvfp4_fp8','calibration':str(root),'component_calibrations':{'target':str(source)},'component_precisions':{'target':'nvfp4'}}
    validate_component_roles(plan)
    plan['component_precisions']['target']='fp8'
    with pytest.raises(ValueError):validate_component_roles(plan)
