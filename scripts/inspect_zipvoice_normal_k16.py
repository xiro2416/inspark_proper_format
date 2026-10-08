"""Record actual normal-K16 engine differences without attributing E2E from counts."""
import collections,json
from pathlib import Path
from run_zipvoice_validation import ROOT,sha


def main():
    artifacts=ROOT/'artifacts/zipvoice/a1007/b64'
    source=artifacts/'fm-inherited';candidate=artifacts/'fm-normtf32k16'
    builds={kind:json.loads((path/'build.json').read_text()) for kind,path in [('control',source),('candidate',candidate)]}
    assert builds['control']['source_sha256']==builds['candidate']['source_sha256']
    assert builds['control']['effective_shape_profile']==builds['candidate']['effective_shape_profile']
    layers={kind:json.loads((path/'inspector.json').read_text())['Layers'] for kind,path in [('control',source),('candidate',candidate)]}
    gemms={kind:{x['Name'].split('_myl')[0]:x for x in rows if x['LayerType']=='gemm'} for kind,rows in layers.items()}
    assert len(gemms['control'])==len(gemms['candidate'])==214 and gemms['control'].keys()==gemms['candidate'].keys()
    changed_tactics=[name for name in gemms['control'] if gemms['control'][name].get('TacticName')!=gemms['candidate'][name].get('TacticName')]
    changed=[];before=ROOT/'.work/zipvoice/b64/code';after=ROOT/'.work/zipvoice/b64_normtf32k16/code'
    for path in before.glob('*.py'):
        other=after/path.name
        if path.read_text()!=other.read_text().replace('zipvoice_int8_b64_normtf32k16','zipvoice_int8_b64'):changed.append(path.name)
    assert set(changed)=={'online_branch_stats_runtime_kernel.py','online_branch_wide_aot.py','online_branch_wide_plugin.py'}
    resources=[json.loads(row) for row in (candidate/'aot-resources.jsonl').read_text().splitlines()]
    normal=[x for x in resources if x['function'] in ('normal_stats_write_aot','normal_stats_read_aot')]
    assert len(normal)==2 and all(x['rna_conversion'] and x['tf32_mma'] and not x['global_scratch_bytes'] and not x['spills'] and x['shared_bytes']<=49152 for x in normal)
    record={'status':'normal_k16_actual_engine_execution_contract_reviewed_quality_e2e_pending','engine_sha256':{k:v['engine_sha256'] for k,v in builds.items()},'build_sha256':{k:sha(p/'build.json') for k,p in [('control',source),('candidate',candidate)]},'inspector_sha256':{k:sha(p/'inspector.json') for k,p in [('control',source),('candidate',candidate)]},'same_source_graph_weights_and_profile':True,'changed_source_modules':changed,'native_gemm_tactics':{'count':214,'changed':changed_tactics},'layer_type_counts':{k:dict(collections.Counter(x['LayerType'] for x in v)) for k,v in layers.items()},'normal_aot_resources':normal,'removed_shuffle_nodes':[{'name':x['Name'],'metadata':x.get('Metadata','')} for x in layers['control'] if x['LayerType']=='Shuffle'],'attribution_limits':['Counts include views/metadata and do not establish removed physical GPU cost','Same214GEMMtactics, but normal math+Kgeometry and optimizer layout/view changes are combined; do not label a measured gain pureTF32 alone','AOT diagnosticcubins versusJIT resources differ: actualwrite146/read128registers; no spills/scratch. TensorRT consumes recordedPTX.'],'protected_float_fusion_prior_review':{'evidence':'source/b64/history/259-F32-native-TF32-fusion-probe.json','scope':'Samehardware/fullB64T760 originalprojection→tanh/value/gate, nearestTF32 andFloat32storage','decision':'No new protectedfloatingfusion change; priorwidegeometry near1.25percent localgain requires52spills, other tested configurations regress14–28percent or exceedsharedcapacity. Current214nativeGEMMtactics unchanged; revisit onlywitha newmechanism/prerequisite.'}}
    (ROOT/'reports/sm89/zipvoice/a1007/b64/history/033-normal-k16-engine-review.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'status':record['status'],'native_tactic_changes':len(changed_tactics),'normal_aot_resources':normal}))


if __name__=='__main__':main()
