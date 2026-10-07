"""Current first-chunk deployment and plain FP32 semantic reference."""
from pathlib import Path
from inspark_infer.runtime.bundle_paths import read_json

def validate(plan):
    if plan.get('schema')==9:
        from inspark_infer.runtime.unified_deployment import validate as current
        return current(plan)
    if plan!={'schema':1,'status':'fp32_reference','precision':'fp32','compute_backend':'eager'}:
        raise ValueError('Historical optimization routes are removed; choose a current release deployment')
    return plan

def load(path):
    value=validate(read_json(path))
    if value['schema']==9:
        from inspark_infer.runtime.unified_deployment import load as current
        return current(path)
    return value

def prepare(engine,plan):
    plan=validate(plan)
    if plan['schema']==9:
        from inspark_infer.runtime.unified_deployment import prepare as current
        return current(engine,plan)
    if engine.sessions or engine.deployment_state!='raw':raise ValueError('Prepare a fresh engine before admission')
    engine.deployment_state='ready'
    return dict(requested=plan,resolved_precision='fp32',compute_backend='eager',online_learning=False)
