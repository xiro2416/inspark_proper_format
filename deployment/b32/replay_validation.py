"""Frozen replay validation, independent of removed historical benchmark modules."""
import hashlib,json
from pathlib import Path
import torch


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def load_capture(path,calibration=None):
    frozen=torch.load(path,map_location='cpu',weights_only=True)
    if (frozen.get('schema')!=1 or frozen.get('kind')!='unified_real_head_acoustic_capture'
            or type(frozen.get('batch')) is not int or frozen.get('batch') not in (1,2,4,8,16,32,64,128) or frozen.get('scheme')!='int8_smoothquant'
            or not frozen.get('waves') or frozen.get('cfm_intervals')!=[[0.,.25],[.25,.5],[.5,.75],[.75,1.]]):
        raise ValueError('Expected authorized-batch real INT8 four-step capture')
    calibration=Path(calibration or frozen['calibration']['path'])
    if digest(calibration)!=frozen['calibration']['sha256']:raise ValueError('Capture calibration changed')
    return frozen,calibration


def validate_plan(frozen,path):
    path=Path(path);plan=json.loads(path.read_text());component=plan.get('component')
    if component not in ('cfm','vocoder'):raise ValueError('Not an acoustic plan')
    source=frozen['plans'][component]
    if type(plan.get('batch')) is not int or type(source.get('batch')) is not int:raise ValueError('Exact integer batch required')
    for key in ('format','batch','frames','sm','gpu_name','trt','kind','precision','prompt_frames'):
        if plan.get(key)!=source.get(key):raise ValueError('Replay plan interface mismatch: '+key)
    for key in ('scheme','alpha','role_specs_sha256','role_manifest'):
        if plan['quantization_recipe'].get(key)!=source['quantization_recipe'].get(key):raise ValueError('Replay quantization changed: '+key)
    if plan['quantization_recipe']['calibration']['sha256']!=source['quantization_recipe']['calibration']['sha256']:
        raise ValueError('Component calibration changed')
    for key in set(plan['provenance'])|set(source['provenance']):
        if key!='onnx_binding' and plan['provenance'].get(key)!=source['provenance'].get(key):
            raise ValueError('Replay original provenance changed: '+key)
    binding=plan['provenance']['onnx_binding'];onnx_path=Path(binding['onnx']['path'])
    if digest(onnx_path)!=binding['onnx']['sha256']:raise ValueError('Replay ONNX identity mismatch')
    for data in binding.get('external_data',[]):
        data_path=Path(data['path']);data_path=data_path if data_path.is_absolute() else onnx_path.parent/data_path
        if digest(data_path)!=data['sha256']:raise ValueError('Replay ONNX external data changed')
    binary=Path(plan['engine']);binary=binary if binary.is_absolute() else path.parent/binary
    if digest(binary)!=plan['sha256']:raise ValueError('Replay engine identity mismatch')
    return plan
