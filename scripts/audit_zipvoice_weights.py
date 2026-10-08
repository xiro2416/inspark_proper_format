"""Verify every retained INT8 weight maps exactly into both FM source graphs."""
import hashlib,json
from pathlib import Path
import onnx
import numpy as np
from safetensors.numpy import load_file

ROOT=Path(__file__).resolve().parents[1]
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    weights=ROOT/'models/zipvoice/int8/model.safetensors'
    assert sha(weights)=='6dcc27eb8ba7f1f97488e10d1e9365f679b9e843d07dcba7b89629fe95214e1b'
    state=load_file(weights)
    recipe=json.loads((ROOT/'models/zipvoice/int8/quantization.json').read_text())
    modules=set(recipe['modules']);assert len(modules)==180
    result={'status':'running','int8_state_sha256':sha(weights),'graphs':{}}
    for kind in ('fm-native','fm-inherited'):
        graph=onnx.load(ROOT/f'models/zipvoice/onnx/{kind}.onnx')
        mapped={}
        for tensor in graph.graph.initializer:
            key=tensor.name.removeprefix('model.')
            if key.endswith('.weight_int8'):
                value=onnx.numpy_helper.to_array(tensor)
                assert key in state and value.dtype==np.int8
                assert value.tobytes()==state[key].tobytes(),key
                mapped[key.removesuffix('.weight_int8')]={'shape':list(value.shape),'sha256':hashlib.sha256(value.tobytes()).hexdigest()}
        assert set(mapped)==modules
        assert not any('fm_decoder.encoders.0.' in k or 'fm_decoder.encoders.1.' in k for k in mapped)
        result['graphs'][kind]={'original_int8_modules':len(mapped),'bijection_exact':True,'protected_first_four_int8_modules':0,'weights':mapped}
    result['status']='all_180_original_int8_weights_exact_in_native_and_inherited_graphs'
    (ROOT/'reports/sm89/zipvoice/a1007/weight-mapping.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result['status'])

if __name__=='__main__':main()
