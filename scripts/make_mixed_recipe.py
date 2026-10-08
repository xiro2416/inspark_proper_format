"""Compose an operator-specific recipe without changing protected roles."""
import argparse,json
from pathlib import Path
from inspark_infer.quantization.mixed import make_recipe

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for flag in ['nvfp4','fp8','weight-layouts','out']:p.add_argument('--'+flag,type=Path,required=True)
 p.add_argument('--vocoder-fp8',type=Path);a=p.parse_args()
 read=lambda path:json.loads(path.read_text())
 shapes={name:row['original_shape'] for component in ['target','draft','cfm','vocoder'] for name,row in read(a.weight_layouts/(component+'.json'))['roles'].items() if 'original_shape' in row}
 overrides={'vocoder':read(a.vocoder_fp8)} if a.vocoder_fp8 else None
 recipe=make_recipe(read(a.nvfp4),read(a.fp8),shapes,overrides)
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(recipe,indent=2)+'\n')
if __name__=='__main__':main()
