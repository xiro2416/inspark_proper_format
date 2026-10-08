"""Map actual compiler tactics to the preserved INT8 module allowlist."""
import argparse,json,re,hashlib
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def scope(name):
 parts=[]
 for p in name.split('.'):
  if p.isdigit():parts[-1]+='.'+p
  else:parts.append(p)
 return '/'+('/'.join(parts))+'/'

def main():
 p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 report=json.loads((a.build/'build.json').read_text());assert report['status']=='built_unvalidated'
 assert hashlib.sha256((a.build/'engine.plan').read_bytes()).hexdigest()==report['engine_sha256']
 layers=json.loads((a.build/'inspector.json').read_text())['Layers'];config=json.loads((ROOT/'models/zipvoice/int8/quantization.json').read_text())
 def native(layer):return bool(re.search(r'(?:int8|s8|i8)',layer.get('TacticName',''),re.I))
 mappings=[]
 for name in config['modules']:
  selected=[l for l in layers if scope(name) in (l['Name'] + ' ' + l.get('Metadata',''))]
  mappings.append({'module':name,'native_int8_tactics':[{'name':l['Name'],'tactic':l.get('TacticName'),'inputs':l.get('Inputs'),'outputs':l.get('Outputs')} for l in selected if native(l)],'mapped_layers':[{'name':l['Name'],'type':l['LayerType'],'tactic':l.get('TacticName')} for l in selected]})
 native_layers=[l for l in layers if native(l)];protected=[l for l in native_layers if re.search(r'/fm_decoder/encoders\.[01]/',l['Name']+' '+l.get('Metadata',''))]
 assert not protected,'Protected first4 layers unexpectedly use INT8'
 result={'status':'compiler_precision_inspected_inference_pending','engine_sha256':report['engine_sha256'],'layers':len(layers),'types':dict(Counter(l['LayerType'] for l in layers)),'native_int8_tactic_count':len(native_layers),'native_int8_mapped_modules':sum(bool(m['native_int8_tactics']) for m in mappings),'native_int8_protected_layers':protected,'module_execution_mapping':mappings,'caveat':'A mapped QDQ layer alone is not native integer execution; missing tactics/fused foreign regions need runtime inspection. Counts do not establish audio correctness or E2E.'}
 a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='module_execution_mapping'}),flush=True)
if __name__=='__main__':main()
