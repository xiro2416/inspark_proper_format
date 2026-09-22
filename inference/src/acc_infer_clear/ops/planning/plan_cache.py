"""Offline-only model/device/toolchain-bound plans, shared by serving workers."""
import ast,hashlib,json,os,tempfile
from dataclasses import asdict
from pathlib import Path
from acc_infer_clear.ops.planning.planner import Tile, DeviceCaps

class PlanCache(dict):
    def __init__(self,engine,group):
        import torch,triton
        root=Path(__file__).resolve().parents[2]
        sources={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted((root/'ops').rglob('*.py'))}
        sources.update({str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted((root/'quantization').rglob('*.py'))})
        weights=Path(engine.config['weights']);paths=[weights/'index_tts2/gpt.pth',weights/'index_tts2/s2mel.pth',weights/'draft_onpolicy100/model.safetensors']
        paths.extend(p for p in (weights/'index_tts2/hf_cache/bigvgan').iterdir() if p.is_file())
        assets={str(p):dict(size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in paths}
        identity=dict(group=group,device=asdict(DeviceCaps.current()),uuid=str(getattr(torch.cuda.get_device_properties(0),'uuid','unknown')),
                      torch=torch.__version__,triton=triton.__version__,cuda=torch.version.cuda,sources=sources,
                      student_sha256=engine.config['student_sha256'],base_assets_stat=assets)
        self.identity=identity;self.key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        self.path=Path(engine.config['cache'])/'kernel_plans'/(self.key+'.json')
        raw=json.loads(self.path.read_text()) if self.path.exists() else {'plans':{}}
        if raw.get('identity',identity)!=identity:raise RuntimeError('Kernel plan identity mismatch')
        super().__init__({ast.literal_eval(k):Tile(**v) for k,v in raw['plans'].items()})
        self.loaded=len(self)
    def save(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        data=dict(identity=self.identity,plans={str(k):asdict(v) for k,v in self.items()})
        with tempfile.NamedTemporaryFile(mode='w',dir=self.path.parent,delete=False) as temporary:
            json.dump(data,temporary,indent=2);name=temporary.name
        os.replace(name,self.path)
        return dict(path=str(self.path),key=self.key,loaded=self.loaded,total=len(self),online_tuning=False)
