"""Pinned current release download and inference commands."""
from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0]=='zipvoice':
        from inspark_infer.runtime.zipvoice.cli import main as zipvoice
        return zipvoice(args[1:])
    model=next((v.split('=',1)[1] for v in args if v.startswith('--model=')),None)
    if '--model' in args and args.index('--model')+1<len(args):model=args[args.index('--model')+1]
    if args[:2]==['trt','ensure'] and model=='zipvoice':
        from inspark_infer.runtime.zipvoice.cli import main as zipvoice
        return zipvoice(['ensure',*args[2:]])
    import argparse,os
    from pathlib import Path
    p=argparse.ArgumentParser(prog='inspark')
    p.add_argument('command',choices=['fetch','infer'])
    p.add_argument('--asset-dir',type=Path,required=True)
    p.add_argument('--precision',choices=['fp8','int8_smoothquant'],default='fp8')
    p.add_argument('--batch',type=int,choices=[1,8,64,128],default=1)
    options,rest=p.parse_known_args(args)
    if options.precision=='int8_smoothquant' and options.batch==128:p.error('INT8 B128 is not published')
    from inspark_infer.api.release import fetch,materialize
    if options.command=='fetch':
        if rest:p.error('Unexpected fetch arguments')
        config,deployment=fetch(options.asset_dir,options.precision,options.batch)
        print(config);print(deployment);return 0
    from inspark_infer.api.release import registry
    info=registry();root=options.asset_dir.resolve()
    bundle=root/'download'/info['engine_prefix']
    config,deployment=materialize(bundle,root/'weights',root/'runtime',options.precision,options.batch)
    os.environ.setdefault('MPI4PY_MPIABI','openmpi')
    os.environ.setdefault('ACC_TRT_SITE',str(Path(sys.prefix)/'lib/python3.12/site-packages'))
    sys.argv=['inspark infer','--config',str(config),'--deployment',str(deployment),'--batch',str(options.batch),*rest]
    from inspark_infer.api.cli import main as infer
    infer();return 0


if __name__ == "__main__":
    raise SystemExit(main())
