"""Inspect inventory, generate analytic shadow manifests, and validate artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from acc_infer_clear.planner_v2.formulas import generate_candidates
from acc_infer_clear.planner_v2.inventory import canonical_inventory, keyed, tunable
from acc_infer_clear.planner_v2.manifest import DeploymentManifest, RolePolicy, load
from acc_infer_clear.planner_v2.model import HardwareProfile


def profile(args):
    return HardwareProfile.current() if args.sm is None else HardwareProfile.synthetic(args.sm, sms=args.sms)


def emit(value, output):
    text = json.dumps(value, indent=2, sort_keys=True)
    if output:
        Path(output).write_text(text)
    else:
        print(text)


def command_inventory(args):
    p = profile(args); signatures = canonical_inventory(p, batches=tuple(args.batches))
    emit(dict(hardware=p.as_dict(), signatures={k: v.as_dict() for k, v in keyed(signatures).items()},
              performance_claim=p.rates is not None), args.output)


def command_candidates(args):
    p = profile(args); signatures = canonical_inventory(p, batches=tuple(args.batches))
    selected = [s for s in signatures if tunable(s) and (not args.role or s.role_key == args.role)]
    rows = {}
    for signature in selected:
        key = f'{signature.role_key}:{signature.shape_key}'
        rows[key] = [dict(schedule=s.as_dict(), estimate=e) for s, e in generate_candidates(p, signature, limit=args.limit)]
    emit(dict(hardware=p.as_dict(), candidates=rows, analytic_only=p.rates is None), args.output)


def command_shadow(args):
    p = profile(args)
    model_hash, source_hash, toolchain = args.model_hash, args.source_hash, {'status': args.toolchain}
    if args.current_config:
        if args.sm is not None:raise ValueError('--current-config cannot be combined with synthetic --sm')
        from acc_infer_clear.config import load as load_config
        from acc_infer_clear.planner_v2.deploy import model_hash as hash_model, source_hash as hash_source, toolchain as current_toolchain
        config=load_config(args.current_config);model_hash=hash_model(config);source_hash=hash_source(ROOT);toolchain=current_toolchain()
    if not model_hash or not source_hash:raise ValueError('Provide hashes or --current-config')
    signatures = canonical_inventory(p, batches=tuple(args.batches)); mapped = keyed(signatures)
    policies = {}
    tunable_signatures = [signature for signature in signatures if tunable(signature)]
    for role in sorted({signature.role_key for signature in tunable_signatures}):
        schedules = {}
        for signature in (s for s in tunable_signatures if s.role_key == role):
            candidates = generate_candidates(p, signature, limit=args.limit)
            if not candidates:
                raise RuntimeError(f'No legal analytic candidate for {role}:{signature.shape_key}')
            schedules[signature.shape_key] = candidates[0][0]
        policies[role] = RolePolicy(
            backend='explicit', global_layout='mk_nk', schedules=schedules,
            legacy_exception=True,
            exception_reason='Analytic shadow has not completed target-GPU calibration',
            remove_when='Offline component and end-to-end gates pass at <=5% regression',
        )
    manifest = DeploymentManifest(
        hardware=p, model_hash=model_hash, source_hash=source_hash,
        toolchain=toolchain, policies=policies, signatures=mapped,
        calibration=dict(kind='analytic-shadow', budget_seconds=7200,
                         performance_claim=False, regression_limit=.05), status='shadow',
    )
    if not args.output:
        emit(manifest.as_dict(), None)
    else:
        manifest.write(args.output)
        print(json.dumps(dict(output=str(Path(args.output).resolve()), manifest_hash=manifest.manifest_hash,
                              roles=len(policies), signatures=len(mapped)), indent=2))


def command_validate(args):
    manifest = load(args.manifest)
    print(json.dumps(dict(valid=True, status=manifest.status, manifest_hash=manifest.manifest_hash,
                          roles=len(manifest.policies), signatures=len(manifest.signatures)), indent=2))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(required=True)
    def common(command):
        command.add_argument('--sm', type=int, choices=(80, 86, 89, 90, 120))
        command.add_argument('--sms', type=int, default=80)
        command.add_argument('--batches', type=int, nargs='+', default=[1,2,3,4,5,6,7,8,16,32])
        command.add_argument('--output')
    p = sub.add_parser('inventory'); common(p); p.set_defaults(fn=command_inventory)
    p = sub.add_parser('candidates'); common(p); p.add_argument('--role'); p.add_argument('--limit', type=int, default=8); p.set_defaults(fn=command_candidates)
    p = sub.add_parser('shadow'); common(p); p.add_argument('--limit', type=int, default=8); p.add_argument('--model-hash'); p.add_argument('--source-hash'); p.add_argument('--toolchain', default='unbound-shadow'); p.add_argument('--current-config',type=Path); p.set_defaults(fn=command_shadow)
    p = sub.add_parser('validate'); p.add_argument('manifest'); p.set_defaults(fn=command_validate)
    args = parser.parse_args(); args.fn(args)


if __name__ == '__main__': main()
