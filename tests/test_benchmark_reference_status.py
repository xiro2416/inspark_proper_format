import importlib.util
from pathlib import Path


def test_reference_benchmark_execution_and_numerical_exit_codes_are_distinct():
    path=Path(__file__).resolve().parents[1]/'benchmarks/benchmark_reference.py'
    spec=importlib.util.spec_from_file_location('benchmark_reference_status',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert module.report_exit_code(dict(execution_pass=False,numerical_pass=False))==1
    assert module.report_exit_code(dict(execution_pass=True,numerical_pass=False))==2
    assert module.report_exit_code(dict(execution_pass=True,numerical_pass=True))==0
    assert module.report_exit_code(dict(execution_pass=True,numerical_pass=None))==0
