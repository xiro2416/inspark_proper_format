from pathlib import Path
import pytest
import torch

from inspark_infer.guardrails.snapshots import (
    compare_regions, cpu_copy, load_bundle, save_bundle, tensor_inventory,
)


def test_snapshot_is_detached_from_producer_static_buffer(tmp_path):
    original = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    frozen = cpu_copy({"inputs": (original,), "output": original})
    original.add_(100)
    assert frozen["output"][0, 0].item() == 0
    record = save_bundle(tmp_path, 0, frozen)
    frozen["output"].zero_()
    assert load_bundle(tmp_path, record)["output"][1, 2].item() == 5
    with pytest.raises(FileExistsError):
        save_bundle(tmp_path, 0, frozen)


def test_snapshot_bf16_scalar_empty_and_tamper(tmp_path):
    value = {"bf16": torch.tensor([1, 2], dtype=torch.bfloat16),
             "scalar": torch.tensor(1), "empty": torch.empty(0)}
    record = save_bundle(tmp_path, 0, value)
    assert tensor_inventory(value) == record["tensors"]
    assert load_bundle(tmp_path, record)["bf16"].dtype == torch.bfloat16
    record["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash"):
        load_bundle(tmp_path, record)


def test_regions_do_not_hide_generated_failures_with_masked_zeros():
    expected = torch.zeros(1, 2, 10)
    actual = expected.clone(); actual[:, :, 9] = .02
    mask = torch.arange(10)[None, None] < 9
    result = compare_regions(expected, actual, "bf16", mask)
    assert result["comparisons"]["masked_prompt"]["pass_gate"]
    assert result["comparisons"]["generated"]["total_elements"] == 2
    assert not result["pass_gate"]


def test_empty_generated_region_is_not_evidence():
    result = compare_regions(torch.zeros(1, 2, 3), torch.zeros(1, 2, 3),
                             "bf16", torch.ones(1, 1, 3, dtype=torch.bool))
    assert not result["pass_gate"]
