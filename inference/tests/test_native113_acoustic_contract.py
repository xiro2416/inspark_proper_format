"""CPU-only engine-contract and routing checks; no TensorRT/CUDA initialization."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from acc_infer_clear.ops.tensorrt import native113 as native


class FakeCudaTensor(torch.Tensor):
    """A CPU tensor exposing only a mocked CUDA device in metadata."""

    @property
    def device(self):
        return getattr(self, "fake_device", torch.device("cuda:0"))


def cuda_metadata_tensor(shape, dtype=torch.float32):
    return torch.zeros(shape, dtype=dtype).as_subclass(FakeCudaTensor)


class EagerReference:
    def __init__(self):
        self.model = self
        self.times = (0.0, 0.5)
        self.identity = {"reference": "cpu-mock"}
        self.invocations = 0

    def parameters(self):
        return iter([SimpleNamespace(device=torch.device("cuda:0"))])

    def buffers(self):
        return iter(())

    def __call__(self, *args):
        self.invocations += 1
        return "eager-output"

    forward = __call__


def fake_trt(component, batch):
    trt = SimpleNamespace(
        float32="fp32", int64="int64", bool="bool",
        TensorIOMode=SimpleNamespace(INPUT="input", OUTPUT="output"),
        TensorLocation=SimpleNamespace(DEVICE="device"),
        TensorFormat=SimpleNamespace(LINEAR="linear"),
    )
    if component == "cfm":
        tensors = {
            "x": ((batch, 80, 310), "fp32", "input"),
            "prompt": ((batch, 80, 310), "fp32", "input"),
            "lengths": ((batch,), "int64", "input"),
            "style": ((batch, 192), "fp32", "input"),
            "mu": ((batch, 310, 512), "fp32", "input"),
            "mask": ((batch, 1, 310), "bool", "input"),
            "output": ((batch, 80, 310), "fp32", "output"),
        }
    else:
        tensors = {
            "mel": ((batch, 80, 52), "fp32", "input"),
            "pcm": ((batch, 1, 13312), "fp32", "output"),
        }
    context = MagicMock()
    context.set_tensor_address.return_value = True
    context.execute_async_v3.return_value = True
    engine = SimpleNamespace(
        tensors=tensors, num_io_tensors=len(tensors),
        get_tensor_name=lambda index: list(tensors)[index],
        get_tensor_shape=lambda name: tensors[name][0],
        get_tensor_dtype=lambda name: tensors[name][1],
        get_tensor_mode=lambda name: tensors[name][2],
        get_tensor_location=lambda name: "device",
        get_tensor_format=lambda name: "linear",
        create_execution_context=MagicMock(return_value=context),
    )

    class Logger:
        ERROR = 1

        def __init__(self, level):
            pass

    trt.Logger = Logger
    trt.Runtime = lambda logger: SimpleNamespace(deserialize_cuda_engine=lambda blob: engine)
    return trt, engine, context


class NativeAcousticContractTest(unittest.TestCase):
    def setUp(self):
        # All generated fixtures stay in the user-authorized workspace.
        self.directory = tempfile.TemporaryDirectory(
            prefix=".acoustic-contract-", dir=Path(__file__).resolve().parents[1])
        self.addCleanup(self.directory.cleanup)

    def plan(self, component, batch, **updates):
        root = Path(self.directory.name)
        blob = b"cpu-only-mock-engine"
        engine = root / (component + ".engine")
        engine.write_bytes(blob)
        values = dict(format=1, batch=batch, frames=310 if component == "cfm" else 52,
                      engine=engine.name, sha256=hashlib.sha256(blob).hexdigest())
        if component == "cfm":
            values["prompt_frames"] = 258
        else:
            values["plugins"] = ["inspark::alias_free", "inspark::deconv1d"]
        values.update(updates)
        path = root / (component + ".json")
        path.write_text(json.dumps(values))
        return path

    def construct(self, component, batch, trt=None, path=None):
        if trt is None:
            trt, engine, context = fake_trt(component, batch)
        else:
            engine = trt.Runtime(None).deserialize_cuda_engine(b"")
            context = engine.create_execution_context.return_value
        reference = EagerReference()
        constructor = native.NativeCFMSolver113 if component == "cfm" else native.NativeVocoder113

        def allocation(*shape, **kwargs):
            return SimpleNamespace(shape=shape, device=kwargs["device"], dtype=kwargs["dtype"],
                                   data_ptr=lambda: 123456)

        with patch.object(native, "_import_trt113", return_value=trt),\
                patch("acc_infer_clear.ops.tensorrt.vocoder_plugin.register"),\
                patch.object(native.torch, "empty", side_effect=allocation) as allocate:
            wrapper = constructor(path or self.plan(component, batch),
                                  reference if component == "cfm" else reference.forward)
        return wrapper, reference, engine, context, allocate

    @staticmethod
    def inputs(wrapper):
        return tuple(cuda_metadata_tensor(shape, dtype) for _, shape, dtype in wrapper.input_signature)

    def test_supported_batches_use_exact_engine_shapes_and_metadata_only_routes(self):
        for component in ("cfm", "vocoder"):
            for batch in (1, 4, 8):
                with self.subTest(component=component, batch=batch):
                    wrapper, _, engine, context, allocate = self.construct(component, batch)
                    shape = (batch, 80, 310) if component == "cfm" else (batch, 1, 13312)
                    self.assertEqual(wrapper.output.shape, shape)
                    allocate.assert_called_once()
                    engine.create_execution_context.assert_called_once()
                    args = self.inputs(wrapper)
                    with patch.object(torch.cuda, "current_stream", side_effect=AssertionError("CUDA")),\
                            patch.object(FakeCudaTensor, "data_ptr", side_effect=AssertionError("pointer")),\
                            patch.object(FakeCudaTensor, "item", side_effect=AssertionError("scalar")):
                        route = wrapper.route_for_signature(*args)
                        self.assertEqual(route, wrapper.describe_route(*args))
                    self.assertEqual(route["kind"], "tensorrt")
                    self.assertEqual(route["backend"], "tensorrt113")
                    self.assertEqual(route["batch"], batch)
                    self.assertEqual(route["engine_batch"], batch)
                    self.assertIsNone(route["reason"])
                    self.assertEqual(json.loads(json.dumps(route)), route)
                    self.assertEqual(wrapper.calls, 0)
                    self.assertEqual(wrapper.fallbacks, 0)
                    context.execute_async_v3.assert_not_called()

    def test_plan_engine_shape_dtype_mode_names_location_and_format_must_match(self):
        for component in ("cfm", "vocoder"):
            for mismatch in ("shape", "dtype", "mode", "names", "location", "format"):
                with self.subTest(component=component, mismatch=mismatch):
                    trt, engine, _ = fake_trt(component, 1)
                    name = "output" if component == "cfm" else "pcm"
                    shape, dtype, mode = engine.tensors[name]
                    if mismatch == "shape":
                        engine.tensors[name] = ((8,) + shape[1:], dtype, mode)
                    elif mismatch == "dtype":
                        engine.tensors[name] = (shape, "bf16", mode)
                    elif mismatch == "mode":
                        engine.tensors[name] = (shape, dtype, "input")
                    elif mismatch == "names":
                        engine.tensors["wrong_output"] = engine.tensors.pop(name)
                    elif mismatch == "location":
                        engine.get_tensor_location = lambda name: "host"
                    else:
                        engine.get_tensor_format = lambda name: "vectorized"
                    with self.assertRaisesRegex(ValueError, "TensorRT acoustic I/O"):
                        self.construct(component, 1, trt=trt)
                    engine.create_execution_context.assert_not_called()

    def test_hash_mismatch_rejected_before_loading_tensorrt(self):
        for component in ("cfm", "vocoder"):
            with self.subTest(component=component):
                path = self.plan(component, 1, sha256="invalid")
                constructor = native.NativeCFMSolver113 if component == "cfm" else native.NativeVocoder113
                with patch.object(native, "_import_trt113") as load,\
                        self.assertRaisesRegex(ValueError, "hash mismatch"):
                    constructor(path, EagerReference())
                load.assert_not_called()

    def test_unsupported_plan_boundaries_rejected(self):
        for component, updates in (("cfm", {"batch": 2}), ("cfm", {"prompt_frames": 257}),
                                   ("cfm", {"frames": 311}), ("vocoder", {"batch": 16}),
                                   ("vocoder", {"frames": 53})):
            with self.subTest(component=component, updates=updates):
                values = {"batch": 1, **updates}
                path = self.plan(component, **values)
                with self.assertRaisesRegex(ValueError, "Expected static TensorRT"):
                    self.construct(component, 1, path=path)

    def test_actual_fallbacks_include_batch_dtype_device_and_layout_reasons(self):
        for component in ("cfm", "vocoder"):
            for mismatch in ("shape", "dtype", "cpu", "wrong_device", "layout"):
                with self.subTest(component=component, mismatch=mismatch):
                    wrapper, reference, _, context, _ = self.construct(component, 4)
                    args = list(self.inputs(wrapper))
                    shape = tuple(args[0].shape)
                    if mismatch == "shape":
                        args[0] = cuda_metadata_tensor((1,) + shape[1:])
                    elif mismatch == "dtype":
                        args[0] = cuda_metadata_tensor(shape, torch.float16)
                    elif mismatch == "cpu":
                        args[0] = torch.zeros(shape)
                    elif mismatch == "wrong_device":
                        args[0].fake_device = torch.device("cuda:1")
                    else:
                        args[0] = cuda_metadata_tensor((shape[0], shape[2], shape[1])).transpose(1, 2)
                    reason = wrapper.input_signature[0][0] + "." + (
                        "device" if mismatch in ("cpu", "wrong_device") else mismatch)
                    route = wrapper.route_for_signature(*args)
                    self.assertEqual(route["kind"], "eager")
                    self.assertEqual(route["reason"], reason)
                    self.assertEqual(wrapper(*args), "eager-output")
                    self.assertEqual(wrapper.stats()["fallback_reasons"], {reason: 1})
                    self.assertEqual(wrapper.fallbacks, 1)
                    self.assertEqual(reference.invocations, 1)
                    context.execute_async_v3.assert_not_called()

    def test_supported_direct_enqueue_is_counted_only_after_success(self):
        for component in ("cfm", "vocoder"):
            with self.subTest(component=component):
                wrapper, _, _, context, _ = self.construct(component, 1)
                args = self.inputs(wrapper)
                with patch.object(torch.cuda, "current_stream", return_value=SimpleNamespace(cuda_stream=7)):
                    self.assertIs(wrapper(*args), wrapper.output)
                    self.assertEqual(wrapper.calls, 1)
                    self.assertEqual(wrapper.fallbacks, 0)
                    context.execute_async_v3.assert_called_once_with(7)
                    context.execute_async_v3.return_value = False
                    with self.assertRaisesRegex(RuntimeError, "enqueue failed"):
                        wrapper(*args)
                    self.assertEqual(wrapper.calls, 1)

    def test_binding_failure_never_enqueues_or_increments_success(self):
        for component in ("cfm", "vocoder"):
            with self.subTest(component=component):
                wrapper, _, _, context, _ = self.construct(component, 1)
                context.set_tensor_address.return_value = False
                with self.assertRaisesRegex(RuntimeError, "refused .* binding"):
                    wrapper(*self.inputs(wrapper))
                context.execute_async_v3.assert_not_called()
                self.assertEqual(wrapper.calls, 0)

    def test_missing_context_rejected_before_output_allocation(self):
        for component in ("cfm", "vocoder"):
            with self.subTest(component=component):
                trt, engine, _ = fake_trt(component, 1)
                engine.create_execution_context.return_value = None
                with self.assertRaisesRegex(RuntimeError, "execution context"):
                    self.construct(component, 1, trt=trt)

    def test_argument_count_and_nonfirst_input_are_validated_without_execution(self):
        wrapper, _, _, context, _ = self.construct("cfm", 1)
        args = list(self.inputs(wrapper))
        self.assertEqual(wrapper.describe_route(*args[:-1])["reason"], "argument_count")
        args[-1] = torch.zeros(args[-1].shape, dtype=torch.bool)
        route = wrapper.describe_route(*args)
        self.assertEqual(route["reason"], "mask.device")
        self.assertEqual(route["kind"], "eager")
        context.execute_async_v3.assert_not_called()

    def test_provenance_routes_freeze_declarations_but_never_claim_loader_weight_checks(self):
        for component in ("cfm", "vocoder"):
            with self.subTest(component=component):
                # A top-level label alone cannot promote a legacy plan.
                path = self.plan(component, 1, provenance_status="recorded_not_audited")
                wrapper, _, _, _, _ = self.construct(component, 1, path=path)
                self.assertEqual(wrapper.stats()["provenance"]["status"], "legacy_unverified")
                provenance = {"schema": 1, "component": component, "status": "recorded_not_audited",
                              "model_sources": [{"role": "declaration-only", "sha256": "a" * 64}]}
                path = self.plan(component, 1, provenance=provenance)
                wrapper, _, _, _, _ = self.construct(component, 1, path=path)
                route = wrapper.describe_route(*self.inputs(wrapper))
                self.assertEqual(route["provenance_status"], "recorded_not_audited")
                self.assertFalse(route["weight_identity_verified"])
                self.assertFalse(route["loader_weights_checked"])
                self.assertEqual(route["plan_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
                # Later plan edits and mutations of returned stats cannot relabel
                # an already constructed/captured runtime route.
                path.write_text("changed after preparation")
                stats = wrapper.stats()
                stats["provenance"]["build_provenance"]["status"] = "fake"
                self.assertEqual(wrapper.describe_route(*self.inputs(wrapper)), route)
                self.assertEqual(wrapper.stats()["provenance"]["build_provenance"]["status"], "recorded_not_audited")


if __name__ == "__main__":
    unittest.main()
