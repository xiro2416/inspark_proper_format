"""Isolated single-process, multi-stream B8 experiment.

This module deliberately monkey-patches only the experiment process.  Existing
runtime modules and deployment JSON files remain unchanged.
"""
from __future__ import annotations

import copy
import threading
from pathlib import Path


_SHARED_TRT_ENGINES = {}


def install_b8_only_graph_policy():
    """Capture only the shape exercised by this experiment, never B16/B32."""
    def only_b8(max_batch):
        return (8,) if max_batch >= 8 else ()

    from acc_infer_clear.runtime import graph_policy, graphs, prefix_graphs
    from acc_infer_clear.runtime.indextts2 import proposal, slot_draft, slot_target
    graph_policy.batches = only_b8
    graphs.batches = only_b8
    prefix_graphs.batches = only_b8
    proposal.batches = only_b8
    slot_draft.batches = only_b8
    slot_target.batches = only_b8


def _shared_engine(path, trt):
    """Deserialize each immutable TensorRT plan once per experiment process."""
    path = str(Path(path).resolve())
    cached = _SHARED_TRT_ENGINES.get(path)
    if cached is None:
        logger = trt.Logger(trt.Logger.ERROR)
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(Path(path).read_bytes())
        if engine is None:
            raise RuntimeError(f"Failed to deserialize {path}")
        cached = (logger, runtime, engine)
        _SHARED_TRT_ENGINES[path] = cached
    return cached[1], cached[2]


def install_shared_engine_classes():
    """Install process-local TRT wrappers with shared engines and private contexts."""
    import hashlib
    import json
    import torch
    from acc_infer_clear.ops.tensorrt import native113 as native

    if getattr(native, "_parallel_b8_experiment_installed", False):
        return

    base_target = native.NativeTargetFull113
    base_draft = native.NativeDraftFull113
    base_cfm = native.NativeCFMSolver113
    base_vocoder = native.NativeVocoder113

    class SharedTarget(base_target):
        def __init__(self, engine_path, batch, max_slots, device):
            self.trt = native._import_trt113(); self.batch = int(batch); self.device = device
            self.runtime, self.engine = _shared_engine(engine_path, self.trt)
            self.context = self.engine.create_execution_context()
            self.cache = torch.empty(24, 2, max_slots, 20, 128, 64, device=device, dtype=torch.bfloat16)
            self.mask = torch.empty(batch, 1, 8, 128, device=device, dtype=torch.bool)
            self.logits = torch.empty(batch, 8, 8194, device=device, dtype=torch.float32)
            self.selected = torch.empty(batch, 8, 6400, device=device, dtype=torch.float32)
            self.final = torch.empty(batch, 8, 1280, device=device, dtype=torch.float32)
            self.calls = 0

    class SharedDraft(base_draft):
        def __init__(self, engine_path, batch, cache, device):
            self.trt = native._import_trt113(); self.batch = int(batch); self.device = device
            self.runtime, self.engine = _shared_engine(engine_path, self.trt)
            self.context = self.engine.create_execution_context(); self.cache = cache
            self.mask = torch.empty(batch, 1, 7, 135, device=device, dtype=torch.bool)
            self.hidden = torch.empty(batch, 7, 1280, device=device, dtype=torch.float32)
            self.base = torch.empty(batch, 7, 8194, device=device, dtype=torch.float32)
            self.calls = 0

    class SharedCFM(base_cfm):
        def __init__(self, plan_path, eager):
            plan_file = Path(plan_path).resolve(); plan = json.loads(plan_file.read_text())
            if int(plan.get("format", -1)) != 1 or int(plan.get("batch", -1)) != 8 or int(plan.get("frames", -1)) != 310:
                raise ValueError("Expected static TensorRT 11.3 B8/F310 CFM plan")
            engine_path = Path(plan["engine"])
            if not engine_path.is_absolute(): engine_path = (plan_file.parent / engine_path).resolve()
            if hashlib.sha256(engine_path.read_bytes()).hexdigest() != plan.get("sha256"):
                raise ValueError("TensorRT 11.3 CFM engine hash mismatch")
            self.trt = native._import_trt113(); self.runtime, self.engine = _shared_engine(engine_path, self.trt)
            self.context = self.engine.create_execution_context(); self.eager = eager
            self.model = eager.model; self.times = eager.times; self.identity = dict(eager.identity)
            self.identity.update(backend="TensorRT 11.3 shared-engine lane", batch=8, frames=310, plan=str(plan_file))
            self.observer = None; self.calls = 0; self.fallbacks = 0
            device = next(eager.model.parameters()).device
            self.output = torch.empty(8, 80, 310, device=device, dtype=torch.float32)

    class SharedVocoder(base_vocoder):
        def __init__(self, plan_path, eager):
            from acc_infer_clear.ops.tensorrt.vocoder_plugin import register
            plan_file = Path(plan_path).resolve(); plan = json.loads(plan_file.read_text())
            if int(plan.get("format", -1)) != 1 or int(plan.get("batch", -1)) != 8 or int(plan.get("frames", -1)) != 52:
                raise ValueError("Expected static TensorRT 11.3 B8/F52 Vocoder plan")
            engine_path = Path(plan["engine"])
            if not engine_path.is_absolute(): engine_path = (plan_file.parent / engine_path).resolve()
            if hashlib.sha256(engine_path.read_bytes()).hexdigest() != plan.get("sha256"):
                raise ValueError("TensorRT 11.3 Vocoder engine hash mismatch")
            self.trt = native._import_trt113(); register()
            self.runtime, self.engine = _shared_engine(engine_path, self.trt)
            self.context = self.engine.create_execution_context(); self.eager = eager
            self.calls = 0; self.fallbacks = 0; self.plan = str(plan_file)
            self.output = torch.empty(8, 1, 13312, device="cuda", dtype=torch.float32)

    native.NativeTargetFull113 = SharedTarget
    native.NativeDraftFull113 = SharedDraft
    native.NativeCFMSolver113 = SharedCFM
    native.NativeVocoder113 = SharedVocoder
    native._parallel_b8_experiment_installed = True


class _LaneModelView:
    """Read-only model/weight view with a lane-private stream and ownership lock."""
    def __init__(self, root):
        import torch
        self._root = root
        self.config = root.config
        self.stream = torch.cuda.Stream()
        self.lock = threading.Lock()
        self.status = "ready"

    def __getattr__(self, name):
        return getattr(self._root, name)

    def _acquire(self, operation):
        if self.status == "closed": raise RuntimeError("Lane model view is closed")
        if not self.lock.acquire(blocking=False): raise RuntimeError("Lane already executing")
        self.status = operation

    def _release(self):
        self.status = "ready"; self.lock.release()

    def close(self):
        self.status = "closed"


def fork_engine(root_engine):
    """Create an Engine state lane without loading another copy of model weights."""
    import torch
    from acc_infer_clear.runtime.indextts2.runtime import Runtime
    from acc_infer_clear.runtime.engine import Engine

    lane = Engine.__new__(Engine)
    lane.torch = torch; lane.config = dict(root_engine.config)
    lane.model = _LaneModelView(root_engine.model); lane.tts = lane.model.tts
    lane.student = lane.model.student
    with torch.cuda.stream(lane.model.stream), torch.inference_mode():
        lane.rt = Runtime(lane.model)
    lane.vocoder = lane.tts.bigvgan.forward; lane.steps = 2
    lane.sessions = {}; lane.stages = []; lane.failures = []; lane.closed = False
    lane.head_graphs = None; lane.deployment_state = "raw"
    lane.overlap_acoustics = False; lane.acoustic_stream = None
    lane.head_batch_barrier = False; lane.device_round_b8 = False
    lane.device_round_bank = None; lane.device_round_batches = set()
    lane.device_round_attempts = 0; lane.device_round_successes = 0; lane.device_round_fallbacks = 0
    lane.profile_cuda = False; lane.profile_spans = []
    return lane


def make_lane_plan(root_plan, lane_index):
    """Rebuild only lane-local runtime state; shared module transforms run once."""
    plan = copy.deepcopy(root_plan)
    plan["status"] = f"parallel_b8_lane_{lane_index}"
    plan["precision"] = "fp32"
    plan["components"] = []
    plan["rnn_precision"] = "bf16"
    plan.pop("cfm_triton_fusions", None)
    plan["acoustic_kernels"] = "off"
    plan["acoustic_plan"] = None
    return plan


def memory_snapshot(torch):
    free, total = torch.cuda.mem_get_info()
    return {
        "free_mib": int(free >> 20), "total_mib": int(total >> 20),
        "torch_allocated_mib": int(torch.cuda.memory_allocated() >> 20),
        "torch_reserved_mib": int(torch.cuda.memory_reserved() >> 20),
    }
