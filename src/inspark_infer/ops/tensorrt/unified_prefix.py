"""Static Target prefill/latent TRT with deployment-time Graph capture only."""
from __future__ import annotations
from inspark_infer.runtime.bundle_paths import read_json

from collections import Counter
import hashlib
import json
from pathlib import Path

import torch

from inspark_infer.build.unified_prefix_export import EXTENTS
from inspark_infer.runtime.prefix_graphs import PackedKV, PrefixGraphs


def validate_plan(plan, kind, batch, calibration_sha256, scheme):
    if kind not in EXTENTS or batch not in (1, 2, 4, 8, 16, 32, 64, 128):
        raise ValueError("Expected prefill/latent at B1/B8/B64/B128")
    if (plan.get("component"), plan.get("kind"), plan.get("batch"), plan.get("frames")) != (
            "target", kind, batch, EXTENTS[kind]):
        raise ValueError("Prefix engine kind/batch/extent mismatch")
    recipe = plan.get("quantization_recipe", {})
    if recipe.get("scheme") != scheme or recipe.get("calibration", {}).get("sha256") != calibration_sha256:
        raise ValueError("Prefix engine calibration/precision mismatch")
    if plan.get("plugins") != []:
        raise ValueError("Prefix engine must declare an empty plugin inventory")


def io_contract(kind, batch):
    length = EXTENTS[kind]
    inputs = {"x": ((batch, length, 1280), torch.float32), "keep": ((batch, length), torch.int64)}
    outputs = ({"last_logits": ((batch, 1, 8194), torch.float32),
                "packed_kv": ((24, 2, batch, 20, length, 64), torch.float32),
                "selected": ((batch, length, 6400), torch.float32),
                "final": ((batch, length, 1280), torch.float32)} if kind == "prefill"
               else {"latent": ((batch, length, 1280), torch.float32)})
    return inputs, outputs


class NativePrefixEngine:
    def __init__(self, plan_path, *, kind, batch, calibration_sha256, scheme, capture_graph=True):
        from inspark_infer.ops.tensorrt.native113 import _import_trt113
        path = Path(plan_path).resolve()
        self.plan = plan = read_json(path)
        validate_plan(plan, kind, batch, calibration_sha256, scheme)
        self.trt = trt = _import_trt113()
        if plan["trt"] != trt.__version__:
            raise ValueError("Prefix TensorRT runtime and engine versions differ")
        self.device = torch.device("cuda", torch.cuda.current_device())
        if plan.get("sm") != int("".join(map(str, torch.cuda.get_device_capability(self.device)))):
            raise ValueError("Prefix engine GPU compute capability mismatch")
        binary = Path(plan["engine"])
        binary = binary if binary.is_absolute() else path.parent / binary
        blob = binary.read_bytes()
        if hashlib.sha256(blob).hexdigest() != plan["sha256"]:
            raise ValueError("Prefix engine hash mismatch")
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(blob)
        if self.engine is None:
            raise RuntimeError("Prefix engine deserialization failed")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Prefix context creation failed")
        expected_inputs, expected_outputs = io_contract(kind, batch)
        types = {trt.float32: torch.float32, trt.int64: torch.int64}
        actual_inputs, actual_outputs = {}, {}
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            dtype = types.get(self.engine.get_tensor_dtype(name))
            table = actual_inputs if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else actual_outputs
            table[name] = (tuple(self.engine.get_tensor_shape(name)), dtype)
        if actual_inputs != expected_inputs or actual_outputs != expected_outputs:
            raise ValueError(f"Prefix engine IO contract mismatch: {actual_inputs}, {actual_outputs}")
        self.inputs = {name: torch.zeros(shape, device=self.device, dtype=dtype)
                       for name, (shape, dtype) in expected_inputs.items()}
        self.inputs["keep"].fill_(1)
        self.outputs = {name: torch.empty(shape, device=self.device, dtype=dtype)
                        for name, (shape, dtype) in expected_outputs.items()}
        for name, value in {**self.inputs, **self.outputs}.items():
            if not self.context.set_tensor_address(name, value.data_ptr()):
                raise RuntimeError(f"Cannot bind prefix tensor: {name}")
        self.graph = None
        self.enqueue_calls = 0
        self.graph_hits = 0
        self.direct_hits = 0
        self.kind, self.batch, self.extent = kind, batch, EXTENTS[kind]
        self.plan_path = str(path)
        if capture_graph:
            self.capture()

    def _enqueue(self):
        if not self.context.execute_async_v3(torch.cuda.current_stream(self.device).cuda_stream):
            raise RuntimeError("Prefix TRT enqueue failed")
        self.enqueue_calls += 1

    def capture(self):
        if self.graph is not None or self.graph_hits or self.direct_hits:
            raise RuntimeError("Capture prefix engine once, before admitting requests")
        for _ in range(3):
            self._enqueue()
        torch.cuda.current_stream(self.device).synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._enqueue()
        self.graph = graph

    def run(self, x, keep):
        actual, length = x.shape[:2]
        self.inputs["x"].zero_()
        self.inputs["keep"].zero_()
        self.inputs["x"][:actual, :length].copy_(x)
        self.inputs["keep"][:actual, :length].copy_(keep)
        self.inputs["keep"][actual:, :1].fill_(1)
        if self.graph is not None:
            self.graph.replay()
            self.graph_hits += 1
        else:
            self._enqueue()
            self.direct_hits += 1
        if self.kind == "latent":
            return self.outputs["latent"][:actual, :length]
        return (self.outputs["last_logits"][:actual], PackedKV(self.outputs["packed_kv"][:, :, :actual]),
                self.outputs["selected"][:actual, :length], self.outputs["final"][:actual, :length])

    def run_cached_prefix(self, prefixes, suffix, token_lengths):
        """Pack audited head embeddings directly into the captured latent input."""
        if self.kind != 'latent' or self.graph is None:
            raise ValueError('Cached-prefix packing requires a captured latent engine')
        actual=len(prefixes)
        lengths=[int(prefix.shape[1])+int(tokens)+1
                 for prefix,tokens in zip(prefixes,token_lengths)]
        if (not 0<actual<=self.batch or len(token_lengths)!=actual or
                suffix.shape[0]!=actual or max(lengths)>self.extent):
            raise ValueError('Cached-prefix input is outside the static latent profile')
        x,keep=self.inputs['x'],self.inputs['keep']
        if getattr(self,'vector_pack',False):
            from inspark_infer.runtime.ragged_pack import shared_rows,pack_prefix_suffix
            shared=shared_rows(prefixes,max(p.shape[1] for p in prefixes))
            if shared is not None:
                p=torch.tensor([v.shape[1] for v in prefixes],device=x.device)
                n=torch.tensor(token_lengths,device=x.device)
                packed,valid=pack_prefix_suffix(self.pack_workspace,shared,suffix,p,n)
                x[:actual].copy_(packed);keep[:actual].copy_(valid)
                x[actual:].zero_();keep[actual:].zero_();keep[actual:,:1].fill_(1)
                self.graph.replay();self.graph_hits+=1
                return self.outputs['latent'][:actual]
        x.zero_();keep.zero_()
        for index,(prefix,tokens,length) in enumerate(zip(prefixes,token_lengths,lengths)):
            end=prefix.shape[1]
            x[index,:end].copy_(prefix[0])
            x[index,end:length].copy_(suffix[index,:tokens+1])
            keep[index,:length].fill_(1)
        keep[actual:,:1].fill_(1)
        self.graph.replay()
        self.graph_hits+=1
        return self.outputs['latent'][:actual]


class NativePrefixBank:
    """Install exact-batch engines with explicit same-recipe unsupported-shape fallback."""
    def __init__(self, engine, prefill_plan, latent_plan, calibration_path, *, capture_graph=True):
        if engine.sessions:
            raise ValueError("Prepare prefix engines before request admission")
        from inspark_infer.quantization.unified import load_artifact
        artifact = load_artifact(calibration_path)
        # Fallback may serve tails. Verify it really is the declared reference,
        # rather than silently falling back to original or legacy dynamic FP8.
        transformer = engine.rt.engine.target.model.transformer
        for path, spec in artifact["role_specs"].items():
            if path.startswith("target.blocks."):
                module = transformer.get_submodule("h." + path.removeprefix("target.blocks."))
                if getattr(module, "spec", None) != spec:
                    raise ValueError(f"Prefix fallback model does not match recipe: {path}")
        digest = hashlib.sha256(Path(calibration_path).read_bytes()).hexdigest()
        self.engine = engine
        self.fallback = PrefixGraphs(engine)
        self.backends = {kind: NativePrefixEngine(path, kind=kind, batch=engine.config["max_batch"],
                            calibration_sha256=digest, scheme=artifact["scheme"], capture_graph=capture_graph)
                         for kind, path in (("prefill", prefill_plan), ("latent", latent_plan))}
        if getattr(engine,'latent_vector_pack',False):
            backend=self.backends['latent'];backend.vector_pack=True
            backend.pack_workspace=backend.inputs['x'].new_empty(backend.batch,backend.extent+1,1280)
        self.hits = {"prefill": 0, "latent": 0}
        self.tail_eager = {"prefill": 0, "latent": 0}
        self.fallback_reasons = {"prefill": Counter(), "latent": Counter()}
        self.installed = False
        self.reuse_rows=None
        self.reuse_prefill_valid=False
        self.latent_reuse_hits=0
        self.latent_reuse_misses=Counter()
        self.latent_reuse=None
        if getattr(engine,'latent_reuse_prefill_kv',False):
            from .unified_ar import StaticEngine
            if getattr(engine,'latent_suffix_plan',None):
                from .cached_latent import OnePassLatentPrefixReplay
                target=StaticEngine(engine.latent_suffix_plan,engine.config['max_batch'])
                self.latent_reuse=OnePassLatentPrefixReplay(target,self.backends['prefill'],engine.tts.gpt.final_norm)
            else:
                from .latent_prefix_reuse import LatentPrefixReplay
                target=StaticEngine(engine.unified_first_chunk.provider.target_engine.plan_path,
                                    engine.config['max_batch'])
                self.latent_reuse=LatentPrefixReplay(target,self.backends['prefill'],engine.tts.gpt.final_norm)
            self.latent_reuse.capture()

    def install(self):
        if self.engine.sessions or self.installed:
            raise ValueError("Install prefix bank exactly once before admission")
        self.engine.rt.target.prefill_body = self.prefill
        self.engine.rt.latent.body = self.latent
        self.engine.unified_prefix = self
        self.engine.prefix_graphs = self
        self.installed = True
        return self.stats()

    def run(self, x, keep, kind):
        backend = self.backends[kind]
        reason = None
        if x.ndim != 3 or keep.ndim != 2 or tuple(keep.shape) != tuple(x.shape[:2]) or x.shape[-1] != 1280:
            raise ValueError("Invalid prefix/keep tensor shape")
        if not 0 < x.shape[0] <= backend.batch:
            reason = "batch"
        elif not 0 < x.shape[1] <= backend.extent:
            reason = "extent"
        elif x.dtype != torch.float32 or keep.dtype not in (torch.int64, torch.int32, torch.bool):
            reason = "dtype"
        elif x.device != backend.device or keep.device != backend.device:
            reason = "device"
        if reason is not None:
            if kind=='prefill':self.reuse_prefill_valid=False
            self.tail_eager[kind] += 1
            self.fallback_reasons[kind][reason] += 1
            return self.fallback.body(x, keep, kind == "prefill")
        result = backend.run(x, keep)
        self.hits[kind] += 1
        if kind=='prefill':self.reuse_prefill_valid=True
        return result

    def remember_prefix_rows(self,jobs):
        self.reuse_rows=[(x.data_ptr(),tuple(x.shape),x.dtype,x.device) for x,keep in jobs]
        self.reuse_prefill_valid=False

    def reused_latents(self,prefixes,codes,token_lengths):
        if not getattr(self.engine,'latent_reuse_prefill_kv',False) or self.latent_reuse is None:
            return None
        reason=None
        if not self.reuse_prefill_valid or self.reuse_rows is None:reason='prefill_not_current'
        elif len(prefixes)!=self.latent_reuse.batch:reason='batch'
        elif [(x.data_ptr(),tuple(x.shape),x.dtype,x.device) for x in prefixes]!=self.reuse_rows:
            reason='prefix_row_ownership'
        elif any(n<1 or n>self.latent_reuse.steps*8+1 or p.shape[1]+n-1>self.latent_reuse.capacity
                 for p,n in zip(prefixes,token_lengths)):reason='extent'
        if reason is not None:
            self.latent_reuse_misses[reason]+=1
            return None
        gpt=self.engine.tts.gpt;width=self.latent_reuse.steps*8
        matrix=torch.full((len(codes),width),gpt.stop_mel_token,device=codes[0].device,dtype=codes[0].dtype)
        for i,(code,n) in enumerate(zip(codes,token_lengths)):
            # Last speech token is not consumed by the N returned latents:
            # they are BOS followed by speech[0:N-1].
            matrix[i,:n-1].copy_(code[0,:n-1])
        positions=torch.arange(1,width+1,device=matrix.device)
        suffix=gpt.mel_embedding(matrix)+gpt.mel_pos_embedding.emb(positions)[None]
        pl=torch.tensor([p.shape[1] for p in prefixes],device=matrix.device)
        sl=torch.tensor([n-1 for n in token_lengths],device=matrix.device)
        result=self.latent_reuse.run_embedded(suffix,pl,sl)
        self.latent_reuse_hits+=1;self.hits['latent']+=1
        return [result[i:i+1,:n].clone() for i,n in enumerate(token_lengths)]

    def prefill(self, x, past, keep, pos):
        if past is not None or pos is not None:
            raise ValueError("Prefix engine supports prefill only, not cached verification")
        return self.run(x, keep, "prefill")

    def latent(self, x, keep):
        return self.run(x, keep, "latent")

    def latent_cached_prefix(self, prefixes, suffix, token_lengths):
        output=self.backends['latent'].run_cached_prefix(prefixes,suffix,token_lengths)
        self.hits['latent']+=1
        return output

    def stats(self):
        return dict(backend="TensorRT static target prefill/latent", native_trtllm_executor=False,
                    prefill_keys=[[self.backends["prefill"].batch, self.backends["prefill"].extent]],
                    latent_keys=[[self.backends["latent"].batch, self.backends["latent"].extent]],
                    hits=dict(self.hits), tail_eager=dict(self.tail_eager),
                    fallback_reasons={key: dict(value) for key, value in self.fallback_reasons.items()},
                    graph_hits={kind: backend.graph_hits for kind, backend in self.backends.items()},
                    direct_hits={kind: backend.direct_hits for kind, backend in self.backends.items()},
                    enqueue_prepare_and_direct={kind: backend.enqueue_calls for kind, backend in self.backends.items()},
                    plans={kind: dict(path=backend.plan_path, sha256=backend.plan["sha256"],
                                     graph_captured=backend.graph is not None) for kind, backend in self.backends.items()},
                    latent_prefix_reuse=dict(hits=self.latent_reuse_hits,misses=dict(self.latent_reuse_misses),
                        captured=self.latent_reuse is not None,
                        target_steps=getattr(self.latent_reuse,'target_steps',0)),
                    online_capture=False, installed=self.installed)
