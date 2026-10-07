"""DSpark framework adapters, shared by FP8 and INT8 compute backends.

The adapters do not imply that the TensorRT-LLM executor has been deployed.
``probe_trtllm_dspark.py`` records installed-runtime and source-only results
separately. No custom GPU kernels are implemented in this package.
"""

from .adapter import DSparkModelAdapter, OfficialRNNProposal, padded_rnn_weights
from .pcg import FrameworkPCG

__all__ = ["DSparkModelAdapter", "OfficialRNNProposal", "FrameworkPCG", "padded_rnn_weights"]
