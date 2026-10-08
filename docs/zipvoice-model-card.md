---
library_name: pytorch
pipeline_tag: text-to-speech
license: apache-2.0
language:
- zh
- en
tags:
- zipvoice
- smoothquant
- int8
- tensorrt
---

# ZipVoice A_1007

This private repository stores the unquantized and INT8 SmoothQuant ZipVoice-Distill weights, source ONNX graphs, Vocos assets and validated TensorRT engine bundles. Code and validation reports are published separately in [xiro2416/inspark_proper_format](https://github.com/xiro2416/inspark_proper_format), under the ZipVoice layout. This model is separate from the IndexTTS2 release in that code repository.

The INT8 recipe uses the original SmoothQuant alpha 0.5 scales, with the first four layers floating and the last twelve layers quantized (1:3). Migration does not recalibrate or alter the source weights. There is no FP8 release in this repository.

Each complete model batch, 1, 2, 4, 8, 16, 32 and 64, has its own FM, text and Vocos engine. A batch is not split into smaller model calls. Engine profiles support total mel frames 600 / 760 / 920 and padded text tokens 52 / 78 / 141 (minimum / optimum / maximum). A single engine per component covers its dynamic profile; these are not three independently fixed frame engines. Shape and stage specific kernel choices and independent length measurements are recorded in the code repository.

The reference audio contains 375 mel frames from a continuous four-second VAD segment. Reference text is the ASR transcription of the actual segment. Reference frames are included in the total frame count. The raw generated waveform is approximately 2.389333, 4.096 or 5.802667 seconds at the three frame endpoints; original silence trimming and tail pause processing determine final WAV duration. Full text must naturally fit the duration and token profile. The inference entry does not implicitly crop the text or reference audio.

Engine compatibility is restricted to the recorded RTX 4090 SM89 with 49140 MiB memory, TensorRT 11.3.0.99 and the pinned ZipVoice Python environment. GPU1 was used at a configured and enforced 400 W power limit. Floating projection and attention paths may use authorized TF32, depending on the retained batch route. Exact engine hashes, selected plugin package, application scheduling and runtime source hashes are bound in each bundle manifest. Source weights can be used independently of those GPU-specific binaries.

## Files and reproducibility

- `eager/model.safetensors`: original floating model weights.
- `int8/model.safetensors` and `int8/quantization.json`: quantized weights and the original recipe.
- `config/`: original architecture and token vocabulary.
- `vocos/`: vocoder weights and configuration.
- `onnx/`: reproducible graph and external tensor data.
- `bundles/zipvoice/sm89/int8/`: engine bundles, profiles, source identities and license notices.
- `weights-manifest.json`: weight/graph file sizes and SHA256 identities.

Use the code repository's `scripts/bootstrap_zipvoice.sh`, `scripts/download_zipvoice_weights.py` and `inspark zipvoice ensure` commands with the pinned private Hub revision from `configs/hardware/sm89/zipvoice_int8_registry.json`. Downloads try hf-mirror first and use the previously authorized official Hub fallback for private assets. Engine downloads verify every binary and source hash and do not silently rebuild incompatible engines.

Latency is measured from prepared CPU conditions, request shaping and H2D to all ordered PCM. It excludes text frontend, initialization, warmup, graph capture and WAV writes. CER, UTMOS and SIM-o support path and row mapping checks; they do not establish perceptual equivalence. The validation corpus is finite. Power reports contain whole-board NVML readings and preserve sampled overshoot instead of clipping readings to the configured limit. A known idle external allocation was preserved, so these are not exclusive-device certifications.

ZipVoice and Vocos provenance and license notices accompany the weights and engine bundles. Private storage does not remove the applicable downstream license obligations.
