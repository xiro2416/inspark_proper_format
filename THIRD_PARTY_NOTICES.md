# Source and model provenance

The necessary IndexTTS2, DSpark/PCG and audio-model definitions were extracted
from the frozen universal implementation. This release adds inference-only
runtime dispatch, explicit CUDA Graph capture, quantization and custom GPU
kernels. Training, profiling and experiment drivers are intentionally absent.

- IndexTTS2 and its derivative CFM student: original `LICENSE` (bilibili Model Use License Agreement) retained at project root. Original model assets remain an external dependency.
- DSpark / DeepSpec components: `licenses/DeepSpec.txt`.
- BigVGAN Python model and anti-aliasing modules: `licenses/BigVGAN.txt`; original copyright headers retained.
- Hugging Face model adaptations and other Apache-2.0 components: `licenses/Apache-2.0.txt`; original notices retained where present. Installed transformers is used instead of copying its large modeling/generation utility files.
- Amphion / Vocos codec components retain their original source headers. Installed Python dependencies retain their respective package licenses.

Official neural weights remain external dependencies and are fetched from their
original repositories at revisions and hashes recorded in
`configs/model_sources.json`. Source normalization and pruning do not imply
ownership of third-party model definitions.
