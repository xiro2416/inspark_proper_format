# SM89 Index publication

`assets-publication.json` records remote hash verification of the private runtime closure. The immutable download pin is `configs/hardware/sm89/indextts/assets.json`.

All eight selected batches and internal B48 AR engines retain the original weights, SmoothQuant calibration and build-input bindings. Runtime-only distribution excludes ONNX rebuild inputs. Public reports contain aggregate telemetry; detailed request histories remain private.

Deployment and hash-checked download commands: [deployment README](../README.md). Readiness scheduling: [B64 results](../ready_pipeline/RESULTS.md).

After download, sequential GPU1 deserialization/context checks can be reproduced with:

```bash
source deployment/multibatch/env.sh
.venv-native/bin/python deployment/publication/validate_engines.py --out outputs/publication/engine-load.json
```

This check creates each context in turn. It does not enqueue model computation; the readiness runner separately checks full first-PCM computation and request lifecycle.

[Publication verification](verification.json): fresh network cache and all engine hashes checked; all50 contexts created sequentially on GPU1; baseline/C frozen replay, partial1/15/17, terminal EOS handoff and noncontiguous64→48 state continuation passed. The unchanged weight archive was manifest-verified and its loader containers rebuilt.
