# ZipVoice INT8 migration acceptance

Matched original vs integrated runner; GPU1, configured/enforced 400 W. Each row is one measured request after warmup, with 4108 MiB pre-existing idle resident memory. These are migration checks, not new optimization gains or steady-state power certification.

|Batch|Total frames|Original ms|Integrated ms|Mean W|P95 W|Sampled max W|
|---:|---:|---:|---:|---:|---:|---:|
|16|582|223.613|233.683|89.443|91.855|91.855|
|16|760|290.953|289.633|107.064|107.064|107.064|
|16|1750|898.495|894.953|146.185|194.669|194.669|
|24|582|327.042|326.058|157.352|190.261|190.261|
|24|760|452.184|450.674|109.559|156.013|186.982|
|24|1750|1354.608|1351.578|216.271|393.421|393.421|
|32|582|457.001|457.336|125.921|129.426|129.426|
|32|760|620.816|615.725|142.202|226.615|226.615|
|32|1750|1827.318|1820.149|266.366|397.902|397.902|
|64|582|1002.276|1000.014|147.498|224.976|224.976|
|64|760|1347.302|1343.037|209.518|370.529|401.837|
|64|1750|3831.336|3805.654|339.629|400.583|400.583|

All 12 representative same-text cases have bitwise-identical selected ODE state and byte-identical saved audio. All four mixed cases select the full-batch text route and produce byte-identical saved audio. Comparison rows are 0, 1 and the last row; all full-batch PCM items were delivered. The canonical TRT adapter was extracted only after verifying identical Engine class ASTs across all four original route implementations. Fresh-download acceptance is recorded separately.

42 CPU checks passed: 35 ZipVoice/layout/API/bundle/cache checks and 7 existing quantization checks. Frontend prompt_mel, token_ids and prompt_rms are bitwise-identical to the retained prepared fixture. CER/UTMOS/SIM-o were not rerun because matched saved waveform bytes are identical. Private inputs and WAV files are not distributed.

Use the pinned private HF registry and [run instructions](../../../docs/zipvoice-int8.md).

All four bundles were fetched into a new local cache from private HF revision `71ac06883ab191d0af877d8218b7a85aa0914993` and ran on GPU1 using only the integrated source and copied caller inputs. Saved row0 PCM matches the original T760 route byte-for-byte for every batch. See download-acceptance.json.
