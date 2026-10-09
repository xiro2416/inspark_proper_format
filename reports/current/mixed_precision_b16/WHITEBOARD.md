# B16 mixed current best

GPU5/RTX6000D/SM120; profile /workspace/.codex/skills/gpu_parameters/rtx6000d-sm120.md (156SM,112MiBL2,83GiB,600W). SameDraft900/CFM800/TRT11.3.0.99/Torch2.13/CUDA13.2. Originalcalib135NVFP4/92FP8/90BF16; noL2gate/repeatedtextreuse.

Migrationcomplete:8exactB16engines allL5/FULL/aux0;CFM56MiBL2othersauto;KV80/Q8target/Q7draft/prefill48/latent80/suffix40prefix48/CFMF310P258full4steps/vocoder52wholegraph+109FP32FIR. AllSHA/staticIO/model-sourceidentitychecked, sourcecoverage restored. NativeF4counts72/12/4/72/72/72/168;CFM64F8Conv,Vocoder72F8correlation,4FP32deconv inheritedgapexplicit. ARDSparkWorker/RNNbridge+TRTcompute/GPU PCG notfullExecutor.

Selectedexistingprefixreadonly/contextviews+overlap,TargetKVwritefusion,graphburst1,lateverify10,CPU8. Headmajor/requestslot KV/RNNstate0/accepted-onlycommit/prefixlatent reuse retained. NoB128microbatch/tail/newmath. All8engines/calib unchanged.

Matched3warm10waves ABBAparallel-exploration:baselineP50108.86/109.64,selected106.77/108.54,mean109.25→107.66ms(1.46%/1.59ms). SelectedB2P95156.6outlier retained; finalexclusive5warm30waves15power completed: baseline108.64/120.38/127.34→selected105.83/117.49/119.21ms,2.587%P50gain; power219.12/411.24/417.79W,peak13.726GiB;480matchingcode/PCM/stateidentical,alloutliersretained. Screeninglarger gainnotcausalclaim. All11variants160rows code/PCM SHA+accept/round/EOSidenticalbaseline,allfallback0. Selected2realwaveprofile confirmsactualGraph/32GPUcoderows/CFM&Vocohits. Floatingaudits inmigration_float_audit.json applicableunchangedengines; CFM1.315/Voco9.518/Target10.451%sameL2reportonly.

RejectedCPU16,burst4(extraTarget/Draftwork);late12cheapcontrolslower provisional. FIR6shapes×8schedules original256/4fastest all,weighted7.882→7.882ms/equaloutputs. FP32deconv0.394ms/wave; noheavybuild for<0.4%bound. Residualfewms/TailB8late1–3rounds+statecopycost/fusionlayout costs remain,stop peruserfewmscondition notglobaloptimality.

Currentartifacts/mixed_b16/selected_deployment.json/runtime.yaml; reproduction migration_queue.py,optimize_queue.py,confirm_queue.py,profile_selected.py. Reports RESULTS.md/summary.json/actual_execution.json,history artifacts/mixed_b16/history; sourceboard preserved. GPU5released; no publication. Finalselected_final_deployment.json/status-only differencefromformalinput; formal_summary.json allhashes/state/actualrouteverified. Parentdecidespublishing; noGPUworkneeded.
