# Compile numerical investigation — SM89

Compilation success is not a numerical audit pass. The same-input BF16 policy
remains `atol=0.01, rtol=0.01`, evaluated elementwise in FP64. No tolerance is
relaxed for compiler-generated kernels.

## Observed failure

The real-model B1 trial
[`compile_bf16_b1_casts.json`](../../reports/sm89/stage2/baselines/compile_bf16_b1_casts.json)
compiled all four components without compiler errors, with
`emulate_precision_casts=True`, but failed the numerical gate:

| Component | Mismatched elements, summed across returned tensors |
| --- | ---: |
| Target | 8,770 |
| Draft | 4,465 |
| CFM | 160 |
| Vocoder | 867 |

This is an experimental result, not an accepted accelerated reference. The
earlier [`compile_bf16_b1.json`](../../reports/sm89/stage2/baselines/compile_bf16_b1.json)
also failed Draft/CFM and could not compile Target because of Transformers'
legacy-cache deprecation logger. Its Vocoder pass does not contradict the later
failure: upstream generated trajectories changed the real mel input. Comparisons
between options must replay identical boundary inputs, not compare mismatch
counts from different trajectories. Published hardware reports should retain
both failures and identify any later trial separately.

## A concrete BF16 rounding hazard

The installed PyTorch `2.8.0+cu128` source contains:

- `torch/_inductor/fx_passes/post_grad.py:1357–1376`:
  `should_prefer_unfused_addmm` / `unfuse_bias_add_to_pointwise` rewrite a GPU
  `aten.addmm(bias, x, weight)` into `x @ weight + bias` when downstream users
  are pointwise. The pass is under `config.pattern_matcher` at line 130.
- `torch/_inductor/config.py:669–679`: `emulate_precision_casts` preserves
  low-precision downcast/upcast boundaries during fusion; it is not a guarantee
  that every algebraic rewrite preserves eager rounding.

These were inspected locally under
`.venv/lib/python3.11/site-packages/torch/`; no installed source was modified.
In the generated Draft module
`.cache/torchinductor/jh/cjhek7jo6ucnf7seyycf2qkh5b5wxz7dismmj6pzjl6kbscboaud.py`,
18 original BF16 `aten.addmm` sites use `extern_kernels.mm` with BF16 output
buffers (for example lines 789–790). A following pointwise kernel promotes that
rounded result and the bias to FP32 and adds them (lines 181–190). The final
FP32 logits projection remains `extern_kernels.addmm` (line 1023).

The rewrite is not numerically interchangeable with fused BF16 `F.linear`.
A one-element CPU counterexample uses BF16 `x=weight=16.125`, `bias=-260`:

```text
F.linear(x, weight, bias).float() = 0.015625
(x @ weight.T).float() + bias.float() = 0
```

The extra BF16 rounding before bias loses the residual and violates even the
unchanged `0.01/0.01` gate. The executable regression is
[`test_quantization_contract.py`](../tests/test_quantization_contract.py),
`test_bf16_bias_unfusion_is_a_documented_rounding_pitfall`. It documents the
hazard; it is deliberately not an Inductor or GPU model pass claim.

## Accuracy-first follow-up

The B1 real-input trial with this rewrite disabled is retained in
`compile_bf16_b1_no_rewrite.json`: Draft and Vocoder passed their captured boundary
comparisons, Target had 2,671 and CFM 122 out-of-tolerance elements. End-to-end
all-first-PCM/complete-EOS medians were 293.54/823.38 ms (three measured waves).
Inputs downstream of AR are not identical to earlier trials; these counts are
not an apples-to-apples improvement percentage. The independently selectable
`sm89_compile_bf16_draft_vocoder.json` leaves Target/CFM eager and is a separate
candidate to measure, not an automatic acceptance of all compiled components.

Disable `pattern_matcher` while retaining `emulate_precision_casts=True`; the
deployment option `compile_pattern_matcher` makes the former independently
switchable. This targets the observed rewrite without copying the model or
changing its arithmetic specification. It still requires same-input GPU audit:
normalization reductions, other fusions and convolution algorithms can also
change numerical results. Splitting compilation by whole layers alone does not
prevent an addmm rewrite inside each layer.

The completed [`compile_bf16_b1_no_rewrite.json`](../../reports/sm89/stage2/baselines/compile_bf16_b1_no_rewrite.json)
trial records `pattern_matcher=False` and `emulate_precision_casts=True`.
Draft and Vocoder pass their sampled same-input comparisons, but Target and CFM
still fail, so the overall numerical gate remains false. These component results
apply only to this trial's captured inputs; changed AR trajectories prevent a
cross-trial error-rate comparison. The
[stage-two index](../../reports/sm89/stage2/README.md) keeps all three trials and
their execution-coverage limitations.

The current audit clones compiled outputs before invoking eager. Inspection of
the four boundaries found no input-mutation/cache-alias explanation for these
failures: Target creates a fresh cache container for tuple inputs, Draft reads
its packed context, and acoustic calls do not advance request state. This is a
static finding, not an exhaustive alias proof; a diagnostic replay should also
check input tensors before/after each call if mutation is suspected.
