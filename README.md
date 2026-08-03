# GLM-5.2 domain-blended MoE expert pruning

Compressing GLM-5.2 (753.86B MoE) to run on one NVIDIA GB10 Spark (128 GB unified)
by pruning routed experts, with a keep-list chosen to serve ten text domains at once.

**Status:** `mix108_maxmin` built, verified, benchmarked, and deployed on the Spark.

## Result

Final artefact: **mix108_maxmin** — keep 108 of 256 experts per layer, selected by a
max-min-fair blend of expert saliency measured across ten domains.
100.32 GiB / 107.7 GB, 324.83B params (from 753.86B). Byte-exactness verified 25/25.

All perplexity below is on **neutral held-out corpora** (`code_v2_eval` 125,461 tokens,
`general_v2_eval` 149,778 tokens) that no model in the table was calibrated on.
Ratios are relative to the unpruned IQ2_M parent.

| model | size | bpw | code PPL | ratio | general PPL | ratio |
|---|---:|---:|---:|---:|---:|---:|
| parent (IQ2_M, 256 experts) | 222.18 GiB | 2.65 | 2.1401 | 1.000 | 3.7753 | 1.000 |
| REAP-50 Q3_K_M (128 experts) | 169.30 GiB | 3.82 | 2.4217 | **1.132** | 6.5462 | 1.734 |
| REAP-50 Q2_K (128 experts) | 129 GiB | 2.91 | — | — | — | — |
| **mix108_maxmin (108 experts)** | **100.32 GiB** | **2.65** | 2.7751 | 1.297 | 6.2115 | **1.645** |
| code108 (108 experts, code-calibrated) | 100.32 GiB | 2.65 | 2.5240 | 1.179 | 10.5173 | 2.786 |
| code96 (96 experts, code-calibrated) | 90.72 GiB | 2.65 | 2.6339 | 1.231 | 12.3949 | 3.283 |

**mix108 beats the published REAP-50 Q3_K_M artefact on general-text perplexity at
59.3% of its size**, and gives up ground on code — a deliberate trade, not a regression.

## The membership control

`mix108_maxmin` and `code108` are the same size, the same bit-width, the same K, and
come from the same byte-exact slicer applied to the same parent file. The **only**
difference is the calibration corpus, and therefore which 108 experts survive. The two
keep-lists share 54.8% of their experts, so about 49 of 108 differ per layer.

| axis | code108 | mix108_maxmin | delta |
|---|---:|---:|---:|
| general PPL ratio | 2.786 | **1.645** | −41% |
| MMLU | 34.15 | **37.03** | **+2.88** (5.0σ) |
| ARC-Challenge | 30.10 | **44.48** | **+14.38** (3.7σ) |
| TruthfulQA | 26.07 | **32.93** | **+6.86** (3.0σ) |
| OOD degenerate probes | 3/5 | **0/5** | — |

This is the cleanest causal experiment in the project: at fixed capacity, corpus
choice alone moves every axis measured. code108's ARC-Challenge score of 30.10 sits
only 5.1 points above the 25% chance floor — it retains 14% of the parent's
above-chance ARC skill, against mix108's 54%.

## Benchmarks

Zero-shot loglikelihood multiple-choice scoring via `llama-perplexity --multiple-choice`
with the `ikawrakow/validation-datasets-for-llama.cpp` binaries. MMLU 13,943 tasks
(99 of 14,042 exceed context), ARC-Challenge 299, TruthfulQA 817.

| model | size | bpw | MMLU | ARC-C | TruthfulQA |
|---|---:|---:|---:|---:|---:|
| parent | 222.18 GiB | 2.65 | **46.91 ±0.42** | **60.87 ±2.83** | **39.17 ±1.71** |
| REAP-50 Q3_K_M | 169.30 GiB | 3.82 | 40.08 ±0.42 | 47.49 ±2.89 | 31.58 ±1.63 |
| REAP-50 Q2_K | 129 GiB | 2.91 | 36.08 ±0.41 | 41.81 ±2.86 | 29.50 ±1.60 |
| **mix108_maxmin** | **100.32 GiB** | **2.65** | 37.03 ±0.41 | 44.48 ±2.88 | **32.93 ±1.65** |
| code108 | 100.32 GiB | 2.65 | 34.15 ±0.40 | 30.10 ±2.66 | 26.07 ±1.54 |
| *random chance* | | | *25.00* | *25.00* | *19.86* |

Three readings:

**At a matched bit budget, mix108 sweeps.** Against REAP-50 Q2_K it wins MMLU by 0.95,
ARC by 2.68 and TruthfulQA by 3.43 — while being 22% smaller, at lower bpw, with 20
fewer experts per layer. No individual gap reaches significance; three-for-three in the
same direction is the signal.

**The MMLU deficit to Q3_K_M is a precision effect, not a selection effect.** REAP-50's
own two builds share an identical keep-list and differ only in quantization, giving a
clean slope: 3.82 → 2.91 bpw costs −4.01 MMLU, −5.69 ARC, −2.08 TruthfulQA, i.e. about
4.4 MMLU points per bpw. Extrapolated to mix108's 2.65 bpw that line predicts ~34.9
MMLU; mix108 measures 37.03. (Caveat: IQ2_M is an i-quant and Q2_K a k-quant, and
i-quants are generally better per bit, so part of the margin is quantization family.)

**mix108 leads on TruthfulQA and trails on MMLU, which is coherent.** TruthfulQA rewards
not asserting plausible falsehoods — calibration rather than stored fact. The blend
restored routing coherence; it did not restore knowledge held in the 148 deleted experts.
Above-chance retention vs parent: TruthfulQA 67.7% (REAP-50 Q3_K_M: 60.7%), MMLU 54.9%
(68.8%), ARC 54.3% (62.7%).

**Cost of pruning itself.** `parent → mix108` is fully controlled — same quantization,
same slicer, only K changes: **−9.88 MMLU, −16.39 ARC**. That shared loss dwarfs the
~3-point spread between pruned artefacts. Pruning at this depth is expensive for
knowledge no matter how the experts are chosen.

## Generation probes

13 frozen prompts (8 in-domain, 5 out-of-domain) at `--temp 0 --seed 1`. `distinct4` is
the unique-4-gram ratio; below ~0.55 indicates looping.

| model | IN distinct4 | IN degen | OOD distinct4 | OOD degen |
|---|---:|---:|---:|---:|
| parent | 0.965 | 0/8 | 0.991 | 0/5 |
| REAP-50 Q3_K_M | 0.952 | 0/8 | 0.883 | 1/5 |
| **mix108 (guarded)** | **0.963** | **0/8** | **0.996** | **0/5** |
| mix108 (plain) | 0.954 | 0/8 | 0.855 | 0/5 |
| code108 | 0.954 | 0/8 | 0.424 | 3/5 |
| code96 | 0.957 | 0/8 | 0.476 | 3/5 |
| code80 | 0.725 | 2/8 | 0.165 | 5/5 |
| random96 | 0.367 | 6/8 | 0.543 | 2/5 |

Guarded = `--repeat-penalty 1.10`. mix108's guarded OOD `distinct4` of 0.996 is
statistically indistinguishable from the unpruned parent's 0.991.

**Sampling matters more than the probe protocol suggests.** At `--temp 0` mix108's
French Revolution answer gives wrong dates and an invented term; at `--temp 0.6
--top-p 0.95`, the serving default, the same prompt returns correct dates (1789–1799),
the correct wars, and the correct monarch. Greedy decoding understates deployed quality
on a pruned model. The probe battery is retained as a conservative gate.

## Calibration corpus

1,931,616 tokens across ten English domains, built by `tools/build_corpora_v2.py`
(seed 20260728). Calibration and evaluation splits are disjoint at document level,
verified by SHA-256. Chat, instruct, maths and reasoning shards are rendered through
the model's real chat template, extracted from the GGUF.

| domain | tokens | source |
|---|---:|---|
| reasoning | 275,133 | open-r1/Mixture-of-Thoughts, `<think>` spans preserved |
| science | 214,916 | neuralwork/arxiver (natural-case markdown) |
| math | 202,826 | GSM8K train |
| books | 197,393 | Project Gutenberg, random windows past front matter |
| wiki | 191,804 | wikitext-103-raw-v1 |
| code_raw | 180,557 | llama.cpp sources |
| web | 179,709 | allenai/c4 (English) |
| news | 178,914 | CNN/DailyMail |
| chat | 157,879 | HuggingFaceH4/ultrachat_200k |
| code_instruct | 152,485 | theblackcat102/evol-codealpaca-v1 |

Because the REAP criterion is a plain sum over tokens, `saliency(A) + saliency(B)`
equals the saliency of A and B concatenated exactly. Each domain is therefore scored
once, and blends are formed post hoc at no additional GPU cost.

**Blend weights** were searched to maximise the worst-covered domain (max-min fairness):

```
code_raw 2.003 | web 1.613 | science 1.225 | code_instruct 1.074
books 0.844 | wiki 0.738 | chat / math / news / reasoning 0.626
```

The pattern is interpretable: the most *orthogonal* domains need the most weight,
because no other domain carries their experts.

**Coverage** — the fraction of a domain's saliency mass retained, relative to what that
domain's own optimal top-108 would keep, reported as the worst case over ten domains:

| keep-list | worst-case coverage |
|---|---:|
| code-only (the code108 strategy) | 0.463 |
| uniform blend | 0.720 |
| code-weighted 2× | 0.774 |
| **max-min optimised** | **0.814** |

For comparison, raising K from 96 to 128 buys only +0.051. **Corpus blending is worth
roughly 8× more than any capacity increase available within the Spark's budget.**

## Key findings

- **Code and general prose share almost no saliency ordering.** Rank correlation between
  raw-code and literary-prose expert saliency is **−0.005**; Jaccard@96 is 0.347, so they
  share only 33 of their top 96 experts. This is the mechanism behind a code-calibrated
  model's collapse on general text.
- **No expert is ever idle.** Across all ten domains, dead-expert count is 0.0 per layer.
  There is no free lunch: any prune removes actively-used capacity.
- **A union keep-list is impossible.** The union of ten domains' top-108 lists is 237.8
  experts per layer — 93% of all 256. Serving many domains requires a blend, not a union.
- **Reasoning traces route more like code than code domains route like each other.**
  reasoning × code_instruct Jaccard@96 is 0.594, against 0.552 for code_instruct ×
  code_raw. The concern that a non-thinking calibration corpus would prune thinking
  experts was not borne out; the domains code-calibration actually gutted were
  encyclopedic, journalistic and literary prose.
- **Calibration volume is not saturated but is deep into diminishing returns.** At 7/8
  of the data, per-domain Jaccard@108 against the full merge ranges 0.963 (wiki) to
  0.994 (math), still rising. Concentration predicts convergence speed: math has the
  highest Gini (0.645) and converges fastest; web the lowest (0.377) and among the
  slowest.
- **Perplexity is a poor predictor of the generative cliff.** It degrades smoothly while
  generation breaks discontinuously. Temperature-0 probes remain the primary gate.
- **K governs residency, not decode speed.** Exactly 8 experts fire per token regardless
  of how many are resident: measured 8.18 tok/s at K=108 versus 8.16 at K=96.

## Prior art and attribution

The saliency criterion is **REAP (Router-weighted Expert Activation Pruning)**:

> Lasby, Lazarevich, Sinnadurai, Lie, Ioannou & Thangarasa,
> *REAP the Experts: Why Pruning Prevails for One-Shot MoE Compression*,
> arXiv:2510.13999 — https://github.com/CerebrasResearch/reap

REAP scores each expert by the mean, over tokens where it is active, of
(router gate × ‖expert output‖), and prunes the lowest scorers. The paper proves that
expert *merging* incurs an irreducible error ("functional subspace collapse") because
summing gates destroys the router's independent, input-dependent control, while pruning
preserves it.

Cerebras's published calibration mix for their released checkpoints is 24,576 samples at
up to 16,384 tokens: evol-codealpaca-v1, Mixture-of-Thoughts, xlam-function-calling-60k
and SWE-smith trajectories — an agentic-reasoner mix roughly 20× larger than ours.

Reference artefacts for comparison: `pipenetwork/GLM-5.2-REAP50-Q3_K_M-GGUF` and
`pipenetwork/GLM-5.2-REAP50-Q2_K-GGUF` — community applications of REAP to GLM-5.2,
keep 128/256, calibration corpus undocumented.

### What we extend

- **Prune depth.** REAP evaluates 25% and 50%; we operate at 57.8%.
- **Quantized-native.** We slice the shipping GGUF directly, byte-for-byte, with
  verification — no requantization pass after compression, and no need for the ~1.5 TB
  BF16 checkpoint.
- **In-engine saliency.** Collected inside llama.cpp, so experts are ranked by how they
  behave in the model that actually runs, quantization noise included.
- **Post-hoc domain blending.** Scoring each domain once and combining afterwards makes
  the whole space of corpus mixtures searchable for the cost of one pass per domain.
- **Random and membership controls.** A random baseline quantifies how much of the
  benefit is the criterion; the code108-vs-mix108 pair isolates corpus choice at fixed
  capacity.
- **Deployment.** The output is an artefact sized to, and running on, a specific 128 GB
  device.

## Iteration 1: code96, and what was wrong with it

The first artefact was **code96** — 96 of 256 experts, calibrated on 85,607 tokens of
which ~85% was llama.cpp source. It ran on the Spark at 8.16 tok/s and was, on
llama.cpp-flavoured code, near-indistinguishable from the parent.

It was evaluated on corpora drawn from its own calibration domain, and that inflated
every headline number. The following claims from the first version of this document are
**retracted**:

- *"code96 beats REAP-50 on code (+6.6% vs +7.9%)."* Both figures were measured on
  llama.cpp-derived text. On neutral corpora it is code96 +23.1% against REAP-50 +13.2%:
  REAP-50 wins. The honest statement is that code96 beats REAP-50 **on the workload it
  was calibrated for**.
- *"General-calibrated saliency costs +31.5%."* That control was calibrated on llama.cpp
  *markdown* — documentation about the same codebase — so it never tested general
  calibration. It was also evaluated on the same corpus it was calibrated on.
- *"Cross-domain Spearman +0.609."* An artefact of the same markdown corpus. Against real
  literary prose the figure is −0.005.
- *"~39 of 256 experts per layer below 0.1% usage share."* True on code only. Measured
  across ten domains, no expert is idle in any layer.
- The damage decomposition (70% saliency / 24% domain targeting) inherits the flawed
  control and should not be quoted.

K=96 was also selected on the code axis alone, and was not a budget decision: the Spark
had residency for K=108 throughout, so code96 left roughly 10 GiB unused.

The corpus was the defect, not the method. The diagnosis was confirmed by a
pre-registered neutral-corpus test before any new artefact was built, and independently
by EvoESAP (arXiv:2603.06003), which reports the same C4-versus-evol-codealpaca effect
on a different model.

## Pipeline

1. `src/moe-saliency.cpp` — llama.cpp eval-callback observer (drop into
   `examples/moe-saliency/`). Accumulates per-layer, per-expert selection count, gate
   sum, and gate × ‖expert output‖ over a calibration corpus. Four non-obvious issues are
   fixed in this version: `ffn_moe_topk` is a NON-CONTIGUOUS view (a linear read silently
   returns argsort rows and yields perfectly uniform, entirely fictitious statistics);
   batches must request logits for every token or the last layer's FFN is pruned to
   output tokens only; the CPU norm loop is OpenMP-parallelised (rows are independent, so
   output is bit-identical); and the fetch buffer is `static thread_local`, which removes
   ~68M page faults per shard and cuts wall time 2.8×.
   `SAL_FILELIST` processes many corpora per model load.
2. `tools/build_corpora_v2.py` — ten-domain corpus builder with SHA-256-verified disjoint
   splits and real chat-template rendering.
3. `tools/merge_saliency.py`, `convergence.py`, `crossdomain.py` — exact shard merging
   with a `sum(cnt) == tokens × 8` invariant check, per-domain convergence curves, and
   the cross-domain Spearman/Jaccard matrix.
4. `tools/analyze_saliency.py` — concentration/Gini, keep-list emission (`--emit K`,
   `--random` for the control).
5. `tools/glm_prune_gguf.py` — whole-expert byte-range slice of an already-quantized
   GGUF. Slices router rows, rewrites expert_count / block_count /
   nextn_predict_layers, drops the MTP block.
6. `tools/verify_candidate.sh` — byte-exactness identity test, code probe, PPL/KLD,
   speed. `run_probes.sh` + `score_probes.py` — 13 frozen probes and objective
   degeneration scoring.

## Deployment: measured on GB10 Spark

`mix108_maxmin` runs on a single NVIDIA GB10 Spark (128 GB LPDDR5X unified, ~273 GB/s,
sm_121, aarch64, 20 × Cortex-X925). llama.cpp builds natively at the same commit with
`-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121 -DLLAMA_USE_PREBUILT_UI=OFF
-DLLAMA_CURL=OFF` under CUDA 13.0 — no patches, no elevated privileges.

| measurement | 8 × RTX 5090 | 1 × GB10 Spark |
|---|---:|---:|
| decode, tg128 | 51.16 ±0.29 tok/s | **8.18 ±0.01 tok/s** |
| prefill, pp512 | 1159.8 ±0.3 tok/s | 244.8 ±2.6 tok/s |
| resident, 64K context | 100.32 GiB across 8 cards | 109 of 121 GB unified |

Serving config: `-c 65536 -np 1 -t 20 --jinja -rea off`. Real latency is ~37 s for a
300-token answer. Only one artefact fits at a time — 100.32 + 90.72 GiB exceeds 121 GB —
so side-by-side comparison on the Spark must be sequential.

**Cross-architecture validation** (measured on code96, same pipeline): perplexity differs
by 0.13% between x86_64/sm_120 and aarch64/sm_121, inside the ±0.030 error bar, so the
two architectures compute identically modulo floating-point reduction order.

**Why decode is ~8.2 tok/s.** Decode is memory-bandwidth-bound, so throughput is set by
bytes read per token, not by expert count:

| component | active params | precision | bytes/token |
|---|---:|---|---:|
| routed experts (8 active) | 22.6B | ~2.4 bpw | 6.8 GB |
| attention (MLA, 78 layers) | ~14B | Q8_0 / Q5_K | ~9.6 GB |
| shared expert, dense FFN, output head | ~4.5B | Q5_K / Q6_K | ~3.1 GB |
| **total** | **~41B** | | **~19.5 GB** |

19.5 GB / 273 GB/s gives a ceiling near 14 tok/s; the measured 8.18 is ~58% bandwidth
efficiency, normal for MoE decode where expert gather is scattered and GEMV is
latency-sensitive. This model predicted its own successor: because exactly 8 experts fire
per token regardless of K, mix108 at K=108 reads the same bytes as code96 at K=96, and
measures 8.18 against 8.16.

**Implication.** Experts are only 6.8 GB of the 19.5 GB read per token, because the
quantization deliberately upcasts everything except the routed experts. Further pruning
cannot improve decode — even zero experts would leave 12.7 GB, a ceiling near 21 tok/s.
The remaining decode lever is quantizing the **non-expert** layers lower: dropping
attention from ~5.5 to ~3 bpw would cut roughly 5 GB/token, about +35% decode, at a
quality cost not yet measured on this model.

**Context.** Comparable configurations, all GLM-5.2:

| configuration | decode | hardware |
|---|---:|---|
| IQ2_M resident | 52.5 tok/s | 8 × RTX 5090 |
| IQ2_M + CPU expert offload | 6.55 tok/s | 1 × RTX 5090, needs 503 GB host RAM |
| **mix108_maxmin** | **8.18 tok/s** | **1 × GB10 Spark, standalone** |

8.18 tok/s suits background and agentic use; it is below the ~20 tok/s commonly cited as
an interactive threshold.

## Caveats and limitations

- **Benchmark numbers are not comparable to published leaderboards.** These are zero-shot
  loglikelihood scores from llama.cpp's `--multiple-choice`, not the standard 5-shot MMLU
  protocol.
- **"Parent" is the IQ2_M 222.18 GiB quantized model**, so every figure isolates pruning
  cost *on top of* quantization. Quantization's own contribution is unmeasured and is
  plausibly larger.
- **Benchmark coverage is knowledge-only.** MMLU, ARC-C and TruthfulQA all probe the same
  broad axis. IFEval, which would measure the instruction-following axis where this
  artefact is strongest, has not been run.
- **World knowledge is imperfect and can be confidently wrong.** At 57.8% expert pruning
  both this artefact and the REAP-50 comparator confabulate; the comparator additionally
  loops on the same prompt. Neither should be treated as a reference source.
- **Probe sets are small.** n=8 in-domain and n=5 out-of-domain; ARC-C is n=299. Only the
  MMLU gaps are individually significant.
- **English only**, by deliberate scope choice.
- **The code axis regressed** relative to code108 (1.179 → 1.297 PPL ratio) as the price
  of general capability.

## Reproducing

Source: `unsloth/GLM-5.2-GGUF` UD-IQ2_M (222.18 GiB), merged to a single file with
`llama-gguf-split --merge`. llama.cpp pinned at `7f575c39d` (b10005).
Flags throughout: `-ngl 99 -fa 1 -ctk q8_0 -ctv q8_0`; probes at
`--temp 0 --seed 1 -rea off -st --simple-io -n 300` (`-rea off` suppresses GLM's
reasoning block; `/nothink` does not). Multiple-choice evals need `-np 8` for MMLU and
ARC, and `-np 16` for TruthfulQA, whose option counts run to ~12. See `env.sh.example`
for paths.

Build hardware: 8 × RTX 5090, 503 GB RAM. Peak ~160 GB RSS per slice; run candidates
sequentially, and drop page caches between model switches.

## Licence and distribution

Derived from an MIT-licensed checkpoint. This repository is **private** pending
clarification of the intellectual-property position — EU novelty rules require filing
before any public disclosure.
