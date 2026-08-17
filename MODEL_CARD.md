---
license: mit
license_link: https://huggingface.co/zai-org/GLM-5.2/blob/main/LICENSE
base_model:
  - zai-org/GLM-5.2
base_model_relation: quantized
library_name: gguf
pipeline_tag: text-generation
language:
  - en
tags:
  - gguf
  - moe
  - mixture-of-experts
  - expert-pruning
  - compression
  - dgx-spark
  - gb10
  - llama-cpp
---

# GLM-5.2 MaxMin108 — a 753B MoE compressed onto one 128 GB desktop

GLM-5.2 with **108 of its 256 routed experts** kept per layer, selected by a max-min fair
criterion across ten text domains. 100.32 GiB. Runs on a single NVIDIA GB10 Spark.

**Headline:** level with [REAP-50](https://huggingface.co/pipenetwork) on knowledge
(MMLU **63.00** vs 62.75), behind it on code (MBPP **0.626** vs 0.708) — at **59% of its
size**, on hardware where REAP-50 does not fit.

**What it costs.** Compression removes **24.29 MMLU points** against the parent
(87.29 → 63.00, −27.8% relative). Factual answers can be fluent and confidently wrong. If a
120B-class model at 4-bit fits your hardware, it will likely score higher and run several
times faster. This artifact demonstrates that a frontier-scale MoE *can* be made to fit, and
documents a selection method that measurably beats the alternatives.

---

## Files

| variant | size | bpw | shards |
|---|---:|---:|---|
| `GLM-5.2-MaxMin108-IQ2_M` | 100.32 GiB | 2.650 | 6 |
| `GLM-5.2-MaxMin108-IQ2_M-fast` | 95.77 GiB | 2.530 | 6 |

Both keep the same 108 experts per layer — the expert tensors are **byte-identical between
them**. They differ only in the precision of *non-expert* tensors (attention projections and
the shared expert).

**Use `-fast` unless you have a reason not to.** It is smaller, ~12–14% faster at decode on
both machines we tested, and frees ~5 GB of resident memory, for roughly 1–3% relative
capability (−1.16 MMLU, −0.034 MBPP). The base variant is the reference artifact every number
here is anchored to, and is marginally faster at prefill.

Load shard `00001`; llama.cpp finds the rest automatically.

---

## Benchmarks

Everything below was run **by us, on the same machines, in the same harnesses**, for all four
models — including the parent and the comparator. Nothing is quoted from another card.

### MMLU — `lm-evaluation-harness` `mmlu_generative`, 5-shot, all 14,042 questions

| model | size | MMLU | humanities | social sci. | STEM | other |
|---|---:|---:|---:|---:|---:|---:|
| GLM-5.2 UD-IQ2_M (parent) | 222.18 GiB | **87.29** ±0.27 | 83.44 | 91.84 | 87.03 | 88.86 |
| REAP-50 Q3_K_M (comparator) | 169.30 GiB | 62.75 ±0.39 | 63.63 | 66.85 | 62.26 | 57.84 |
| **MaxMin108** | 100.32 GiB | **63.00** ±0.39 | 62.74 | 68.61 | 59.25 | 61.67 |
| **MaxMin108-fast** | 95.77 GiB | 61.84 ±0.40 | 62.64 | 67.57 | 57.82 | 59.06 |

MaxMin108 and REAP-50 differ by 0.25 points against ±0.39 error bars — a **tie**, not a win.
The interesting part is that the tie holds at 59% of the size. Per category REAP-50 leads on
STEM (+3.01) and humanities (+0.89); MaxMin108 leads on social sciences (+1.76) and other
(+3.83). The STEM gap is the one real signal, and is what you would expect from REAP-50's
higher bit budget (3.82 bpw vs 2.65) and larger expert count (128 vs 108).

### MBPP — 3-shot, 500 problems, pass@1, code executed

| model | size | pass@1 | retained vs parent |
|---|---:|---:|---:|
| parent | 222.18 GiB | **0.710** ±0.020 | — |
| REAP-50 Q3_K_M | 169.30 GiB | 0.708 ±0.020 | **99.7%** |
| **MaxMin108** | 100.32 GiB | 0.626 ±0.022 | 88.2% |
| **MaxMin108-fast** | 95.77 GiB | 0.592 ±0.022 | 83.4% |

**A finding worth its own line: REAP-50 is statistically indistinguishable from the parent on
MBPP (0.708 vs 0.710) while losing 24.54 MMLU points.** Pruning half the experts costs
almost nothing on short-form code generation and roughly a quarter of world knowledge. That
dissociation — expert pruning destroys knowledge far faster than procedural coding ability —
is the most transferable result here, and it holds for our artifact too: MaxMin108 retains
**88.2%** of parent MBPP against only **72.2%** of parent MMLU.

Note this **inverts** what the perplexity ratios below suggest. Code perplexity degrades more
than general perplexity, yet downstream code *capability* degrades less. Perplexity ratio and
task capability disagree here; we report both rather than choosing the flattering one.

MaxMin108 is genuinely behind REAP-50 on MBPP (−8.2 points, ~2.8σ). Unlike the MMLU tie, that
gap is real.

### Perplexity — `llama-perplexity`, `-c 4096 --chunks 32`, held-out corpora

Neither model was calibrated on these (half llama.cpp source + half evol-codealpaca for code;
wiki/web/chat for general). Ratios are relative to the parent.

| model | code PPL | ratio | general PPL | ratio |
|---|---:|---:|---:|---:|
| parent | 2.1401 | 1.000 | 3.7753 | 1.000 |
| REAP-50 Q3_K_M | 2.4217 | 1.132 | 6.5462 | 1.734 |
| **MaxMin108** | 2.7751 | 1.297 | 6.2115 | **1.645** |
| **MaxMin108-fast** | 2.8260 | 1.321 | 6.2790 | 1.663 |

### Generation quality — 13 frozen prompts, greedy decoding

Degeneration is scored objectively: 4-gram distinctness < 0.55, a line repeated ≥ 4×, or a
word repeated ≥ 5× consecutively.

| model | in-domain | out-of-domain | OOD distinct-4 |
|---|---|---|---|
| parent | 0/8 | 0/5 | 0.991 |
| REAP-50 Q3_K_M | 0/8 | **1/5** | 0.883 |
| **MaxMin108** | 0/8 | **0/5** | 0.855 (0.996 sampled) |
| **MaxMin108-fast** | 0/8 | **0/5** | 0.984 |
| code-only ablation | 0/8 | **3/5** | 0.424 |

At **greedy decoding**, where the matched-bit-budget REAP-50 Q2_K variant is documented by its
own authors to collapse into repetition, both artifacts here produce zero degenerate outputs
across all thirteen prompts.

### Speed and residency

`llama-bench`, llama.cpp build `b10005`, `-p 512 -n 128 -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0
-r 3`. Each pair measured in a single session.

**1 × GB10 Spark** (128 GB unified, ~273 GB/s):

| | decode (tg128) | prefill (pp512) | resident @ 64K ctx |
|---|---:|---:|---:|
| MaxMin108 | 8.24 ±0.02 tok/s | 244.12 ±1.82 tok/s | 109 of 121 GB |
| **MaxMin108-fast** | **9.36 ±0.02** (+13.6%) | 243.94 ±3.28 | **104 of 121 GB** |
| parent / REAP-50 | — does not fit — | | |

**8 × RTX 5090:**

| | decode (tg128) | prefill (pp512) |
|---|---:|---:|
| MaxMin108 | 51.21 ±0.27 tok/s | 1166.00 ±6.76 tok/s |
| **MaxMin108-fast** | **57.58 ±0.38** (+12.4%) | 1145.34 ±4.44 (−1.8%) |

Two things worth stating plainly. First, the decode gain is **not** hardware-specific: +13.6%
on a bandwidth-starved 273 GB/s desktop and +12.4% on eight datacentre GPUs. Second, `-fast`
is **1.8% slower at prefill** on the 5090 box, outside the error bars — lower-bit K-quants
cost more dequantisation work, which shows up in the compute-bound prefill regime. It is a
small regression and we would rather report it than have you find it.

Serving config for the residency figure: `-c 65536 -np 1 -t 20 --jinja -rea off`. Roughly
37 s for a 300-token answer on the Spark. Decode is memory-bandwidth-bound and exactly 8
experts fire per token regardless of how many are resident — so pruning changes what *fits*,
not how fast it runs. The `-fast` speedup comes entirely from requantising non-expert tensors.

---

## Method

### Expert selection

Per-expert saliency uses the REAP criterion — gate weight × ‖expert output‖ — accumulated over
**1.93 M calibration tokens across ten disjoint English domains** (code_raw, code_instruct,
web, wiki, chat, math, news, books, science, reasoning), measured on the parent with a custom
llama.cpp observer.

The novel part is **how the domains are combined**. Rather than ranking experts by aggregate
saliency, weights are chosen to **maximise the worst-served domain's coverage** — the share of
a domain's routing mass that survives, relative to what that domain's own optimal top-108
would keep. An iterative reweighting search converged to:

```
code_raw 2.003 | web 1.613 | science 1.225 | code_instruct 1.074 | books 0.844
wiki 0.738 | chat 0.626 | math 0.626 | news 0.626 | reasoning 0.626
```

Seven of the ten domains land at exactly 0.814 coverage — a shared minimum is the signature of
a genuine max-min fixed point.

| strategy | worst-case coverage @ K=108 |
|---|---:|
| single domain (code only) | 0.463 |
| uniform blend | 0.720 |
| **max-min blend** | **0.814** |

Raising K from 108 to 128 buys only +0.050 by comparison. **The weighting matters more than
the capacity.**

### Why this matters — the ablations

Five alternative keep-lists were built at identical size, bit budget, K and slicer, with only
the calibration corpus differing:

| keep-list | OOD degenerate | OOD distinct-4 |
|---|---|---|
| code-only, K=96 | 3/5 | 0.476 |
| code-only, K=80 | 5/5 | 0.165 |
| code-only, K=108 | 3/5 | 0.424 |
| general-only, K=96 | 4/5 | 0.421 |
| random, K=96 | 2/5 | 0.543 |
| **max-min, K=108** | **0/5** | **0.855** |

Single-domain expert pruning does not merely narrow a model — it silently destroys
out-of-domain coherence. Every one of these alternatives loops or repeats on prompts outside
its calibration domain.

### Non-expert requantisation (the `-fast` variant)

Non-expert tensors are only 13.9% of the artifact but roughly **69% of the bytes read per
token**, because the source quantisation deliberately upcasts everything except the routed
experts. Requantising them is the remaining decode lever.

`-fast` changes 616 tensors: `attn_output` Q5_K→Q3_K, `attn_q_b` Q8_0→Q4_K, `attn_q_a`
Q5_K→Q4_K, `attn_k_b` Q8_0→Q5_0, `attn_v_b` Q8_0→Q5_K, the shared expert Q5_K/Q6_K→Q3_K, and
`token_embd` Q5_K→Q4_K, using an importance matrix computed on the same ten-domain corpus.

Deliberately **left untouched**: every `indexer.*` tensor (GLM-5.2's DSA sparse-attention
selector — precision loss there changes *which tokens are attended*, not just values),
`attn_kv_a_mqa` (the compressed MLA latent everything flows through), and `output.weight`.

A measured side-finding: **effective memory bandwidth falls as quantisation deepens** —
160.7 GB/s at the parent's non-expert precision, 151.5 for a conservative recipe, 138.8 for
`-fast`. Cheaper unpacking (Q8_0) buys back part of what the extra bytes cost, which is why
removing 24% of the bytes read yields 13.6% more speed rather than the naive 32% — and why
prefill gets slightly slower.

### Verification

Expert tensors in both released variants are **byte-identical slices** of the source GGUF's
expert tensors — 25 of 25 sampled tensors across layers 3, 20, 40, 60 and 77, zero failures.
No expert weight was requantised, merged, or otherwise modified. Router rows and `exp_probs_b`
are sliced to match; the MTP block is dropped. Both shard sets were reloaded after splitting
and reproduce their unsplit perplexity exactly.

---

## Limitations

- **World knowledge is substantially reduced** — 24.29 MMLU points below the parent. Answers
  can be fluent and confidently wrong. Aggressive pruning usually manifests as looping; these
  artifacts do not loop, which arguably makes the remaining errors *harder* to spot.
- **Code is behind REAP-50** (MBPP 0.626 vs 0.708, ~2.8σ), though ahead of what the perplexity
  ratio implies.
- **English only.** GLM-5.2 is bilingual (en/zh); calibration was English-only, so the experts
  carrying Chinese were preferentially pruned. Chinese capability is discarded and unmeasured.
- **Derived from an IQ2_M quantisation, not BF16.** All figures isolate the cost of *pruning on
  top of* quantisation; quantisation's own cost is not measured here. `-fast` additionally
  involves a dequantise→requantise round trip on non-expert tensors.
- **`-fast` is 1.8% slower at prefill** on multi-GPU hardware. Decode is faster on both
  machines tested.
- **Two benchmarks.** MMLU and MBPP cover knowledge and short-form code. We attempted IFEval
  for instruction-following on two separate occasions and could not complete it: llama.cpp's
  server strictly parses model output against the template's declared reasoning and tool-call
  grammars and returns HTTP 500 on a parse failure. Roughly 1–4% of IFEval prompts triggered
  this on the pruned models across four server configurations; the parent never did. **That
  the compressed models occasionally emit unparseable structured output is itself a
  compression artifact**, and one we would rather report than hide.
- **The 5 out-of-domain probes are a small sample.** The 3/5 → 0/5 effect is large, but n=5.
- Sampled decoding (`--temp 0.6 --top-p 0.95`) produces noticeably better output than greedy;
  the greedy numbers above are a deliberately conservative floor.

---

## Credits

This work rests on three others and would not exist without them:

- **[zai-org/GLM-5.2](https://huggingface.co/zai-org/GLM-5.2)** — the base model (MIT).
- **[unsloth/GLM-5.2-GGUF](https://huggingface.co/unsloth/GLM-5.2-GGUF)** — the UD-IQ2_M
  quantisation these artifacts are sliced from. Every expert tensor here is a byte-identical
  copy of theirs, the released files still carry their importance-matrix metadata, and their
  dynamic-quantisation choices — which tensors to protect at higher precision — are inherited
  wholesale.
- **[Cerebras REAP](https://github.com/CerebrasResearch/reap)** (ICLR 2026) — the saliency
  criterion this work extends.
- **[pipenetwork/GLM-5.2-REAP50](https://huggingface.co/pipenetwork)** — the comparator, and
  the reason a meaningful baseline existed to measure against.

---

## Reproducing this

The keep-lists are published alongside the models. Either artifact can be rebuilt from
Unsloth's GGUF in about fifteen minutes of disk I/O — no GPU time, no saliency campaign:

```bash
python tools/glm_prune_gguf.py \
  --src GLM-5.2-UD-IQ2_M-merged.gguf \
  --dst MaxMin108.gguf \
  --keeplist keep_mix108_maxmin.json \
  --drop-blocks 78
```

Tooling, corpus manifests, per-domain saliency aggregates and the full experiment log:
**https://github.com/Dorijan10/glm-expert-pruning**

## Usage

```bash
llama-server -m GLM-5.2-MaxMin108-IQ2_M-fast-00001-of-00006.gguf \
  --alias glm-5.2-maxmin108 -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 \
  -c 65536 -np 1 -t 20 --jinja -rea off
```

`-rea off` is recommended: without it GLM-5.2's template emits a high reasoning-effort
directive that can consume an entire short generation budget on hidden reasoning tokens.

Requires llama.cpp with GLM-DSA support (build `b10005` / `7f575c39d` or newer).
