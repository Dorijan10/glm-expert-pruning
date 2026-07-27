# GLM-5.2 domain-calibrated MoE expert pruning

Compressing GLM-5.2 (753.86B MoE) to run on one NVIDIA GB10 Spark (128 GB unified)
by pruning the routed experts a coding workload does not use.

**Status:** artefact selected and characterised; Spark deployment in progress.

## Result

Final artefact: **code96** — keep 96 of 256 experts per layer, code-calibrated.
90.72 GiB / 97.4 GB, 290.85B params (from 753.86B). Fits one Spark with ~24 GB
headroom for KV cache and activations.

| K | size | code PPL | vs parent | in-domain degen | OOD degen |
|---:|---:|---:|---:|---:|---:|
| 256 (parent IQ2_M) | 222.18 GiB | 1.9383 | — | 0/8 | 0/5 |
| 108 | 100.32 GiB | 2.0374 | +5.1% | 0/8 | 3/5 |
| **96 (selected)** | **90.72 GiB** | **2.0669** | **+6.6%** | **0/8** | 3/5 |
| 80 | 78 GiB | 2.1268 | +9.7% | 2/8 | 5/5 |

Controls at K=96 — identical slicer, bit-widths and corpus, all byte-exactness
verified, one variable changed:

| selection | code PPL | vs parent | probe output |
|---|---:|---:|---|
| code-calibrated saliency | 2.0669 | +6.6% | correct, commented C++ |
| general-calibrated saliency | 2.5495 | +31.5% | malformed C++ |
| random | 3.9740 | +105.0% | degeneration loop |

Damage decomposition: of the +105.0 points of excess perplexity that random
selection costs, the REAP saliency ranking removes 73.5 (70%) and domain
calibration removes a further 24.9 (24%), leaving +6.6. Both are load-bearing.

Speed vs parent, same session: prefill **+74%** (1222.5 vs 702.7 tok/s), decode
−2.7% (51.2 vs 52.5). Pruning buys capacity, not decode speed — only 8 of 256
experts are active per token, so bytes-per-token is unchanged.

## Prior art and attribution

The saliency criterion is **REAP (Router-weighted Expert Activation Pruning)**:

> Lasby, Lazarevich, Sinnadurai, Lie, Ioannou & Thangarasa,
> *REAP the Experts: Why Pruning Prevails for One-Shot MoE Compression*,
> arXiv:2510.13999 — https://github.com/CerebrasResearch/reap

REAP scores each expert by the mean, over tokens where it is active, of
(router gate × ‖expert output‖), and prunes the lowest scorers. The paper proves
that expert *merging* incurs an irreducible error ("functional subspace collapse")
because summing gates destroys the router's independent, input-dependent control,
while pruning preserves it. They report near-lossless 50% pruning on
Qwen3-Coder-480B (code avg 0.660 → 0.644) and Kimi-K2 (0.659 → 0.646).

Reference artefact for comparison: `pipenetwork/GLM-5.2-REAP50-Q3_K_M-GGUF` —
a community application of REAP to GLM-5.2 (general calibration, keep 128/256,
Q3_K_M, 169.3 GiB). We measured it at +7.9% code PPL on our corpus.

### What we replicated

REAP's Table A6 shows Qwen3-30B at 50% pruning calibrated on general text (c4)
collapsing from 0.557 to **0.003** code average. Our gen96-vs-code96 control
reproduces that ablation on a different model (GLM-5.2 vs Qwen3-30B), at 25× the
scale, in a different format (GGUF vs PyTorch), at a deeper prune ratio — same
qualitative outcome. We also independently rediscovered their warning that
perplexity is a poor proxy for generative quality.

### What we extend

- **Prune depth.** REAP evaluates 25% and 50%; we operate at 62.5% and locate the
  failure boundary between 62.5% and 68.75%.
- **Quantized-native.** We slice the shipping GGUF directly, byte-for-byte, with
  verification — no requantization pass after compression.
- **In-engine saliency.** Collected inside llama.cpp, so no ~1.5 TB BF16
  checkpoint is needed to calibrate a 754B model.
- **Random control.** REAP compares frequency and activation-norm criteria; adding
  a random baseline quantifies how much of the benefit is the criterion itself.
- **Deployment.** The output is an artefact sized to a specific 128 GB device.

## Key findings

- GLM-5.2 routing is strongly non-uniform: mean Gini 0.53, ~39 of 256 experts per
  layer below 0.1% usage share.
- Expert ranking is reproducible (within-domain Spearman +0.961) and
  domain-dependent (cross-domain +0.609; ~33 of the top 96 differ between code and
  general text, against a ~9-expert noise floor).
- **Perplexity is a poor predictor of the cliff.** It degrades smoothly
  (+5.1 → +6.6 → +9.7%) while generation breaks discontinuously
  (clean → clean → broken). Use temperature-0 probes as the primary gate.
- code96 is a coding specialist, not a general assistant. In-domain it is
  statistically indistinguishable from the unpruned parent; out-of-domain it
  degenerates and loses world knowledge. Sampler guardrails (repeat-penalty 1.10)
  do **not** rescue it — the same 3/5 prompts fail, confirming absent knowledge
  rather than unstable decoding.
- Layers L7/L9/L10 are least tolerant of pruning, independently reproducing the
  published observation that GLM-5.2's layers 7–17 are the most sensitive.

## Pipeline

1. `moe-saliency.cpp` — llama.cpp eval-callback observer (drop into
   `examples/moe-saliency/`). Accumulates per-layer, per-expert selection count,
   gate sum, and gate × ‖expert output‖ over a calibration corpus.
   Two non-obvious bugs are fixed in this version: `ffn_moe_topk` is a
   NON-CONTIGUOUS view (needs strided reads — a linear read silently returns
   argsort rows and yields perfectly uniform, entirely fictitious statistics), and
   batches must request logits for every token or the last layer's FFN is pruned
   to output tokens only.
2. `analyze_saliency.py` — concentration/Gini, within- vs cross-domain
   reliability, keep-list emission (`--emit K`, `--random` for the control).
3. `glm_prune_gguf.py` — whole-expert byte-range slice of an already-quantized
   GGUF. Slices router rows, rewrites expert_count / block_count /
   nextn_predict_layers, drops the MTP block.
4. `verify_candidate.sh` — byte-exactness identity test, code probe, PPL/KLD,
   speed. `run_probes.sh` + `score_probes.py` — 13 frozen probes and objective
   degeneration scoring (unique-4-gram ratio, longest repeated line, longest
   repeated-word run).

## Reproducing

Source: `unsloth/GLM-5.2-GGUF` UD-IQ2_M (222.18 GiB), merged to a single file with
`llama-gguf-split --merge`. llama.cpp pinned at `7f575c39d` (b10005).
Flags throughout: `-ngl 99 -fa 1 -ctk q8_0 -ctv q8_0`; probes at
`--temp 0 --seed 1 -rea off -st --simple-io -n 300` (`-rea off` suppresses GLM's
reasoning block; `/nothink` does not). See `env.sh.example` for paths.

Build hardware: 8 × RTX 5090, 503 GB RAM. Peak ~160 GB RSS per slice; run
candidates sequentially.

## Deployment target

NVIDIA GB10 Spark — 128 GB LPDDR5X unified (~273 GB/s), sm_121, aarch64.
llama.cpp builds natively at the same commit with
`-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121 -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_CURL=OFF`
under CUDA 13.0 — no patches or elevated privileges required.
Decode prediction to verify: ~15–18 tok/s (~9–11 GB active weights per token
÷ 273 GB/s).

## Licence and distribution

Derived from an MIT-licensed checkpoint. This repository is **private** pending
clarification of the intellectual-property position — EU novelty rules require
filing before any public disclosure.
