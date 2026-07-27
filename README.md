# GLM-5.2 domain-calibrated MoE expert pruning

Compressing GLM-5.2 (753.86B MoE) to run on one NVIDIA GB10 Spark (128 GB unified)
by pruning the routed experts a coding workload does not use.

## Result

Final artifact: **code96** — keep 96 of 256 experts per layer, code-calibrated.
90.72 GiB / 97.4 GB, 290.85B params. Fits one Spark with ~23 GB headroom.

| K | size | code PPL | vs parent | in-domain degen | OOD degen |
|---:|---:|---:|---:|---:|---:|
| 256 (parent IQ2_M) | 222.18 GiB | 1.9383 | — | 0/8 | 0/5 |
| 108 | 100.32 GiB | 2.0374 | +5.1% | 0/8 | 3/5 |
| **96** | **90.72 GiB** | **2.0669** | **+6.6%** | **0/8** | 3/5 |
| 80 | 78 GiB | 2.1268 | +9.7% | 2/8 | 5/5 |

Controls at K=96 (identical slicer, bits, corpus; all byte-exact):
code-calibrated +6.6% | general-calibrated +31.5% | random selection +105.0%.

Speed vs parent (same session): prefill +74% (1222.5 vs 702.7 tok/s),
decode -2.7% (51.2 vs 52.5) — pruning buys capacity, not decode speed,
because only 8 of 256 experts are active per token.

## Key findings

- GLM-5.2 routing is strongly non-uniform: mean Gini 0.53, ~39 of 256 experts
  per layer below 0.1% usage share.
- Expert ranking is reproducible (within-domain Spearman +0.961) and
  domain-dependent (cross-domain +0.609; ~33 of the top 96 differ between
  code and general text).
- **PPL is a poor predictor of the cliff.** Perplexity degrades smoothly
  (+5.1 → +6.6 → +9.7%) while generation quality breaks discontinuously
  (clean → clean → broken). Use temp-0 probes as the primary gate.
- code96 is a coding specialist: in-domain output is indistinguishable from
  the unpruned parent, out-of-domain it degenerates and loses world knowledge.
  Sampler guardrails (repeat-penalty 1.10) do NOT rescue OOD — the knowledge
  is absent, not merely unstable.

## Pipeline

1. `moe-saliency.cpp` — llama.cpp eval-callback observer (drop in
   `examples/moe-saliency/`). Accumulates per-layer, per-expert selection
   count, gate sum, and gate x ||expert output|| over a calibration corpus.
   Two bugs worth knowing about are fixed in this version: `ffn_moe_topk` is a
   NON-CONTIGUOUS view (needs strided reads) and batches must request logits
   for every token or the last layer's FFN is pruned to output tokens only.
2. `analyze_saliency.py` — concentration/Gini, within- vs cross-domain
   reliability, emits keep-lists (`--emit K`, `--random` for the control).
3. `glm_prune_gguf.py` — whole-expert byte-range slice of an already-quantized
   GGUF. No requantization: surviving experts are bit-identical. Slices router
   rows, rewrites expert_count/block_count/nextn_predict_layers, drops the MTP
   block.
4. `verify_candidate.sh` — byte-exactness identity test, code probe, PPL/KLD,
   speed. `run_probes.sh` / `score_probes.py` — 13 frozen probes and objective
   degeneration scoring.

## Reproducing

Source model: `unsloth/GLM-5.2-GGUF` UD-IQ2_M (222.18 GiB), merged to a single
file with `llama-gguf-split --merge`. llama.cpp pinned at `7f575c39d` (b10005).
Measurement flags throughout: `-ngl 99 -fa 1 -ctk q8_0 -ctv q8_0`;
probes `--temp 0 --seed 1 -rea off -st --simple-io -n 300`
(`-rea off` suppresses GLM's reasoning block; `/nothink` does not).

See `env.sh.example` for paths.
