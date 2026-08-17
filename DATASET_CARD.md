---
license: mit
task_categories:
  - text-generation
language:
  - en
tags:
  - moe
  - expert-pruning
  - mixture-of-experts
  - interpretability
size_categories:
  - n<1K
---

# GLM-5.2 expert saliency and keep-lists

Everything needed to reproduce
[turintech/GLM-5.2-MaxMin108-GGUF](https://huggingface.co/turintech/GLM-5.2-MaxMin108-GGUF)
from [unsloth/GLM-5.2-GGUF](https://huggingface.co/unsloth/GLM-5.2-GGUF): about fifteen
minutes of disk I/O, no GPU, no calibration run.

## Contents

- **`keep_lists/`**: which experts survive, per layer. `keep_mix108_maxmin.json` produces the
  published model. The others are the ablations (code-only, general-only and random selection
  at several K), four of which degenerate on out-of-domain prompts.
- **`saliency_merged/`**: per-expert saliency using the REAP criterion, gate weight times the
  norm of the expert output, accumulated over 1.93 M tokens. One file per domain across ten
  domains, plus the pooled total. 75 MoE layers by 256 experts, with `cnt`, `gate` and `sal`
  per expert.
- **`weights_maxmin.json`**: the max-min fair blend weights.
- **`corpus_manifest.json`**: sha256-verified composition of the calibration corpus.
- **`results/`**: raw `lm-evaluation-harness` output for every benchmark in the model card.
- **`extras/`**: the importance matrix used for the `-fast` variant (regenerating it takes
  about 32 minutes of GPU time), and the trained routers from the router-repair experiment
  described below.

## Rebuilding the model

```bash
git clone https://github.com/Dorijan10/glm-expert-pruning
python tools/glm_prune_gguf.py \
  --src GLM-5.2-UD-IQ2_M-merged.gguf \
  --dst MaxMin108.gguf \
  --keeplist keep_lists/keep_mix108_maxmin.json \
  --drop-blocks 78
```

Surviving experts are copied byte for byte, and nothing is requantised.

## Why the saliency data may be useful on its own

Across all ten domains, **no expert is ever idle**: mean dead-expert count is 0.0 per layer.
Pruning always removes actively-used capacity. Code and literary prose share essentially zero
saliency ordering (Spearman near 0), and the union of per-domain top-108 lists spans 238 of
256 experts, which is why single-corpus selection breaks out-of-domain generation.

`extras/routers_trained_v2.tgz` holds the output of a router-repair experiment that failed
informatively. Retraining the router against the parent's MoE output improved layer-local
reconstruction error by 9.90% and made real perplexity 2.61% worse, degrading monotonically
from the very first step of interpolation between the original and trained weights. The
verbatim-sliced router appears to sit at a local optimum of perplexity.

Method, evaluation and known costs: see the
[model card](https://huggingface.co/turintech/GLM-5.2-MaxMin108-GGUF).