# Handover: GLM-5.2 compression campaign

Written 2026-08-19 for a fresh session. Covers what was built, what was published, what was
proven not to work, and what is available to continue with.

---

## 1. What exists now

Two GGUF artifacts are published and public on Hugging Face under the TurinTech org:

**`turintech/GLM-5.2-MaxMin108-GGUF`**

| variant | size | bpw | note |
|---|---:|---:|---|
| `GLM-5.2-MaxMin108-IQ2_M` | 100.32 GiB | 2.650 | reference artifact |
| `GLM-5.2-MaxMin108-IQ2_M-fast` | 95.77 GiB | 2.530 | recommended; faster, smaller |

Six shards each (50 GB per-file HF limit forces splitting). 324.83B parameters, down from
GLM-5.2's 753.86B, by keeping 108 of 256 routed experts per layer.

**`turintech/GLM-5.2-MaxMin108-saliency`** (dataset) holds the keep-lists, per-domain
saliency, raw eval output, the importance matrix, and the trained routers from the failed
router experiment.

**`github.com/Dorijan10/glm-expert-pruning`** holds the tooling, the observer source, the
logs, `MODEL_CARD.md`, `DATASET_CARD.md` and `requirements.txt`. Personal account, not the
org, because the company git is not reachable externally. Head commit `d715e90` plus later
card syncs.

### Headline measurements

All from the same harness, run on the same machines, for all four models including the parent
and the comparator.

| model | size | MMLU (5-shot generative) | MBPP pass@1 |
|---|---:|---:|---:|
| GLM-5.2 UD-IQ2_M (parent) | 222.18 GiB | 87.29 ±0.27 | 0.710 ±0.020 |
| REAP-50 Q3_K_M (comparator) | 169.30 GiB | 62.75 ±0.39 | 0.708 ±0.020 |
| **MaxMin108** | 100.32 GiB | **63.00 ±0.39** | 0.626 ±0.022 |
| **MaxMin108-fast** | 95.77 GiB | 61.84 ±0.40 | 0.592 ±0.022 |

Defensible claim: **level with REAP-50 on knowledge, behind on code, at 59% of its size**, on
hardware where REAP-50 does not fit. Not "beats REAP-50"; the MMLU difference is inside the
error bars.

Speed, same-session pairs: GB10 Spark decode 8.24 to 9.36 tok/s (+13.6%); 8x RTX 5090 decode
51.21 to 57.58 (+12.4%); prefill on the 5090 box drops 1.8% for the fast variant. Spark
residency at 64K context is 109 GB (base) and 104 GB (fast) of 121 GB.

---

## 2. The two methods that worked

### Max-min fair expert selection

Saliency is the REAP criterion (gate weight times the norm of the expert output) accumulated
over 1.93 M calibration tokens across **ten labelled domains** (code_raw, code_instruct, web,
wiki, chat, math, news, books, science, reasoning).

The contribution is the objective, not the criterion. Define per-domain coverage as the
routing mass a keep-list retains for domain *d*, **normalised by what that domain's own
optimal top-K would retain**. Then choose domain weights that **maximise the minimum coverage
across domains** rather than the sum or mean.

Result: worst-case coverage 0.463 (single-domain selection) to 0.720 (uniform blend) to
**0.814** (max-min). Raising K from 108 to 128 buys only +0.050 by comparison, so the
weighting matters more than the capacity.

Why it matters: five keep-lists built at identical size, bit budget, K and slicer, varying
only the calibration corpus. Four of them **degenerate on out-of-domain prompts**, looping at
greedy decoding, 3 of 5 or worse. The max-min version is 0 of 5.

### Non-expert requantisation

Non-expert tensors are 13.9% of the artifact but roughly **69% of the bytes read per token**,
because only 8 of 108 experts are read per token while every non-expert tensor is read every
token. The source quantisation upcasts everything except routed experts, so this is where the
remaining decode headroom sits.

The `-fast` variant requantises 616 tensors (attention projections, shared expert,
token_embd) while **deliberately protecting** the DSA sparse-attention indexer tensors, the
compressed MLA latent `attn_kv_a_mqa`, and `output.weight`.

Measured side-finding worth carrying forward: **effective memory bandwidth falls as
quantisation deepens** (160.7 to 151.5 to 138.8 GB/s across the recipes). Cheaper unpacking
buys back part of what the extra bytes cost, which is why removing 24% of the bytes read
yields 13.6% more speed rather than the naive 32%, and why prefill gets slightly slower.

---

## 3. What was proven not to work

**Router repair is dead for this artifact.** This consumed roughly two weeks and is the most
useful negative result available. Do not redo it.

- **Sigmoid vacuity.** GLM gates with per-expert sigmoid plus a selection-only bias, then
  renormalises. Sigmoid is elementwise, so deleting expert rows does not change any surviving
  expert's score. The sliced router already computes the teacher's distribution restricted to
  survivors. KD against that target has **zero gradient**.
- **83/17 split.** Of the selection error, 83% is irreducible (the experts are gone) and only
  17% is router-repairable.
- **Substitute mapping refuted.** Re-choosing which 8 experts to use buys about 1%;
  re-weighting the ones already chosen buys 95%.
- **Shrinkage artifact.** Layer-local MSE fits wanted `expert_weights_scale` around 1.80;
  real perplexity wanted 2.83. The optimal least-squares scalar is below 1 for any imperfect
  predictor, which is generic shrinkage, not a mechanism. **A layer-local optimum can
  mis-transfer in direction, not just magnitude.**
- **Negative transfer.** Training routers properly improved layer-local reconstruction by
  9.90% across all 75 layers and made real perplexity **2.61% worse**, degrading monotonically
  from the first step of interpolation. Worse than random noise of twice the magnitude.
- **Local optimum.** Random perturbation, trained direction, temperature up and down, expert
  count up and down: every direction tested makes it worse. The verbatim-sliced router sits at
  a local optimum of perplexity.

**Other closed avenues:** expert merging (violates byte-exact preservation and REAP's own
paper argues pruning beats merging at 50%); no-renorm (`expert_weights_norm=False`, worse at
5 of 8 layers); per-layer non-uniform K (blocked by llama.cpp's single `expert_count` KV);
a Q3_K-free requant recipe (measured, strictly worse than the shipped one).

---

## 4. Evaluation: the trap that nearly sank the release

The campaign's internal harness was `llama-perplexity --multiple-choice`, zero-shot
likelihood scoring. It gave the **parent** an MMLU of 46.91. That is not a leaderboard-scale
number, and publishing 37.03 for the compressed model would have made a respectable model
look broken.

Switching to `lm-evaluation-harness` `mmlu_generative` 5-shot moved mix108 from 37.03 to
**63.00** and, more seriously, **overturned a conclusion**: the old harness said REAP-50 beat
mix108 by 3.06 MMLU at 5 sigma. The standard harness says they are level. That was a harness
artifact.

Practical notes for any future eval:

- llama.cpp's server **cannot do loglikelihood tasks**: `echo=true` is ignored on
  `/v1/completions`, so no prompt-token logprobs. Every loglikelihood task (ARC, HellaSwag,
  Winogrande, TruthfulQA MC) is unavailable. Use **generative** task variants.
- IFEval could not be completed, twice. llama.cpp's server strictly parses model output
  against the template's declared reasoning and tool-call grammars and returns HTTP 500 on a
  parse failure. Roughly 1 to 4% of prompts triggered it on the pruned models across four
  server configurations; the parent never did. Deterministic generation means retries cannot
  help. That the compressed models occasionally emit unparseable structured output is itself
  a compression artifact and is reported in the card.
- Working invocation is in the repo; the key arguments are `--model local-completions`,
  `tokenizer=zai-org/GLM-5.2`, `tokenized_requests=False`, and `num_concurrent` matched to the
  server's `-np`.
- `-c` is divided across slots in llama-server. MMLU 5-shot prompts run to about 3,077 tokens,
  so slots need 4096 or more.

---

## 5. Infrastructure

**8x RTX 5090 box** (`ezc-turintech-n08b`, Ubuntu 22.04): working as of 2026-08-19. 8 GPUs,
257 GiB VRAM total, driver 580.173.02, CUDA toolkits 12.9 and 13.3 both present with nvcc
resolving to 13.3.

Everything lives under `/data/container-runtime/glm` (898 GB, root-owned so new directories
need `sudo mkdir` plus `chown`):

- `GLM-5.2-GGUF/` — the UD-IQ2_M parent (6 shards, 222.18 GiB) and REAP-50 Q3_K_M (5 shards)
- `GLM-5.2-pcuenq/GLM-5.2-q3_k_m.gguf` — **334 GiB Q3-class parent**, never used
- `oracle/` — the 10-domain calibration corpora, 59 GB
- `llama.cpp/` — source tree with GLM-DSA support
- `build-n08b/` — freshly built llama-bench, llama-perplexity, llama-server, llama-quantize

Root filesystem is small (98 GB, roughly 5 GB free) and Docker holds 429 GB on the data mount
with about 200 GB reclaimable. Do not build or stage anything on root.

**The GB10 Spark** is released: services stopped and disabled, model files deleted, 261 GB
free. Redeploying means re-downloading roughly 96 GB from Hugging Face.

**Rescue set** on Dori's laptop at `~/Downloads/rescue`: full logs, corpora, tooling,
importance matrix, trained routers, `env.sh`, `requirements.txt`.

### Verified state as of 2026-08-19

Prepared and working: driver 580.173.02 with all 8 GPUs visible; fresh build at
`$GLM/build-n08b`; the tooling repo cloned to `$GLM/repo`; a venv at `$GLM/venv` with gguf,
huggingface_hub, lm-eval, transformers; the published keep-lists and saliency pulled to
`$GLM/saliency-data`; and an `env.sh` exporting `GLM`, `B`, `PY`, `PARENT_IQ2`, `PARENT_Q3`,
`REAP50`, `ORACLE`, `TMPDIR`.

The ten-domain corpora are at **`$GLM/repo/corpora`** (10 `*_calib.txt`, 12 `*_eval.txt`).
`$GLM/oracle` is unrelated PR25407 benchmark data, not the calibration corpora.

Two open items:

**The build is not the campaign pin.** `7f575c39d` does not exist in this clone, so the build
is `0badc06ab` (build 9959). Numbers from it are not directly comparable to the published
card; re-measure baselines and say so.

**The 334 GiB Q3 parent does not load, and the fix is not trivial.** It carries indexer
tensors on only 22 of 79 layers (110 tensors at 5 per layer), while the IQ2 parent carries
them on all 79. The difference is exactly 285 = (79 − 22) × 5. Stock llama.cpp demands
indexer tensors on every layer and refuses the file.

`$GLM/glm-dsa.patch` addresses exactly this, but it is **incomplete**: it applies cleanly yet
fails to compile, because its `glm-dsa.cpp` hunk references `hparams.is_indexer_full_impl`,
`hparams.is_indexer_full()` and `LLM_KV_ATTENTION_INDEXER_TYPES`, none of which exist in this
tree. The supporting changes to `llama-hparams.h` and `llama-arch.h/cpp` are missing from the
patch file. Making the Q3 parent usable means writing that support by hand, or finding the
upstream PR that introduced per-layer indexer types. The patch was reverted and the build
restored.

Note that `GGUFReader` reads one file at a time, so it cannot inspect the split IQ2 parent
from shard 1, which is metadata only.

---

## 6. Working conventions that matter

- **`setsid nohup cmd &` prints `[1]+ Done` for the wrapper immediately.** That is not the
  job finishing. Only `pgrep -af <binary>` is authoritative. Block with
  `while pgrep -f '<binary>[.]py' >/dev/null; do sleep 30; done`, using the bracket so the
  monitor does not match its own pattern.
- **Heredocs and `< /dev/null` conflict.** `python - <<'EOF' ... < /dev/null` silently runs an
  empty program. Write scripts to a file.
- **`export TMPDIR=/work/tmp`** (or equivalent on the big mount) before any GGUF write;
  `GGUFWriter(use_temp_file=True)` spools through `TMPDIR` and will fill a small root.
- **In gguf-py, `str(t.tensor_type)` renders the integer**, not the enum name. Use
  `GGMLQuantizationType(int(tt)).name`.
- Model loads take minutes and llama.cpp prints nothing during them. A frozen log after
  `done_getting_tensors` is normal.
- Verify byte-exactness after any slicing: sample expert tensors across layers 3, 20, 40, 60
  and 77 and compare against `src[keep_indices]`. The campaign standard is 25 of 25, zero
  failures.

---

## 7. Open threads

**Not started, feasible now:** the K=80 at 3 bpw precision experiment. The question is whether
spending the byte budget on **expert precision** rather than **expert count** wins. REAP-50's
own two builds imply roughly 4.4 MMLU points per bpw, predicting about +0.66 MMLU, roughly
five times what adding experts buys. It was blocked on needing a Q3-class parent; that parent
is now on the box.

**In progress:** a pre-registered experiment testing whether Artemis, pointed at repositories
that do not contain the answer, proposes the two methods above. Prompts, rubric and protocol
are in `ARTEMIS_DISCOVERY_PREREG.md`. Run 1 of Task 1 scored top of the rubric, deriving the
per-domain normalisation and the maximin objective, and adding leave-one-domain-out validation
the campaign never ran. Four more runs are needed before any claim.

**Considered and parked:** benchmarking on Apple Silicon. GLM-DSA on Metal is confirmed
working by pipenetwork's own card. No provider rents Apple hardware for less than 24 hours
(an Apple licensing rule). AWS `mac-m4max.metal` is 128 GB at $6.25/hr with a $150 floor and
would be tight; M3 Ultra 256 GB fits comfortably but is monthly only.

**Housekeeping:** the model and dataset cards cross-link, but the GitHub repo is under a
personal account. LinkedIn posts are drafted but unpublished; the recommended order leads with
the router negative result, then the knowledge-versus-code dissociation, then the release.

---

## 8. Framing for the next model

The user's stated intent is to try something **radically different** for the next release
rather than incrementally improving MaxMin108.

Two findings from this campaign that should shape whatever comes next:

**Expert pruning destroys knowledge far faster than procedural skill.** REAP-50 lost 24.54
MMLU points while remaining statistically identical to the parent on MBPP. MaxMin108 retains
88.2% of parent MBPP against 72.2% of parent MMLU. Knowledge appears distributed across
experts; procedural coding ability appears redundant across them. Any method that targets
knowledge retention specifically is attacking the real loss.

**Perplexity ratio and task capability disagreed.** Code perplexity degraded more than general
perplexity while code capability degraded less. Choosing a compression recipe on perplexity
ratios can optimise the wrong thing.

The campaign's credibility rests on reporting the 24-point MMLU loss, a prefill regression,
two harness errors of its own, and a benchmark it could not complete. Whatever comes next
should hold that standard.
