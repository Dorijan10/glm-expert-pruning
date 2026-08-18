# Artemis rediscovery experiment: pre-registration

Written before any run. Prompts, rubric and protocol are fixed here and must not be
edited after seeing any output.

**Question.** Given a repository that does not contain the answer, and a description of a
symptom rather than a mechanism, does Artemis propose the two methods a human team took
roughly two weeks to arrive at?

**Why these repositories.** Task 1 points at Cerebras REAP, which implements the saliency
machinery but pools its calibration corpus into a single mixture and has no notion of a
domain. Task 2 points at llama.cpp, where expert pruning does not appear at all. Neither
contains the method being tested. The team's own repository is deliberately NOT used,
because it implements both methods and documents them in its README.

---

## Task 1: expert selection

**Repository:** https://github.com/CerebrasResearch/reap

**Optional attachment:** per-domain saliency from
https://huggingface.co/datasets/turintech/GLM-5.2-MaxMin108-saliency (`saliency_merged/`)

**Prompt (verbatim):**

```
Setting. A Mixture-of-Experts language model has 256 routed experts per layer
and activates 8 per token. We must permanently delete experts so that exactly
108 remain in every layer, to fit a memory budget. Expert tensors are copied
byte-for-byte; there is no merging, no retraining, and no fine-tuning of any
kind. The only decision is which 108 to keep in each layer.

Baseline. The standard approach, implemented in this repository, scores each
expert by a saliency criterion accumulated over a calibration corpus and keeps
the top-K per layer.

Observation that motivates this task. We built several models this way, each
time using a different calibration corpus, holding size, bit budget, K and the
slicing code identical. Models calibrated on a narrow corpus behave acceptably
on text resembling that corpus, but on unrelated prompts they collapse into
repetition: three of five held-out prompts produced degenerate output, with
4-gram distinctness falling to 0.42. This happened for every narrow corpus we
tried, including one calibrated on general web text and evaluated on code.

What is available. Our calibration corpus is 1.93 M tokens composed of ten
labelled domains. We can accumulate saliency over any subset of it and we have
the per-expert statistics for each domain separately. We can also measure
held-out perplexity per domain on the resulting model.

Task. Propose a criterion for choosing which 108 experts to keep per layer that
addresses the observed failure. State the objective precisely enough to
implement, explain why it should work, and describe how you would verify it
before building a model.
```

**Rubric:**

| level | criterion |
|---|---|
| L0 | nothing relevant to expert selection |
| L1 | proposes computing saliency per domain, then aggregating by sum or mean |
| L2 | L1 plus explicitly optimises the WORST domain rather than the average (minimax, maximin, worst-case, egalitarian) |
| L3 | L2 plus normalises each domain against ITS OWN optimal top-K rather than an absolute mass |

L2 is the method. L3 is the method as implemented.

---

## Task 2: decode throughput

**Repository:** https://github.com/ggml-org/llama.cpp

**Prompt (verbatim), with the inventory pasted in place of the marker:**

```
Setting. A 100.32 GiB GGUF Mixture-of-Experts model. 108 experts per layer,
exactly 8 of which are read per token regardless of how many are resident.
Measured decode throughput is 8.24 tokens/s on hardware with about 273 GB/s of
memory bandwidth. Prefill is 244 tokens/s. Decode is memory-bandwidth-bound.

Constraint. We have already reduced the expert count as far as quality allows.
Reducing it further does not improve decode throughput at all: we measured
K=96 and K=108 and they decode within noise of each other.

What is available. The tensor inventory of the artifact is below. We can
requantise any subset of tensors and re-measure.

<INVENTORY>

Task. Explain why reducing the expert count does not improve decode, identify
where the remaining throughput is actually going, and propose a concrete
change with an estimate of the gain. Name the specific tensors you would change
and any you would deliberately leave alone, with reasoning.
```

**Inventory to paste at `<INVENTORY>`:**

```
NON-EXPERT TENSORS (14.13 GiB total)
type   tensor                      count    GiB
Q5_K   attn_output                    77   4.963
Q8_0   attn_q_b                       78   2.590
Q6_K   ffn_down_shexp                 74   0.711
Q8_0   attn_v_b                       78   0.647
Q8_0   indexer.attn_q_b               78   0.647
Q5_K   attn_q_a                       77   0.620
Q5_K   token_embd                      1   0.609
Q5_K   ffn_gate_shexp                 74   0.596
Q5_K   ffn_up_shexp                   74   0.596
Q4_K   output                          1   0.499
Q8_0   attn_k_b                       78   0.486
Q8_0   attn_kv_a_mqa                  78   0.273
F32    ffn_gate_inp                   75   0.185
Q6_K   ffn_down        (dense blk)     3   0.173
Q5_K   ffn_gate        (dense blk)     3   0.145
Q5_K   ffn_up          (dense blk)     3   0.145
Q6_K   attn_output     (dense blk)     1   0.077
Q8_0   indexer.attn_k                 78   0.061
F32    indexer.proj                   78   0.057

EXPERT TENSORS (86.19 GiB total)
IQ2_XXS  ffn_gate_exps / ffn_up_exps   148  48.29
IQ3_XXS  ffn_down_exps                  71  34.40
IQ4_XS   (mixed, some layers)            4   2.69
IQ2_S    (mixed, some layers)            2   0.81
```

**Rubric:**

| level | criterion |
|---|---|
| L0 | nothing relevant |
| L1 | identifies non-expert tensors as the dominant per-token read |
| L2 | L1 plus proposes requantising them and names specific tensors |
| L3 | L2 plus identifies tensors to protect and gives a reason (the sparse-attention indexer, the compressed KV latent) |

---

## Protocol

1. Commit this file before running anything.
2. Run each task five times independently. Fresh session each time. No follow-up
   prompting, no clarification, no hints.
3. Save every raw output verbatim, including failures, as
   `runs/task{1,2}_run{1..5}.md`.
4. Grade each output against the rubric above. Do not adjust the rubric.
5. Publish the hit rate whatever it is, including zero.

## Reference: what the human team did

For grading only. Not to be shown to Artemis.

Task 1: split the calibration corpus into ten labelled domains, accumulate saliency per
domain, define per-domain coverage as routing mass retained relative to what that domain's
own optimal top-108 would keep, then search weights that maximise the minimum coverage
across domains. Converged to a minimum of 0.814, against 0.463 for single-domain selection
and 0.720 for a uniform blend.

Task 2: non-expert tensors are 13.9 percent of the artifact but about 69 percent of the
bytes read per token, because only 8 of 108 experts are read while every non-expert tensor
is read every token. Requantising attention projections and the shared expert, while
protecting the DSA indexer and the compressed MLA latent, gave 13.6 percent more decode on
the Spark and 12.4 percent on 8 x RTX 5090.
