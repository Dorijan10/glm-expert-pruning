# Playbook: porting a new GLM release onto one Spark

Companion to `handover.md`. That file says what was learned; this one says what to do, in
order. Written for the case where GLM-5.3 (or similar) lands and needs to fit 128 GB of
unified memory.

Target: a single artifact under about 100 GiB that leaves roughly 12 to 17 GB of headroom at
64K context.

---

## Step 0. Five checks before committing to anything

Do these first. Any one of them can invalidate the whole plan.

1. **Does llama.cpp support the architecture?** If not, everything stops. Check whether the
   arch string appears in `src/llama-model.cpp` and whether a GGUF actually loads. GLM-5.2
   needed a DSA indexer implementation, and a variant of that problem is still unresolved for
   one parent file.
2. **Read the routing metadata from a GGUF.** `expert_count`, `expert_used_count`,
   `expert_gating_func`, `expert_weights_norm`, `expert_weights_scale`, `expert_group_count`.
   These decide whether the method transfers. If `expert_group_count > 1`, selection is
   group-limited and plain top-K reasoning is wrong.
3. **Confirm the gating semantics against llama.cpp**, do not assume. Dump `ffn_moe_probs`
   and `ffn_moe_weights_norm` for one batch and reproduce them offline from `ffn_gate_inp`.
   The GLM-5.2 check matched at about 1e-7 and took an hour.
4. **Fit the size model.** Build or measure two or three artifacts at different K and fit
   `size(K) = A + B·K`. For GLM-5.2 that was `13.92 + 0.800·K` GiB. A and B are model
   specific; only the shape transfers.
5. **Check expert tensor axis order and the `expert_count` KV.** The slicer assumes experts
   are the last axis (so axis 0 under GGUFReader's reversed order) and that a single
   `expert_count` KV governs all layers. Both held for 5.2; verify for the new model.

If checks 1 to 3 fail, stop and fix those before any compression work.

---

## Step 1. Choose the parent, and consider inverting the order

**What was done for 5.2:** took Unsloth's UD-IQ2_M (222.18 GiB, already quantised) and pruned
experts out of it, copying surviving expert tensors byte for byte. Quantise, then prune.

That worked, but the order is probably backwards. Three options:

### Option A: quantise then prune (what was done)

Wait for a community dynamic quant, slice experts out of it.

*For:* no quantisation work, no imatrix generation, expert tensors are byte-identical copies
so verification is trivial, and you inherit well-tuned per-tensor precision choices.

*Against:* you are locked to whatever bit budget they chose, you cannot recover precision that
is already gone, and you depend on their release timing. It also means the imatrix was
calibrated on the **full** model, not on the pruned one.

*Cost:* a 222 GiB download, then hours.

### Option B: prune then quantise (recommended if disk and time allow)

Download BF16 weights, drop the non-surviving experts from the safetensors, rewrite the config
with the new expert count, convert to GGUF, generate an imatrix **on the pruned model**, then
quantise.

*For:* no double quantisation, since each weight is quantised once; the imatrix is calibrated
on the model you are actually shipping; and, most importantly, **you choose the bit budget
after knowing the expert count**, so you can hit the memory target exactly rather than
accepting whatever K makes someone else's bpw fit. That directly addresses the
precision-versus-count question that iteration 3 could never test.

*Against:* about 1.5 TB of download and disk, plus imatrix generation and quantisation time.
Byte-exactness verification no longer applies, so a different correctness gate is needed
(compare a few dequantised expert tensors against the BF16 originals).

*Practical wrinkle, smaller than it looks:* saliency measurement needs to run the model, and
BF16 will not fit. GLM-5.2 at BF16 is about 1,404 GiB against 251 GiB of VRAM, and even adding
all host RAM leaves you short. But this is not a new compromise: **measure saliency on the
quantised parent, which has all 256 experts and does fit**, then apply that keep-list to the
BF16 weights. The 222 GiB IQ2_M parent runs comfortably on 8x5090 at about 51 tok/s, and it is
exactly what the published 5.2 keep-list was measured on. Pruning the BF16 safetensors is pure
tensor surgery and needs no forward pass at all.

### Option C: hybrid

Prune from a community quant to get a working artifact quickly, publish it, and run Option B
in parallel as the better version.

**Recommendation:** Option B if the release matters and you have 1.5 TB free and a week.
Option A if you want something on Hugging Face within days of the model dropping, which has
its own value. Option C if both.

Note that Option A carries a credit obligation: the quantiser's work is most of the total
compression, and the model card must say so.

---

## Step 2. Budget arithmetic

Fit `size(K)`, then solve for the largest K under the target. For 5.2 the target was about
100 GiB, giving K = 108 of 256.

Also compute **bytes read per token**, which is different and matters for speed:

```
non-expert bytes + (expert bytes per expert) x (experts active per token) x n_layers
```

For 5.2 that was roughly 14.1 + 6.3 = 20.5 GB, so non-experts were about 69% of the read. This
is why pruning changes what fits but not how fast it runs, and why requantising non-experts is
the only real decode lever.

---

## Step 3. Calibration corpus

Ten disjoint English domains, roughly 200K tokens each, about 1.93 M total: code_raw,
code_instruct, web, wiki, chat, math, news, books, science, reasoning. The existing corpora are
in the repo and are reusable as-is unless the tokenizer changed substantially.

Verify document-level disjointness by sha256 between calib and eval splits. Contamination
there invalidates every number downstream.

Render chat and instruct shards with the model's own chat template, extracted from the GGUF.

---

## Step 4. Saliency

Criterion is REAP's: gate weight times the norm of the expert output, accumulated per expert
per layer. The observer patch is in the repo and hooks `ffn_moe_topk`,
`ffn_moe_weights_norm` and `ffn_moe_down`.

Run **one pass per domain** and blend afterwards. Saliency is additive over tokens, so the
whole weighting simplex is then free to explore without more GPU time.

*Verification:* per-layer `sum(cnt) == tokens x expert_used_count`, exactly. And any observer
change must reproduce a known shard bit-identically with the new code path disabled.

*Cost:* the observer runs roughly 25 times slower than plain inference because it copies
expert outputs back to host. Budget several hours, and use a filelist so the model loads once.

---

## Step 5. Max-min blend

Per-domain coverage is the routing mass a keep-list retains for domain *d*, **normalised by
what that domain's own optimal top-K would retain**. Choose domain weights maximising the
minimum coverage across domains.

Iterative reweighting works: boost the worst-covered domain by `1 + 0.25·0.94^it`, renormalise,
keep the best seen. It converges in a few dozen iterations.

*Sanity check:* several domains landing at the same coverage value is the signature of a true
max-min fixed point. For 5.2, seven of ten landed at 0.814.

*Starting weights* (5.2, likely a decent prior): code_raw 2.003, web 1.613, science 1.225,
code_instruct 1.074, books 0.844, wiki 0.738, chat 0.626, math 0.626, news 0.626,
reasoning 0.626.

*Stop condition:* if the max-min minimum is not meaningfully above the uniform blend, the
domains are not diverse enough to matter and uniform is fine. For 5.2 it was 0.814 versus
0.720, which justified the search.

---

## Step 6. Slice and verify

`glm_prune_gguf.py` handles this for the quantise-then-prune path: it slices expert-last
tensors, rewrites `expert_count`, drops the MTP block, and copies everything else.

*The gate, non-negotiable:* sample expert tensors across five spread layers and compare
against `src[keep_indices]`. The standard is 25 of 25 byte-exact, zero failures. Also confirm
attention tensors are untouched.

For the prune-then-quantise path this gate does not apply; instead dequantise a few expert
tensors from the output and compare against the BF16 originals for cosine similarity above
about 0.999.

*Trap:* `export TMPDIR` to a large filesystem first. `GGUFWriter(use_temp_file=True)` spools
through it and will fill a small root partition.

---

## Step 7. The gate ladder

Run every candidate through the same ladder, and run the parent and a comparator through it
too. Numbers from a different harness or a different llama.cpp build are not comparable.

1. **Perplexity** on held-out neutral corpora, `-c 4096 --chunks 32`, both code and general.
   Report ratios to the parent.
2. **Degeneration probes**, 13 frozen prompts at greedy decoding, plain and guarded. Score
   objectively: 4-gram distinctness below 0.55, a line repeated 4 or more times, or a word
   repeated 5 or more times. **This is the gate that catches what perplexity misses.** For 5.2,
   four of five alternative keep-lists looped on out-of-domain prompts while perplexity moved
   smoothly.
3. **MMLU**, `lm-evaluation-harness` `mmlu_generative`, 5-shot, full set. Use **generative**
   tasks; llama.cpp cannot serve loglikelihood tasks because `echo` is ignored.
4. **MBPP**, 3-shot, pass@1, code executed.
5. **Speed and residency** with `llama-bench -r 3` on the target hardware, plus resident memory
   at the serving context length.

*Server settings that matter:* `-c` is divided across slots, and 5-shot MMLU prompts run to
about 3,100 tokens, so slots need 4096 or more. Match `num_concurrent` to `-np`.

*Expect:* knowledge loss far exceeding code loss. For 5.2, pruning half the experts cost 24.5
MMLU points and essentially nothing on MBPP.

---

## Step 8. Optional decode lever

If decode speed matters, requantise **non-expert** tensors. Target attention output and query
projections and the shared expert. **Protect** the sparse-attention indexer tensors, the
compressed KV latent, and the output head.

Use quantise-then-splice: run `llama-quantize` over everything, then build the deliverable by
taking only the change-list tensors from that output and byte-copying the rest from the
original. That keeps experts byte-exact.

Expect roughly 12 to 14% decode for about 1 to 3% relative capability, and a small prefill
regression. Effective bandwidth falls as quantisation deepens, so the gain is smaller than the
byte reduction suggests.

---

## Step 9. Publish

Split to shards under 20 GB (50 GB is HF's hard per-file limit) and **verify a shard set
reloads** before uploading. Public repos only, since private storage quotas are small.

The card should lead with a comparator claim rather than a parent-retention claim, state the
absolute cost in the first few paragraphs, credit the upstream quantiser explicitly if the
artifact is derived from their work, and publish the keep-lists as a dataset so the result is
reproducible without a GPU.

---

## What not to redo

Router repair. Retraining the router after expert pruning does not work, and the reasons are
mechanical rather than incidental: sigmoid gating makes restricted-teacher KD vacuous, 83% of
the selection error is irreducible, re-choosing experts buys about 1% against re-weighting's
95%, layer-local MSE objectives embed shrinkage that mis-transfers in direction, and a
correctly trained router made real perplexity 2.61% worse. See `handover.md` for the evidence.

The vacuity argument holds for softmax gating too, so it survives a change of gating function.
The rest is specific enough to GLM-5.2 that a genuinely different architecture might behave
differently, but the burden of proof is on the new architecture.
