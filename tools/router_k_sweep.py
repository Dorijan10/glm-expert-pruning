#!/usr/bin/env python3
"""Re-tune `expert_used_count` (top-k arity) for the 108-expert population.

WHY THIS IS A DIFFERENT MECHANISM FROM THE OTHER ROUTER REPAIRS
--------------------------------------------------------------
`exp_probs_b` re-fitting and router temperature both change *which* of the 108
survivors fire, or how sharply the chosen 8 are mixed. Neither changes how much
surviving capacity a token gets to use. But the parent→mix108 gap (-9.88 MMLU,
-16.39 ARC) is dominated by knowledge held in the 148 *deleted* experts, and no
reordering of survivors can restore deleted knowledge.

`expert_used_count` is the only lever available that changes per-token capacity
without touching a single expert weight. The parent chose top-8 of 256; the
slicer inherits top-8 of 108 unchanged. Raising k lets a token recruit *more*
surviving experts to cover for the specialists that were deleted — partial
compensation by breadth, since depth is gone. Lowering k tests the opposite
reading: that a thinner pool makes the marginal 8th expert a poor match that
only dilutes the mixture.

Under this architecture's routing (sigmoid affinities, `expert_weights_norm=true`,
`expert_weights_scale=2.5`) the k winners are renormalised, so raising k does not
inflate the output scale -- it redistributes the same total weight over more
experts. That makes k a genuinely well-behaved knob rather than a gain change.

MEASURED IN-REPO PRECEDENT (the reason this outranks the other levers)
---------------------------------------------------------------------
`tools/g0_queue.sh` already ran this sweep on the *code96* artefact, and
`logs/ALL_PPL_RESULTS.txt` records it:

    k     code PPL   general PPL
    8     2.0669     3.6246
    10    2.0386     3.5887      <- both axes improve
    12    2.0290     3.5952      <- best code
    16    2.0407     4.4102      <- general collapses

So k=10-12 beat the inherited k=8 on *both* axes simultaneously (-1.8% code,
-1.0% general), and k=16 is past the cliff. That is measured evidence, on this
architecture, from this harness -- which no other repair proposal in this run
has. It was never tried on `mix108_maxmin`, and never carried through to
MMLU/ARC/TruthfulQA, which is where the gap actually lives. README warns
"perplexity is a poor predictor of the generative cliff", so the PPL sweep alone
cannot answer the question; the MC gate is the discriminating test.

ZERO-MODIFICATION PROPERTY
--------------------------
k is a u32 KV field (`glm-dsa.expert_used_count`), not a tensor. It is applied at
load time with `--override-kv`, exactly as `g0_queue.sh` does. So sweeping it
writes **no bytes at all**: every expert tensor AND every router tensor stays
byte-identical, `verify_candidate.sh` passes an unmodified 25/25, and size/bpw
are untouched. Of every repair in this run this is the only one with a perfect
artefact-preservation story.

DECODE COST IS NOT FREE, AND IS REPORTED HONESTLY
-------------------------------------------------
README's "K governs residency, not decode speed" holds only because exactly 8
experts fire per token regardless of how many are resident. Raising k breaks that
premise: k experts fire, so expert bytes/token scales with k. From README's decode
table (routed experts 6.8 GB of 19.5 GB/token at k=8, ~0.85 GB per expert):

    bytes/token(k) ~= 19.5 - 6.8 + 0.85*k
    tok/s(k)       ~= 8.18 * 19.5 / bytes/token(k)      (bandwidth-bound)

    k=6  -> ~17.8 GB -> ~9.0 tok/s      k=12 -> ~22.9 GB -> ~7.0 tok/s
    k=8  -> ~19.5 GB ->  8.18 tok/s     k=14 -> ~24.6 GB -> ~6.5 tok/s
    k=10 -> ~21.2 GB -> ~7.5 tok/s

This script measures tok/s at every k rather than trusting that model, and
applies a PREDECLARED acceptance rule so the trade cannot be rationalised after
the fact (see `--min-gain-per-pct-slowdown`).

usage: router_k_sweep.py --model CAND --ks 6 8 10 12 14
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

BUILD = os.environ.get('B', '/work/build-compress')
ARCH_KEY = 'glm-dsa.expert_used_count'

# frozen baseline row for mix108_maxmin at the inherited k=8 (README lines 20-24, 63-70)
BASE = dict(k=8, code_ppl=2.7751, general_ppl=6.2115, mmlu=37.03, arc=44.48, tqa=32.93,
            tg128=51.16, spark_tok_s=8.18)
PARENT = dict(code_ppl=2.1401, general_ppl=3.7753, mmlu=46.91, arc=60.87, tqa=39.17)


def _run(cmd, log):
    with open(log, 'w') as lf:
        subprocess.call(cmd, stdout=lf, stderr=lf)
    return Path(log).read_text(errors='replace')


def ppl_at_k(model, corpus, k, log, chunks=16):
    """Held-out fitness PPL with k applied at load time. No bytes are written."""
    txt = _run([f'{BUILD}/bin/llama-perplexity', '-m', model, '-ngl', '99',
                '-fa', '1', '-ctk', 'q8_0', '-ctv', 'q8_0', '-c', '4096',
                '--chunks', str(chunks), '-f', corpus,
                '--override-kv', f'{ARCH_KEY}=int:{k}'], log)
    m = re.findall(r'Final estimate: PPL = ([0-9.]+) \+/- ([0-9.]+)', txt)
    if not m:
        raise RuntimeError(f"no PPL in {log}")
    return float(m[-1][0]), float(m[-1][1])


def speed_at_k(model, k, log):
    """tok/s at this k -- the cost side of the trade, measured not assumed."""
    txt = _run([f'{BUILD}/bin/llama-bench', '-m', model, '-ngl', '99', '-fa', '1',
                '-ctk', 'q8_0', '-ctv', 'q8_0', '-p', '512', '-n', '128',
                '-r', '2', '-d', '0',
                '-o', 'json' if False else 'md',
                '--override-kv', f'{ARCH_KEY}=int:{k}'], log)
    tg = pp = None
    for ln in txt.splitlines():
        if 'tg128' in ln:
            n = re.findall(r'([0-9]+\.[0-9]+)', ln)
            if n:
                tg = float(n[-2]) if len(n) > 1 else float(n[-1])
        if 'pp512' in ln:
            n = re.findall(r'([0-9]+\.[0-9]+)', ln)
            if n:
                pp = float(n[-2]) if len(n) > 1 else float(n[-1])
    return tg, pp


def predicted_spark_tok_s(k):
    """Bandwidth-bound estimate from README's decode table (per-expert ~0.85 GB)."""
    b = 19.5 - 6.8 + 0.85 * k
    return BASE['spark_tok_s'] * 19.5 / b, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--ks', type=int, nargs='+', default=[6, 8, 10, 12, 14])
    ap.add_argument('--fitness', default='/work/corpora/router_fitness.txt',
                    help='held-out selection signal; never a frozen eval corpus')
    ap.add_argument('--outdir', default='/work/logs/r3')
    ap.add_argument('--no-speed', action='store_true')
    # PREDECLARED acceptance rule: a k>8 candidate must buy at least this much
    # held-out PPL improvement (%) for each 1% of decode throughput it costs.
    ap.add_argument('--min-gain-per-pct-slowdown', type=float, default=0.10)
    a = ap.parse_args()

    O = Path(a.outdir); O.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(a.fitness):
        sys.exit(f"FATAL: {a.fitness} missing -- run build_router_fitness.py first")
    if 8 not in a.ks:
        a.ks = sorted(set(a.ks) | {8})   # the inherited value must be in the sweep

    print(f"=== expert_used_count sweep on {os.path.basename(a.model)} ===")
    print(f"fitness corpus: {a.fitness} (held-out calib; frozen sets untouched)")
    print(f"acceptance: >= {a.min_gain_per_pct_slowdown}% PPL gain per 1% slowdown\n")

    rows = []
    for k in sorted(a.ks):
        p, e = ppl_at_k(a.model, a.fitness, k, O / f'ppl_k{k}.log')
        tg = pp = None
        if not a.no_speed:
            tg, pp = speed_at_k(a.model, k, O / f'bench_k{k}.log')
        pred, gb = predicted_spark_tok_s(k)
        rows.append(dict(k=k, ppl=p, err=e, tg128=tg, pp512=pp,
                         pred_spark_tok_s=round(pred, 2), gb_per_token=round(gb, 1)))
        print(f"  k={k:<3} held-out PPL {p:.4f} +/- {e:.4f}   "
              f"tg128 {tg if tg else 'n/a'}   "
              f"pred Spark {pred:.2f} tok/s ({gb:.1f} GB/token)")

    base = next(r for r in rows if r['k'] == 8)
    for r in rows:
        r['ppl_gain_pct'] = 100.0 * (base['ppl'] - r['ppl']) / base['ppl']
        if r['tg128'] and base['tg128']:
            r['slowdown_pct'] = 100.0 * (base['tg128'] - r['tg128']) / base['tg128']
        else:
            r['slowdown_pct'] = 100.0 * (1 - r['pred_spark_tok_s'] / BASE['spark_tok_s'])
        r['outside_error_bar'] = (base['ppl'] - r['ppl']) > r['err']

    # apply the predeclared rule
    ok = []
    for r in rows:
        if r['k'] == 8:
            continue
        if not r['outside_error_bar']:
            r['verdict'] = 'reject: gain inside error bar'
        elif r['slowdown_pct'] <= 0:
            r['verdict'] = 'accept: faster or equal and better'
        elif r['ppl_gain_pct'] / r['slowdown_pct'] >= a.min_gain_per_pct_slowdown:
            r['verdict'] = 'accept: gain justifies the decode cost'
            ok.append(r)
        else:
            r['verdict'] = (f"reject: {r['ppl_gain_pct']:.2f}% gain for "
                            f"{r['slowdown_pct']:.1f}% slowdown")
        if r['slowdown_pct'] <= 0 and r['outside_error_bar']:
            ok.append(r)

    print("\n| k | held-out PPL | gain % | tg128 | slowdown % | verdict |")
    print("|---:|---:|---:|---:|---:|---|")
    for r in rows:
        v = 'inherited baseline' if r['k'] == 8 else r.get('verdict', '')
        tg = f"{r['tg128']:.2f}" if r['tg128'] else 'n/a'
        print(f"| {r['k']} | {r['ppl']:.4f} | {r['ppl_gain_pct']:+.2f} | {tg} | "
              f"{r['slowdown_pct']:+.1f} | {v} |")

    best = min(ok, key=lambda r: r['ppl']) if ok else base
    print(f"\nchosen k = {best['k']}"
          + ("" if ok else "  (no k>8 cleared the predeclared rule; keeping inherited 8)"))
    print("NOTE: this selection used ONLY the held-out fitness corpus. The frozen "
          "code_v2_eval / general_v2_eval, the 13 probes and the MC bins are spent "
          "once, on this single chosen k.")

    json.dump(dict(rows=rows, chosen_k=best['k'],
                   acceptance_rule=dict(
                       min_gain_per_pct_slowdown=a.min_gain_per_pct_slowdown,
                       predeclared=True),
                   baseline_row=BASE, parent_row=PARENT,
                   bytes_written=0,
                   tensors_modified=0,
                   note='k is a u32 KV applied with --override-kv; no tensor '
                        'and no byte of the artefact is modified',
                   fitness_corpus=a.fitness),
              open(O / 'k_sweep.json', 'w'), indent=1)
    print(f"wrote {O/'k_sweep.json'}")
    return best['k']


if __name__ == '__main__':
    sys.exit(0 if main() else 0)
