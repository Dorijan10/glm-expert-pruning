#!/usr/bin/env python3
"""Router load-balance diagnostics, and the target load profile for a re-fit.

Two jobs:

1. `stats()` -- summarise how routing mass is distributed over the surviving
   experts in a layer: max share, entropy (normalised to [0,1]), Gini, and the
   dead-expert count at the project's existing <0.1%-share threshold (the same
   threshold `analyze_saliency.py` uses, so numbers stay comparable to the
   README's "no expert is ever idle" claim). This is the guard against the
   failure mode Balanced-KD reports: gate mass collapsing onto a handful of
   experts. Any router edit that flattens or concentrates load beyond the
   parent's own profile is suspect, and these are the numbers that catch it.

2. `target_load()` -- the reference profile a bias re-fit should aim at. NOT
   flat-uniform: the parent model genuinely specialises, and README's max-min
   blend exists precisely to preserve that structure across ten domains, so
   forcing uniform load would fight the model and destroy the domain argument.
   The defensible target is the PARENT's realised load over the 108 survivors,
   renormalised -- i.e. "route to the survivors in the same proportions the
   256-expert model did", which is the closest thing to an unpruned equilibrium
   that the survivor set can represent.

The counts come from the existing observer output (`cnt` = number of times an
expert was selected, summed over tokens; `merge_saliency.py` validates
`sum(cnt) == tokens*8`). No new GPU pass is needed for the target: the parent's
per-domain counts are already on disk in `logs/blend_*.json` / `merged_*.json`.
"""
import json

import numpy as np

DEAD_SHARE = 0.001  # <0.1% share, matching analyze_saliency.py


def _share(cnt):
    c = np.asarray(cnt, dtype=float)
    s = c.sum()
    return c / s if s > 0 else np.full_like(c, 1.0 / max(len(c), 1))


def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    c = np.cumsum(x)
    if c[-1] <= 0:
        return 0.0
    return float((n + 1 - 2 * np.sum(c) / c[-1]) / n)


def stats(cnt):
    """Load-balance summary for one layer's selection counts."""
    p = _share(cnt)
    n = len(p)
    nz = p[p > 0]
    ent = -np.sum(nz * np.log(nz))
    return dict(n_expert=n,
                max_share=float(p.max()),
                min_share=float(p.min()),
                entropy_norm=float(ent / np.log(n)) if n > 1 else 1.0,
                gini=gini(p),
                dead=int((p < DEAD_SHARE).sum()),
                p95_over_p50=float(np.percentile(p, 95) / max(np.percentile(p, 50), 1e-12)))


def summarise(per_layer):
    """Mean/worst across layers. `per_layer` is {il: cnt array}."""
    rows = {il: stats(c) for il, c in per_layer.items()}
    keys = ('max_share', 'entropy_norm', 'gini', 'dead')
    agg = {f'mean_{k}': float(np.mean([r[k] for r in rows.values()])) for k in keys}
    agg['worst_max_share'] = float(np.max([r['max_share'] for r in rows.values()]))
    agg['worst_entropy_norm'] = float(np.min([r['entropy_norm'] for r in rows.values()]))
    agg['total_dead'] = int(np.sum([r['dead'] for r in rows.values()]))
    agg['n_layers'] = len(rows)
    return agg, rows


def target_load(blend_path, keeplist_path):
    """Parent's realised load restricted to survivors and renormalised.

    Returns {il: target_share[108]} summing to 1 per layer.
    """
    b = json.load(open(blend_path))
    keep = {int(k): sorted(int(x) for x in v)
            for k, v in json.load(open(keeplist_path))['keep'].items()}
    out = {}
    for il, idx in keep.items():
        cnt = np.asarray(b['layers'][str(il)]['cnt'], dtype=float)
        sub = cnt[idx]
        tot = sub.sum()
        out[il] = sub / tot if tot > 0 else np.full(len(idx), 1.0 / len(idx))
    return out


def report(per_layer, label='', target=None):
    agg, rows = summarise(per_layer)
    print(f"--- router load balance {label} ---")
    print(f"  layers={agg['n_layers']}  mean_max_share={agg['mean_max_share']:.5f}  "
          f"worst_max_share={agg['worst_max_share']:.5f}")
    print(f"  mean_entropy_norm={agg['mean_entropy_norm']:.5f}  "
          f"worst={agg['worst_entropy_norm']:.5f}  "
          f"mean_gini={agg['mean_gini']:.4f}  dead_total={agg['total_dead']}")
    if target is not None:
        d = [float(np.abs(_share(per_layer[il]) - target[il]).sum() / 2)
             for il in per_layer if il in target]
        if d:
            print(f"  mean TV distance to parent target = {np.mean(d):.5f} "
                  f"(0 = matches parent's restricted load profile)")
    return agg


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--sal', required=True, help='observer JSON for the artefact')
    ap.add_argument('--blend', help='parent blend json for the target profile')
    ap.add_argument('--keeplist')
    a = ap.parse_args()
    d = json.load(open(a.sal))
    per_layer = {int(k): np.asarray(v['cnt'], dtype=float)
                 for k, v in d['layers'].items()}
    tgt = (target_load(a.blend, a.keeplist)
           if a.blend and a.keeplist else None)
    report(per_layer, label=a.sal.split('/')[-1], target=tgt)
