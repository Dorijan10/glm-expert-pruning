#!/usr/bin/env python3
"""Max-min-fair blend of per-domain saliency, and keep-list emission.

Reads merged_<domain>.json (from merge_saliency.py), searches domain weights that
maximise the WORST-covered domain's saliency-mass coverage at K, writes
weights_maxmin.json and blend_<name>.json in the merged-file schema so
analyze_saliency.py --emit K can consume it directly.

usage: make_blend.py [K] [merged_dir]
"""
import json, sys
import numpy as np
from pathlib import Path

K = int(sys.argv[1]) if len(sys.argv) > 1 else 108
P = Path(sys.argv[2] if len(sys.argv) > 2 else "/work/logs/g3")

D = {p.stem.replace("merged_", ""): json.load(open(p))
     for p in sorted(P.glob("merged_*.json")) if p.name != "merged_all.json"}
names = sorted(D)
L = list(D[names[0]]["layers"])
S = {d: {k: np.array(D[d]["layers"][k]["sal"], float) for k in L} for d in names}
N = {d: {k: S[d][k] / D[d]["tokens"] for k in L} for d in names}
OWN = {d: {k: np.sort(S[d][k])[::-1][:K].sum() for k in L} for d in names}

def cover(W):
    return np.array([np.mean([
        S[d][k][np.argsort(-sum(W[x] * N[x][k] for x in names))[:K]].sum() / OWN[d][k]
        for k in L]) for d in names])

# iterative max-min: boost the worst-covered domain, step decaying toward 1.0
W = {d: 1.0 for d in names}
best = (-1.0, None)
for it in range(60):
    c = cover(W)
    if c.min() > best[0]:
        best = (c.min(), dict(W))
    W[names[int(np.argmin(c))]] *= 1.0 + 0.25 * (0.94 ** it)
    m = sum(W.values()) / len(W)
    W = {d: W[d] / m for d in W}

Wb = best[1]
c = cover(Wb)
print(f"K={K}  best MIN coverage={best[0]:.4f}  mean={c.mean():.4f}")
for d in sorted(Wb, key=lambda x: -Wb[x]):
    print(f"  {d:<15}weight {Wb[d]:.3f}   coverage {c[names.index(d)]:.3f}")
json.dump(Wb, open(P / "weights_maxmin.json", "w"), indent=1)

BLENDS = {"maxmin": Wb,
          "uniform": {d: 1.0 for d in names},
          "code2x": {d: (2.0 if d.startswith("code") else 1.0) for d in names}}
for bn, W in BLENDS.items():
    out = {"arch": D[names[0]]["arch"], "n_expert": D[names[0]]["n_expert"],
           "tokens": sum(D[d]["tokens"] for d in names),
           "blend": bn, "weights": W, "layers": {}}
    for k in L:
        out["layers"][k] = {
            "sal":  sum(W[d] * N[d][k] for d in names).tolist(),
            "cnt":  sum(np.array(D[d]["layers"][k]["cnt"], float) for d in names).tolist(),
            "gate": sum(np.array(D[d]["layers"][k]["gate"], float) for d in names).tolist()}
    json.dump(out, open(P / f"blend_{bn}.json", "w"))
    print("wrote", P / f"blend_{bn}.json")
