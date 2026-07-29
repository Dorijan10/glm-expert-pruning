#!/usr/bin/env python3
"""Pairwise Spearman and Jaccard@K between merged per-domain saliency,
plus per-domain concentration stats. numpy only."""
import json, sys
import numpy as np
from pathlib import Path

SRC = Path("/work/logs/g3"); K = int(sys.argv[1]) if len(sys.argv) > 1 else 96

def rankdata(a):
    a = np.asarray(a, float); o = np.argsort(a)
    r = np.empty(len(a), float); r[o] = np.arange(1, len(a)+1)
    u, inv, c = np.unique(a, return_inverse=True, return_counts=True)
    if (c > 1).any():
        s = np.zeros(len(u)); np.add.at(s, inv, r); r = (s/c)[inv]
    return r

def gini(v):
    v = np.sort(np.asarray(v, float)); n = len(v)
    return float((2*np.arange(1, n+1) - n - 1).dot(v) / (n*v.sum())) if v.sum() else 0.0

D = {}
for p in sorted(SRC.glob("merged_*.json")):
    if p.name == "merged_all.json": continue
    d = json.load(open(p))
    D[p.stem.replace("merged_","")] = {k: np.array(v["sal"], float)
                                       for k, v in d["layers"].items()}
names = sorted(D)

print("Per-domain concentration (mean over 75 layers)\n")
print(f"{'domain':<14}{'Gini':>7}{'mass@'+str(K):>9}{'dead':>7}")
for n in names:
    L = D[n]
    g = np.mean([gini(L[k]) for k in L])
    m = np.mean([np.sort(L[k])[::-1][:K].sum()/L[k].sum() for k in L])
    dead = np.mean([(L[k] == 0).sum() for k in L])
    print(f"{n:<14}{g:>7.3f}{m:>9.3f}{dead:>7.1f}")

def pair(a, b, fn):
    return np.mean([fn(D[a][k], D[b][k]) for k in D[a]])
sp = lambda x, y: np.corrcoef(rankdata(x), rankdata(y))[0,1]
jc = lambda x, y: len(set(np.argsort(-x)[:K].tolist()) &
                      set(np.argsort(-y)[:K].tolist()))/K

for label, fn in (("Spearman", sp), (f"Jaccard@{K}", jc)):
    print(f"\n{label}\n")
    print(f"{'':<14}" + "".join(f"{n[:7]:>9}" for n in names))
    for a in names:
        print(f"{a:<14}" + "".join(f"{pair(a,b,fn):>9.3f}" for b in names))
