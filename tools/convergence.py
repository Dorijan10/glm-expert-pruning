#!/usr/bin/env python3
"""Jaccard@K of the cumulative first-n shards vs the full merge, per domain.
Flat by n=3 => volume was never the lever. Still climbing at n=8 => that
domain needs more tokens before G4."""
import json, re, sys
import numpy as np
from pathlib import Path
from collections import defaultdict

SRC = Path("/work/logs/g2"); K = int(sys.argv[1]) if len(sys.argv) > 1 else 96

def acc(paths):
    tot = None; tok = 0
    for p in paths:
        d = json.load(open(p)); tok += d["tokens"]
        cur = {k: np.array(v["sal"], float) for k, v in d["layers"].items()}
        tot = cur if tot is None else {k: tot[k] + cur[k] for k in tot}
    return tot, tok

def topk(v): return set(np.argsort(-v)[:K].tolist())

doms = defaultdict(list)
for p in sorted(SRC.glob("sal_*_calib_*.json")):
    doms[re.match(r"sal_(.+)_calib_\d+\.json", p.name).group(1)].append(p)

print(f"Jaccard@{K} of cumulative shards 1..n vs full merge\n")
print(f"{'domain':<14}" + "".join(f"n={i+1:<6}" for i in range(8)) + "  tokens")
for dom in sorted(doms):
    ps = sorted(doms[dom]); full, tok = acc(ps)
    row = []
    for n in range(1, len(ps)+1):
        part, _ = acc(ps[:n])
        js = [len(topk(part[k]) & topk(full[k]))/K for k in full]
        row.append(np.mean(js))
    print(f"{dom:<14}" + "".join(f"{x:<8.3f}" for x in row) + f"  {tok:,}")
