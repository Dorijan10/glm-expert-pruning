#!/usr/bin/env python3
"""Merge saliency shards. The REAP criterion is a sum over tokens, so merging
is exact elementwise addition -- merge(A,B) == saliency(A concat B)."""
import json, sys, re
from pathlib import Path
from collections import defaultdict

SRC = Path("/work/logs/g2"); DST = Path("/work/logs/g3"); DST.mkdir(exist_ok=True)

def merge(paths):
    out = None
    for p in paths:
        d = json.load(open(p))
        if out is None:
            out = {"arch": d["arch"], "n_expert": d["n_expert"], "tokens": 0,
                   "layers": {k: {f: list(v[f]) for f in ("cnt","gate","sal")}
                              for k, v in d["layers"].items()},
                   "shards": []}
            out["tokens"] = d["tokens"]; out["shards"].append(Path(p).name)
            continue
        assert set(d["layers"]) == set(out["layers"]), f"layer mismatch in {p}"
        out["tokens"] += d["tokens"]; out["shards"].append(Path(p).name)
        for k, v in d["layers"].items():
            for f in ("cnt","gate","sal"):
                acc = out["layers"][k][f]
                for i, x in enumerate(v[f]): acc[i] += x
    return out

def check(m, label):
    bad = [k for k, v in m["layers"].items() if abs(sum(v["cnt"]) - m["tokens"]*8) > 0.5]
    print(f"{label:22s} tokens={m['tokens']:>9,} shards={len(m['shards']):>2} "
          f"layers={len(m['layers'])} " + ("OK" if not bad else f"*** BAD {bad[:3]}"))
    return not bad

doms = defaultdict(list)
for p in sorted(SRC.glob("sal_*_calib_*.json")):
    doms[re.match(r"sal_(.+)_calib_\d+\.json", p.name).group(1)].append(p)

allp = []
for dom in sorted(doms):
    m = merge(sorted(doms[dom])); check(m, dom)
    json.dump(m, open(DST/f"merged_{dom}.json","w"))
    allp += doms[dom]
m = merge(sorted(allp)); check(m, "ALL")
json.dump(m, open(DST/"merged_all.json","w"))
print(f"\nwrote {len(doms)+1} files to {DST}")
