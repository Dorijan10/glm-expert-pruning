import numpy as np, json, sys
from collections import defaultdict

NU = 8
def load(path):
    raw = np.fromfile(path, dtype=np.int32)
    rec = 2 + NU + NU
    assert raw.size % rec == 0, f"{path}: {raw.size} not divisible by {rec}"
    a = raw.reshape(-1, rec)
    lay, tok = a[:,0], a[:,1]
    eid = a[:,2:2+NU]
    w   = a[:,2+NU:].view(np.float32)
    return lay, tok, eid, w

keep = {int(k): np.array(sorted(int(x) for x in v))
        for k, v in json.load(open('/work/repo/logs/keep_mix108_maxmin.json'))['keep'].items()}

pl, pt, pe, pw = load('/work/logs/g6/trace_parent.bin')
sl, st, se, sw = load('/work/logs/g6/trace_mix108.bin')
print(f"parent recs={len(pl)}  student recs={len(sl)}")

P = {(l,t):(pe[i],pw[i]) for i,(l,t) in enumerate(zip(pl,pt))}
S = {(l,t):(se[i],sw[i]) for i,(l,t) in enumerate(zip(sl,st))}
common = sorted(set(P) & set(S))
print(f"common (layer,token) pairs = {len(common)}")

agg = defaultdict(lambda: dict(a=0, o=0, n=0, l1=0.0))
for k in common:
    l, _ = k
    pe_i, pw_i = P[k]; se_i, sw_i = S[k]
    kl = keep[l]
    s_glob = kl[se_i]                      # local -> global
    Sp, Ss = set(pe_i.tolist()), set(s_glob.tolist())
    d = agg[l]
    d['a'] += len(Sp & set(kl.tolist()))
    d['o'] += len(Sp & Ss)
    d['n'] += NU
    pv = np.zeros(256); pv[pe_i] = pw_i
    sv = np.zeros(256); sv[s_glob] = sw_i
    d['l1'] += float(np.abs(pv - sv).sum())

print(f"\n{'layer':>5} {'A(avail)':>9} {'O(overlap)':>11} {'R=O/A':>8} {'L1':>7}")
rows = []
for l in sorted(agg):
    d = agg[l]; n = d['n'] // NU
    A, O = d['a']/d['n'], d['o']/d['n']
    R = O/A if A > 0 else float('nan')
    L1 = d['l1']/n
    rows.append((l, A, O, R, L1))
    print(f"{l:5d} {A:9.4f} {O:11.4f} {R:8.4f} {L1:7.4f}")

r = np.array(rows)
print(f"\nMEAN   A={r[:,1].mean():.4f}  O={r[:,2].mean():.4f}  R={r[:,3].mean():.4f}  L1={r[:,4].mean():.4f}")
for name, lo, hi in (("early blk3-27", 3, 27), ("mid blk28-52", 28, 52), ("deep blk53-77", 53, 77)):
    m = (r[:,0] >= lo) & (r[:,0] <= hi)
    print(f"{name:15s} A={r[m,1].mean():.4f}  O={r[m,2].mean():.4f}  R={r[m,3].mean():.4f}  L1={r[m,4].mean():.4f}")
print(f"\nR slope vs depth: {np.polyfit(r[:,0], r[:,3], 1)[0]:+.6f} per layer")
print(f"R at blk.77 = {r[-1,3]:.4f}   [DECISION: <0.90 proceed to P0 | >0.97 reallocate to G5]")
np.savetxt('/work/logs/g6/trace_metrics.csv', r, delimiter=',',
           header='layer,A,O,R,L1', comments='', fmt='%.6f')
