import json, argparse, numpy as np
ap = argparse.ArgumentParser()
ap.add_argument('--code', required=True)
ap.add_argument('--general'); ap.add_argument('--eval')
ap.add_argument('--keep', type=int, nargs='+', default=[192,160,128,108,96,80])
ap.add_argument('--metric', default='sal', choices=['sal','gate','cnt'])
ap.add_argument('--emit', type=int); ap.add_argument('--out'); ap.add_argument('--random', action='store_true')
a = ap.parse_args()

def load(p, m):
    d = json.load(open(p)); return d, {int(k): np.array(v[m], float) for k, v in d['layers'].items()}
dc, C = load(a.code, a.metric)
G = load(a.general, a.metric)[1] if a.general else None
E = load(a.eval,    a.metric)[1] if a.eval    else None
NE = dc['n_expert']; L = sorted(C)
print(f"layers={len(L)} n_expert={NE} tokens={dc['tokens']} metric={a.metric}\n")

def gini(x):
    x = np.sort(x); n = len(x); c = np.cumsum(x)
    return (n + 1 - 2*np.sum(c)/c[-1]) / n
def spear(x, y):
    return np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0,1]
def jac(x, y, K):
    s=set(np.argsort(-x)[:K]); t=set(np.argsort(-y)[:K]); return len(s&t)/len(s|t)

print("CONCENTRATION  (mass captured by best K experts)")
print("  K    mass%   uniform%  ratio    worst-layer     dead(<0.1% share)")
for K in a.keep:
    fr = np.array([np.sort(C[l])[::-1][:K].sum()/(C[l].sum() or 1) for l in L])
    u = K/NE
    dead = np.mean([(C[l]/(C[l].sum() or 1) < 0.001).sum() for l in L])
    print(f"{K:4d}  {100*fr.mean():6.2f}   {100*u:6.2f}   {fr.mean()/u:5.3f}   L{L[int(fr.argmin())]} {100*fr.min():5.2f}%   ~{dead:.0f}/layer")
print(f"\nmean Gini across layers: {np.mean([gini(C[l]) for l in L]):.4f}   (0 = uniform, 1 = one expert)")

if E is not None:
    w=[spear(C[l],E[l]) for l in L]
    print(f"\nRELIABILITY  within-domain (code_calib vs code_eval): {np.mean(w):+.3f} +/- {np.std(w):.3f}")
if G is not None:
    c=[spear(C[l],G[l]) for l in L]
    print(f"SPECIALIZATION cross-domain (code vs general)        : {np.mean(c):+.3f} +/- {np.std(c):.3f}")
if E is not None and G is not None:
    print("\n  K    Jaccard_within  Jaccard_cross  chance")
    for K in a.keep:
        jw=np.mean([jac(C[l],E[l],K) for l in L]); jc=np.mean([jac(C[l],G[l],K) for l in L])
        print(f"{K:4d}      {jw:.3f}          {jc:.3f}       {(K/NE)/(2-K/NE):.3f}")
    print("\n  interpretation: within >> cross  => ranking reliable AND domains differ")
    print("                  within ~= cross  => ranking is noise")

if a.emit:
    K = a.emit
    if a.random:
        rng = np.random.default_rng(0)
        keep = {str(l): sorted(int(x) for x in rng.choice(NE, K, replace=False)) for l in L}; tag='random'
    else:
        keep = {str(l): sorted(int(x) for x in np.argsort(-C[l])[:K]) for l in L}; tag=a.metric
    out = a.out or f"/work/logs/keep_{K}.json"
    json.dump({'n_expert': NE, 'metric': tag, 'source': a.code, 'keep': keep}, open(out,'w'))
    print(f"\nwrote {out} (K={K}, {tag})")
