import argparse, numpy as np
from gguf import GGUFReader
ap = argparse.ArgumentParser()
ap.add_argument('--gguf', default='/work/GLM-5.2-GGUF/cand/mix108_rtest.gguf')
ap.add_argument('--save', default='/work/g6/routers_orig.npz')
ap.add_argument('--eps', type=float, default=0.0)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--init', action='store_true', help='save originals and exit')
a = ap.parse_args()
D, NES = 6144, 108
names = [f'blk.{il}.ffn_gate_inp.weight' for il in range(3, 78)]

if a.init:
    r = GGUFReader(a.gguf, 'r')
    T = {t.name: t for t in r.tensors}
    np.savez(a.save, **{n: np.array(T[n].data, dtype=np.float32) for n in names})
    print(f'saved {len(names)} routers -> {a.save}'); raise SystemExit

orig = np.load(a.save)
r = GGUFReader(a.gguf, 'r+')
T = {t.name: t for t in r.tensors}
rng = np.random.default_rng(a.seed)
tot = 0.0
for n in names:
    W0 = orig[n].reshape(NES, D)
    if a.eps == 0.0:
        Wn = W0
    else:
        z = rng.standard_normal(W0.shape).astype(np.float32)
        z *= np.linalg.norm(W0) / np.linalg.norm(z)     # match Frobenius norm
        Wn = W0 + a.eps * z
        tot += float(np.linalg.norm(Wn - W0) / np.linalg.norm(W0))
    T[n].data[:] = Wn.reshape(T[n].data.shape)
del r
print(f'patched 75 routers  eps={a.eps}  mean rel|dW|={tot/75 if a.eps else 0:.4f}')
