import numpy as np, gguf
from gguf import GGUFReader

NU, NE, D, FF = 8, 108, 6144, 2048
r = GGUFReader('/work/GLM-5.2-GGUF/cand/mix108_maxmin.gguf')
T = {t.name: t for t in r.tensors}
def sig(x): return 1.0/(1.0+np.exp(-x))

for il in (3, 40, 77):
    h  = np.fromfile(f'/work/g6/hid_gate/L{il:02d}.f16', dtype=np.float16).reshape(-1, D).astype(np.float32)
    ref= np.fromfile(f'/work/g6/moe_gate/M{il:02d}.f16', dtype=np.float16).reshape(-1, D).astype(np.float32)
    n = min(len(h), len(ref)); h, ref = h[:n], ref[:n]

    W = T[f'blk.{il}.ffn_gate_inp.weight'].data.reshape(NE, D).astype(np.float32)
    b = T[f'blk.{il}.exp_probs_b.bias'].data.reshape(-1).astype(np.float32)
    p   = sig(h @ W.T)
    sel = np.argsort(-(p + b), axis=1, kind='stable')[:, :NU]
    g   = np.take_along_axis(p, sel, axis=1)
    g  /= g.sum(1, keepdims=True)

    tg, tu, td = (T[f'blk.{il}.ffn_{k}_exps.weight'] for k in ('gate','up','down'))
    y = np.zeros((n, D), dtype=np.float32)
    for j in np.unique(sel):
        rows, slot = np.where(sel == j)
        Hj = h[rows]
        Wg = gguf.quants.dequantize(tg.data[j:j+1], tg.tensor_type)[0].astype(np.float32).reshape(FF, D)
        Wu = gguf.quants.dequantize(tu.data[j:j+1], tu.tensor_type)[0].astype(np.float32).reshape(FF, D)
        Wd = gguf.quants.dequantize(td.data[j:j+1], td.tensor_type)[0].astype(np.float32).reshape(D, FF)
        G  = Hj @ Wg.T
        A  = (G * sig(G)) * (Hj @ Wu.T)
        y[rows] += g[rows, slot][:, None] * (A @ Wd.T)

    rn = np.linalg.norm(ref)
    for scale in (1.0, 2.5):
        e = np.linalg.norm(scale*y - ref) / rn
        cos = float(np.mean(np.sum(scale*y*ref,1) /
                    (np.linalg.norm(scale*y,axis=1)*np.linalg.norm(ref,axis=1)+1e-20)))
        print(f"L{il:02d} scale={scale:<4} rel_err={e:.4e}  mean_cos={cos:.6f}")
    print(f"L{il:02d} null baseline (y=0): rel_err=1.0    ||ref||={rn:.3f}\n", flush=True)
