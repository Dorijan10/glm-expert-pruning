import numpy as np
from gguf import GGUFReader

NU, NE, D = 8, 108, 6144
GG = '/work/GLM-5.2-GGUF/cand/mix108_maxmin.gguf'
r = GGUFReader(GG); T = {t.name: t for t in r.tensors}

raw = np.fromfile('/work/g6/trace_gate.bin', dtype=np.int32).reshape(-1, 2+NU+NU)
lay, tok = raw[:,0], raw[:,1]
eid, w = raw[:,2:2+NU], raw[:,2+NU:].view(np.float32)

def sig(x): return 1.0/(1.0+np.exp(-x))

for il in (3, 40, 77):
    h = np.fromfile(f'/work/g6/hid_gate/L{il:02d}.f16', dtype=np.float16).reshape(-1, D).astype(np.float32)
    W = T[f'blk.{il}.ffn_gate_inp.weight'].data.reshape(NE, D).astype(np.float32)
    b = T[f'blk.{il}.exp_probs_b.bias'].data.reshape(-1).astype(np.float32)
    m = lay == il
    ids, wn = eid[m], w[m]
    n = min(len(h), len(ids)); h, ids, wn = h[:n], ids[:n], wn[:n]

    logits = h @ W.T
    for tag, p in (("sigmoid", sig(logits)), ("softmax",
                   np.exp(logits-logits.max(1,keepdims=True))/np.exp(logits-logits.max(1,keepdims=True)).sum(1,keepdims=True))):
        sel = np.argsort(-(p + b), axis=1, kind='stable')[:, :NU]
        idm = np.mean([len(set(a) & set(c)) for a, c in zip(sel, ids)]) / NU
        g = np.take_along_axis(p, np.sort(ids, axis=1), axis=1)
        g = g / np.maximum(g.sum(1, keepdims=True), 1e-20)
        ref = np.take_along_axis(wn, np.argsort(ids, axis=1), axis=1)
        err = np.abs(g - ref).max()
        print(f"L{il:02d} {tag:8s} top8_id_match={idm:.4f}  max_weight_err={err:.3e}")
