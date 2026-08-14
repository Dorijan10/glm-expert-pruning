import numpy as np, os
from gguf import GGUFReader
GG='/work/GLM-5.2-GGUF/cand/mix108_rtest.gguf'
r=GGUFReader(GG,'r+'); T={t.name:t for t in r.tensors}
orig=np.load('/work/g6/routers_orig.npz')
n=0; tot=0.0
for il in range(3,78):
    f=f'/work/g6/routers_v2/W{il:02d}.npy'
    if not os.path.exists(f): print('MISSING',f); continue
    nm=f'blk.{il}.ffn_gate_inp.weight'
    W=np.load(f).astype(np.float32)
    W0=orig[nm].reshape(W.shape)
    tot+=float(np.linalg.norm(W-W0)/np.linalg.norm(W0))
    T[nm].data[:]=W.reshape(T[nm].data.shape)
    n+=1
del r
print(f'wrote {n}/75 routers, mean relDW={tot/max(n,1):.4f}')
