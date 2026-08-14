import numpy as np, sys
from gguf import GGUFReader
t=float(sys.argv[1])
GG='/work/GLM-5.2-GGUF/cand/mix108_rtest.gguf'
r=GGUFReader(GG,'r+'); T={x.name:x for x in r.tensors}
orig=np.load('/work/g6/routers_orig.npz')
n=0; tot=0.0
for il in range(3,78):
    nm=f'blk.{il}.ffn_gate_inp.weight'
    W1=np.load(f'/work/g6/routers_v2/W{il:02d}.npy').astype(np.float32)
    W0=orig[nm].reshape(W1.shape)
    W=W0+t*(W1-W0)
    tot+=float(np.linalg.norm(W-W0)/np.linalg.norm(W0))
    T[nm].data[:]=W.reshape(T[nm].data.shape); n+=1
del r
print(f't={t} wrote {n}/75 mean relDW={tot/n:.4f}')
