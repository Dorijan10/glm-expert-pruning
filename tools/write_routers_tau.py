import numpy as np, sys
from gguf import GGUFReader
tau=float(sys.argv[1])
r=GGUFReader('/work/GLM-5.2-GGUF/cand/mix108_rtest.gguf','r+')
T={x.name:x for x in r.tensors}; orig=np.load('/work/g6/routers_orig.npz')
for il in range(3,78):
    nm=f'blk.{il}.ffn_gate_inp.weight'
    T[nm].data[:]=(tau*orig[nm]).reshape(T[nm].data.shape)
del r
print(f'tau={tau} written')
