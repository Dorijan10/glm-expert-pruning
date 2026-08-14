import json, time, numpy as np, torch, gguf
import torch.nn.functional as F
from gguf import GGUFReader
D, FF, NU, NEP, NES = 6144, 2048, 8, 256, 108
il, dev, NTOK = 71, "cuda:0", 1200
PARENT='/work/GLM-5.2-GGUF/merged/GLM-5.2-IQ2_M-merged.gguf'
STUD  ='/work/GLM-5.2-GGUF/cand/mix108_maxmin.gguf'
def log(*s): print(f'[{time.strftime("%H:%M:%S")}]',*s,flush=True)

keep=np.array(sorted(int(x) for x in json.load(open('/work/repo/logs/keep_mix108_maxmin.json'))['keep'][str(il)]))
keep_t=torch.from_numpy(keep).to(dev)
h=torch.from_numpy(np.fromfile(f'/work/g6/hidden_held/L{il:02d}.f16',dtype=np.float16)
                   .reshape(-1,D)[:NTOK].copy()).to(dev)
TP={t.name:t for t in GGUFReader(PARENT).tensors}; TS={t.name:t for t in GGUFReader(STUD).tensors}
Wp=torch.from_numpy(TP[f'blk.{il}.ffn_gate_inp.weight'].data.reshape(NEP,D).astype(np.float32)).to(dev)
bp=torch.from_numpy(TP[f'blk.{il}.exp_probs_b.bias'].data.reshape(-1).astype(np.float32)).to(dev)

log('dequant 256 experts ...')
G=torch.empty(NEP,FF,D,dtype=torch.float16,device=dev); U=torch.empty_like(G)
Dn=torch.empty(NEP,D,FF,dtype=torch.float16,device=dev)
tg,tu,td=(TP[f'blk.{il}.ffn_{k}_exps.weight'] for k in ('gate','up','down'))
for j in range(NEP):
    G[j] =torch.from_numpy(gguf.quants.dequantize(tg.data[j:j+1],tg.tensor_type)[0].reshape(FF,D)).half()
    U[j] =torch.from_numpy(gguf.quants.dequantize(tu.data[j:j+1],tu.tensor_type)[0].reshape(FF,D)).half()
    Dn[j]=torch.from_numpy(gguf.quants.dequantize(td.data[j:j+1],td.tensor_type)[0].reshape(D,FF)).half()
log('bank ready')

@torch.no_grad()
def ex(j, x): return ((F.silu(x@G[j].T)*(x@U[j].T))@Dn[j].T)

with torch.no_grad():
    p=torch.sigmoid(h.float()@Wp.T); tsel=torch.topk(p+bp,NU,1).indices
    tg_=p.gather(1,tsel); tg_=tg_/tg_.sum(1,keepdim=True)
    Y=torch.zeros(NTOK,D,device=dev)
    for j in torch.unique(tsel).tolist():
        r,s=(tsel==j).nonzero(as_tuple=True); Y[r]+=tg_[r,s,None]*ex(j,h[r]).float()
    E=torch.empty(NTOK,NES,D,device=dev)              # survivor bank per token
    for i,j in enumerate(keep.tolist()): E[:,i,:]=ex(j,h).float()
    ps=p[:,keep_t]; ssel=torch.topk(ps+bp[keep_t],NU,1).indices
    sg=ps.gather(1,ssel); sg=sg/sg.sum(1,keepdim=True)
    cur=torch.zeros(NTOK,D,device=dev)
    for k in range(NU): cur+=sg[:,k,None]*E[torch.arange(NTOK),ssel[:,k]]

den=(Y**2).sum(1)+1e-12
def rel(yh): return ((yh-Y)**2).sum(1).div(den).mean().item()

def lstsq(cols):                                       # cols (N,k) indices
    A=torch.gather(E,1,cols.unsqueeze(-1).expand(-1,-1,D))
    Gm=A@A.transpose(1,2)+1e-3*torch.eye(cols.shape[1],device=dev)
    w=torch.linalg.solve(Gm,(A@Y.unsqueeze(-1)))
    return (w*A).sum(1), w

log('--- ORACLE DECOMPOSITION (L71, %d held tokens) ---'%NTOK)
print(f'  current router              relL2 = {rel(cur):.6f}')
print(f'  best weights on current top8 relL2 = {rel(lstsq(ssel)[0]):.6f}')
res=Y.clone(); sel=torch.zeros(NTOK,0,dtype=torch.long,device=dev)
for step in range(NU):                                  # greedy OMP
    sc=(E@res.unsqueeze(-1)).squeeze(-1).abs()/(E.norm(dim=2)+1e-9)
    if sel.numel(): sc.scatter_(1,sel,-1e9)
    sel=torch.cat([sel,sc.argmax(1,keepdim=True)],1)
    yh,_=lstsq(sel); res=Y-yh
print(f'  best 8 by greedy selection   relL2 = {rel(yh):.6f}')
allc=torch.arange(NES,device=dev).expand(NTOK,-1)
print(f'  unconstrained all 108        relL2 = {rel(lstsq(allc)[0]):.6f}   <- absolute floor')
