import argparse, json, time, numpy as np, torch, gguf
import torch.nn.functional as F
from gguf import GGUFReader
ap = argparse.ArgumentParser()
ap.add_argument('--layer', type=int, required=True)
ap.add_argument('--dev', type=int, default=0)
ap.add_argument('--lr', type=float, default=3e-6)
ap.add_argument('--epochs', type=int, default=30)
ap.add_argument('--batch', type=int, default=4096)
ap.add_argument('--tag', default='')
a = ap.parse_args()
D, FF, NU, NEP, NES = 6144, 2048, 8, 256, 108
il, dev = a.layer, f'cuda:{a.dev}'
PARENT='/work/GLM-5.2-GGUF/merged/GLM-5.2-IQ2_M-merged.gguf'
STUD  ='/work/GLM-5.2-GGUF/cand/mix108_maxmin.gguf'
DOMS=['code_raw','web','science','code_instruct','books','wiki','chat','math','news','reasoning']
CNT_F=[16591,18433,12778,6886,8783,8365,6822,6105,6008,7049]
CNT_H=[1682,1883,1273,627,834,784,616,594,565,721]
WGT=dict(zip(DOMS,[2.003,1.613,1.225,1.074,.844,.738,.626,.626,.626,.626]))
ST=[0,1682,3565,4838,5465,6299,7083,7699,8293,8858]
def log(*s): print(f'[{time.strftime("%H:%M:%S")}] L{il:02d} lr={a.lr:g}',*s,flush=True)
def wvec(cnt):
    tot,S=sum(cnt),sum(WGT.values()); w=np.empty(tot,np.float32); o=0
    for d,c in zip(DOMS,cnt): w[o:o+c]=(WGT[d]/S)/(c/tot); o+=c
    return torch.from_numpy(w).to(dev)

keep=np.array(sorted(int(x) for x in
  json.load(open('/work/repo/logs/keep_mix108_maxmin.json'))['keep'][str(il)]))
keep_t=torch.from_numpy(keep).to(dev)
ld=lambda p:torch.from_numpy(np.fromfile(p,dtype=np.float16).reshape(-1,D).copy()).to(dev)
H =ld(f'/work/g6/hidden_full/L{il:02d}.f16'); Wt=wvec(CNT_F)
Hh=ld(f'/work/g6/hidden_held/L{il:02d}.f16')
sidx=torch.from_numpy(np.concatenate([np.arange(s,s+120) for s in ST])).to(dev)
Hs=Hh[sidx]

TP={t.name:t for t in GGUFReader(PARENT).tensors}
TS={t.name:t for t in GGUFReader(STUD).tensors}
Wp=torch.from_numpy(TP[f'blk.{il}.ffn_gate_inp.weight'].data.reshape(NEP,D).astype(np.float32)).to(dev)
bp=torch.from_numpy(TP[f'blk.{il}.exp_probs_b.bias'].data.reshape(-1).astype(np.float32)).to(dev)
W0=torch.from_numpy(TS[f'blk.{il}.ffn_gate_inp.weight'].data.reshape(NES,D).astype(np.float32)).to(dev)
bs=bp[keep_t].contiguous()

log('dequant ...')
G=torch.empty(NEP,FF,D,dtype=torch.float16,device=dev); U=torch.empty_like(G)
Dn=torch.empty(NEP,D,FF,dtype=torch.float16,device=dev)
tg,tu,td=(TP[f'blk.{il}.ffn_{k}_exps.weight'] for k in ('gate','up','down'))
for j in range(NEP):
    G[j] =torch.from_numpy(gguf.quants.dequantize(tg.data[j:j+1],tg.tensor_type)[0].reshape(FF,D)).half()
    U[j] =torch.from_numpy(gguf.quants.dequantize(tu.data[j:j+1],tu.tensor_type)[0].reshape(FF,D)).half()
    Dn[j]=torch.from_numpy(gguf.quants.dequantize(td.data[j:j+1],td.tensor_type)[0].reshape(D,FF)).half()
log('bank ready')

@torch.no_grad()
def experts(h,sel):
    E=torch.zeros(h.shape[0],NU,D,dtype=torch.float16,device=dev)
    for j in torch.unique(sel).tolist():
        r,s=(sel==j).nonzero(as_tuple=True); x=h[r]
        E[r,s]=(F.silu(x@G[j].T)*(x@U[j].T))@Dn[j].T
    return E
@torch.no_grad()
def teacher(h):
    Y=torch.empty(h.shape[0],D,device=dev)
    for i in range(0,h.shape[0],a.batch):
        c=h[i:i+a.batch]; p=torch.sigmoid(c.float()@Wp.T)
        sel=torch.topk(p+bp,NU,1).indices
        g=p.gather(1,sel); g=g/g.sum(1,keepdim=True)
        Y[i:i+a.batch]=(g.unsqueeze(-1)*experts(c,sel).float()).sum(1)
    return Y
Yt=teacher(H); Ys=teacher(Hs)
log('targets done')

def fwd(h,W):
    p=torch.sigmoid(h.float()@W.T)
    sel=torch.topk(p+bs,NU,1).indices
    g=p.gather(1,sel); g=g/g.sum(1,keepdim=True)
    return (g.unsqueeze(-1)*experts(h,keep_t[sel]).float()).sum(1)
def rel(y,t,w=None):
    r=((y-t)**2).sum(1)/((t**2).sum(1)+1e-12)
    return r.mean() if w is None else (w*r).sum()/w.sum()
@torch.no_grad()
def ev(W): return rel(fwd(Hs,W),Ys).item()

try: ceil=json.load(open(f'/work/g6/scale/L{il:02d}.json'))
except FileNotFoundError: ceil={'sum1':float('nan'),'current':float('nan')}
base=ev(W0)
log(f'BASE strat={base:.6f}  ceiling sum1={ceil["sum1"]:.6f}  (json current={ceil["current"]:.6f})')
W=torch.nn.Parameter(W0.clone()); opt=torch.optim.Adam([W],lr=a.lr)
best,bw=base,W0.clone()
for ep in range(a.epochs):
    perm=torch.randperm(H.shape[0],device=dev)
    for i in range(0,H.shape[0],a.batch):
        idx=perm[i:i+a.batch]
        loss=rel(fwd(H[idx],W),Yt[idx],Wt[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    e=ev(W); rd=(W-W0).norm().item()/W0.norm().item()
    m=''
    if e<best: best,bw,m=e,W.detach().clone(),' *'
    log(f'ep {ep:3d} strat={e:.6f} gain={100*(base-e)/base:+.2f}% relDW={rd:.4f}{m}')
frac=100*(base-best)/max(base-ceil['sum1'],1e-9)
log(f'BEST={best:.6f}  gain={100*(base-best)/base:+.2f}%  CEILING_CAPTURED={frac:.1f}%')
np.save(f'/work/g6/routers_v2/W{il:02d}{a.tag}.npy', bw.cpu().numpy())
