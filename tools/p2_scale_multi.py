import argparse, json, time, numpy as np, torch, gguf
import torch.nn.functional as F
from gguf import GGUFReader

ap = argparse.ArgumentParser()
ap.add_argument('--layer', type=int, required=True)
ap.add_argument('--dev', type=int, default=0)
ap.add_argument('--per', type=int, default=120)
a = ap.parse_args()
D, FF, NU, NEP, NES = 6144, 2048, 8, 256, 108
il, dev = a.layer, f'cuda:{a.dev}'
PARENT='/work/GLM-5.2-GGUF/merged/GLM-5.2-IQ2_M-merged.gguf'
STUD  ='/work/GLM-5.2-GGUF/cand/mix108_maxmin.gguf'
ST=[0,1682,3565,4838,5465,6299,7083,7699,8293,8858]
def log(*s): print(f'[{time.strftime("%H:%M:%S")}] L{il:02d}',*s,flush=True)

keep=np.array(sorted(int(x) for x in
   json.load(open('/work/repo/logs/keep_mix108_maxmin.json'))['keep'][str(il)]))
keep_t=torch.from_numpy(keep).to(dev)
idx=np.concatenate([np.arange(s,s+a.per) for s in ST])
h=torch.from_numpy(np.fromfile(f'/work/g6/hidden_held/L{il:02d}.f16',
     dtype=np.float16).reshape(-1,D)[idx].copy()).to(dev)
N=h.shape[0]

TP={t.name:t for t in GGUFReader(PARENT).tensors}
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

with torch.no_grad():
    ex=lambda j,x: ((F.silu(x@G[j].T)*(x@U[j].T))@Dn[j].T).float()
    p=torch.sigmoid(h.float()@Wp.T)
    tsel=torch.topk(p+bp,NU,1).indices
    tg_=p.gather(1,tsel); tg_=tg_/tg_.sum(1,keepdim=True)
    Y=torch.zeros(N,D,device=dev)
    for j in torch.unique(tsel).tolist():
        r,s=(tsel==j).nonzero(as_tuple=True); Y[r]+=tg_[r,s,None]*ex(j,h[r])
    E=torch.empty(N,NES,D,device=dev)
    for i,j in enumerate(keep.tolist()): E[:,i,:]=ex(j,h)
    ps=p[:,keep_t]
    ssel=torch.topk(ps+bp[keep_t],NU,1).indices
    praw=ps.gather(1,ssel)
    sg=praw/praw.sum(1,keepdim=True)
    ar=torch.arange(N,device=dev)
    cur=torch.zeros(N,D,device=dev); nr=torch.zeros(N,D,device=dev)
    for k in range(NU):
        Ek=E[ar,ssel[:,k]]
        cur+=sg[:,k,None]*Ek; nr+=praw[:,k,None]*Ek
    avail=torch.isin(tsel,keep_t).float().mean().item()

    den=(Y**2).sum(1)+1e-12
    rel=lambda yh: ((yh-Y)**2).sum(1).div(den).mean().item()
    def bestc(x):
        return ((x*Y).sum(1)/den).sum()/(((x*x).sum(1)/den).sum()+1e-20)
    def ls(cols,simplex=False,target=1.0):
        A=torch.gather(E,1,cols.unsqueeze(-1).expand(-1,-1,D))
        Gm=A@A.transpose(1,2)
        Gm=Gm+1e-6*(torch.diagonal(Gm,dim1=1,dim2=2).sum(1)/cols.shape[1])[:,None,None]*torch.eye(cols.shape[1],device=dev)
        w=torch.linalg.solve(Gm,A@Y.unsqueeze(-1))
        if simplex:
            one=torch.ones(N,cols.shape[1],1,device=dev)
            Gi=torch.linalg.solve(Gm,one)
            w=w+Gi*((target-(w*one).sum(1,keepdim=True))/((Gi*one).sum(1,keepdim=True)))
        return (w*A).sum(1)
    ag=bestc(cur).item(); cnr=bestc(nr).item()
    at=(((cur*Y).sum(1))/((cur*cur).sum(1)+1e-20)).unsqueeze(1)
    allc=torch.arange(NES,device=dev).expand(N,-1)
    r=dict(layer=il, n=N, avail=avail,
           current=rel(cur), alpha=ag, kv_norm=2.5*ag, scale_global=rel(ag*cur),
           norenorm=rel(cnr*nr), kv_nonorm=cnr, scale_token=rel(at*cur),
           sum1=rel(ls(ssel,True)),
           combo=min((rel(ls(ssel,True,t)),round(t,2)) for t in [x/20 for x in range(8,25)]), free=rel(ls(ssel)), floor=rel(ls(allc)))
json.dump(r,open(f'/work/g6/scale/L{il:02d}.json','w'),indent=1)
log('DONE ' + ' '.join(f'{k}={v:.4f}' for k,v in r.items() if k!='layer'))
