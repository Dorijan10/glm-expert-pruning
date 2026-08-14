exec(open('/work/tools/p2_oracle.py').read().split("den=(Y**2)")[0].replace(
  "h=torch.from_numpy(np.fromfile(f'/work/g6/hidden_held/L{il:02d}.f16',dtype=np.float16)\n"
  "                   .reshape(-1,D)[:NTOK].copy()).to(dev)",
  "st=[0,1682,3565,4838,5465,6299,7083,7699,8293,8858]\n"
  "idx=np.concatenate([np.arange(s,s+120) for s in st])\n"
  "h=torch.from_numpy(np.fromfile(f'/work/g6/hidden_held/L{il:02d}.f16',dtype=np.float16)"
  ".reshape(-1,D)[idx].copy()).to(dev)"))
N=h.shape[0]
den=(Y**2).sum(1)+1e-12
def rel(yh): return ((yh-Y)**2).sum(1).div(den).mean().item()
num=(cur*Y).sum(1); dn=(cur*cur).sum(1)
ag=(num/den).sum()/((dn/den).sum())
at=(num/dn).unsqueeze(1)
def lstsq(cols, simplex=False):
    A=torch.gather(E,1,cols.unsqueeze(-1).expand(-1,-1,D))
    Gm=A@A.transpose(1,2)+1e-3*torch.eye(cols.shape[1],device=dev)
    w=torch.linalg.solve(Gm,A@Y.unsqueeze(-1))
    if simplex:
        one=torch.ones(N,cols.shape[1],1,device=dev)
        Gi=torch.linalg.solve(Gm,one)
        w=w+Gi*((1-(w*one).sum(1,keepdim=True))/(Gi*one).sum(1,keepdim=True))
    return (w*A).sum(1)
print(f'  tokens={N} (stratified 120 x 10 domains)')
print(f'  current router                 relL2 = {rel(cur):.6f}')
print(f'  + best GLOBAL scale a={ag:.4f}      relL2 = {rel(ag*cur):.6f}')
print(f'  + best PER-TOKEN scale         relL2 = {rel(at*cur):.6f}')
print(f'  + best weights, sum-to-1       relL2 = {rel(lstsq(ssel,True)):.6f}')
print(f'  + best weights, FREE           relL2 = {rel(lstsq(ssel)):.6f}')
allc=torch.arange(NES,device=dev).expand(N,-1)
print(f'  unconstrained all 108          relL2 = {rel(lstsq(allc)):.6f}  <- floor')
print(f'  implied expert_weights_scale   = {2.5*ag:.4f}  (currently 2.5)')
