import sys, numpy as np
from gguf import GGUFReader, GGMLQuantizationType
tn = lambda t: GGMLQuantizationType(int(t)).name
o = GGUFReader('/work/GLM-5.2-GGUF/cand/mix108_maxmin.gguf')
d = GGUFReader(sys.argv[1])
O = {t.name: t for t in o.tensors}; D = {t.name: t for t in d.tensors}
print(f'tensors src={len(O)} dst={len(D)}')
ok = bad = 0
PROT = ['ffn_gate_exps.weight','ffn_up_exps.weight','ffn_down_exps.weight',
        'ffn_gate_inp.weight','exp_probs_b.bias','indexer.attn_q_b.weight',
        'indexer.attn_k.weight','indexer.proj.weight','attn_kv_a_mqa.weight']
for il in (3,20,40,60,77):
    for b in PROT:
        n = f'blk.{il}.{b}'
        if n not in D: print('  MISSING', n); bad += 1; continue
        same = np.array_equal(O[n].data, D[n].data)
        ok += same; bad += (not same)
        if not same: print('  CHANGED', n)
print(f'protected byte-exact={ok} failures={bad}')
ch = {}
for n, t in D.items():
    a, b = tn(O[n].tensor_type), tn(t.tensor_type)
    if a != b: ch[(a,b)] = ch.get((a,b), 0) + 1
print('changed:', {f'{a}->{b}': c for (a,b), c in sorted(ch.items())})
print('total changed:', sum(ch.values()))
print('indexer/kv_a touched:', any('indexer' in n or 'kv_a_mqa' in n
      for n in D if tn(D[n].tensor_type) != tn(O[n].tensor_type)))
