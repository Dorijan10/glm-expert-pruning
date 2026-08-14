import numpy as np, json, sys
from gguf import GGUFReader
src = GGUFReader('/work/GLM-5.2-GGUF/merged/GLM-5.2-IQ2_M-merged.gguf')
dst = GGUFReader(sys.argv[1])
keep = {int(k): sorted(int(x) for x in v) for k, v in json.load(open(sys.argv[2]))['keep'].items()}
S = {t.name: t for t in src.tensors}; D = {t.name: t for t in dst.tensors}
print(f"src={len(S)} dst={len(D)} (expect src-27)  blk78 present: {any(n.startswith('blk.78.') for n in D)}", flush=True)
ok = bad = 0
for il in (3, 20, 40, 60, 77):
    for b in ('ffn_gate_exps.weight','ffn_up_exps.weight','ffn_down_exps.weight','ffn_gate_inp.weight','exp_probs_b.bias'):
        n = f'blk.{il}.{b}'
        if n not in D:
            print("  MISSING", n, flush=True); bad += 1; continue
        e, g = S[n].data[keep[il]], D[n].data
        same = e.shape == g.shape and np.array_equal(e, g)
        ok += same; bad += (not same)
        print(f"  {'ok ' if same else 'BAD'} {n}", flush=True)
print("attn_output untouched:", np.array_equal(S['blk.40.attn_output.weight'].data, D['blk.40.attn_output.weight'].data), flush=True)
print(f"RESULT byte-exact={ok} failures={bad}", flush=True)
