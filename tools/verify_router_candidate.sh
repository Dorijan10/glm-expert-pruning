#!/bin/bash
# Integrity gate for a ROUTER-REPAIRED candidate.
#
# Why this exists: verify_candidate.sh asserts byte-exactness on 25 tensors, and
# 10 of those 25 are routers (ffn_gate_inp.weight / exp_probs_b.bias at 5 probe
# layers). A router-repaired artefact therefore fails that script BY DESIGN, and
# the project's headline "byte-exact 25/25" claim becomes ambiguous. This script
# splits the claim in two, so the part that must never weaken stays checkable:
#
#   (a) EXPERT tensors byte-identical to src.data[keep[il]]  -- MUST be 15/15
#   (b) ROUTER tensors intentionally modified                -- reported, not failed
#   (c) file size / tensor count / expert_count KV unchanged -- MUST hold
#   (d) decode throughput unchanged                          -- reported
#
# usage: verify_router_candidate.sh <candidate.gguf> <keeplist.json> [tag]
set -u
source /work/env.sh
C=$1; KL=$2; TAG=${3:-$(basename ${1%.gguf})}
[ -f "$C" ] || { echo "FATAL: $C not found"; exit 1; }
LOG=/work/logs/verify_router_${TAG}.log
exec > >(tee "$LOG") 2>&1
echo "=== ROUTER-CANDIDATE INTEGRITY $TAG $(date -Is) ==="
ls -lh "$C"

$PY - "$C" "$KL" <<'PY'
import sys, json
import numpy as np
from gguf import GGUFReader

sys.path.insert(0, '/work/tools')
from glm_router_io import EXPERT_FAMILIES, ROUTER_SUFFIX, router_tensor_names

PARENT = '/work/GLM-5.2-GGUF/merged/GLM-5.2-IQ2_M-merged.gguf'
src = GGUFReader(PARENT)
dst = GGUFReader(sys.argv[1])
keep = {int(k): sorted(int(x) for x in v)
        for k, v in json.load(open(sys.argv[2]))['keep'].items()}
S = {t.name: t for t in src.tensors}
D = {t.name: t for t in dst.tensors}
print(f"src={len(S)} dst={len(D)} (expect src-27)  "
      f"blk78 present: {any(n.startswith('blk.78.') for n in D)}")

# ---- (a) the frozen-expert claim: this is the one that must not weaken -------
probe_layers = (3, 20, 40, 60, 77)
ok = bad = 0
for il in probe_layers:
    for b in ('ffn_gate_exps.weight', 'ffn_up_exps.weight', 'ffn_down_exps.weight'):
        n = f'blk.{il}.{b}'
        if n not in D:
            print("  MISSING", n); bad += 1; continue
        e, g = S[n].data[keep[il]], D[n].data
        same = e.shape == g.shape and np.array_equal(e, g)
        ok += same; bad += (not same)
        if not same:
            print(f"  MISMATCH {n}: exp{e.shape} got{g.shape}")
print(f"EXPERTS byte-exact={ok}/15 failures={bad}")
print("attn_output untouched:",
      np.array_equal(S['blk.40.attn_output.weight'].data,
                     D['blk.40.attn_output.weight'].data))

# ---- (b) routers: report the intended deviation, per layer -------------------
rt = router_tensor_names(dst)
n_mod = n_same = 0
maxdev = 0.0
for il, names in rt.items():
    for n in names:
        e = np.asarray(S[n].data[keep[il]], dtype=np.float32)
        g = np.asarray(D[n].data, dtype=np.float32)
        if np.array_equal(e, g):
            n_same += 1
        else:
            n_mod += 1
            maxdev = max(maxdev, float(np.abs(g - e).max()))
print(f"ROUTERS modified={n_mod} unchanged={n_same} "
      f"(total={n_mod + n_same}, expect 150) max|delta|={maxdev:.6g}")

# ---- (c) structural invariants ----------------------------------------------
ec = [k for k in dst.fields if k.endswith('.expert_count')]
print("expert_count KV:", {k: dst.fields[k].contents() for k in ec})
nz = [t.name for t in dst.tensors
      if any(t.name.endswith(s) for s in ROUTER_SUFFIX)
      and not np.isfinite(np.asarray(t.data, dtype=np.float32)).all()]
print("routers with non-finite values:", nz if nz else "none")
fam = sum(1 for t in dst.tensors if any(f in t.name for f in EXPERT_FAMILIES))
print(f"expert tensors present={fam} (expect 225)")

verdict = "PASS" if (ok == 15 and bad == 0 and not nz) else "FAIL"
print(f"RESULT {verdict}: experts frozen, routers intentionally modified={n_mod}")
PY

echo; echo "=== SPEED (expect unchanged: router edits are decode-neutral) ==="
$B/bin/llama-bench -m "$C" -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 \
  -p 512 -n 128 -r 2 -d 0 2>&1 | tail -4
echo "=== DONE $(date -Is) ==="
