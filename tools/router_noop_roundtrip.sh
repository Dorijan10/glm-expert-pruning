#!/bin/bash
# Correctness gate for the in-place router patcher.
#
# The claim being tested is the one every later search depends on: patching a
# router in place changes EXACTLY the intended bytes and nothing else. A patcher
# that silently corrupted a neighbouring tensor would still produce plausible
# perplexity numbers, so the whole search would be measuring garbage. Hence a
# no-op round-trip before any real edit:
#
#   1. dump all 150 router tensors to a baseline bundle
#   2. sha256 the whole artefact
#   3. patch the routers back with their OWN values (a no-op edit)
#   4. sha256 again  -> MUST be identical, byte for byte
#   5. rerun held-out PPL -> MUST be identical to the pre-patch value
#
# Step 4 is the strong test; step 5 catches the case where the file is
# byte-identical but the engine reads routers from somewhere else than assumed.
#
# usage: router_noop_roundtrip.sh <candidate.gguf>
set -u
exec 9>/work/logs/.router_noop.lock
flock -n 9 || { echo "ALREADY RUNNING"; exit 1; }
source /work/env.sh
C=${1:-$ROOT/cand/mix108_maxmin.gguf}
mkdir -p /work/logs/r1
LOG=/work/logs/r1/noop_roundtrip.log
exec > >(tee "$LOG") 2>&1
echo "=== ROUTER PATCHER NO-OP ROUND-TRIP $(date -Is) ==="
echo "candidate: $C"

FIT=/work/corpora/router_fitness.txt
[ -s "$FIT" ] || { echo "FATAL: $FIT missing -- run build_router_fitness.py first"; exit 1; }

ppl() {  # $1=tag -> held-out fitness PPL, never a frozen eval corpus
  local L=/work/logs/r1/ppl_noop_$1.log
  $B/bin/llama-perplexity -m "$C" -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 \
    -c 4096 --chunks 16 -f "$FIT" > "$L" 2>&1
  grep -oP 'Final estimate: PPL = \K[0-9.]+' "$L"
}

echo; echo "--- 1. dump baseline router bundle ---"
$PY - "$C" <<'PY'
import sys, numpy as np
sys.path.insert(0, '/work/tools')
from glm_router_io import dump_routers
b = dump_routers(sys.argv[1])
np.savez('/work/logs/r1/routers_baseline.npz', **b)
tot = sum(v.nbytes for v in b.values())
print(f"dumped {len(b)} router tensors, {tot:,} bytes "
      f"({tot/2**20:.1f} MiB) -- expect 150 tensors")
PY

echo; echo "--- 2. PPL before patch (held-out fitness slice) ---"
P0=$(ppl before); echo "PPL before = $P0"

echo; echo "--- 3. sha256 before ---"
H0=$(sha256sum "$C" | cut -d' ' -f1); echo "$H0"

echo; echo "--- 4. no-op patch (write routers back with their own values) ---"
$PY - "$C" <<'PY'
import sys, numpy as np
sys.path.insert(0, '/work/tools')
from glm_router_io import patch_routers
z = np.load('/work/logs/r1/routers_baseline.npz')
patch_routers(sys.argv[1], {k: z[k] for k in z.files})
PY

echo; echo "--- 5. sha256 after ---"
H1=$(sha256sum "$C" | cut -d' ' -f1); echo "$H1"
if [ "$H0" = "$H1" ]; then echo "SHA256 IDENTICAL -- patcher touched only the intended bytes"
else echo "*** SHA256 CHANGED -- patcher is unsafe, do not proceed"; exit 1; fi

echo; echo "--- 6. PPL after patch ---"
P1=$(ppl after); echo "PPL after  = $P1"
[ "$P0" = "$P1" ] && echo "PPL IDENTICAL ($P0)" || echo "*** PPL MOVED: $P0 -> $P1"

echo; echo "--- 7. integrity verifier (experts must be frozen) ---"
/work/tools/verify_router_candidate.sh "$C" /work/logs/keep_mix108_maxmin.json noop

echo "=== NO-OP ROUND-TRIP DONE $(date -Is) ==="
