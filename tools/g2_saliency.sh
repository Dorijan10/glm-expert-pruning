#!/usr/bin/env bash
set -u
exec 9>/work/logs/g2/.g2.lock
flock -n 9 || { echo "ALREADY RUNNING — aborting"; exit 1; }
source /work/env.sh
export OMP_WAIT_POLICY=passive
for S in /work/oracle/shards/*.txt; do
  N=$(basename "$S" .txt)
  OUT=/work/logs/g2/sal_${N}.json
  if [ -s "$OUT" ] && python3 -c "import json,sys;json.load(open('$OUT'))" 2>/dev/null; then
    echo "skip $N"; continue
  fi
  echo "=== $N $(date -Is)"
  SAL_OUT="$OUT" $B/bin/llama-moe-saliency -m "$M" -ngl 99 -fa 1 \
    -ctk q8_0 -ctv q8_0 -c 4096 -b 512 -f "$S" \
    > /work/logs/g2/run_${N}.log 2>&1 \
    || { echo "*** FAILED $N"; rm -f "$OUT"; }
  python3 -c "
import json;d=json.load(open('$OUT'))
t=d['tokens'];L=d['layers']
bad=[k for k in L if sum(L[k]['cnt'])!=t*8]
print(f'    tokens={t} layers={len(L)} ' + ('OK' if not bad else f'*** BAD {bad[:3]}'))" 2>/dev/null
done
echo "=== G2 DONE $(date -Is)"
