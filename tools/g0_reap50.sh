#!/usr/bin/env bash
set -u
exec 9>/work/logs/g0/.reap50.lock
flock -n 9 || { echo "ALREADY RUNNING — aborting"; exit 1; }
source /work/env.sh
R=$ROOT/REAP50/GLM-5.2-REAP50-Q3_K_M-00001-of-00005.gguf
for C in corpus_code_eval corpus; do
  L=/work/logs/g0/ppl_${C}_reap50.log
  [ -f "$L" ] && grep -q 'Final estimate' "$L" && { echo "skip $C"; continue; }
  echo "=== reap50 $C $(date -Is)"
  $B/bin/llama-perplexity -m "$R" -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 \
    -c 4096 --chunks 32 -f $ORACLE/$C.txt > "$L" 2>&1
  grep -H 'Final estimate' "$L"
done
echo "=== REAP50 DONE $(date -Is)"
