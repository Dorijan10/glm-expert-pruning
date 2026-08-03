#!/usr/bin/env bash
set -u
exec 9>/work/logs/g7/.mc.lock
flock -n 9 || { echo "ALREADY RUNNING"; exit 1; }
source /work/env.sh
BINS=$(ls /work/data/evalbins/*.bin)
run(){ local M_=$1 T_=$2
  sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
  for BF in $BINS; do
    N=$(basename $BF .bin)
    L=/work/logs/g7/mc_${N}_${T_}.log
    [ -s "$L" ] && grep -qiE 'Final result|acc' "$L" && { echo "skip $T_ $N"; continue; }
    echo "=== $T_ $N $(date -Is)"
    $B/bin/llama-perplexity -m "$M_" -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -c 4096 -np 8 \
      --multiple-choice -bf "$BF" > "$L" 2>&1
    tail -3 "$L"
  done
}
run "$ROOT/cand/mix108_maxmin.gguf" mix108
run "$ROOT/REAP50/GLM-5.2-REAP50-Q3_K_M-00001-of-00005.gguf" reap50
run "$M" parent
echo "=== MC DONE $(date -Is)"
