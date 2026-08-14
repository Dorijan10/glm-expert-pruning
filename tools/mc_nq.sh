#!/usr/bin/env bash
set -u
source /work/env.sh
mkdir -p /work/logs/g7
run(){ local M_=$1 T_=$2
  for BF in $(ls /work/data/evalbins/*.bin); do
    N=$(basename $BF .bin)
    L=/work/logs/g7/mc_${N}_${T_}.log
    [ -s "$L" ] && grep -qiE 'Final result' "$L" && { echo "skip $T_ $N"; continue; }
    if [[ "$N" == *truthful* ]]; then OPTS="-c 16384 -np 16"; else OPTS="-c 4096 -np 8"; fi
    echo "=== $T_ $N $OPTS $(date -Is)"
    $B/bin/llama-perplexity -m "$M_" -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 $OPTS \
      --multiple-choice -bf "$BF" > "$L" 2>&1
    grep -h 'Final result' "$L" || tail -2 "$L"
  done
}
run "$ROOT/cand/mix108_nqA.gguf" nqA
run "$ROOT/cand/mix108_nqB.gguf" nqB
echo "=== MC DONE $(date -Is)"
