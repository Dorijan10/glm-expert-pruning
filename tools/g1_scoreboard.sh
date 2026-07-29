#!/usr/bin/env bash
set -u
exec 9>/work/logs/g1/.score.lock
flock -n 9 || { echo "ALREADY RUNNING"; exit 1; }
source /work/env.sh
V=/work/oracle/v2
run(){ local M_=$1 T_=$2 C_=$3
  local L=/work/logs/g1/ppl_${C_}_${T_}.log
  [ -f "$L" ] && grep -q 'Final estimate' "$L" && { echo "skip $T_ $C_"; return; }
  echo "=== $T_ $C_ $(date -Is)"
  $B/bin/llama-perplexity -m "$M_" -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 \
    -c 4096 --chunks 32 -f $V/${C_}.txt > "$L" 2>&1
  grep -H 'Final estimate' "$L"
}
for C in code_v2_eval general_v2_eval; do run "$M" parent $C; done
for C in code_v2_eval general_v2_eval; do run "$ROOT/REAP50/GLM-5.2-REAP50-Q3_K_M-00001-of-00005.gguf" reap50 $C; done
for C in code_v2_eval general_v2_eval; do run "$ROOT/cand/code108.gguf" code108 $C; done
for C in code_v2_eval general_v2_eval; do run "$ROOT/cand/code96.gguf"  code96  $C; done
echo "=== SCOREBOARD DONE $(date -Is)"
