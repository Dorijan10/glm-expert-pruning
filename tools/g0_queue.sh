#!/usr/bin/env bash
set -u
exec 9>/work/logs/g0/.queue.lock
flock -n 9 || { echo "ANOTHER QUEUE IS ALREADY RUNNING — aborting"; exit 1; }
source /work/env.sh
PPL() {
  local M_=$1 C_=$2 T_=$3; shift 3
  local L=/work/logs/g0/ppl_${T_}.log
  [ -f "$L" ] && grep -q 'Final estimate' "$L" && { echo "skip $T_"; return; }
  echo "=== $T_ $(date -Is)"
  $B/bin/llama-perplexity -m "$M_" -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 \
    -c 4096 --chunks 32 -f "$C_" "$@" > "$L" 2>&1
  grep -H 'Final estimate' "$L"
}

PPL "$M" "$ORACLE/corpus.txt"            general_parent
PPL "$M" "$ORACLE/corpus_code_eval.txt"  code_parent

for K in 8 10 12 16; do
  PPL "$ROOT/cand/code96.gguf" "$ORACLE/corpus_code_eval.txt" "code_code96_k$K" \
      --override-kv glm-dsa.expert_used_count=int:$K
  PPL "$ROOT/cand/code96.gguf" "$ORACLE/corpus.txt"           "general_code96_k$K" \
      --override-kv glm-dsa.expert_used_count=int:$K
done
echo "=== QUEUE DONE $(date -Is)"
