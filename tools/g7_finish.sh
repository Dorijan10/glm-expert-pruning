#!/usr/bin/env bash
set -u
exec 9>/work/logs/g7/.fin.lock
flock -n 9 || { echo "ALREADY RUNNING"; exit 1; }
source /work/env.sh
# 1. the membership control — code108 on MMLU + ARC
for BF in /work/data/evalbins/mmlu-test.bin /work/data/evalbins/arc-challenge-validation.bin; do
  N=$(basename $BF .bin); L=/work/logs/g7/mc_${N}_code108.log
  [ -s "$L" ] && grep -q 'Final result' "$L" && { echo "skip $N"; continue; }
  echo "=== code108 $N $(date -Is)"
  $B/bin/llama-perplexity -m "$ROOT/cand/code108.gguf" -ngl 99 -fa 1 \
    -ctk q8_0 -ctv q8_0 -c 4096 -np 8 --multiple-choice -bf "$BF" > "$L" 2>&1
  grep -H 'Final result' "$L"
done
# 2. TruthfulQA for all four, at -np 16 (variable option counts up to ~12)
BF=/work/data/evalbins/truthful-qa-validation.bin
for SPEC in "mix108:$ROOT/cand/mix108_maxmin.gguf" \
            "reap50:$ROOT/REAP50/GLM-5.2-REAP50-Q3_K_M-00001-of-00005.gguf" \
            "code108:$ROOT/cand/code108.gguf" \
            "parent:$M"; do
  T=${SPEC%%:*}; MM=${SPEC#*:}; L=/work/logs/g7/mc_truthful-qa-validation_${T}.log
  [ -s "$L" ] && grep -q 'Final result' "$L" && { echo "skip tqa $T"; continue; }
  echo "=== $T truthfulqa $(date -Is)"
  $B/bin/llama-perplexity -m "$MM" -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 \
    -c 16384 -np 16 --multiple-choice -bf "$BF" > "$L" 2>&1
  grep -H 'Final result' "$L"
done
echo "=== FINISH DONE $(date -Is)"
