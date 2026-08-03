#!/usr/bin/env bash
set -u
source /work/env.sh
TQA=/work/data/evalbins/truthful-qa-validation.bin

# 1. parent TruthfulQA retry — LEAN config: 222 GiB model leaves only ~4 GiB/card
L=/work/logs/g7/mc_truthful-qa-validation_parent.log
if ! grep -q 'Final result' "$L" 2>/dev/null; then
  echo "=== parent truthfulqa retry $(date -Is)"
  $B/bin/llama-perplexity -m "$M" -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 \
    -c 4096 -np 16 -b 512 -ub 256 --multiple-choice -bf "$TQA" > "$L" 2>&1
  grep -H 'Final result' "$L" || { echo "parent TQA still failing:"; tail -3 "$L"; }
fi

# 2. REAP-50 Q2_K — 129 GiB, plenty of headroom, same config as the other models
Q=$(ls $ROOT/REAP50_Q2K/*00001*.gguf); echo "model: $Q"
for BF in /work/data/evalbins/mmlu-test.bin /work/data/evalbins/arc-challenge-validation.bin; do
  N=$(basename $BF .bin); L=/work/logs/g7/mc_${N}_reap50q2k.log
  [ -s "$L" ] && grep -q 'Final result' "$L" && { echo "skip $N"; continue; }
  echo "=== reap50q2k $N $(date -Is)"
  $B/bin/llama-perplexity -m "$Q" -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 \
    -c 4096 -np 8 --multiple-choice -bf "$BF" > "$L" 2>&1
  grep -H 'Final result' "$L"
done
L=/work/logs/g7/mc_truthful-qa-validation_reap50q2k.log
echo "=== reap50q2k truthfulqa $(date -Is)"
$B/bin/llama-perplexity -m "$Q" -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 \
  -c 16384 -np 16 --multiple-choice -bf "$TQA" > "$L" 2>&1
grep -H 'Final result' "$L"
echo "=== Q2K DONE $(date -Is)"
