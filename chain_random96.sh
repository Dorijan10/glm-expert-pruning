#!/bin/bash
source /work/env.sh
while pgrep -f "glm_prune_gguf.py --src.*gen96" >/dev/null || pgrep -f "verify_candidate.sh.*gen96" >/dev/null; do sleep 30; done
echo "=== gen96 chain clear, starting random96 @ $(date +%H:%M:%S) ==="
PYTHONUNBUFFERED=1 $PY /work/tools/glm_prune_gguf.py --src "$MM" \
  --dst "$ROOT/cand/random96.gguf" --keeplist /work/logs/keep_random_96.json \
  --drop-blocks 78 > /work/logs/slice_random96.log 2>&1
/work/tools/verify_candidate.sh "" "$ROOT/cand/random96.gguf" /work/logs/keep_random_96.json \
  > /work/logs/verify_random96.log 2>&1
echo "=== random96 done @ $(date +%H:%M:%S) ==="
