#!/bin/bash
source /work/env.sh
MM="$ROOT/merged/GLM-5.2-IQ2_M-merged.gguf"
echo "=== SLICE code80 $(date +%H:%M:%S) ==="
PYTHONUNBUFFERED=1 $PY /work/tools/glm_prune_gguf.py --src "$MM" \
  --dst "$ROOT/cand/code80.gguf" --keeplist /work/logs/keep_code_80.json --drop-blocks 78
echo "=== VERIFY code80 $(date +%H:%M:%S) ==="
/work/tools/verify_candidate.sh "" "$ROOT/cand/code80.gguf" /work/logs/keep_code_80.json
echo "=== PROBES code80 $(date +%H:%M:%S) ==="
/work/tools/run_probes.sh "$ROOT/cand/code80.gguf" code80
echo "=== ALL code80 DONE $(date +%H:%M:%S) ==="
