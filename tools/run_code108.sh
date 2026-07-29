#!/bin/bash
source /work/env.sh
MM="$ROOT/merged/GLM-5.2-IQ2_M-merged.gguf"
echo "=== SLICE code108 $(date +%H:%M:%S) ==="
PYTHONUNBUFFERED=1 $PY /work/tools/glm_prune_gguf.py --src "$MM" \
  --dst "$ROOT/cand/code108.gguf" --keeplist /work/logs/keep_code_108.json --drop-blocks 78
echo "=== VERIFY code108 $(date +%H:%M:%S) ==="
/work/tools/verify_candidate.sh "" "$ROOT/cand/code108.gguf" /work/logs/keep_code_108.json
echo "=== PROBES code108 $(date +%H:%M:%S) ==="
/work/tools/run_probes.sh "$ROOT/cand/code108.gguf" code108
echo "=== ALL code108 DONE $(date +%H:%M:%S) ==="
