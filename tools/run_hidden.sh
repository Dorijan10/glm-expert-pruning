#!/bin/bash
source /work/env.sh
export OMP_WAIT_POLICY=passive
SAL_HIDDEN="$1" SAL_FILELIST="$2" \
  $B/bin/llama-moe-saliency -m "$ROOT/cand/mix108_maxmin.gguf" -ngl 99 -fa 1 \
  -ctk q8_0 -ctv q8_0 -c 4096 -b 512
echo "EXIT=$?"
