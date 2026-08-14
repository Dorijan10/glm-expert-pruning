#!/bin/bash
source /work/env.sh
export OMP_WAIT_POLICY=passive
MODEL="$1"; TAG="$2"
printf '/work/oracle/trace_probe.txt /work/logs/g6/sal_trace_%s.json\n' "$TAG" > /tmp/tracelist_$TAG.txt
SAL_FILELIST=/tmp/tracelist_$TAG.txt SAL_TRACE=/work/logs/g6/trace_$TAG.bin \
  $B/bin/llama-moe-saliency -m "$MODEL" -ngl 99 -fa 1 \
  -ctk q8_0 -ctv q8_0 -c 4096 -b 512
echo "EXIT=$?"
