#!/bin/bash
source /work/env.sh
export OMP_WAIT_POLICY=passive
SAL_HIDDEN=/work/g6/hid_gate SAL_MOEOUT=/work/g6/moe_gate SAL_TRACE=/work/g6/trace_gate.bin \
SAL_FILELIST=/work/g6/gate_filelist.txt \
  $B/bin/llama-moe-saliency -m "$ROOT/cand/mix108_maxmin.gguf" -ngl 99 -fa 1 \
  -ctk q8_0 -ctv q8_0 -c 4096 -b 512
echo "EXIT=$?"
