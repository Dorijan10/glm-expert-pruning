#!/bin/bash
source /work/env.sh
export OMP_WAIT_POLICY=passive
SAL_FILELIST=/tmp/regress_list.txt $B/bin/llama-moe-saliency -m "$M" -ngl 99 -fa 1 \
  -ctk q8_0 -ctv q8_0 -c 4096 -b 512
echo "EXIT=$?"
