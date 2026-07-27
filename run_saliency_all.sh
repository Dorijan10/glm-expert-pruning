#!/bin/bash
source /work/env.sh
run() {
  echo "=== START $1 -> $2 @ $(date +%H:%M:%S) ==="
  SAL_OUT=/work/logs/$2 $B/bin/llama-moe-saliency -m "$M" \
    -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -c 4096 -b 512 -f "$ORACLE/$1"
  echo "=== END $2 @ $(date +%H:%M:%S) ==="
}
run corpus_code_calib.txt sal_code.json
run corpus.txt            sal_general.json
run corpus_code_eval.txt  sal_code_eval.json
echo "=== ALL DONE @ $(date +%H:%M:%S) ==="
