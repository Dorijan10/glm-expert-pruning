#!/bin/bash
source /work/env.sh
gen() {  # <corpus> <refname>
  echo "=== GEN $2 from $1 @ $(date +%H:%M:%S) ==="
  $B/bin/llama-perplexity -m "$M" -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 \
    -f "$ORACLE/$1" -c 4096 --chunks 32 --kl-divergence-base "$ORACLE/$2"
  echo "=== VALIDATE $2 (must be ~0% RMS / ~100% same-top) ==="
  $B/bin/llama-perplexity -m "$M" -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 \
    -f "$ORACLE/$1" -c 4096 --chunks 32 --kl-divergence-base "$ORACLE/$2" --kl-divergence 2>&1 | egrep -i 'rms|same top'
}
gen corpus_code_eval.txt ref_iq2m_code_c4096.bin
gen corpus.txt           ref_iq2m_c4096_ch32.bin
echo "=== ALL REFS DONE @ $(date +%H:%M:%S) ==="
