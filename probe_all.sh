#!/bin/bash
source /work/env.sh
echo "=== PARENT RE-BENCH (same-session baseline) $(date +%H:%M:%S) ==="
$B/bin/llama-bench -m "$M" -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -p 512 -n 128 -r 2 -d 0 2>&1 | tail -4
for pair in "parent:$M" "code96:$ROOT/cand/code96.gguf" "gen96:$ROOT/cand/gen96.gguf" "random96:$ROOT/cand/random96.gguf"; do
  tag=${pair%%:*}; mdl=${pair#*:}
  echo "=== PROBES $tag $(date +%H:%M:%S) ==="
  /work/tools/run_probes.sh "$mdl" "$tag"
done
echo "=== ALL PROBES DONE $(date +%H:%M:%S) ==="
