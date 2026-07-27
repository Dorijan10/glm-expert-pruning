#!/bin/bash
# usage: run_probes.sh <model.gguf> <tag>
source /work/env.sh
MODEL="$1"; TAG="$2"
OUT=/work/logs/probes_$TAG.txt
: > "$OUT"
echo "### MODEL=$MODEL TAG=$TAG START=$(date +%H:%M:%S)" >> "$OUT"
run_set() {
  local file=$1 kind=$2 i=0
  while IFS= read -r prompt; do
    [ -z "$prompt" ] && continue
    i=$((i+1))
    { echo; echo "=====[$kind-$i] $prompt"; } >> "$OUT"
    echo "" | $B/bin/llama-cli -m "$MODEL" -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 \
      -c 4096 --temp 0 --seed 1 -rea off -st --simple-io -n 300 \
      -p "$prompt" 2>/dev/null | sed -n '/^> /,$p' >> "$OUT"
  done < "$file"
}
run_set "$ORACLE/probes_indomain.txt" IN
run_set "$ORACLE/probes_ood.txt" OOD
{ echo; echo "### DONE $(date +%H:%M:%S)"; } >> "$OUT"
