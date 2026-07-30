#!/usr/bin/env bash
set -u
exec 9>/work/logs/g2/.g2.lock
flock -n 9 || { echo "ALREADY RUNNING"; exit 1; }
source /work/env.sh
export OMP_WAIT_POLICY=passive
BL=/work/logs/g2/blacklist.txt; touch $BL

for ATTEMPT in 1 2 3 4 5 6; do
  L=/work/logs/g2/filelist_${ATTEMPT}.txt; : > $L
  for S in /work/oracle/shards/*.txt; do
    N=$(basename "$S" .txt); OUT=/work/logs/g2/sal_${N}.json
    [ -s "$OUT" ] && python3 -c "import json;json.load(open('$OUT'))" 2>/dev/null && continue
    grep -qxF "$S" $BL && continue
    echo "$S $OUT" >> $L
  done
  CNT=$(wc -l < $L)
  echo "=== attempt $ATTEMPT: $CNT remaining $(date -Is)"
  [ "$CNT" -eq 0 ] && break
  SAL_FILELIST=$L $B/bin/llama-moe-saliency -m "$M" -ngl 99 -fa 1 \
    -ctk q8_0 -ctv q8_0 -c 4096 -b 512 \
    > /work/logs/g2/mf_attempt_${ATTEMPT}.log 2>&1
  RC=$?
  echo "    exit=$RC  done=$(ls /work/logs/g2/sal_*.json 2>/dev/null | wc -l)/80"
  if [ $RC -ne 0 ]; then
    LAST=$(grep '^\[sal\] === ' /work/logs/g2/mf_attempt_${ATTEMPT}.log | tail -1 | awk '{print $4}')
    if [ -n "$LAST" ] && [ ! -s "/work/logs/g2/sal_$(basename $LAST .txt).json" ]; then
      if grep -qxF "$LAST" /work/logs/g2/died_once.txt 2>/dev/null; then
        echo "$LAST" >> $BL; echo "    BLACKLISTED $LAST (died twice)"
      else
        echo "$LAST" >> /work/logs/g2/died_once.txt; echo "    died on $LAST (first time)"
      fi
    fi
    sleep 30
  fi
done
echo "=== G2 FINISHED $(date -Is)  total=$(ls /work/logs/g2/sal_*.json 2>/dev/null | wc -l)/80"
