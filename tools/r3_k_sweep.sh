#!/bin/bash
# expert_used_count re-tune: sweep on held-out fitness, then spend the frozen
# gates once on the single chosen k.
#
# The distinguishing property of this repair: k is a u32 KV field, applied with
# --override-kv at load time. NO byte of the artefact is written, so every expert
# tensor AND every router tensor stays byte-identical and the ORIGINAL
# verify_candidate.sh passes an unmodified 25/25. That is a stronger artefact
# guarantee than any tensor-editing repair can offer.
#
# Gate order:
#   1  k sweep       -- selects ONLY on corpora/router_fitness.txt
#   2  integrity     -- original 25/25 byte-exactness, unmodified script
#   3  load balance  -- does raising k concentrate or spread routing mass?
#   4  13 probes     -- plain + guarded at the chosen k
#   5  frozen PPL    -- spent once, at the chosen k
#   6  MC benchmarks -- spent once; MMLU/ARC is where the -9.88/-16.39 gap lives
#
# usage: r3_k_sweep.sh ["6 8 10 12 14"]
set -u
exec 9>/work/logs/.r3.lock
flock -n 9 || { echo "ALREADY RUNNING"; exit 1; }
source /work/env.sh
KS=${1:-"6 8 10 12 14"}
C="$ROOT/cand/mix108_maxmin.gguf"
KL=/work/logs/keep_mix108_maxmin.json
mkdir -p /work/logs/r3
LOG=/work/logs/r3/run.log
exec > >(tee "$LOG") 2>&1
echo "=== R3 expert_used_count SWEEP ks=[$KS] $(date -Is) ==="
echo "artefact: $C (read-only throughout -- k is a load-time KV override)"

[ -s /work/corpora/router_fitness.txt ] || $PY /work/tools/build_router_fitness.py

# ---- 1. sweep on held-out fitness only --------------------------------------
echo; echo "--- 1. k sweep (held-out fitness; frozen sets untouched) ---"
$PY /work/tools/router_k_sweep.py --model "$C" --ks $KS \
    --outdir /work/logs/r3 || { echo "FATAL: sweep failed"; exit 1; }
K=$($PY -c "import json;print(json.load(open('/work/logs/r3/k_sweep.json'))['chosen_k'])")
echo "chosen k = $K"
OV="--override-kv glm-dsa.expert_used_count=int:$K"
if [ "$K" = "8" ]; then
  echo "NOTE: the sweep kept the inherited k=8 -- no k>8 cleared the predeclared"
  echo "      quality-per-slowdown rule. Recording that as the result; the gates"
  echo "      below then simply re-confirm the published baseline row."
fi

# ---- 2. integrity: unmodified 25/25, because nothing was written -------------
echo; echo "--- 2. integrity (original script; expect byte-exact 25/25) ---"
/work/tools/verify_candidate.sh "" "$C" "$KL" 2>&1 | \
  sed -n '/BYTE-EXACTNESS/,/RESULT/p'
echo "bytes written by this repair: 0   tensors modified: 0"

# ---- 3. does higher k concentrate or spread routing mass? -------------------
echo; echo "--- 3. load balance at k=$K ---"
SJ=/work/logs/r3/sal_k${K}.json
if [ ! -s "$SJ" ]; then
  echo "/work/corpora/router_search_train.txt $SJ" > /work/logs/r3/fl_k${K}.txt
  SAL_FILELIST=/work/logs/r3/fl_k${K}.txt $B/bin/llama-moe-saliency \
    -m "$C" -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -c 4096 -b 512 $OV \
    > /work/logs/r3/sal_k${K}.log 2>&1
fi
[ -s "$SJ" ] && $PY /work/tools/router_load_stats.py --sal "$SJ" \
    --blend /work/logs/blend_maxmin.json --keeplist "$KL"
echo "(at k=$K each token selects $K experts, so sum(cnt) == tokens*$K here,"
echo " not tokens*8 -- the observer's invariant is arity-dependent.)"

# ---- 4. the 13 frozen probes at the chosen k -------------------------------
echo; echo "--- 4. frozen generation probes at k=$K ---"
probe_at_k() {   # $1=outfile $2=extra sampling flags
  local OUT=$1; shift
  : > "$OUT"
  for SET in indomain ood; do
    KIND=IN; [ "$SET" = ood ] && KIND=OOD
    i=0
    while IFS= read -r p; do
      [ -z "$p" ] && continue
      i=$((i+1))
      { echo; echo "=====[$KIND-$i] $p"; } >> "$OUT"
      echo "" | $B/bin/llama-cli -m "$C" -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 \
        -c 4096 --seed 1 -rea off -st --simple-io -n 300 $OV "$@" \
        -p "$p" 2>/dev/null | sed -n '/^> /,$p' >> "$OUT"
    done < "$ORACLE/probes_${SET}.txt"
  done
}
probe_at_k /work/logs/probes_mix108_k${K}.txt --temp 0
probe_at_k /work/logs/probes_mix108_k${K}_guard.txt \
  --temp 0.6 --top-p 0.95 --repeat-penalty 1.10
$PY /work/tools/score_probes.py | tee /work/logs/r3/probe_scores.txt
echo "baseline to hold: IN 0.963 (0/8 degenerate) | OOD guarded 0.996 (0/5)"

# ---- 5. frozen neutral PPL, spent once at the chosen k ---------------------
echo; echo "--- 5. frozen eval PPL at k=$K ---"
V=/work/oracle/v2
for CORP in code_v2_eval general_v2_eval; do
  L=/work/logs/r3/ppl_${CORP}_k${K}.log
  [ -f "$L" ] && grep -q 'Final estimate' "$L" && { echo "skip $CORP"; continue; }
  $B/bin/llama-perplexity -m "$C" -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 \
    -c 4096 --chunks 32 -f "$V/${CORP}.txt" $OV > "$L" 2>&1
  grep -H 'Final estimate' "$L"
done

# ---- 6. MC: the axis the gap actually lives on -----------------------------
echo; echo "--- 6. multiple-choice at k=$K (MMLU ~35 min) ---"
for BF in /work/data/evalbins/mmlu-test.bin \
          /work/data/evalbins/arc-challenge-validation.bin; do
  N=$(basename $BF .bin); L=/work/logs/r3/mc_${N}_k${K}.log
  [ -s "$L" ] && grep -q 'Final result' "$L" && { echo "skip $N"; continue; }
  $B/bin/llama-perplexity -m "$C" -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 \
    -c 4096 -np 8 --multiple-choice -bf "$BF" $OV > "$L" 2>&1
  grep -H 'Final result' "$L"
done
L=/work/logs/r3/mc_truthful-qa-validation_k${K}.log
if ! { [ -s "$L" ] && grep -q 'Final result' "$L"; }; then
  $B/bin/llama-perplexity -m "$C" -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 \
    -c 16384 -np 16 --multiple-choice \
    -bf /work/data/evalbins/truthful-qa-validation.bin $OV > "$L" 2>&1
fi
grep -H 'Final result' "$L" 2>/dev/null

# ---- report in the README's own table shape --------------------------------
$PY - "$K" <<'PY'
import re, sys, json, glob
K = sys.argv[1]
def one(pat, key):
    f = glob.glob(pat)
    if not f: return None
    m = re.findall(key, open(f[0], errors='replace').read())
    return float(m[-1]) if m else None
FE = r'Final estimate: PPL = ([0-9.]+)'
FR = r'Final result: ([0-9.]+)'
c = one(f'/work/logs/r3/ppl_code_v2_eval_k{K}.log', FE)
g = one(f'/work/logs/r3/ppl_general_v2_eval_k{K}.log', FE)
mm = one(f'/work/logs/r3/mc_mmlu-test_k{K}.log', FR)
ar = one(f'/work/logs/r3/mc_arc-challenge-validation_k{K}.log', FR)
tq = one(f'/work/logs/r3/mc_truthful-qa-validation_k{K}.log', FR)
sw = json.load(open('/work/logs/r3/k_sweep.json'))
row = next((r for r in sw['rows'] if str(r['k']) == K), {})
print("\n| model | size | bpw | code PPL | ratio | general PPL | ratio | MMLU | ARC-C | TruthfulQA |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
print("| parent (256 experts) | 222.18 GiB | 2.65 | 2.1401 | 1.000 | 3.7753 | 1.000 | 46.91 | 60.87 | 39.17 |")
print("| mix108_maxmin (k=8, inherited) | 100.32 GiB | 2.65 | 2.7751 | 1.297 | 6.2115 | 1.645 | 37.03 | 44.48 | 32.93 |")
def f(x, d=4): return f"{x:.{d}f}" if x is not None else "n/a"
if c and g:
    print(f"| mix108_maxmin (k={K}, re-tuned) | 100.32 GiB | 2.65 | {f(c)} | {c/2.1401:.3f} "
          f"| {f(g)} | {g/3.7753:.3f} | {f(mm,2)} | {f(ar,2)} | {f(tq,2)} |")
print(f"\ndecode at k={K}: predicted Spark {row.get('pred_spark_tok_s','?')} tok/s "
      f"({row.get('gb_per_token','?')} GB/token) vs 8.18 at k=8 -- the cost side of the trade")
print("gap to close: -9.88 MMLU / -16.39 ARC (parent -> mix108)")
PY
echo "=== R3 DONE $(date -Is) ==="
