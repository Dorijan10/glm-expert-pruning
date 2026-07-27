#!/bin/bash
source /work/env.sh
PID=$1; C=$2; KL=$3
[ -n "$PID" ] && { echo "waiting for pid $PID ..."; while ps -p $PID >/dev/null 2>&1; do sleep 20; done; }
echo "=== slice log tail ==="; tail -4 /work/logs/slice_$(basename ${C%.gguf}).log 2>/dev/null
[ -f "$C" ] || { echo "FATAL: $C not created"; exit 1; }
ls -lh "$C"

echo; echo "=== 1. BYTE-EXACTNESS ==="
$PY - "$C" "$KL" <<'PY'
import numpy as np, json, sys
from gguf import GGUFReader
src = GGUFReader('/work/GLM-5.2-GGUF/merged/GLM-5.2-IQ2_M-merged.gguf')
dst = GGUFReader(sys.argv[1])
keep = {int(k): sorted(int(x) for x in v) for k, v in json.load(open(sys.argv[2]))['keep'].items()}
S = {t.name: t for t in src.tensors}; D = {t.name: t for t in dst.tensors}
print(f"src={len(S)} dst={len(D)} (expect src-27)  blk78 present: {any(n.startswith('blk.78.') for n in D)}")
ok = bad = 0
for il in (3, 20, 40, 60, 77):
    for b in ('ffn_gate_exps.weight','ffn_up_exps.weight','ffn_down_exps.weight','ffn_gate_inp.weight','exp_probs_b.bias'):
        n = f'blk.{il}.{b}'
        if n not in D: print("  MISSING", n); bad += 1; continue
        e, g = S[n].data[keep[il]], D[n].data
        same = e.shape == g.shape and np.array_equal(e, g)
        ok += same; bad += (not same)
        if not same: print(f"  MISMATCH {n}: exp{e.shape} got{g.shape}")
print("attn_output untouched:", np.array_equal(S['blk.40.attn_output.weight'].data, D['blk.40.attn_output.weight'].data))
print(f"RESULT byte-exact={ok} failures={bad}")
PY

echo; echo "=== 2. CODE PROBE (temp 0) ==="
echo "" | $B/bin/llama-cli -m "$C" -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 \
  -c 2048 --temp 0 --seed 1 -rea off -st --simple-io -n 200 \
  -p "Write a C++ function that reverses a singly linked list in place, and explain the pointer updates." 2>&1 | tail -22

echo; echo "=== 3. PPL / KLD vs IQ2_M (baseline code PPL = 1.9392) ==="
$B/bin/llama-perplexity -m "$C" -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 \
  -f "$ORACLE/corpus_code_eval.txt" -c 4096 --chunks 32 \
  --kl-divergence-base "$ORACLE/ref_iq2m_code_c4096.bin" --kl-divergence \
  > /work/logs/kld_$(basename ${C%.gguf}).log 2>&1
egrep -i 'Mean PPL|RMS|Same top' /work/logs/kld_$(basename ${C%.gguf}).log | head -6

echo; echo "=== 4. SPEED ==="
$B/bin/llama-bench -m "$C" -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -p 512 -n 128 -r 2 -d 0 2>&1 | tail -4
echo "=== DONE $(date +%H:%M:%S) ==="
