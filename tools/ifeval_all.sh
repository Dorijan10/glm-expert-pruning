#!/bin/bash
source /work/env.sh
run(){ M_=$1; T_=$2; CTX=$3; NP=$4
  pkill -f 'llama-server.*8090'; sleep 5
  setsid nohup $B/bin/llama-server -m "$M_" --alias glm-eval -ngl 99 -fa 1 \
    -ctk q8_0 -ctv q8_0 -c $CTX -np $NP -rea off --reasoning-format none --host 127.0.0.1 --port 8090 \
    > /work/logs/g6/srv_if_$T_.log 2>&1 < /dev/null &
  for i in $(seq 120); do sleep 10; curl -s -m3 localhost:8090/health | grep -q ok && break; done
  grep -E 'n_ctx_slot|out of memory' /work/logs/g6/srv_if_$T_.log | tail -2
  echo "server up: $T_ $(date -Is)"
  $PY -m lm_eval --model local-completions \
    --model_args model=glm-eval,tokenizer=zai-org/GLM-5.2,base_url=http://127.0.0.1:8090/v1/completions,num_concurrent=$NP,max_retries=1,tokenized_requests=False \
    --tasks ifeval --output_path /work/logs/g6/if_$T_
  echo "=== $T_ rc=$? $(date -Is)"
}
run "$ROOT/cand/mix108_maxmin.gguf" maxmin 65536 16
run "$ROOT/cand/mix108_nqB.gguf"    nqB    65536 16
run "$ROOT/REAP50/GLM-5.2-REAP50-Q3_K_M-00001-of-00005.gguf" reap50 32768 8
pkill -f 'llama-server.*8090'
echo "ALL DONE $(date -Is)" > /work/logs/g6/ifeval_all.DONE
