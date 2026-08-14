#!/bin/bash
source /work/env.sh
MODEL="$1"; TAG="$2"
pkill -f 'llama-server.*8090'; sleep 5
setsid nohup $B/bin/llama-server -m "$MODEL" --alias glm-eval -ngl 99 -fa 1 \
  -ctk q8_0 -ctv q8_0 -c 32768 -np 8 -rea off --host 127.0.0.1 --port 8090 \
  > /work/logs/g6/srv_$TAG.log 2>&1 < /dev/null &
for i in $(seq 60); do sleep 10; curl -s -m3 localhost:8090/health | grep -q ok && break; done
curl -s -m3 localhost:8090/health; echo " server up: $TAG"
$PY -m lm_eval --model local-completions \
  --model_args model=glm-eval,tokenizer=zai-org/GLM-5.2,base_url=http://127.0.0.1:8090/v1/completions,num_concurrent=8,max_retries=1,tokenized_requests=False \
  --tasks mmlu_generative --num_fewshot 5 --output_path /work/logs/g6/mmlu_$TAG
echo "=== $TAG rc=$? $(date -Is)"
