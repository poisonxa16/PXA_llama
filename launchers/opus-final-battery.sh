#!/bin/bash
# FINAL tuned opus-minime battery — winner config from the sweep: MTP n_max=3, ctx=16384, anti-rep, q8 KV.
# Validates n3@16384 fits + full e2e quality: coding (inspect output), tool-calls, reasoning, repetition.
# (Same qwen35moe arch as the 122B bigger-brother → this config is the 122B starting point too.)
export PATH=/usr/local/bin:/usr/bin:/bin:/sbin:/usr/sbin
NAME=opus-final; GPU=GPU-aad5ef40-9b80-8fd0-4391-dfe595f42640
MODEL=/qwen-models/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-Q2_K.gguf
BUILD=/mnt/user/appdata/ik_llama/ik_llama.cpp/build-speed
PORT=8077; BASE=http://127.0.0.1:$PORT
LOG=/tmp/hammer/opus_final.log; : > "$LOG"; rm -f /tmp/hammer/FINAL_DONE /tmp/of_*.json
docker rm -f opus-tune opus-battery opus-final >/dev/null 2>&1
docker run -d --name $NAME --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=$GPU \
  -e LD_LIBRARY_PATH=/build/bin:/build/src:/build/ggml/src:/build/examples/mtmd \
  -p $PORT:8080 -v "$BUILD":/build:ro -v /mnt/user/models/qwen:/qwen-models:ro \
  nvidia/cuda:12.8.1-devel-ubuntu24.04 /build/bin/llama-server -m "$MODEL" \
  -c 16384 -np 1 -ngl 99 -fa on -ctk q8_0 -ctv q8_0 --jinja \
  --spec-type "mtp:n_max=3,p_min=0.5" \
  --temp 0.6 --top-p 0.9 --top-k 40 --min-p 0.05 \
  --repeat-penalty 1.08 --repeat-last-n 256 --presence-penalty 0.1 \
  --host 0.0.0.0 --port 8080 >/dev/null 2>&1
ok=0; for i in $(seq 1 90); do sleep 3; curl -sf -m3 $BASE/health >/dev/null 2>&1 && { ok=1; break; }; done
if [ $ok -eq 0 ]; then echo "LOAD FAILED: $(docker logs $NAME 2>&1|grep -iE 'out of memory|error|assert'|tail -2)" >>"$LOG"; echo DONE>/tmp/hammer/FINAL_DONE; exit 1; fi
echo "=== FINAL: Claude-35B-A3B Q2_K | pxa_llama | MTP n3 | ctx=16384 | anti-rep ===" >>"$LOG"
echo "VRAM(GPU1)=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 1) | MTP=$(docker logs $NAME 2>&1|grep -ioE 'MTP context ready'|head -1)" >>"$LOG"

echo "" >>"$LOG"; echo "##### CODING #####" >>"$LOG"
ci=0
for tag in BST rate-limiter; do ci=$((ci+1))
  if [ "$tag" = BST ]; then P="Write a Python binary search tree class with insert, search, delete, and inorder traversal. Include docstrings and type hints. Just the code."; else P="Write a Python token-bucket rate limiter class with type hints and a short usage example. Just the code."; fi
  curl -s -m180 $BASE/completion -d "{\"prompt\":\"$P\",\"n_predict\":460,\"temperature\":0.3}" -o /tmp/of_c$ci.json
  tps=$(node -e "try{console.log(JSON.parse(require('fs').readFileSync('/tmp/of_c$ci.json')).timings.predicted_per_second.toFixed(1))}catch(e){console.log('ERR')}")
  out=$(node -e "try{console.log(JSON.parse(require('fs').readFileSync('/tmp/of_c$ci.json')).content)}catch(e){console.log('PARSE_ERR')}")
  reploop=$(printf '%s' "$out"|sort|uniq -c|sort -rn|awk '$1>3&&length($0)>10{print}'|head -1)
  defs=$(printf '%s' "$out"|grep -cE "def |class ")
  echo "--- $tag ($tps tok/s) | def/class: $defs | rep-loop: ${reploop:-none} ---" >>"$LOG"
  printf '%s\n' "$out"|head -30 >>"$LOG"
done

echo "" >>"$LOG"; echo "##### TOOL CALLS (e2e OpenAI chat) #####" >>"$LOG"
TOOLS='[{"type":"function","function":{"name":"get_weather","description":"Get weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}},{"type":"function","function":{"name":"calculate","description":"Evaluate an arithmetic expression","parameters":{"type":"object","properties":{"expr":{"type":"string"}},"required":["expr"]}}}]'
ti=0
for q in "What is the weather in Tokyo right now?" "Use the calculator to compute 17*23." "What is the weather in Paris and what is 99 minus 44? Use the tools."; do ti=$((ti+1))
  curl -s -m90 $BASE/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"x\",\"messages\":[{\"role\":\"user\",\"content\":\"$q\"}],\"tools\":$TOOLS,\"tool_choice\":\"auto\",\"temperature\":0.2}" -o /tmp/of_t$ti.json
  tc=$(node -e "try{const m=JSON.parse(require('fs').readFileSync('/tmp/of_t$ti.json')).choices[0].message;const t=m.tool_calls;console.log(t?t.map(x=>x.function.name+'('+x.function.arguments+')').join(', '):'NO_CALL: '+(m.content||'').slice(0,90))}catch(e){console.log('ERR '+e.message)}")
  echo "[$q] -> $tc" >>"$LOG"
done

echo "" >>"$LOG"; echo "##### REASONING (not-stupid) #####" >>"$LOG"
RP="A bat and a ball cost 1.10 dollars total. The bat costs 1.00 dollar more than the ball. How much does the ball cost? Think briefly, then give the final number."
curl -s -m120 $BASE/completion -d "{\"prompt\":\"$RP\",\"n_predict\":220,\"temperature\":0.3}" -o /tmp/of_r.json
node -e "try{const c=JSON.parse(require('fs').readFileSync('/tmp/of_r.json')).content;console.log('ans: '+c.replace(/\n+/g,' ').trim().slice(0,240))}catch(e){console.log('ERR')}" >>"$LOG"

echo "" >>"$LOG"; echo "MTP final acceptance: $(docker logs $NAME 2>&1|grep -oiE 'acceptance rate = [0-9.]+'|tail -1)" >>"$LOG"
echo "OOM count: $(docker logs $NAME 2>&1|grep -c 'out of memory')" >>"$LOG"
echo "BATTERY DONE" >>"$LOG"; echo DONE>/tmp/hammer/FINAL_DONE
