#!/usr/bin/env bash
# pxa_llama — decode-throughput probe. Measures single-stream and aggregate-concurrent
# tok/s using llama-server's own timing fields. Use to compare ik vs mainline vs the fork.
#
# Usage: pxa-llama-speed-test.sh <base_url> [concurrency] [n_predict]
set -u
BASE="${1:?need base_url}"; C="${2:-4}"; NP="${3:-200}"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
PROMPT="Write a detailed technical explanation of how a transformer decoder generates tokens autoregressively, including KV cache, attention, and sampling. Be thorough."

one() { # $1 idx
  local body
  body=$(printf '{"prompt":%s,"n_predict":%d,"temperature":0.7,"cache_prompt":false}' "$(printf '%s' "$PROMPT" | jq -Rs .)" "$NP")
  curl -s -m 300 -X POST "$BASE/completion" -H 'Content-Type: application/json' -d "$body" > "$TMP/s_$1.json"
}

echo "=== SINGLE-STREAM ($BASE, n_predict=$NP) ==="
one 0
sp=$(jq -r '(.timings.predicted_per_second // empty)' "$TMP/s_0.json" 2>/dev/null)
echo "  single decode: ${sp:-?} tok/s"

echo "=== CONCURRENT x$C (aggregate) ==="
t0=$(date +%s.%N)
for i in $(seq 1 $C); do one "c$i" & done
wait
t1=$(date +%s.%N)
total_tok=0; sum_ps=0; n=0
for i in $(seq 1 $C); do
  tk=$(jq -r '(.timings.predicted_n // .tokens_predicted // 0)' "$TMP/s_c$i.json" 2>/dev/null)
  ps=$(jq -r '(.timings.predicted_per_second // 0)' "$TMP/s_c$i.json" 2>/dev/null)
  total_tok=$((total_tok + ${tk%.*}))
  sum_ps=$(awk -v a="$sum_ps" -v b="${ps:-0}" 'BEGIN{printf "%.4f", a+b}')
  n=$((n+1))
done
awk -v tt="$total_tok" -v t0="$t0" -v t1="$t1" -v sp="$sum_ps" -v n="$n" 'BEGIN{
  wall=t1-t0; if(wall<=0)wall=1;
  printf "  per-stream avg: %.2f tok/s\n", sp/n;
  printf "  AGGREGATE: %.2f tok/s over %.2fs wall (%d tokens, C=%d)\n", tt/wall, wall, tt, n;
}'
