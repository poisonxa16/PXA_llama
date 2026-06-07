#!/usr/bin/env bash
# pxa_llama — concurrency-correctness + speed harness for hybrid/recurrent models.
#
# PURPOSE: prove whether a llama-server build isolates per-sequence RECURRENT STATE
# under concurrent decoding (np>1). The ik_llama hybrid bug = recurrent state bleeds
# across slots -> a request sees another request's content (cross-contamination) or
# garbage. Mainline is clean. This harness fires K distinct concurrent requests, each
# carrying a unique codeword, and checks every reply contains ONLY its own codeword.
#
# Usage:  pxa-llama-concurrency-test.sh <base_url> [K] [mode]
#   base_url : e.g. http://127.0.0.1:8088   (llama-server root)
#   K        : number of concurrent requests (default 6)
#   mode     : "chat" (/v1/chat/completions, default) or "completion" (/completion)
#
# Exit 0 = clean (no contamination, no garbage). Exit 1 = contamination/garbage found.
set -u
BASE="${1:?need base_url}"; K="${2:-6}"; MODE="${3:-chat}"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
echo "=== pxa_llama concurrency-correctness test: $BASE  K=$K  mode=$MODE ==="

# Each request: a unique codeword + a unique tiny fact. The model must echo its OWN
# codeword and answer its OWN fact. Recurrent-state bleed shows up as the wrong
# codeword/fact appearing in a reply.
declare -a WORDS=(ZEBRA MANGO FALCON COBALT TUNDRA QUASAR NIMBUS VORTEX PRISM GECKO)
declare -a NUMS=(11 22 33 44 55 66 77 88 99 10)

fire() {
  local i="$1" word="$2" num="$3"
  local prompt="You are in test slot $i. Your secret codeword is ${word}_${i}. Your secret number is ${num}. Reply with EXACTLY this and nothing else: 'codeword=${word}_${i} number=${num}'."
  local url body
  if [ "$MODE" = "completion" ]; then
    url="$BASE/completion"
    body=$(printf '{"prompt":%s,"n_predict":40,"temperature":0,"cache_prompt":false}' "$(printf '%s' "$prompt" | jq -Rs .)")
  else
    url="$BASE/v1/chat/completions"
    body=$(printf '{"messages":[{"role":"user","content":%s}],"max_tokens":40,"temperature":0,"stream":false}' "$(printf '%s' "$prompt" | jq -Rs .)")
  fi
  curl -s -m 120 -X POST "$url" -H 'Content-Type: application/json' -d "$body" > "$TMP/resp_$i.json" 2>&1
}

# fire all K concurrently
for i in $(seq 0 $((K-1))); do fire "$i" "${WORDS[$i]}" "${NUMS[$i]}" & done
wait

fail=0
for i in $(seq 0 $((K-1))); do
  word="${WORDS[$i]}"; num="${NUMS[$i]}"
  # extract text
  txt=$(jq -r '(.choices[0].message.content // .content // .choices[0].text // "")' "$TMP/resp_$i.json" 2>/dev/null)
  [ -z "$txt" ] && txt=$(cat "$TMP/resp_$i.json")
  own="${word}_${i}"
  # 1) must contain own codeword
  has_own=0; echo "$txt" | grep -qF "$own" && has_own=1
  # 2) must NOT contain any OTHER slot's codeword (contamination)
  contam=""
  for j in $(seq 0 $((K-1))); do
    [ "$j" = "$i" ] && continue
    other="${WORDS[$j]}_${j}"
    if echo "$txt" | grep -qF "$other"; then contam="$contam $other"; fi
  done
  status="OK"
  if [ -n "$contam" ]; then status="CONTAMINATED by:$contam"; fail=1
  elif [ "$has_own" = 0 ]; then status="MISSING-OWN/garbage"; fail=1; fi
  printf "  slot %-2s want=%-10s -> [%s]  reply=%s\n" "$i" "$own" "$status" "$(echo "$txt" | tr '\n' ' ' | cut -c1-70)"
done

echo "--- verdict: $([ $fail = 0 ] && echo CLEAN || echo DIRTY) ---"
exit $fail
