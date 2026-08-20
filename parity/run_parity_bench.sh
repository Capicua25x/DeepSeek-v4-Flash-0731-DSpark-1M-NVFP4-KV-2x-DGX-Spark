#!/usr/bin/env bash
# Parity bench Tier A: {mmlu_pro, aime25, ifeval} x 3 seeds x 1 endpoint.
# Usage: run_parity_bench.sh <endpoint-name> <base_url> [api_key]
set -u
EP="$1"; URL="$2"; KEY="${3:-}"
OUT="${PARITY_OUT:-./parity-results}"; mkdir -p "$OUT"
LM="${LM_EVAL:-lm_eval}"
for SEED in 1234 2345 3456; do
  for SPEC in "mmlu_pro:80:24000" "aime25:30:30000" "ifeval:80:16000"; do
    TASK="${SPEC%%:*}"; REST="${SPEC#*:}"; LIMIT="${REST%%:*}"; TOKS="${REST#*:}"
    TAG="${EP}_${TASK}_s${SEED}"
    [ -f "$OUT/$TAG.done" ] && continue
    echo "=== $TAG @ $(date +%H:%M)"
    OPENAI_API_KEY="$KEY" $LM --model local-chat-completions \
      --model_args "model=deepseek-v4-flash,base_url=$URL,num_concurrent=6,max_retries=3,timeout=1800" \
      --tasks "$TASK" --limit "$LIMIT" --apply_chat_template \
      --gen_kwargs "{\"temperature\":0.6,\"top_p\":0.95,\"reasoning_effort\":\"max\",\"max_gen_toks\":$TOKS,\"until\":[]}" \
      --output_path "$OUT/$TAG" --seed "$SEED" > "$OUT/$TAG.log" 2>&1 \
      && touch "$OUT/$TAG.done"
    echo "    exit=$? nulls=$(grep -c 'null content' "$OUT/$TAG.log")"
  done
done
echo "CAMPAIGN $EP COMPLETE $(date +%H:%M)"
