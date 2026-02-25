#!/usr/bin/env bash
set -euo pipefail

MODEL="models/llama-3.2-1b-q4_k_m.gguf"
PROMPT_FILE="workloads/long_prompt.txt"

if [[ ! -f "$MODEL" ]]; then
  echo "Missing model: $MODEL" >&2
  exit 1
fi
if [[ ! -f "$PROMPT_FILE" ]]; then
  echo "Missing prompt file: $PROMPT_FILE" >&2
  exit 1
fi

CMD=(./build/korith_dynamic "$MODEL" --prompt "$(cat "$PROMPT_FILE")" --n-predict 512)

echo "===== COLD RUN ====="
"${CMD[@]}"

echo "===== WARM RUN 1 ====="
"${CMD[@]}"

echo "===== WARM RUN 2 ====="
"${CMD[@]}"
