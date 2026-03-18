#!/usr/bin/env bash
# Run korith_dynamic with AMF on 4K context, 10 requests, with AI reasoning layer
set -euo pipefail

MODEL="/home/korith/.axropus/models/llama-3.2-1b-q4.gguf"
BINARY="/home/korith/korith/build/korith_dynamic"
AMF_PATH="/home/korith/.axropus/amf"
DATA_DIR="/home/korith/.axropus/data"
PORT=8080
REPO_DIR="/home/korith/korith"

mkdir -p "$AMF_PATH" "$DATA_DIR"

export KORITH_AMF_PATH="$AMF_PATH"
export KORITH_DATA_DIR="$DATA_DIR"
export KORITH_DEFAULT_MODEL_PATH="$MODEL"
export KORITH_KV_FP8=0
export KORITH_API_KEY_SALT="test-salt"

echo "[1/3] Starting korith_dynamic gateway on port $PORT..."
"$BINARY" --model "$MODEL" --port $PORT --ctx-size 4096 --n-gpu-layers 99 &
BINARY_PID=$!
echo "korith_dynamic PID: $BINARY_PID"

# Wait for gateway to be healthy
echo "[2/3] Waiting for gateway..."
for i in $(seq 1 20); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        echo "Gateway ready."
        break
    fi
    sleep 2
done

echo "[3/3] Running 10 inference requests at 4K context with AMF..."
PROMPT="The following is a long technical document about distributed systems and KV cache optimization in large language models. $(python3 -c "print('token ' * 900)")"

for i in $(seq 1 10); do
    T0=$(date +%s%3N)
    RESULT=$(curl -sf -X POST "http://127.0.0.1:$PORT/v1/completions" \
        -H "Content-Type: application/json" \
        -d "{\"prompt\": \"${PROMPT} Question: What is KV cache?\", \"max_tokens\": 32, \"temperature\": 0}" \
        2>/dev/null || echo '{"error":"failed"}')
    T1=$(date +%s%3N)
    MS=$((T1 - T0))

    if echo "$RESULT" | grep -q "error"; then
        echo "  Run $i: ERROR"
    else
        echo "  Run $i: ${MS}ms"
    fi
done

echo ""
echo "Checking AMF store..."
ls -lh "$AMF_PATH/" 2>/dev/null || echo "(empty)"

kill $BINARY_PID 2>/dev/null || true
echo "Done."
