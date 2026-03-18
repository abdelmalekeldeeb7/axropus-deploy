#!/usr/bin/env bash
# Axropus node installer — runs on the customer's GPU via SSH
# Clones the repo, builds korith_dynamic, starts the gateway + AI reasoning agent.
set -euo pipefail

AX_HOME="${HOME}/.axropus"
REPO_DIR="${AX_HOME}/korith"
LOG="${AX_HOME}/agent.log"
CONFIG_PATH="${AX_HOME}/gateway_config.json"
BACKEND_URL="${AXROPUS_BACKEND_URL:-https://axropus-backend.up.railway.app}"
PYTHON="${AXROPUS_PYTHON:-python3}"

mkdir -p "${AX_HOME}" "${AX_HOME}/models"
chmod 700 "${AX_HOME}"

echo "[AXROPUS] Cloning / updating repo..."
if [[ ! -d "${REPO_DIR}/.git" ]]; then
    git clone --depth=1 https://github.com/Abdelmalekeldeeb7/axropus-deploy.git "${REPO_DIR}" \
        >> "${LOG}" 2>&1
else
    cd "${REPO_DIR}" && git pull --ff-only >> "${LOG}" 2>&1
fi
cd "${REPO_DIR}"

echo "[AXROPUS] Building korith_dynamic (skipping model download)..."
SKIP_MODEL=1 bash scripts/h100_setup.sh >> "${LOG}" 2>&1

echo "[AXROPUS] Installing Python dependencies..."
"${PYTHON}" -m pip install --quiet numpy httpx >> "${LOG}" 2>&1 \
    || "${PYTHON}" -m pip install --quiet --break-system-packages numpy httpx >> "${LOG}" 2>&1

# AI reasoning deps — optional, gateway falls back gracefully if missing
"${PYTHON}" -m pip install --quiet transformers bitsandbytes accelerate >> "${LOG}" 2>&1 || true

echo "[AXROPUS] Writing gateway config..."
cat > "${CONFIG_PATH}" <<EOF
{
  "api_key": "${AXROPUS_API_KEY:-}",
  "deployment_id": "${AXROPUS_DEPLOYMENT_ID:-}",
  "model_family": "${AXROPUS_MODEL_FAMILY:-}",
  "model_size": "${AXROPUS_MODEL_SIZE:-}",
  "runtime": "${AXROPUS_RUNTIME:-korith}",
  "reasoning": true,
  "backend_url": "${BACKEND_URL}"
}
EOF
chmod 600 "${CONFIG_PATH}"

echo "[AXROPUS] Starting gateway agent..."
AXROPUS_API_KEY="${AXROPUS_API_KEY:-}" \
AXROPUS_DEPLOYMENT_ID="${AXROPUS_DEPLOYMENT_ID:-}" \
AXROPUS_BACKEND_URL="${BACKEND_URL}" \
nohup "${PYTHON}" "${REPO_DIR}/platform/runtime/gateway_agent.py" \
    --config "${CONFIG_PATH}" \
    >> "${LOG}" 2>&1 &

AGENT_PID=$!
echo "${AGENT_PID}" > "${AX_HOME}/agent.pid"
echo "[AXROPUS] Gateway agent started (PID ${AGENT_PID})"

# Register as systemd service if available (survives reboots)
SERVICE_NAME="axropus-agent"
if command -v systemctl >/dev/null 2>&1; then
    mkdir -p "${HOME}/.config/systemd/user"
    cat > "${HOME}/.config/systemd/user/${SERVICE_NAME}.service" <<EOF2
[Unit]
Description=Axropus Gateway Agent
After=network-online.target

[Service]
Type=simple
ExecStart=${PYTHON} ${REPO_DIR}/platform/runtime/gateway_agent.py --config ${CONFIG_PATH}
Restart=always
RestartSec=5
Environment=AXROPUS_API_KEY=${AXROPUS_API_KEY:-}
Environment=AXROPUS_DEPLOYMENT_ID=${AXROPUS_DEPLOYMENT_ID:-}
Environment=AXROPUS_BACKEND_URL=${BACKEND_URL}

[Install]
WantedBy=default.target
EOF2
    systemctl --user daemon-reload >> "${LOG}" 2>&1 || true
    systemctl --user enable "${SERVICE_NAME}" >> "${LOG}" 2>&1 || true
fi

echo "[AXROPUS] installer finished"
