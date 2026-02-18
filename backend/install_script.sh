#!/usr/bin/env bash
set -euo pipefail

AX_HOME="${HOME}/.axropus"
CONFIG_PATH="${AX_HOME}/config.json"
AGENT_WRAPPER="${AX_HOME}/axropus-agent.sh"
SERVICE_NAME="axropus-agent"

mkdir -p "${AX_HOME}" "${AX_HOME}/models"
chmod 700 "${AX_HOME}"

cat > "${CONFIG_PATH}" <<EOF
{
  "api_key": "${AXROPUS_API_KEY:-}",
  "runtime": "${AXROPUS_RUNTIME:-}",
  "model_family": "${AXROPUS_MODEL_FAMILY:-}",
  "model_size": "${AXROPUS_MODEL_SIZE:-}",
  "draft_model": "${AXROPUS_DRAFT_MODEL:-}",
  "python_path": "${AXROPUS_PYTHON:-python3}",
  "zero_data": true
}
EOF
chmod 600 "${CONFIG_PATH}"

cat > "${AGENT_WRAPPER}" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
while true; do
  axropus agent start --config "${HOME}/.axropus/config.json" >> "${HOME}/.axropus/agent.log" 2>&1 || true
  sleep 2
done
EOF
chmod 700 "${AGENT_WRAPPER}"

if command -v systemctl >/dev/null 2>&1; then
  mkdir -p "${HOME}/.config/systemd/user"
  cat > "${HOME}/.config/systemd/user/${SERVICE_NAME}.service" <<EOF
[Unit]
Description=Axropus customer-side runtime wrapper
After=network-online.target

[Service]
Type=simple
ExecStart=${AGENT_WRAPPER}
Restart=always
RestartSec=3
Environment=AXROPUS_HOME=${AX_HOME}

[Install]
WantedBy=default.target
EOF

  systemctl --user daemon-reload || true
  systemctl --user enable "${SERVICE_NAME}" || true
  systemctl --user restart "${SERVICE_NAME}" || true
else
  nohup "${AGENT_WRAPPER}" >/dev/null 2>&1 &
fi

echo "[AXROPUS] installer finished"

