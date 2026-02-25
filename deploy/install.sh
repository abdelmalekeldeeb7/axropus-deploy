#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPLOY_DIR="${ROOT_DIR}/deploy"
STATE_DIR="${DEPLOY_DIR}/state"
BACKUP_DIR="${STATE_DIR}/backups"
STAMP="$(date +%Y%m%d_%H%M%S)"

mkdir -p "${BACKUP_DIR}"
mkdir -p "${ROOT_DIR}/platform_data"

ensure_cmd() {
  local cmd="$1"
  if command -v "${cmd}" >/dev/null 2>&1; then
    return 0
  fi
  return 1
}

install_docker_if_missing() {
  if ensure_cmd docker; then
    return 0
  fi
  if [[ "${EUID}" -ne 0 ]]; then
    echo "docker missing; rerun install.sh as root to install docker automatically"
    exit 1
  fi
  apt-get update
  apt-get install -y ca-certificates curl gnupg
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  chmod a+r /etc/apt/keyrings/docker.asc
  . /etc/os-release
  echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu ${VERSION_CODENAME} stable" \
    | tee /etc/apt/sources.list.d/docker.list >/dev/null
  apt-get update
  apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
}

install_nvidia_runtime_if_missing() {
  if docker info 2>/dev/null | grep -q "Runtimes:.*nvidia"; then
    return 0
  fi
  if [[ "${EUID}" -ne 0 ]]; then
    echo "nvidia runtime not found; rerun install.sh as root to install nvidia-container-toolkit"
    return 0
  fi
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update
  apt-get install -y nvidia-container-toolkit
  nvidia-ctk runtime configure --runtime=docker
  systemctl restart docker
}

backup_current_state() {
  local config_src="${DEPLOY_DIR}/config/config.yaml"
  local ledger_src="${ROOT_DIR}/platform_data/ledger.sqlite"
  if [[ -f "${config_src}" ]]; then
    cp "${config_src}" "${BACKUP_DIR}/config_${STAMP}.yaml"
  fi
  if [[ -f "${ledger_src}" ]]; then
    cp "${ledger_src}" "${BACKUP_DIR}/ledger_${STAMP}.sqlite"
  fi
  cat > "${STATE_DIR}/last_backup.env" <<EOF
BACKUP_STAMP=${STAMP}
CONFIG_BACKUP=${BACKUP_DIR}/config_${STAMP}.yaml
LEDGER_BACKUP=${BACKUP_DIR}/ledger_${STAMP}.sqlite
EOF
}

ensure_api_key_salt_configured() {
  local config_file="${DEPLOY_DIR}/config/config.yaml"
  local placeholder="korith_pilot_change_this_before_prod"
  if [[ ! -f "${config_file}" ]]; then
    echo "missing config file: ${config_file}"
    exit 1
  fi
  if grep -q "api_key_salt:" "${config_file}"; then
    if grep -q "${placeholder}" "${config_file}"; then
      local generated
      generated="$(tr -d '-' < /proc/sys/kernel/random/uuid)"
      sed -i "s/${placeholder}/${generated}/g" "${config_file}"
      echo "configured api_key_salt in deploy/config/config.yaml"
    fi
  else
    local generated
    generated="$(tr -d '-' < /proc/sys/kernel/random/uuid)"
    cat >> "${config_file}" <<EOF

security:
  api_key_salt: "${generated}"
EOF
    echo "added security.api_key_salt to deploy/config/config.yaml"
  fi
}

main() {
  local compose_cmd=("docker" "compose")
  if ! docker compose version >/dev/null 2>&1; then
    compose_cmd=("docker-compose")
  fi
  install_docker_if_missing
  install_nvidia_runtime_if_missing
  backup_current_state
  ensure_api_key_salt_configured
  "${compose_cmd[@]}" -f "${DEPLOY_DIR}/docker-compose.yml" build
  "${compose_cmd[@]}" -f "${DEPLOY_DIR}/docker-compose.yml" up -d
  python3 "${ROOT_DIR}/platform/korith_platform.py" keys create --org local --config "${DEPLOY_DIR}/config/config.yaml" > "${STATE_DIR}/bootstrap_key.json"
  echo "Korith platform started."
  echo "Router health: http://127.0.0.1:8000/health"
  echo "Gateway health: http://127.0.0.1:8080/health"
  echo "Bootstrap API key saved to: ${STATE_DIR}/bootstrap_key.json"
}

main "$@"
