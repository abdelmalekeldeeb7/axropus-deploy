#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPLOY_DIR="${ROOT_DIR}/deploy"
STATE_DIR="${DEPLOY_DIR}/state"
LAST_BACKUP="${STATE_DIR}/last_backup.env"

if [[ ! -f "${LAST_BACKUP}" ]]; then
  echo "No backup metadata found at ${LAST_BACKUP}"
  exit 1
fi

# shellcheck disable=SC1090
source "${LAST_BACKUP}"

compose_cmd=("docker" "compose")
if ! docker compose version >/dev/null 2>&1; then
  compose_cmd=("docker-compose")
fi
"${compose_cmd[@]}" -f "${DEPLOY_DIR}/docker-compose.yml" down || true

if [[ -n "${CONFIG_BACKUP:-}" && -f "${CONFIG_BACKUP}" ]]; then
  cp "${CONFIG_BACKUP}" "${DEPLOY_DIR}/config/config.yaml"
fi

if [[ -n "${LEDGER_BACKUP:-}" && -f "${LEDGER_BACKUP}" ]]; then
  cp "${LEDGER_BACKUP}" "${ROOT_DIR}/platform_data/ledger.sqlite"
fi

echo "Rollback complete."
