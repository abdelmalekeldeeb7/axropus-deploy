#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="${ROOT_DIR}/deploy/docker-compose.yml"
STATE_KEY="${ROOT_DIR}/deploy/state/bootstrap_key.json"
GATEWAY_URL="${KORITH_GATEWAY_URL:-http://127.0.0.1:8080}"

usage() {
  cat <<'EOF'
Usage:
  scripts/korith_deploy_easy.sh up
  scripts/korith_deploy_easy.sh key
  scripts/korith_deploy_easy.sh smoke
  scripts/korith_deploy_easy.sh submit-demo
  scripts/korith_deploy_easy.sh down
  scripts/korith_deploy_easy.sh rollback
EOF
}

load_key() {
  if [[ -n "${KORITH_API_KEY:-}" ]]; then
    echo "${KORITH_API_KEY}"
    return 0
  fi
  if [[ ! -f "${STATE_KEY}" ]]; then
    return 1
  fi
  python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["api_key"])' "${STATE_KEY}"
}

require_key() {
  local key
  if ! key="$(load_key)"; then
    echo "No API key found. Run: scripts/korith_deploy_easy.sh up"
    exit 1
  fi
  export KORITH_API_KEY="${key}"
}

cmd="${1:-help}"
case "${cmd}" in
  up)
    bash "${ROOT_DIR}/deploy/install.sh"
    if [[ -f "${STATE_KEY}" ]]; then
      echo "Deployment complete."
      echo "Export key:"
      echo "export KORITH_API_KEY=\"$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[\"api_key\"])' "${STATE_KEY}")\""
    fi
    ;;
  key)
    require_key
    echo "${KORITH_API_KEY}"
    ;;
  smoke)
    require_key
    echo "Gateway health:"
    curl -fsS "${GATEWAY_URL}/health"
    echo
    echo "Authenticated health:"
    curl -fsS -H "Authorization: Bearer ${KORITH_API_KEY}" "${GATEWAY_URL}/v1/health"
    echo
    echo "Metrics reachable:"
    curl -fsS "${GATEWAY_URL}/metrics" | head -n 20
    ;;
  submit-demo)
    require_key
    python3 "${ROOT_DIR}/platform/korith_platform.py" submit "${ROOT_DIR}/demo/jobs/ticket_triage.json" --url "${GATEWAY_URL}" --api-key "${KORITH_API_KEY}"
    ;;
  down)
    docker compose -f "${COMPOSE_FILE}" down
    ;;
  rollback)
    bash "${ROOT_DIR}/deploy/rollback.sh"
    ;;
  *)
    usage
    exit 1
    ;;
esac
