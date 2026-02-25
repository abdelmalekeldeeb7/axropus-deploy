#!/usr/bin/env bash
set -euo pipefail

model="${1:-${KORITH_MODEL:-}}"
if [[ -z "${model}" ]]; then
  echo "usage: KORITH_MODEL=/path/to/model.gguf $0"
  exit 1
fi

if [[ ! -x "./build/korith_dynamic" ]]; then
  echo "missing ./build/korith_dynamic (build first)"
  exit 1
fi

out="$(KORITH_ENABLE_AMF=0 timeout 5s ./build/korith_dynamic "${model}" 2>&1 || true)"
echo "${out}" | rg "\\[SPEC_DISABLED\\]" >/dev/null
echo "SPEC_DISABLED observed"
