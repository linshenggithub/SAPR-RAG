#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [[ -f demo/.env ]]; then
  set -a
  source demo/.env
  set +a
fi

source config/env_3090.sh
CONDA_ROOT="$(dirname "$(dirname "$SAPR_CONDA_BIN")")"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ROOT}/envs/reasonrag/bin/python}"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_BIN" -m uvicorn demo.backend.app:app \
  --host 127.0.0.1 \
  --port 8200 \
  --workers 1 \
  --proxy-headers \
  --forwarded-allow-ips=127.0.0.1 \
  --no-access-log
