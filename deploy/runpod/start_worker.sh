#!/usr/bin/env bash
set -euo pipefail

ROLE="${1:?usage: start_worker.sh site|helper}"
ENV_FILE="${ECHO_ENV_FILE:-/workspace/Echo/.env.runpod}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  source "$ENV_FILE"
  set +a
fi
STATE_DIR="${ECHO_STATE_DIR:-/workspace/compute-echo-state}"

if [[ "$ROLE" == "helper" ]]; then
  export ECHO_WORKER_TOKEN="${HELPER_WORKER_TOKEN:?missing HELPER_WORKER_TOKEN}"
  export ECHO_CONTROL_TOKEN="${HELPER_CONTROL_TOKEN:?missing HELPER_CONTROL_TOKEN}"
  exec echo-audit worker --backend cuda --ledger "$STATE_DIR/helper.sqlite3" \
    --host 0.0.0.0 --port 8102 --telemetry-role remote_helper \
    --profile-path "$STATE_DIR/helper-initial-profile.json"
elif [[ "$ROLE" == "site" ]]; then
  export ECHO_WORKER_TOKEN="${SITE_WORKER_TOKEN:?missing SITE_WORKER_TOKEN}"
  export ECHO_CONTROL_TOKEN="${SITE_CONTROL_TOKEN:?missing SITE_CONTROL_TOKEN}"
  export ECHO_HELPER_TOKEN="${HELPER_WORKER_TOKEN:?missing HELPER_WORKER_TOKEN}"
  exec echo-audit worker --backend cuda --ledger "$STATE_DIR/site.sqlite3" \
    --helper-url "${HELPER_API_URL:?missing HELPER_API_URL}" \
    --host 0.0.0.0 --port 8101 --telemetry-role declared_site \
    --profile-path "$STATE_DIR/facility-initial-profile.json"
else
  echo "role must be site or helper" >&2
  exit 2
fi
