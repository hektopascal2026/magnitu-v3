#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${MAGNITU_DATA_DIR:-/app/data}"
CONFIG_FILE="${DATA_DIR}/magnitu_config.json"

mkdir -p "${DATA_DIR}/models"

if [ ! -f "${CONFIG_FILE}" ]; then
  cp /app/magnitu_config.example.json "${CONFIG_FILE}"
  echo "Initialized ${CONFIG_FILE} from example config."
fi

exec "$@"
