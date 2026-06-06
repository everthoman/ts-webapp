#!/usr/bin/env bash
# Launch the Thompson Sampling + GNINA docking web app.
# Uses the `ts_gnina` conda env (Python 3.11 + rdkit + fastapi + uvicorn).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

ENV_NAME="${TS_GNINA_ENV:-ts_gnina}"
HOST="${TS_WEBAPP_HOST:-0.0.0.0}"
# Reachable at http://130.237.250.75:5014
PORT="${TS_WEBAPP_PORT:-5014}"

# conda run works whether or not the env is active in the current shell.
exec conda run --no-capture-output -n "$ENV_NAME" \
    uvicorn ts_webapp:app --host "$HOST" --port "$PORT"
