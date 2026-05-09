#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$script_dir"
export PYTHONUTF8=1
export PYTHONPATH="${script_dir}:${PYTHONPATH:-}"

PARAMS_JSON="${1:?usage: train.sh <params.json | base64-json>}"

python -m pipeline.train_master "$PARAMS_JSON"
exit_code=$?
echo "Exit code: $exit_code"
exit "$exit_code"
