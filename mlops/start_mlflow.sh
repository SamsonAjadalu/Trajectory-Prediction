#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT_DIR/mlflow-artifacts"

exec python -m mlflow server \
  --host 0.0.0.0 \
  --port "${MLFLOW_PORT:-5000}" \
  --backend-store-uri "sqlite:///$ROOT_DIR/mlflow.db" \
  --default-artifact-root "$ROOT_DIR/mlflow-artifacts"
