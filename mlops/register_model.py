"""Register a completed MLflow run and assign the serving alias."""

from __future__ import annotations

import argparse
import os

import mlflow
from mlflow import MlflowClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--model-name', default='trajectory-transformer')
    parser.add_argument('--alias', default='champion')
    parser.add_argument('--tracking-uri', default=os.getenv('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5000'))
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    model_uri = f"runs:/{args.run_id}/model"
    registration = mlflow.register_model(model_uri, args.model_name)

    client = MlflowClient(tracking_uri=args.tracking_uri)
    client.set_registered_model_alias(args.model_name, args.alias, registration.version)
    client.set_model_version_tag(args.model_name, registration.version, 'source_run_id', args.run_id)

    print(f"registered_model={args.model_name}")
    print(f"version={registration.version}")
    print(f"alias={args.alias}")
    print(f"model_uri=models:/{args.model_name}@{args.alias}")


if __name__ == '__main__':
    main()
