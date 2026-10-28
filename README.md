# Trajectory Prediction for Autonomous Driving

This repository contains three trajectory-prediction tracks for nuScenes-based
motion forecasting:

- `baseline/` — constant-velocity baseline and summaries
- `rnn-model/` — RNN/LSTM/GRU experiments
- `transformer_model/` — Transformer model and the MLflow/FastAPI workflow

The common task is to predict twelve future trajectory points (six seconds at
2 Hz) from four observed `(x, y)` points. The tracked preprocessed tensors are
under `transformer_model/data/`; raw nuScenes data is not required for the
training and serving workflow below.

Existing experiment commands remain available:

```bash
python baseline/scripts/baseline.py
python baseline/scripts/summary_baseline_results.py
python rnn-model/training/train.py
python rnn-model/training/eval.py
python transformer_model/training/example_inference.py
```

## Environment

The existing local CUDA environment is:

```bash
conda activate ece1508
cd /media/robotswithai/data11/Trajectory_Prediction
python -m pip install -r mlops/requirements.txt -r serving/requirements.txt
```

The serving dependencies are pinned separately in
`serving/requirements.txt` and are installed by the Docker build.

## Start MLflow

In terminal 1:

```bash
cd /media/robotswithai/data11/Trajectory_Prediction
./mlops/start_mlflow.sh
```

The tracking UI and API are available at `http://127.0.0.1:5000`.

## Train and track the Transformer

In terminal 2:

```bash
conda activate ece1508
cd /media/robotswithai/data11/Trajectory_Prediction
python transformer_model/training/train.py \
  --agent car \
  --features baseline \
  --data-dir transformer_model/data \
  --checkpoint-dir transformer_model/checkpoints/car_baseline_final \
  --num-epochs 100 \
  --batch-size 512 \
  --learning-rate 5e-4 \
  --seed 42 \
  --mlflow-tracking-uri http://127.0.0.1:5000 \
  --mlflow-experiment trajectory-transformer \
  --mlflow-run-name car-baseline-seed42-final
```

The command writes the best checkpoint and evaluation history under
`transformer_model/checkpoints/car_baseline_final/`, and logs hyperparameters,
feature configuration, loss, ADE, FDE, checkpoint artifacts, and the PyTorch
model artifact to MLflow. The command prints the MLflow run ID.

Evaluate the same checkpoint locally:

```bash
python transformer_model/training/eval.py \
  transformer_model/checkpoints/car_baseline_final/best_model.pt \
  --agent car \
  --features baseline \
  --data-dir transformer_model/data \
  --batch-size 512
```

## Register the selected model

Replace `RUN_ID` with the ID printed by training:

```bash
python mlops/register_model.py \
  --run-id RUN_ID \
  --model-name trajectory-transformer \
  --alias champion \
  --tracking-uri http://127.0.0.1:5000
```

The service loads `models:/trajectory-transformer@champion`, so a new
registered version can be promoted without changing the API code.

## Run the API locally

```bash
conda activate ece1508
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
export MODEL_URI=models:/trajectory-transformer@champion
uvicorn serving.app:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Prediction request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"history":[[0.0,0.0],[1.0,0.1],[2.0,0.2],[3.0,0.3]]}'
```

The response contains a `trajectory` array with shape `[12, 2]`.

## Build and run Docker

Keep the MLflow server running, then build the serving image:

```bash
docker build -f serving/Dockerfile -t trajectory-transformer-api:local .
docker run --rm --name trajectory-transformer-api \
  --add-host=host.docker.internal:host-gateway \
  -v "$PWD/mlflow-artifacts:/media/robotswithai/data11/Trajectory_Prediction/mlflow-artifacts" \
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
  -e MODEL_URI=models:/trajectory-transformer@champion \
  -p 8000:8000 \
  trajectory-transformer-api:local
```

Then run the same `/health` and `/predict` curl commands above. The Docker
image installs CPU-only PyTorch; training remains in the CUDA environment. The
volume is required because this local MLflow setup uses filesystem-backed
artifacts, and it is writable so MLflow can create its model metadata sidecar.
Model weights and MLflow state remain local and are ignored by Git.

For a machine-checkable smoke test while the container is running:

```bash
python serving/smoke_test.py --url http://127.0.0.1:8000
```

## Tests

```bash
conda activate ece1508
python -m pytest -q tests/test_pipeline.py
```

The tests verify preserved tensor discovery, expected tensor shapes, model
output shape, and finite predictions. `serving/smoke_test.py` verifies a real
request against the Dockerized service and checks the `[12, 2]` response.

## Verified local run

The reproducible baseline run used `car`, `baseline`, seed `42`, 100 epochs,
batch size `512`, and the scene split at 700. It produced MLflow run
`2ecf457cd58a43aea3ed7220b72225b1` and registered model version 1 under the
`champion` alias. On the 23,563-sample held-out split, the selected checkpoint
(epoch 59) achieved ADE `3.6782 m` and FDE `6.8662 m`.
