"""FastAPI inference service for the registered trajectory Transformer."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

import mlflow
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    history: list[list[float]] = Field(..., description='Four observed (x, y) points')

    @field_validator('history')
    @classmethod
    def validate_history(cls, value: list[list[float]]) -> list[list[float]]:
        if len(value) != 4 or any(len(frame) != 2 for frame in value):
            raise ValueError('history must have shape [4, 2]')
        if any(not torch.isfinite(torch.tensor(frame, dtype=torch.float32)).all() for frame in value):
            raise ValueError('history must contain only finite numbers')
        return value


class PredictionResponse(BaseModel):
    trajectory: list[list[float]]
    model_uri: str


MODEL_URI = os.getenv('MODEL_URI', 'models:/trajectory-transformer@champion')
TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5000')


@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(TRACKING_URI)
    try:
        # Training may have run on CUDA; map storages directly to CPU so the
        # API image does not require CUDA libraries or a GPU.
        app.state.model = mlflow.pytorch.load_model(
            MODEL_URI, map_location=torch.device('cpu')
        )
        app.state.model.to(torch.device('cpu'))
        app.state.model.eval()
    except Exception as exc:  # pragma: no cover - exercised by deployment smoke tests
        raise RuntimeError(f'Unable to load registered model {MODEL_URI}: {exc}') from exc
    yield


app = FastAPI(title='Trajectory Transformer API', version='1.0.0', lifespan=lifespan)


@app.get('/health')
def health() -> dict[str, str]:
    return {'status': 'ok', 'model_uri': MODEL_URI}


@app.post('/predict', response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        x = torch.tensor([request.history], dtype=torch.float32)
        with torch.inference_mode():
            prediction = app.state.model(x).detach().cpu()
        if tuple(prediction.shape) != (1, 12, 2) or not torch.isfinite(prediction).all():
            raise ValueError(f'unexpected model output shape or values: {tuple(prediction.shape)}')
        return PredictionResponse(trajectory=prediction[0].tolist(), model_uri=MODEL_URI)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'prediction failed: {exc}') from exc
