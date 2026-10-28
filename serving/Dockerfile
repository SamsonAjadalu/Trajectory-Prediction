FROM python:3.10-slim

WORKDIR /app

COPY serving/requirements.txt /app/requirements.txt
# The API image is CPU-only.  Keep the CUDA-enabled training environment
# separate and install the official CPU PyTorch wheel here.
RUN sed '/^torch==/d' /app/requirements.txt > /tmp/requirements-no-torch.txt \
    && pip install --no-cache-dir -r /tmp/requirements-no-torch.txt \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.5.1+cpu

COPY serving/app.py /app/app.py

ENV MLFLOW_TRACKING_URI=http://host.docker.internal:5000
ENV MODEL_URI=models:/trajectory-transformer@champion

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
