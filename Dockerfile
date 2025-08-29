FROM python:3.11 AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Create venv and install requirements
RUN python -m venv .venv
COPY requirements.txt ./
RUN .venv/bin/pip install --no-cache-dir -r requirements.txt
COPY onnx_models/final_model_augv2_89/model.onnx /app/onnx_models/final_model_augv2_89/model.onnx
COPY backend/model/final_model_augv2_89 /app/backend/model/final_model_augv2_89

# Runtime image
FROM python:3.11-slim
WORKDIR /app



COPY --from=builder /app/.venv .venv/
COPY . .

EXPOSE 8080

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "1", "-b", "0.0.0.0:$PORT", "main:app"]

