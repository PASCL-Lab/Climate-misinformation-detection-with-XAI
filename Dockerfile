FROM python:3.10-slim-bullseye

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install build tools (if needed by some Python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install CPU-only Torch and other dependencies
RUN pip install --no-cache-dir torch==2.2.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir typing_extensions uvicorn fastapi

# Copy app files
COPY main.py .
COPY onnx_models/final_model_augv2_89/model.onnx ./onnx_models/final_model_augv2_89/model.onnx
COPY backend/model/final_model_augv2_89 ./backend/model/final_model_augv2_89

# Expose Fly.io port
EXPOSE 8080

# Start FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
