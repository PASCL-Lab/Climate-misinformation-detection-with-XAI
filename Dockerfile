
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Copy requirements file first (for better Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy your HuggingFace models from backend directory
COPY backend/models/final_model_augv2_89 ./models/

# Copy ONNX model folder from root directory
COPY onnx_models/final_model_augv2_89/ ./onnx_models/final_model_augv2_89/

# Copy your main application file from root
COPY main.py .

# Copy the entire backend directory (for other backend files)
COPY backend/ ./backend/

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=30s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose the port your FastAPI app runs on
EXPOSE 8000

# Start the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]