FROM python:3.11 AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Create venv and install requirements
RUN python -m venv .venv
COPY requirements.txt ./
RUN .venv/bin/pip install --no-cache-dir -r requirements.txt

# Runtime image
FROM python:3.11-slim
WORKDIR /app



COPY --from=builder /app/.venv .venv/
COPY . .

EXPOSE 8080

CMD ["/app/.venv/bin/uvicorn", "main:app", "--host=0.0.0.0", "--port=8080", "--workers=1"]
