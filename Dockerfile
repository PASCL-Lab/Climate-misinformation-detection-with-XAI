# Use Python 3.9 slim image
FROM python:3.10-slim-bullseye

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app

# Set working directory
WORKDIR $APP_HOME

# Copy only necessary files
COPY requirements.txt ./ 
COPY main.py ./ 
COPY onnx_models/final_model_augv2_89/model.onnx ./model.onnx
COPY backend/model/final_model_augv2_89 ./backend/model/final_model_augv2_89



# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt \
    && find /usr/local/lib/python*/site-packages -name "__pycache__" -type d -exec rm -rf {} +


# Expose the port Fly.io will use
EXPOSE 8080

# Start the app with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
