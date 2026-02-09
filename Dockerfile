# CloudTrail Anomaly Detection Container
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directory for MLflow artifacts
RUN mkdir -p /app/mlruns

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MLFLOW_TRACKING_URI=file:./mlruns
# ENV AWS_REGION=us-east-1

# Expose MLflow UI port 
EXPOSE 5000

# Default command: run training pipeline
CMD ["python", "main.py"]


# Build image
#docker build -t threat-hunting:latest .

# Run training pipeline
#docker run --rm \
  #-v $(pwd)/mlruns:/app/mlruns \
  #-e AWS_ACCESS_KEY_ID="your-key" \
  #-e AWS_SECRET_ACCESS_KEY="your-secret" \
  #threat-hunting:latest

# Run inference streaming
#docker run --rm \
  #-v $(pwd)/mlruns:/app/mlruns \
  #-e AWS_ACCESS_KEY_ID="your-key" \
  #-e AWS_SECRET_ACCESS_KEY="your-secret" \
  #threat-hunting:latest \
  #python scripts/inference.py

# Run MLflow UI
#docker run -p 5000:5000 \
  #-v $(pwd)/mlruns:/app/mlruns \
  #threat-hunting:latest \
  #mlflow ui --host 0.0.0.0
