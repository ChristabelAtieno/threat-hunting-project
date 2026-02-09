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

