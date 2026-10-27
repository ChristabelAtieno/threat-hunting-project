# CloudTrail Threat Hunting container
# Default: score the last hour of CloudTrail events with the champion model.
#   docker run --rm -e AWS_REGION=us-east-1 -e MLFLOW_TRACKING_URI=... threat-hunt
# Train instead:
#   docker run --rm ... threat-hunt train --source lookup --days 30
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .
COPY scripts ./scripts

# Run as an unprivileged user; /app/state holds the scoring state file and outputs
RUN useradd --create-home --uid 10001 hunter && mkdir -p /app/state && chown hunter /app/state
USER hunter

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MLFLOW_DISABLE_AGENT_HINT=1
# Point MLFLOW_TRACKING_URI at your tracking server (or mount a store) so the model can be loaded
# ENV MLFLOW_TRACKING_URI=http://mlflow:5000
# ENV AWS_REGION=us-east-1

ENTRYPOINT ["python", "main.py"]
CMD ["score", "--hours", "1", "--regions", "all", "--state-file", "/app/state/score_state.json", "--output", "/app/state/incidents.jsonl"]
