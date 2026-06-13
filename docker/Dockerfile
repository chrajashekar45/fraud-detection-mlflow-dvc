FROM python:3.11-slim

WORKDIR /app

# Install system dependency required by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Only copy what the API actually needs
COPY api/        ./api/
COPY app/        ./app/
COPY models/model.pkl          ./models/model.pkl
COPY models/eval_report.json   ./models/eval_report.json
COPY models/latency_report.json ./models/latency_report.json
COPY models/pruned_feature_cols.json ./models/pruned_feature_cols.json

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
