# Fraud Detection MLOps Project

End-to-end payment fraud detection project built for an AI/ML Engineer portfolio. The project combines a LightGBM fraud model with a reproducible DVC pipeline, AWS S3-backed data/model versioning, remote MLflow experiment tracking on DagsHub, a FastAPI inference service, and a Streamlit demo UI.

The model is trained on the IEEE-CIS Fraud Detection dataset, which contains 590,540 transactions and hundreds of transaction, card, address, device, identity, and Vesta-engineered features.

## Project Highlights

- Real-time fraud scoring API with FastAPI
- Streamlit demo UI for interactive transaction testing
- LightGBM classifier trained with time-based validation
- DVC pipeline for cleaning, feature engineering, training, and latency benchmarking
- AWS S3 remote storage for DVC-tracked data and model artifacts
- DagsHub MLflow tracking for online experiment logging
- Docker files for containerized deployment
- GitHub Actions CI/CD deployment to EC2
- Pytest test suite covering API behavior and prediction edge cases

## Architecture

```text
Raw Kaggle CSVs
    |
    v
cleaning/clean.py
    -> data/clean/clean_data.csv
    -> data/clean/cleaning_report.json
    |
    v
features/Engineer.py
    -> data/features/featured_data.csv
    -> data/features/model_feature_cols.json
    -> data/features/feature_report.json
    |
    v
models/train.py
    -> models/model.pkl
    -> models/pruned_feature_cols.json
    -> models/eval_report.json
    -> DagsHub MLflow metrics/params/artifacts
    |
    v
models/optimize.py
    -> models/latency_report.json
    |
    v
api/main.py
    -> FastAPI prediction service
    |
    v
app/streamlit_app.py
    -> Streamlit demo interface
```

## Tech Stack

| Area | Tools |
| --- | --- |
| Model | LightGBM, scikit-learn |
| Data processing | pandas, NumPy |
| API | FastAPI, Pydantic, Uvicorn |
| UI | Streamlit |
| Testing | Pytest, HTTPX |
| Pipeline/versioning | DVC |
| Remote artifact storage | AWS S3 |
| Experiment tracking | MLflow on DagsHub |
| Deployment | Docker, Docker Compose, EC2, GitHub Actions |

## Repository Structure

```text
fraud-detection-mlflow-dvc/
|-- api/
|   |-- main.py                    # FastAPI service and inference schemas
|-- app/
|   |-- streamlit_app.py           # Streamlit demo UI
|-- cleaning/
|   |-- clean.py                   # Raw data cleaning pipeline
|-- features/
|   |-- Engineer.py                # Feature engineering pipeline
|-- models/
|   |-- train.py                   # LightGBM training + MLflow logging
|   |-- optimize.py                # Latency benchmark
|   |-- eval_report.json           # Git-tracked model metrics
|   |-- latency_report.json        # Git-tracked latency report
|-- data/
|   |-- raw/                       # DVC-tracked raw CSVs
|   |-- clean/                     # DVC-generated cleaned data
|   |-- features/                  # DVC-generated feature data
|-- docker/
|   |-- Dockerfile
|   |-- docker-compose.yml
|-- .github/
|   |-- workflows/
|       |-- ci.yml
|       |-- deploy.yml
|-- tests/
|   |-- test_api.py
|-- dvc.yaml                      # DVC pipeline definition
|-- dvc.lock                      # DVC pipeline lock file
|-- EC2_DEPLOYMENT_GUIDE.md       # Full EC2 rebuild/deployment guide
|-- requirements.txt
|-- README.md
```

## Model Results

Current evaluation results from `models/eval_report.json`:

| Metric | Value |
| --- | ---: |
| AUC-ROC | 0.8674 |
| AUC-PR | 0.4960 |
| F1 | 0.4693 |
| Decision threshold | 0.74 |
| Features used | 270 |
| Best iteration | 220 |

Confusion matrix on the time-based test split:

|  | Predicted Legit | Predicted Fraud |
| --- | ---: | ---: |
| Actual Legit | 110,873 | 3,171 |
| Actual Fraud | 1,846 | 2,218 |

Top model features:

1. `feat_card1_fraud_rate`
2. `log_C13`
3. `addr1`
4. `card1`
5. `feat_card1_txn_count`

## Latency Results

Current latency benchmark from `models/latency_report.json`:

| Scenario | p50 | p95 | p99 | Target |
| --- | ---: | ---: | ---: | ---: |
| Single prediction | 1.784 ms | 2.019 ms | 2.097 ms | < 50 ms |
| Batch of 100 | 2.658 ms | 3.015 ms | 3.083 ms | < 500 ms |

The API currently serves the LightGBM pickle model. ONNX export was considered, but the baseline latency already meets the target comfortably.

## Setup

Create and activate a virtual environment:

```powershell
cd E:\RESUME_PROJECTS\fraud-detection-mlflow-dvc
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Data And Model Versioning With DVC

Large artifacts are not stored directly in Git. They are tracked with DVC and pushed to AWS S3.

Configured DVC remote:

```text
s3remote -> s3://fraud-detection-mlflow-dvc/dvc-store
```

AWS setup used for this project:

```text
S3 bucket: fraud-detection-mlflow-dvc
Region: ap-southeast-2
IAM user: dvc-s3-user
```

The IAM user should have least-privilege access to this bucket only.

To pull DVC-tracked artifacts:

```powershell
dvc pull
```

To push updated DVC artifacts:

```powershell
dvc push
```

To check pipeline state:

```powershell
dvc status
```

To reproduce the full pipeline:

```powershell
dvc repro
```

To view DVC pipeline metrics:

```powershell
dvc metrics show
```

## DVC Pipeline

The DVC pipeline has four stages:

```text
clean -> features -> train -> optimize
```

Run all stages:

```powershell
dvc repro
```

Run only the training stage and its dependencies if needed:

```powershell
dvc repro train
```

Pipeline outputs:

- `data/clean/clean_data.csv`
- `data/features/featured_data.csv`
- `models/model.pkl`
- `models/pruned_feature_cols.json`

Small JSON reports are kept readable in Git:

- `data/clean/cleaning_report.json`
- `data/features/feature_report.json`
- `data/features/model_feature_cols.json`
- `models/eval_report.json`
- `models/latency_report.json`

## Remote MLflow Tracking

This project uses hosted MLflow tracking on DagsHub instead of local `mlruns/` storage.

Set the tracking credentials in PowerShell before running training:

```powershell
$env:MLFLOW_TRACKING_URI="https://dagshub.com/<your-username>/fraud-detection-mlflow-dvc.mlflow"
$env:MLFLOW_TRACKING_USERNAME="<your-dagshub-username>"
$env:MLFLOW_TRACKING_PASSWORD="<your-dagshub-token>"
```

Do not commit credentials. Keep them in environment variables or a local ignored `.env` file.

The training script logs:

- model type
- number of selected features
- best iteration
- threshold
- split strategy
- imbalance strategy
- AUC-ROC
- AUC-PR
- F1
- confusion matrix counts
- evaluation report artifacts

The large model artifact remains managed by DVC rather than being duplicated in MLflow.

## Run The FastAPI Backend

Start the API:

```powershell
uvicorn api.main:app --reload --port 8000
```

Open Swagger docs:

```text
http://localhost:8000/docs
```

Health check:

```text
GET http://localhost:8000/health
```

Expected response:

```json
{
  "status": "ok",
  "model_loaded": true,
  "features": 270,
  "threshold": 0.74
}
```

Prediction endpoint:

```text
POST http://localhost:8000/predict
```

Example request:

```json
{
  "TransactionAmt": 2500.0,
  "feat_hour": 3,
  "feat_is_night": 1,
  "feat_card1_fraud_rate": 0.85,
  "feat_m_mismatch_count": 4
}
```

Example response:

```json
{
  "risk_score": 0.8234,
  "is_fraud": true,
  "top_features": [
    "feat_card1_fraud_rate",
    "log_C13",
    "addr1",
    "card1",
    "feat_card1_txn_count"
  ],
  "latency_ms": 2.4
}
```

## Run The Streamlit UI

Start FastAPI first, then run Streamlit in a second terminal:

```powershell
streamlit run app/streamlit_app.py
```

Open:

```text
http://localhost:8501
```

The Streamlit UI exposes a small set of interpretable inputs for demo use. The trained model uses 270 features internally. Missing fields are filled by the FastAPI request schema defaults.

In production, users would not manually enter all 270 features. A transaction backend or feature store would provide the full feature vector automatically.

## Run Tests

```powershell
python -m pytest tests -v
```

Current status:

```text
14 passed
```

Tests cover:

- health endpoint
- model info endpoint
- prediction response shape
- score range validation
- boolean fraud decision
- latency target
- high-risk vs low-risk scenario
- zero and very large amount edge cases
- unknown field handling

## Docker

Build and run the FastAPI backend and Streamlit frontend with Docker Compose:

```powershell
docker-compose -f docker/docker-compose.yml up -d --build
```

The Compose stack exposes:

```text
FastAPI   -> http://localhost:8000
Streamlit -> http://localhost:8501
```

Inside Docker, Streamlit calls FastAPI through the service name:

```text
FASTAPI_URL=http://fraud-api:8000
```

This avoids the common Docker networking mistake where `localhost` inside the UI container points to the UI container itself instead of the backend container.

## CI/CD

GitHub Actions workflows:

```text
.github/workflows/ci.yml
.github/workflows/deploy.yml
```

CI runs on push and pull requests:

- checks out the repo
- installs Python dependencies
- configures AWS credentials
- verifies AWS identity
- pulls DVC model artifacts from S3
- runs the API test suite

Deployment runs after CI succeeds on `main`:

- SSHs into the EC2 instance
- pulls the latest GitHub code
- installs lightweight DVC dependencies on the host
- pulls model artifacts from S3
- rebuilds and restarts the Docker Compose stack

Required GitHub repository secrets:

```text
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
EC2_HOST
EC2_USERNAME
EC2_SSH_KEY
```

For a full rebuild guide, see [EC2_DEPLOYMENT_GUIDE.md](EC2_DEPLOYMENT_GUIDE.md).

## Git And Artifact Rules

Commit to Git:

- source code
- tests
- Docker files
- DVC metadata
- small JSON reports
- README and configuration files

Do not commit:

- `.venv/`
- `.env`
- `mlruns/`
- raw CSV files
- generated large CSV files
- `models/model.pkl`
- DagsHub or AWS credentials

Large data and model artifacts belong in DVC/S3.

## Key MLOps Decisions

**Why DVC?**  
DVC keeps large datasets and model artifacts out of Git while preserving reproducibility through `dvc.yaml`, `dvc.lock`, and S3-backed storage.

**Why AWS S3 for DVC remote?**  
S3 is a common production object store, and using it demonstrates cloud-based data/model artifact management.

**Why DagsHub MLflow instead of local MLflow?**  
Hosted MLflow avoids local `mlruns/` disk growth and provides an online experiment dashboard for portfolio review.

**Why not expose all 270 model features in Streamlit?**  
The UI is for demo and scenario testing. In production, feature values would come from backend transaction systems, historical aggregations, and feature pipelines.

**Why time-based split?**  
Fraud is temporal. A random split can leak future transaction patterns into training and overstate real-world performance.

**Why LightGBM?**  
LightGBM is strong for large tabular datasets, trains quickly, handles nonlinear feature interactions well, and provides low-latency inference.

## Resume Bullet

Built an end-to-end MLOps fraud detection system using LightGBM, FastAPI, Streamlit, DVC, AWS S3, DagsHub MLflow, Docker, and GitHub. Implemented reproducible data pipelines, remote experiment tracking, cloud-backed artifact versioning, API testing, and real-time fraud inference with p99 latency under 3 ms.

## Interview Talking Points

- Git stores source code and DVC metadata, while S3 stores large data/model artifacts.
- DVC makes the ML pipeline reproducible with explicit stages and dependencies.
- MLflow is hosted on DagsHub to avoid local `mlruns/` disk growth.
- The Streamlit app is a demo interface; in production, another backend system or feature store would provide the feature vector.
- EC2 serves Dockerized FastAPI and Streamlit containers.
- GitHub Actions handles CI tests and CD deployment after a successful push to `main`.
- The model uses a time-based split to reduce future-data leakage.
- The API uses a curated request schema for demo inference while the model internally consumes 270 features.
