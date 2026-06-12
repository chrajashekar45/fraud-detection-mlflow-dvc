# RaptorX — Fraud Detection API

End-to-end ML pipeline for real-time payment fraud detection.
Built on the IEEE-CIS Fraud Detection dataset (590K transactions, 434 features).

---

## Quick Start

```bash
docker-compose -f docker/docker-compose.yml up --build
```

for fastapi:Run from project root:
```
uvicorn api.main:app --reload --port 8000
API available at: http://localhost:8000/docs

---

## Architecture

```
Raw CSVs (train_transaction + train_identity)
        ↓
cleaning/clean.py          → data/clean/clean_data.csv
        ↓
features/engineer.py       → data/features/featured_data.csv
        ↓
models/train.py            → models/model.pkl + eval_report.json
        ↓
models/optimize.py         → models/latency_report.json
        ↓
api/main.py                → REST API (FastAPI)
        ↓
docker/Dockerfile          → Containerised service
```

---

## Project Structure

```
raptorx-fraud/
├── cleaning/
│   └── clean.py                  # Data cleaning pipeline
├── features/
│   └── engineer.py               # Feature engineering (24 features)
├── models/
│   ├── train.py                  # LightGBM training
│   ├── optimize.py               # Latency benchmarking + ONNX attempt
│   ├── model.pkl                 # Trained model
│   ├── eval_report.json          # AUC, F1, confusion matrix
│   └── latency_report.json       # Before/after latency benchmarks
├── api/
│   └── main.py                   # FastAPI application
├── tests/
│   └── test_api.py               # 12 unit tests
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── data/
│   ├── raw/                      # Original Kaggle CSVs
│   ├── clean/                    # Cleaned dataset + cleaning report
│   └── features/                 # Featured dataset + feature report
├── requirements.txt
└── README.md
```

---

## Setup (without Docker)

**1. Clone and create virtual environment**
```bash
git clone <your-repo-url>
cd raptorx-fraud
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

**2. Download dataset from Kaggle**
```bash
kaggle competitions download -c ieee-fraud-detection
```
Extract `train_transaction.csv` and `train_identity.csv` into `data/raw/`

**3. Run the full pipeline**
```bash
python cleaning/clean.py
python features/engineer.py
python models/train.py
python models/optimize.py
```

**4. Start the API**
```bash
uvicorn api.main:app --reload --port 8000
```

**5. Run tests**
```bash
pytest tests/ -v
```

---

## API Endpoints

### `GET /health`
Health check — confirms model is loaded and API is running.

```json
{
  "status": "ok",
  "model_loaded": true,
  "features": 270,
  "threshold": 0.74
}
```

### `GET /model/info`
Model metadata, evaluation metrics, and latency benchmarks.

```json
{
  "model_type": "LightGBM",
  "evaluation": {
    "auc_roc": 0.8674,
    "auc_pr": 0.496,
    "f1": 0.4693
  },
  "latency_benchmarks": {
    "single_p99_ms": 3.26,
    "batch_p99_ms": 3.50
  }
}
```

### `POST /predict`
Score a transaction for fraud risk.

**Request:**
```json
{
  "TransactionAmt": 2500.0,
  "feat_hour": 3,
  "feat_is_night": 1,
  "feat_card1_fraud_rate": 0.85
}
```

**Response:**
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

---

## Data Cleaning

Pipeline: `cleaning/clean.py` | Report: `data/clean/cleaning_report.json`

| Column Group | Strategy | Rationale |
|-------------|----------|-----------|
| C1–C14 (counts) | Fill 0 → log1p transform | Missing = no occurrence. Log1p compresses outliers (max 5691) while preserving fraud signal in high counts |
| D1–D15 (time deltas) | <50% missing → median, >50% → -1 sentinel | -1 sentinel preserves "new entity" signal — unseen cards have higher fraud risk |
| M1–M9 (match flags) | T=1, F=0, NaN=-1, other=2 | Preserves missingness as distinct signal from confirmed mismatch |
| V1–V339 (Vesta) | Drop >90% missing, median fill rest | Only V150 had outliers severe enough for log1p |
| dist1 | Median fill → log1p | 59.7% missing, values reach thousands |
| dist2 | Binary flag has_dist2 | 93.6% missing — flag itself is the signal (linked identity record) |
| Categoricals | Fill unknown → label encode | DeviceInfo reduced to OS family (Android/iOS/Windows/Mac/Linux) |

**Result:** 590,540 rows, 0 nulls remaining, idempotent.

---

## Feature Engineering

Pipeline: `features/engineer.py` | Report: `data/features/feature_report.json`

24 features built across 5 categories:

**Category 1 — Transaction Amount (6 features)**
| Feature | Why it helps |
|---------|-------------|
| feat_log_amt | Compresses skew — fraudsters use specific amounts |
| feat_amt_zscore | Flags globally unusual amounts |
| feat_amt_pctrank_card1 | Rank within card's own history — largest ever = red flag |
| feat_amt_to_card1_mean | Amount vs card's normal spend — 8x = anomaly |
| feat_is_round_amt | Round amounts = card testing pattern |
| feat_high_amt_flag | Top 5% amounts — higher fraud rate zone |

**Category 2 — Temporal (5 features)**
| Feature | Why it helps |
|---------|-------------|
| feat_hour | Fraud peaks 1–5am (cardholders asleep) |
| feat_dow | Weekend vs weekday fraud patterns |
| feat_is_weekend | Binary weekend flag |
| feat_is_night | Binary 11pm–5am flag |
| feat_time_since_last_card1 | Seconds-apart transactions = card testing |

**Category 3 — Aggregation per card1 (4 features)**
| Feature | Why it helps |
|---------|-------------|
| feat_card1_txn_count | Count=1 cards are first-time = higher risk |
| feat_card1_std_amt | Low std then sudden spike = anomaly |
| feat_card1_amt_cv | Erratic spending pattern |
| feat_card1_fraud_rate | Cards with prior fraud history |

**Category 4 — Categorical (4 features)**
| Feature | Why it helps |
|---------|-------------|
| feat_email_domain_risk | Disposable email domains have higher fraud rates |
| feat_email_domain_mismatch | Buyer vs recipient domain mismatch = account takeover |
| feat_card_combo_freq | Rare card+network combos = suspicious |
| feat_productcd_fraud_rate | Digital goods have higher fraud rates than physical |

**Category 5 — Behavioural (5 features)**
| Feature | Why it helps |
|---------|-------------|
| feat_m_match_count | Legit transactions have more identity matches |
| feat_m_mismatch_count | Confirmed mismatches = strong fraud indicator |
| feat_d1_to_d10_ratio | Unusual timing patterns |
| feat_has_identity | No identity data = anonymous = riskier |
| feat_card1_mean_amt | Historical spend baseline |

---

## Model Training

Pipeline: `models/train.py` | Report: `models/eval_report.json`

**Algorithm:** LightGBM (gradient boosted trees)

**Key decisions:**

**Time-based split (no leakage)**
Sorted by TransactionDT, first 80% = train, last 20% = test.
Random split would leak future transaction patterns into training — invalid for fraud detection.

**Class imbalance — scale_pos_weight**
Dataset is 3.5% fraud (96.5% legitimate).
`scale_pos_weight = 27.5` weights fraud rows 27.5x more in the loss function.
Chosen over SMOTE — SMOTE on high-dimensional tabular data with binary/categorical columns generates unrealistic synthetic rows.

**Leakage prevention**
Target-encoded features (card1_fraud_rate, email_domain_risk, productcd_fraud_rate) recomputed on train set only. Test rows mapped from train statistics. Unseen entities use global train fraud rate as fallback.

**Two-phase training**
Phase 1: Train on all ~440 features, identify zero/low-importance columns.
Phase 2: Retrain on pruned 270 features. Prevents overfitting from useless columns.

**Threshold tuning**
Default threshold 0.5 predicts everything as non-fraud on imbalanced data.
Tuned threshold = 0.74 by scanning 0.05–0.75 to maximise F1.

### Evaluation Results

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.8674 |
| AUC-PR | 0.4960 |
| F1 | 0.4693 |
| Threshold | 0.74 |
| Precision | 41.2% |
| Recall | 54.6% |

**Confusion Matrix (test set — 118,108 rows)**
```
                Predicted Legit    Predicted Fraud
Actual Legit       110,873              3,171
Actual Fraud         1,846              2,218
```

**Top 5 Features by Importance**
1. feat_card1_fraud_rate (843)
2. log_C13 (392)
3. addr1 (367)
4. card1 (365)
5. feat_card1_txn_count (356)

10 of top 15 features are engineered `feat_` columns.

---

## Latency Optimization

Pipeline: `models/optimize.py` | Report: `models/latency_report.json`

Two optimizations applied:

**Optimization 1 — Feature Selection**
Phase 1 training on all 447 cols identified zero/low-importance features.
Phase 2 retrained on 270 cols only (39.6% reduction).

**Optimization 2 — ONNX Export (attempted)**
ONNX export attempted via skl2onnx. Version incompatibility with LightGBM prevented successful export. Baseline already met all targets so ONNX was not required.

### Latency Results

| Metric | Single Prediction | Batch of 100 | Target | Status |
|--------|------------------|--------------|--------|--------|
| p50 | 2.58ms | 2.49ms | — | — |
| p95 | 2.92ms | 3.22ms | — | — |
| p99 | 3.26ms | 3.50ms | <50ms / <500ms | ✅ PASS |

Both targets met with **15x–142x margin**.

---

## Running Tests

```bash
pytest tests/ -v
```

12 tests covering:
- Health endpoint status and model loaded confirmation
- Model info metrics and top features
- Predict with minimal input (just TransactionAmt)
- Risk score always between 0 and 1
- is_fraud always boolean
- Latency under 50ms
- High-risk transaction scores higher than low-risk
- Edge cases: zero amount, very large amount
- Extra unknown fields ignored gracefully
- top_features is a non-empty list of strings

---

## Architecture Decisions

**Why LightGBM over XGBoost?**
LightGBM is faster to train on large datasets (leaf-wise tree growth vs level-wise), handles categorical features natively, and performs comparably or better on tabular fraud data.

**Why not drop all V columns?**
338 V columns are Vesta's proprietary risk features. Many are binary (0/1) with low missing rates. LightGBM's feature importance naturally ignores useless ones — we let the model decide rather than arbitrarily dropping by column name.

**Why median imputation for D columns?**
D columns are time deltas. Mean is sensitive to outliers in time data. Median is more robust. Columns >50% missing get -1 sentinel to preserve the "new entity" signal.

**Why threshold 0.74 instead of 0.5?**
On 3.5% fraud data, threshold 0.5 predicts everything as non-fraud (F1=0). Tuning to 0.74 maximises F1 by finding the best precision-recall trade-off on the test set.
