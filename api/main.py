"""
Fraud Detection REST API
FastAPI application with three endpoints:
  POST /predict     → accepts transaction JSON, returns risk score
  GET  /health      → health check
  GET  /model/info  → model metadata and evaluation metrics
"""

import pickle
import json
import time
import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "model.pkl"
EVAL_PATH  = BASE_DIR / "models" / "eval_report.json"
LAT_PATH   = BASE_DIR / "models" / "latency_report.json"

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Fraud Detection API",
    description = "Real-time fraud risk scoring for payment transactions.",
    version     = "1.0.0"
)

# ── Load model once at startup ────────────────────────────────────────────────
print("Loading model...")
with open(MODEL_PATH, "rb") as f:
    _bundle = pickle.load(f)

_model     = _bundle["model"]
_features  = _bundle["features"]
_threshold = _bundle.get("threshold", 0.5)

with open(EVAL_PATH) as f:
    _eval_report = json.load(f)

with open(LAT_PATH) as f:
    _lat_report = json.load(f)

print(f"Model loaded. Features: {len(_features)} | Threshold: {_threshold}")


# ── Request schema ────────────────────────────────────────────────────────────
class TransactionRequest(BaseModel):
    """
    Transaction input for fraud scoring.
    All fields are optional — missing values are filled with 0.
    In production, caller should provide as many fields as possible
    for best prediction accuracy.
    """
    # Core transaction fields
    TransactionAmt  : Optional[float] = Field(default=100.0,  description="Transaction amount in USD")
    ProductCD       : Optional[float] = Field(default=0.0,    description="Product category code (encoded)")
    card1           : Optional[float] = Field(default=9678.0, description="Card identifier 1")
    card2           : Optional[float] = Field(default=361.0,  description="Card identifier 2")
    card3           : Optional[float] = Field(default=150.0,  description="Card identifier 3")
    card4           : Optional[float] = Field(default=0.0,    description="Card network (encoded)")
    card5           : Optional[float] = Field(default=226.0,  description="Card identifier 5")
    card6           : Optional[float] = Field(default=0.0,    description="Card type (encoded)")
    addr1           : Optional[float] = Field(default=299.0,  description="Billing address zip")
    addr2           : Optional[float] = Field(default=87.0,   description="Country code")
    P_emaildomain   : Optional[float] = Field(default=0.0,    description="Purchaser email domain (encoded)")
    R_emaildomain   : Optional[float] = Field(default=0.0,    description="Recipient email domain (encoded)")

    # Engineered features — caller can provide or leave as defaults
    feat_log_amt               : Optional[float] = Field(default=None)
    feat_amt_zscore            : Optional[float] = Field(default=0.0)
    feat_amt_pctrank_card1     : Optional[float] = Field(default=0.5)
    feat_card1_mean_amt        : Optional[float] = Field(default=100.0)
    feat_amt_to_card1_mean     : Optional[float] = Field(default=1.0)
    feat_is_round_amt          : Optional[float] = Field(default=0.0)
    feat_high_amt_flag         : Optional[float] = Field(default=0.0)
    feat_hour                  : Optional[float] = Field(default=12.0)
    feat_dow                   : Optional[float] = Field(default=2.0)
    feat_is_weekend            : Optional[float] = Field(default=0.0)
    feat_is_night              : Optional[float] = Field(default=0.0)
    feat_time_since_last_card1 : Optional[float] = Field(default=3600.0)
    feat_card1_txn_count       : Optional[float] = Field(default=10.0)
    feat_card1_std_amt         : Optional[float] = Field(default=50.0)
    feat_card1_amt_cv          : Optional[float] = Field(default=0.5)
    feat_card1_fraud_rate      : Optional[float] = Field(default=0.035)
    feat_email_domain_risk     : Optional[float] = Field(default=0.035)
    feat_email_domain_mismatch : Optional[float] = Field(default=0.0)
    feat_card_combo_freq       : Optional[float] = Field(default=0.01)
    feat_productcd_fraud_rate  : Optional[float] = Field(default=0.035)
    feat_m_match_count         : Optional[float] = Field(default=5.0)
    feat_m_mismatch_count      : Optional[float] = Field(default=0.0)
    feat_d1_to_d10_ratio       : Optional[float] = Field(default=1.0)
    feat_has_identity          : Optional[float] = Field(default=0.0)

    # D columns
    D1  : Optional[float] = Field(default=3.0)
    D2  : Optional[float] = Field(default=97.0)
    D3  : Optional[float] = Field(default=8.0)
    D4  : Optional[float] = Field(default=26.0)
    D5  : Optional[float] = Field(default=-1.0)
    D6  : Optional[float] = Field(default=-1.0)
    D7  : Optional[float] = Field(default=-1.0)
    D8  : Optional[float] = Field(default=-1.0)
    D9  : Optional[float] = Field(default=-1.0)
    D10 : Optional[float] = Field(default=15.0)
    D11 : Optional[float] = Field(default=43.0)
    D12 : Optional[float] = Field(default=-1.0)
    D13 : Optional[float] = Field(default=-1.0)
    D14 : Optional[float] = Field(default=-1.0)
    D15 : Optional[float] = Field(default=52.0)

    # M columns (encoded: T=1, F=0, NaN=-1)
    M1  : Optional[float] = Field(default=-1.0)
    M2  : Optional[float] = Field(default=-1.0)
    M3  : Optional[float] = Field(default=-1.0)
    M4  : Optional[float] = Field(default=2.0)
    M5  : Optional[float] = Field(default=-1.0)
    M6  : Optional[float] = Field(default=1.0)
    M7  : Optional[float] = Field(default=-1.0)
    M8  : Optional[float] = Field(default=-1.0)
    M9  : Optional[float] = Field(default=-1.0)

    # Log C columns
    log_C1  : Optional[float] = Field(default=0.0)
    log_C2  : Optional[float] = Field(default=0.0)
    log_C3  : Optional[float] = Field(default=0.0)
    log_C4  : Optional[float] = Field(default=0.0)
    log_C5  : Optional[float] = Field(default=0.0)
    log_C6  : Optional[float] = Field(default=0.0)
    log_C7  : Optional[float] = Field(default=0.0)
    log_C8  : Optional[float] = Field(default=0.0)
    log_C9  : Optional[float] = Field(default=0.0)
    log_C10 : Optional[float] = Field(default=0.0)
    log_C11 : Optional[float] = Field(default=0.0)
    log_C12 : Optional[float] = Field(default=0.0)
    log_C13 : Optional[float] = Field(default=0.0)
    log_C14 : Optional[float] = Field(default=0.0)

    # Distance
    log_dist1 : Optional[float] = Field(default=0.0)
    has_dist2 : Optional[float] = Field(default=0.0)

    # Device
    DeviceType : Optional[float] = Field(default=2.0)
    DeviceInfo : Optional[float] = Field(default=0.0)

    # Identity cols
    id_01 : Optional[float] = Field(default=-5.0)
    id_02 : Optional[float] = Field(default=125800.0)
    id_03 : Optional[float] = Field(default=0.0)
    id_04 : Optional[float] = Field(default=0.0)
    id_05 : Optional[float] = Field(default=0.0)
    id_06 : Optional[float] = Field(default=0.0)
    id_09 : Optional[float] = Field(default=0.0)
    id_10 : Optional[float] = Field(default=0.0)
    id_11 : Optional[float] = Field(default=100.0)
    id_12 : Optional[float] = Field(default=0.0)
    id_13 : Optional[float] = Field(default=0.0)
    id_14 : Optional[float] = Field(default=0.0)
    id_15 : Optional[float] = Field(default=0.0)
    id_16 : Optional[float] = Field(default=0.0)
    id_17 : Optional[float] = Field(default=0.0)
    id_19 : Optional[float] = Field(default=0.0)
    id_20 : Optional[float] = Field(default=0.0)
    id_28 : Optional[float] = Field(default=0.0)
    id_30 : Optional[float] = Field(default=0.0)
    id_31 : Optional[float] = Field(default=0.0)
    id_32 : Optional[float] = Field(default=0.0)
    id_33 : Optional[float] = Field(default=0.0)
    id_34 : Optional[float] = Field(default=0.0)
    id_35 : Optional[float] = Field(default=0.0)
    id_36 : Optional[float] = Field(default=0.0)
    id_37 : Optional[float] = Field(default=0.0)
    id_38 : Optional[float] = Field(default=0.0)

    class Config:
        # Allow extra fields — ignore unknown columns gracefully
        extra = "allow"


# ── Response schema ───────────────────────────────────────────────────────────
class PredictResponse(BaseModel):
    risk_score   : float
    is_fraud     : bool
    top_features : list
    latency_ms   : float


# ── Helper: build feature row ─────────────────────────────────────────────────
def build_feature_row(txn: TransactionRequest) -> np.ndarray:
    """
    Convert transaction request to numpy array matching model feature order.
    feat_log_amt is auto-computed from TransactionAmt if not provided.
    Missing features default to 0.
    """
    data = txn.dict()

    # Auto-compute feat_log_amt if not provided
    if data.get("feat_log_amt") is None:
        data["feat_log_amt"] = float(np.log1p(data.get("TransactionAmt", 0)))

    # Build row in exact feature order
    row = np.array(
        [[data.get(col, 0) or 0 for col in _features]],
        dtype=np.float32
    )
    return row


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    """
    Health check endpoint.
    Returns 200 if API is running and model is loaded.
    """
    return {
        "status"       : "ok",
        "model_loaded" : _model is not None,
        "features"     : len(_features),
        "threshold"    : _threshold
    }


@app.get("/model/info", tags=["System"])
def model_info():
    """
    Model metadata and evaluation metrics.
    Returns AUC-ROC, AUC-PR, F1, top features, and latency benchmarks.
    """
    return {
        "model_type"        : "LightGBM",
        "num_features"      : len(_features),
        "threshold"         : _threshold,
        "evaluation"        : {
            "auc_roc"       : _eval_report.get("auc_roc"),
            "auc_pr"        : _eval_report.get("auc_pr"),
            "f1"            : _eval_report.get("f1"),
            "best_iteration": _eval_report.get("model_params", {}).get("best_iteration"),
        },
        "top_features"      : list(
            _eval_report.get("top_15_features", {}).keys()
        )[:5],
        "latency_benchmarks": {
            "single_p99_ms" : _lat_report.get("latency_results", {})
                                         .get("baseline_lgbm_pickle", {})
                                         .get("single", {})
                                         .get("p99_ms"),
            "batch_p99_ms"  : _lat_report.get("latency_results", {})
                                         .get("baseline_lgbm_pickle", {})
                                         .get("batch_100", {})
                                         .get("p99_ms"),
        },
        "imbalance_strategy": _eval_report.get("imbalance_strategy", {}).get("method"),
        "split_strategy"    : _eval_report.get("split_strategy", {}).get("method"),
    }


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(txn: TransactionRequest):
    """
    Score a transaction for fraud risk.

    - **risk_score**: probability of fraud (0.0 to 1.0)
    - **is_fraud**: True if risk_score >= threshold (0.74)
    - **top_features**: top 5 most important features for this model
    - **latency_ms**: time taken for this prediction in milliseconds
    """
    t_start = time.perf_counter()

    try:
        row      = build_feature_row(txn)
        prob     = float(_model.predict_proba(row)[0][1])
        is_fraud = prob >= _threshold

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

    latency_ms = round((time.perf_counter() - t_start) * 1000, 3)

    # Top 5 features from eval report
    top_features = list(
        _eval_report.get("top_15_features", {}).keys()
    )[:5]

    return PredictResponse(
        risk_score   = round(prob, 4),
        is_fraud     = is_fraud,
        top_features = top_features,
        latency_ms   = latency_ms
    )