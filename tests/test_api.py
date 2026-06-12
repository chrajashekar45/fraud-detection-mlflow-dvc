"""
RaptorX — API Unit Tests
Tests for /health, /model/info, and /predict endpoints.
Run with: pytest tests/ -v
"""

import sys
from pathlib import Path

# Add project root to path so api.main can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


# ── /health ───────────────────────────────────────────────────────────────────

def test_health_returns_200():
    """Health endpoint must return 200 status."""
    response = client.get("/health")
    assert response.status_code == 200


def test_health_model_loaded():
    """Health endpoint must confirm model is loaded."""
    response = client.get("/health")
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["features"] > 0
    assert "threshold" in data


# ── /model/info ───────────────────────────────────────────────────────────────

def test_model_info_returns_200():
    """Model info endpoint must return 200 status."""
    response = client.get("/model/info")
    assert response.status_code == 200


def test_model_info_has_metrics():
    """Model info must contain evaluation metrics."""
    response = client.get("/model/info")
    data = response.json()
    assert "auc_roc" in data["evaluation"]
    assert "auc_pr"  in data["evaluation"]
    assert "f1"      in data["evaluation"]
    assert data["evaluation"]["auc_roc"] > 0.5  # sanity check


def test_model_info_has_top_features():
    """Model info must return top features list."""
    response = client.get("/model/info")
    data = response.json()
    assert "top_features" in data
    assert len(data["top_features"]) > 0


# ── /predict ──────────────────────────────────────────────────────────────────

def test_predict_minimal_input():
    """Predict must work with only TransactionAmt provided."""
    response = client.post("/predict", json={"TransactionAmt": 100.0})
    assert response.status_code == 200
    data = response.json()
    assert "risk_score"   in data
    assert "is_fraud"     in data
    assert "top_features" in data
    assert "latency_ms"   in data


def test_predict_risk_score_range():
    """Risk score must always be between 0 and 1."""
    response = client.post("/predict", json={"TransactionAmt": 250.0})
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["risk_score"] <= 1.0


def test_predict_is_fraud_is_bool():
    """is_fraud field must be a boolean."""
    response = client.post("/predict", json={"TransactionAmt": 100.0})
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["is_fraud"], bool)


def test_predict_latency_under_50ms():
    """Single prediction latency must be under 50ms."""
    response = client.post("/predict", json={"TransactionAmt": 100.0})
    assert response.status_code == 200
    data = response.json()
    assert data["latency_ms"] < 50.0, (
        f"Latency {data['latency_ms']}ms exceeds 50ms target"
    )


def test_predict_high_risk_transaction():
    """
    High-risk transaction (large amount, night time, high fraud rate card,
    identity mismatches) should return a higher risk score than a
    low-risk transaction.
    """
    high_risk = {
        "TransactionAmt"        : 2500.0,
        "feat_hour"             : 3,
        "feat_is_night"         : 1,
        "feat_card1_fraud_rate" : 0.85,
        "feat_is_round_amt"     : 1,
        "feat_m_mismatch_count" : 4.0,
        "feat_has_identity"     : 0,
        "log_C13"               : 6.5,
    }
    low_risk = {
        "TransactionAmt"        : 45.0,
        "feat_hour"             : 14,
        "feat_is_night"         : 0,
        "feat_card1_fraud_rate" : 0.001,
        "feat_is_round_amt"     : 0,
        "feat_m_mismatch_count" : 0.0,
        "feat_has_identity"     : 1,
        "log_C13"               : 1.0,
    }
    r_high = client.post("/predict", json=high_risk)
    r_low  = client.post("/predict", json=low_risk)

    assert r_high.status_code == 200
    assert r_low.status_code  == 200

    score_high = r_high.json()["risk_score"]
    score_low  = r_low.json()["risk_score"]

    assert score_high > score_low, (
        f"Expected high-risk score ({score_high}) > "
        f"low-risk score ({score_low})"
    )


def test_predict_zero_amount():
    """Edge case: zero amount transaction must not crash."""
    response = client.post("/predict", json={"TransactionAmt": 0.0})
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["risk_score"] <= 1.0


def test_predict_very_large_amount():
    """Edge case: very large amount must not crash."""
    response = client.post("/predict", json={"TransactionAmt": 999999.0})
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["risk_score"] <= 1.0


def test_predict_extra_fields_ignored():
    """Extra unknown fields in request must be ignored gracefully."""
    response = client.post("/predict", json={
        "TransactionAmt" : 100.0,
        "unknown_field"  : "should_be_ignored",
        "another_extra"  : 12345
    })
    assert response.status_code == 200


def test_predict_top_features_list():
    """top_features must be a non-empty list of strings."""
    response = client.post("/predict", json={"TransactionAmt": 100.0})
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["top_features"], list)
    assert len(data["top_features"]) > 0
    assert all(isinstance(f, str) for f in data["top_features"])