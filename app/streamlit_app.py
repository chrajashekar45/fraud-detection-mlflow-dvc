import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="RaptorX Fraud Detection",
    page_icon="🛡️",
    layout="wide",
)

st.title("RaptorX Fraud Detection")
st.caption("Real-time payment fraud risk scoring using a LightGBM model.")

with st.sidebar:
    st.header("API Settings")
    api_url = st.text_input("FastAPI URL", value=API_URL)

st.subheader("Transaction Details")

col1, col2, col3 = st.columns(3)

with col1:
    transaction_amt = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=100.0,
        step=10.0,
    )

    hour = st.slider(
        "Transaction Hour",
        min_value=0,
        max_value=23,
        value=12,
    )

with col2:
    is_night = st.checkbox("Night Transaction")
    is_weekend = st.checkbox("Weekend Transaction")
    is_round_amt = st.checkbox("Round Amount")

with col3:
    card_fraud_rate = st.slider(
        "Card Fraud Rate",
        min_value=0.0,
        max_value=1.0,
        value=0.035,
        step=0.005,
    )

    mismatch_count = st.slider(
        "Identity Mismatch Count",
        min_value=0,
        max_value=9,
        value=0,
    )

payload = {
    "TransactionAmt": transaction_amt,
    "feat_hour": hour,
    "feat_is_night": 1 if is_night else 0,
    "feat_is_weekend": 1 if is_weekend else 0,
    "feat_is_round_amt": 1 if is_round_amt else 0,
    "feat_card1_fraud_rate": card_fraud_rate,
    "feat_m_mismatch_count": mismatch_count,
}

if st.button("Predict Fraud Risk", type="primary"):
    try:
        response = requests.post(
            f"{api_url}/predict",
            json=payload,
            timeout=10,
        )

        if response.status_code != 200:
            st.error(f"API error: {response.status_code}")
            st.code(response.text)
        else:
            result = response.json()

            risk_score = result["risk_score"]
            is_fraud = result["is_fraud"]

            metric_col1, metric_col2, metric_col3 = st.columns(3)

            with metric_col1:
                st.metric("Risk Score", f"{risk_score * 100:.2f}%")

            with metric_col2:
                decision = "Fraud Risk" if is_fraud else "Legitimate"
                st.metric("Decision", decision)

            with metric_col3:
                st.metric("Latency", f"{result['latency_ms']} ms")

            if is_fraud:
                st.error("This transaction is classified as high risk.")
            else:
                st.success("This transaction is classified as low risk.")

            st.subheader("Top Model Features")
            st.write(result["top_features"])

    except requests.exceptions.ConnectionError:
        st.error("Could not connect to FastAPI. Make sure the API is running on port 8000.")

    except requests.exceptions.Timeout:
        st.error("The API request timed out.")

    except Exception as e:
        st.error(f"Unexpected error: {e}")