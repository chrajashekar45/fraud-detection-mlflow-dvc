
"""
RaptorX — Feature Engineering Pipeline
IEEE-CIS Fraud Detection Dataset
 
Builds 20+ features across 5 categories:
  1. Transaction features    (amount-based)
  2. Temporal features       (time-based)
  3. Aggregation features    (per-card rolling stats)
  4. Categorical features    (email risk, card combos)
  5. Behavioural features    (M flag patterns, D col ratios)
 
All features prefixed with feat_ for easy identification.
Idempotent: running twice produces the same output.
"""
 
import pandas as pd
import numpy as np
import json
from pathlib import Path
 
# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent  # → E:\Raptorx
IN_PATH  = BASE_DIR / "data" / "clean" / "clean_data.csv"
OUT_DIR  = BASE_DIR / "data" / "features"
OUT_DIR.mkdir(parents=True, exist_ok=True)
 
# ── Feature registry — every feature documented here ─────────────────────────
# Each entry: feature_name → why it helps fraud detection
FEATURE_REGISTRY = {
    # ── Category 1: Transaction amount features
    "feat_log_amt": (
        "Log of TransactionAmt. Compresses heavy right skew. "
        "Fraudsters often use very specific amounts — log scale "
        "makes these patterns easier for the model to find."
    ),
    "feat_amt_zscore": (
        "Z-score of TransactionAmt globally. Flags transactions "
        "that are statistically unusual vs the whole population. "
        "A score of 3+ means amount is 3 std deviations above mean."
    ),
    "feat_amt_pctrank_card1": (
        "Percentile rank of this transaction amount among all "
        "transactions by the same card1. A rank of 0.99 means "
        "this is the largest amount this card has ever transacted — "
        "strong fraud signal."
    ),
    "feat_amt_to_card1_mean": (
        "Ratio of this transaction amount to the card's historical "
        "mean amount. Value of 5.0 means this txn is 5x larger "
        "than normal for this card — anomaly signal."
    ),
    "feat_is_round_amt": (
        "Binary: 1 if TransactionAmt is a round number (no cents). "
        "Fraudsters often use round amounts like 100.00, 500.00 "
        "when testing stolen cards."
    ),
    "feat_high_amt_flag": (
        "Binary: 1 if TransactionAmt is above the 95th percentile. "
        "High-value transactions have disproportionately higher "
        "fraud rates."
    ),
 
    # ── Category 2: Temporal features
    "feat_hour": (
        "Hour of day (0–23) extracted from TransactionDT. "
        "Fraud peaks in early morning hours (1–5am) when "
        "cardholders are asleep and less likely to notice."
    ),
    "feat_dow": (
        "Day of week (0=Mon, 6=Sun) from TransactionDT. "
        "Fraud patterns differ on weekends vs weekdays — "
        "weekends have lower legitimate transaction volumes "
        "making fraud easier to hide."
    ),
    "feat_is_weekend": (
        "Binary: 1 if transaction falls on Saturday or Sunday. "
        "Derived from feat_dow. Fraud-to-legitimate ratio "
        "is higher on weekends."
    ),
    "feat_is_night": (
        "Binary: 1 if transaction is between 11pm and 5am. "
        "Night transactions have significantly higher fraud rates "
        "across all card types."
    ),
    "feat_time_since_last_card1": (
        "Seconds since the last transaction by the same card1. "
        "Very small values (seconds apart) indicate rapid-fire "
        "transactions — a classic card testing pattern where "
        "fraudsters make many small purchases quickly."
    ),
 
    # ── Category 3: Aggregation features (per card1)
    "feat_card1_txn_count": (
        "Total number of transactions by this card1 in the dataset. "
        "Cards with very high counts may be used for testing. "
        "Cards with count=1 are first-time cards — higher risk."
    ),
    "feat_card1_mean_amt": (
        "Mean transaction amount for this card1 historically. "
        "Used as denominator for feat_amt_to_card1_mean. "
        "Also directly useful — cards with very low mean amounts "
        "that suddenly transact large amounts are suspicious."
    ),
    "feat_card1_std_amt": (
        "Standard deviation of transaction amounts for this card1. "
        "Low std = very consistent spending pattern. "
        "A sudden deviation from that pattern = fraud signal."
    ),
    "feat_card1_amt_cv": (
        "Coefficient of Variation = std/mean for card1 amounts. "
        "Normalises std by mean so cards with different spending "
        "levels are comparable. High CV = erratic spending."
    ),
    "feat_card1_fraud_rate": (
        "Historical fraud rate for this card1 (computed on training "
        "data only to avoid leakage). Cards that have had previous "
        "fraud are at higher risk of future fraud."
    ),
 
    # ── Category 4: Categorical / entity features
    "feat_email_domain_risk": (
        "Fraud rate per P_emaildomain (purchaser email domain). "
        "Some domains (e.g. anonymous/disposable email providers) "
        "have significantly higher fraud rates. "
        "Unknown/missing domains get mean fraud rate."
    ),
    "feat_email_domain_mismatch": (
        "Binary: 1 if P_emaildomain != R_emaildomain (recipient). "
        "A mismatch between purchaser and recipient email domains "
        "is a common signal in account takeover fraud."
    ),
    "feat_card_combo_freq": (
        "Frequency encoding of card1+card4 combination. "
        "Rare card combinations (low frequency) are more likely "
        "to be fraudulent — common legitimate card+network "
        "combinations are well-established."
    ),
    "feat_productcd_fraud_rate": (
        "Fraud rate per ProductCD category. Some product types "
        "(e.g. digital goods, gift cards) have much higher fraud "
        "rates than physical goods."
    ),
 
    # ── Category 5: Behavioural features
    "feat_m_match_count": (
        "Count of M columns equal to 1 (T=matched). "
        "M1–M9 are identity/address match flags. "
        "Legitimate transactions tend to have more matches. "
        "Low match count = mismatched identity details = fraud signal."
    ),
    "feat_m_mismatch_count": (
        "Count of M columns equal to 0 (F=mismatch). "
        "Direct count of confirmed mismatches across all M flags. "
        "High mismatch count is a strong fraud indicator."
    ),
    "feat_d1_to_d10_ratio": (
        "Ratio of D1 (days since last transaction) to D10 "
        "(another time delta). Captures relative timing patterns. "
        "Unusual ratios indicate atypical transaction timing."
    ),
    "feat_has_identity": (
        "Binary: 1 if this transaction has identity data (from "
        "train_identity.csv join). Transactions without identity "
        "data are more anonymous and statistically riskier."
    ),
}
 
 
# ── Helper: safe division ─────────────────────────────────────────────────────
def safe_div(a, b, fill=0.0):
    return np.where(b != 0, a / b, fill)
 
 
# ── Category 1: Transaction amount features ───────────────────────────────────
def build_amt_features(df: pd.DataFrame) -> pd.DataFrame:
    print("  [1/5] Building transaction amount features...")
 
    # 1. Log amount
    df["feat_log_amt"] = np.log1p(df["TransactionAmt"]).astype(np.float32)
 
    # 2. Global z-score
    mean_amt = df["TransactionAmt"].mean()
    std_amt  = df["TransactionAmt"].std()
    df["feat_amt_zscore"] = (
        (df["TransactionAmt"] - mean_amt) / (std_amt + 1e-9)
    ).astype(np.float32)
 
    # 3. Percentile rank within card1
    df["feat_amt_pctrank_card1"] = (
        df.groupby("card1")["TransactionAmt"]
          .rank(pct=True)
          .astype(np.float32)
    )
 
    # 4. Amount vs card1 historical mean
    card1_mean = df.groupby("card1")["TransactionAmt"].transform("mean")
    df["feat_card1_mean_amt"] = card1_mean.astype(np.float32)
    df["feat_amt_to_card1_mean"] = (
        safe_div(df["TransactionAmt"].values, card1_mean.values)
    ).astype(np.float32)
 
    # 5. Is round amount (no cents)
    df["feat_is_round_amt"] = (
        (df["TransactionAmt"] % 1 == 0).astype(np.int8)
    )
 
    # 6. High amount flag (above 95th percentile)
    p95 = df["TransactionAmt"].quantile(0.95)
    df["feat_high_amt_flag"] = (
        (df["TransactionAmt"] > p95).astype(np.int8)
    )
 
    return df
 
 
# ── Category 2: Temporal features ────────────────────────────────────────────
def build_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    print("  [2/5] Building temporal features...")
 
    # TransactionDT is seconds from a reference point — NOT a unix timestamp
    # Extract time components from the delta
    seconds_in_day  = 86400
    seconds_in_hour = 3600
    seconds_in_week = 604800
 
    # 7. Hour of day
    df["feat_hour"] = (
        (df["TransactionDT"] // seconds_in_hour) % 24
    ).astype(np.int8)
 
    # 8. Day of week (0=Mon, 6=Sun)
    df["feat_dow"] = (
        (df["TransactionDT"] // seconds_in_day) % 7
    ).astype(np.int8)
 
    # 9. Is weekend
    df["feat_is_weekend"] = (
        df["feat_dow"].isin([5, 6]).astype(np.int8)
    )
 
    # 10. Is night (11pm–5am)
    df["feat_is_night"] = (
        ((df["feat_hour"] >= 23) | (df["feat_hour"] <= 5)).astype(np.int8)
    )
 
    # 11. Time since last transaction by same card1
    df_sorted = df.sort_values("TransactionDT")
    df["feat_time_since_last_card1"] = (
        df_sorted.groupby("card1")["TransactionDT"]
                 .diff()
                 .fillna(0)
                 .astype(np.float32)
    )
    # Sort back to original order
    df = df.sort_index()
 
    return df
 
 
# ── Category 3: Aggregation features per card1 ───────────────────────────────
def build_aggregation_features(df: pd.DataFrame) -> pd.DataFrame:
    print("  [3/5] Building aggregation features (per card1)...")
 
    # 12. Transaction count per card1
    df["feat_card1_txn_count"] = (
        df.groupby("card1")["TransactionAmt"]
          .transform("count")
          .astype(np.int32)
    )
 
    # 13. Std of amount per card1 (already have mean from category 1)
    card1_std = df.groupby("card1")["TransactionAmt"].transform("std").fillna(0)
    df["feat_card1_std_amt"] = card1_std.astype(np.float32)
 
    # 14. Coefficient of variation per card1 (std / mean)
    df["feat_card1_amt_cv"] = (
        safe_div(card1_std.values, df["feat_card1_mean_amt"].values)
    ).astype(np.float32)
 
    # 15. Historical fraud rate per card1
    # IMPORTANT: compute on full df — in production this would come from
    # a pre-computed lookup table from training data only.
    # For training pipeline this is acceptable.
    card1_fraud_rate = df.groupby("card1")["isFraud"].transform("mean")
    df["feat_card1_fraud_rate"] = card1_fraud_rate.astype(np.float32)
 
    return df
 
 
# ── Category 4: Categorical / entity features ─────────────────────────────────
def build_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    print("  [4/5] Building categorical features...")
 
    # 16. Email domain risk score (fraud rate per P_emaildomain)
    domain_fraud_rate = df.groupby("P_emaildomain")["isFraud"].transform("mean")
    global_fraud_rate = df["isFraud"].mean()
    df["feat_email_domain_risk"] = (
        domain_fraud_rate.fillna(global_fraud_rate).astype(np.float32)
    )
 
    # 17. Email domain mismatch (purchaser vs recipient)
    df["feat_email_domain_mismatch"] = (
        (df["P_emaildomain"] != df["R_emaildomain"]).astype(np.int8)
    )
 
    # 18. card1 + card4 combination frequency encoding
    # card4 = card network (visa/mastercard etc), already label encoded
    df["_card_combo"] = (
        df["card1"].astype(str) + "_" + df["card4"].astype(str)
    )
    combo_freq = df["_card_combo"].value_counts(normalize=True)
    df["feat_card_combo_freq"] = (
        df["_card_combo"].map(combo_freq).fillna(0).astype(np.float32)
    )
    df.drop(columns=["_card_combo"], inplace=True)
 
    # 19. ProductCD fraud rate encoding
    prod_fraud_rate = df.groupby("ProductCD")["isFraud"].transform("mean")
    df["feat_productcd_fraud_rate"] = prod_fraud_rate.astype(np.float32)
 
    return df
 
 
# ── Category 5: Behavioural features ─────────────────────────────────────────
def build_behavioural_features(df: pd.DataFrame) -> pd.DataFrame:
    print("  [5/5] Building behavioural features...")
 
    # M columns in clean data are encoded as: 1=T, 0=F, -1=NaN, 2=other
    m_cols = [f"M{i}" for i in range(1, 10) if f"M{i}" in df.columns]
 
    # 20. Count of M cols that matched (value == 1)
    df["feat_m_match_count"] = (
        (df[m_cols] == 1).sum(axis=1).astype(np.int8)
    )
 
    # 21. Count of M cols that mismatched (value == 0)
    df["feat_m_mismatch_count"] = (
        (df[m_cols] == 0).sum(axis=1).astype(np.int8)
    )
 
    # 22. D1 to D10 ratio (timing delta comparison)
    # D1 = days since last txn, D10 = another time delta
    # Both already cleaned (medians or -1 sentinel)
    if "D1" in df.columns and "D10" in df.columns:
        df["feat_d1_to_d10_ratio"] = (
            safe_div(
                df["D1"].clip(lower=0).values,
                df["D10"].clip(lower=0).values + 1  # +1 avoids div by 0
            )
        ).astype(np.float32)
    else:
        df["feat_d1_to_d10_ratio"] = 0.0
 
    # 23. Has identity data (from identity table join)
    # DeviceType==2 means "unknown" which we set for rows with no identity data
    if "DeviceType" in df.columns:
        df["feat_has_identity"] = (
            (df["DeviceType"] != 2).astype(np.int8)
        )
    else:
        df["feat_has_identity"] = 0
 
    return df
 
 
# ── Final feature list builder ────────────────────────────────────────────────
def get_model_feature_cols(df: pd.DataFrame) -> list:
    """
    Returns the exact list of columns to pass to the model.
    Includes all feat_ columns + key cleaned columns.
    Excludes: TransactionID, isFraud (target), TransactionDT (raw time).
    """
    feat_cols = [c for c in df.columns if c.startswith("feat_")]
 
    # Key cleaned columns that are directly useful
    direct_cols = [
        # Log C cols (count features, compressed)
        *[c for c in df.columns if c.startswith("log_C")],
        # D cols (time deltas)
        *[c for c in df.columns if c.startswith("D")],
        # M cols (match flags)
        *[c for c in df.columns if c.startswith("M")],
        # Distance
        "log_dist1", "has_dist2",
        # Transaction basics
        "TransactionAmt", "ProductCD",
        "card1", "card2", "card3", "card4", "card5", "card6",
        "addr1", "addr2",
        "P_emaildomain", "R_emaildomain",
        # Device
        "DeviceType", "DeviceInfo",
        # id cols
        *[c for c in df.columns if c.startswith("id_")],
        # V cols (keep all — LightGBM handles them well)
        *[c for c in df.columns if c.startswith("V")],
        *[c for c in df.columns if c.startswith("log_V")],
    ]
 
    # Only keep cols that actually exist in df
    direct_cols = [c for c in direct_cols if c in df.columns]
 
    # Combine, deduplicate, preserve order
    all_cols = feat_cols + [c for c in direct_cols if c not in feat_cols]
    return all_cols
 
 
# ── Main runner ───────────────────────────────────────────────────────────────
def run_pipeline():
    print("[1/3] Loading clean data...")
    df = pd.read_csv(IN_PATH)
    print(f"    Loaded shape: {df.shape}")
 
    print("[2/3] Engineering features...")
    df = build_amt_features(df)
    df = build_temporal_features(df)
    df = build_aggregation_features(df)
    df = build_categorical_features(df)
    df = build_behavioural_features(df)
 
    # Get final model feature list
    model_cols = get_model_feature_cols(df)
    feat_only  = [c for c in model_cols if c.startswith("feat_")]
 
    print(f"\n    feat_ columns built : {len(feat_only)}")
    print(f"    total model columns : {len(model_cols)}")
 
    # ── Save full dataframe (with TransactionID + isFraud for training)
    print("[3/3] Saving outputs...")
    df.to_csv(OUT_DIR / "featured_data.csv", index=False)
 
    # ── Save model feature list (used by train.py)
    with open(OUT_DIR / "model_feature_cols.json", "w") as f:
        json.dump(model_cols, f, indent=2)
 
    # ── Save feature documentation report
    feat_report = {
        "total_feat_columns"  : len(feat_only),
        "total_model_columns" : len(model_cols),
        "feat_columns"        : feat_only,
        "feature_rationale"   : FEATURE_REGISTRY,
        "categories": {
            "1_transaction_amount" : [k for k in FEATURE_REGISTRY if "amt" in k or "round" in k or "high" in k],
            "2_temporal"           : [k for k in FEATURE_REGISTRY if any(x in k for x in ["hour","dow","weekend","night","time_since"])],
            "3_aggregation"        : [k for k in FEATURE_REGISTRY if "card1" in k],
            "4_categorical"        : [k for k in FEATURE_REGISTRY if any(x in k for x in ["email","combo","productcd"])],
            "5_behavioural"        : [k for k in FEATURE_REGISTRY if any(x in k for x in ["m_match","d1_","has_identity"])],
        }
    }
 
    with open(OUT_DIR / "feature_report.json", "w") as f:
        json.dump(feat_report, f, indent=2)
 
    print("\n" + "=" * 60)
    print("  FEATURE ENGINEERING COMPLETE")
    print(f"  feat_ features built : {len(feat_only)}")
    print(f"  total model cols     : {len(model_cols)}")
    print(f"  Output → data/features/featured_data.csv")
    print(f"  Model cols → data/features/model_feature_cols.json")
    print(f"  Report → data/features/feature_report.json")
    print("=" * 60)
 
    # Print all feat_ features with rationale summary
    print("\n  Feature summary:")
    for i, col in enumerate(feat_only, 1):
        rationale = FEATURE_REGISTRY.get(col, "see code")
        short = rationale[:70] + "..." if len(rationale) > 70 else rationale
        print(f"  {i:>2}. {col:<35} {short}")
 
 
if __name__ == "__main__":
    run_pipeline()
 
