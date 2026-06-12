import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
# Make paths absolute so it works from anywhere
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUT_DIR = BASE_DIR / "data" / "clean"
OUT_DIR.mkdir(parents=True, exist_ok=True)

 
# ── Column group definitions ──────────────────────────────────────────────────
C_COLS      = [f"C{i}"  for i in range(1, 15)]      # C1–C14  : count features
D_COLS      = [f"D{i}"  for i in range(1, 16)]      # D1–D15  : time-delta features
M_COLS      = [f"M{i}"  for i in range(1, 10)]      # M1–M9   : match flags T/F
V_COLS      = [f"V{i}"  for i in range(1, 340)]     # V1–V339 : Vesta features
DIST_COLS   = ["dist1", "dist2"]                     # distance features (transaction table)
CARD_NUM    = ["card1", "card2", "card3", "card5"]   # numeric card fields
CARD_CAT    = ["card4", "card6"]                     # categorical card fields
ADDR_COLS   = ["addr1", "addr2"]
EMAIL_COLS  = ["P_emaildomain", "R_emaildomain"]
CAT_TX      = ["ProductCD"] + CARD_CAT + EMAIL_COLS
ID_NUM_COLS = [f"id_{str(i).zfill(2)}" for i in range(1, 12)]   # id_01–id_11
ID_CAT_COLS = [f"id_{str(i).zfill(2)}" for i in range(12, 39)]  # id_12–id_38
 
 
# ── Step 1: Load & Merge ──────────────────────────────────────────────────────
def load_and_merge(report: dict) -> pd.DataFrame:
    print("[1/9] Loading transaction data...")
    tx  = pd.read_csv(RAW_DIR / "train_transaction.csv")
 
    print("[2/9] Loading identity data...")
    id_ = pd.read_csv(RAW_DIR / "train_identity.csv")
 
    print("[3/9] Merging on TransactionID (left join)...")
    df  = tx.merge(id_, on="TransactionID", how="left")
 
    report["original_shape"]          = list(df.shape)
    report["transactions_with_id"]    = int(id_["TransactionID"].isin(tx["TransactionID"]).sum())
    report["transactions_without_id"] = int(df["DeviceType"].isna().sum())
    print(f"    Merged shape : {df.shape}")
    return df
 
 
# ── Step 2: C columns ─────────────────────────────────────────────────────────
def clean_c_cols(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    """
    C cols are count features (cards linked, addresses used, etc).
    Missing = no occurrence → fill 0.
    Outliers: DO NOT clip — high count = fraud signal (e.g. 500 cards used).
    Apply log1p to compress scale: 4685→8.45, 164→5.10, 1→0.69, 0→0.
    THEN drop raw C cols — log versions carry same info on better scale.
    No redundancy in final dataset.
    """
    print("[4/9] Cleaning C columns (fill 0 → log1p → drop raw)...")
    present   = [c for c in C_COLS if c in df.columns]
    col_stats = {}
 
    for col in present:
        n_missing = int(df[col].isna().sum())
        raw_max   = float(df[col].max())
        raw_p99   = float(df[col].quantile(0.99))
 
        # 1. Fill missing with 0
        df[col] = df[col].fillna(0)
 
        # 2. Create log version
        log_col     = f"log_{col}"
        df[log_col] = np.log1p(df[col]).astype(np.float32)
 
        col_stats[col] = {
            "missing_filled_zero": n_missing,
            "raw_max"            : round(raw_max, 2),
            "raw_p99"            : round(raw_p99, 2),
            "outlier_strategy"   : "log1p — preserves fraud signal in high counts",
            "log_col_created"    : log_col
        }
 
    # 3. Drop all raw C cols — log versions are strictly better
    df.drop(columns=present, inplace=True)
    print(f"    Dropped {len(present)} raw C cols | kept log_ versions only")
 
    report["C_columns"] = {
        "count"           : len(present),
        "strategy"        : "fill 0 → log1p → drop raw (log versions kept)",
        "raw_cols_dropped": present,
        "details"         : col_stats
    }
    return df
 
 
# ── Step 3: D columns ─────────────────────────────────────────────────────────
def clean_d_cols(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    """
    D cols are time deltas (days since last txn, account age, etc).
    >50% missing → fill -1 sentinel (new entity never seen = higher fraud risk).
    <50% missing → fill median.
    Raw D cols kept as-is (no log transform needed — values are day counts,
    not counts with extreme outliers like C cols).
    """
    print("[5/9] Cleaning D columns (median or -1 sentinel)...")
    present   = [c for c in D_COLS if c in df.columns]
    col_stats = {}
 
    for col in present:
        n_missing   = int(df[col].isna().sum())
        pct_missing = round(n_missing / len(df) * 100, 1)
        median_val  = df[col].median()
 
        if pct_missing > 50:
            fill_val = -1
            strategy = f"fill -1 sentinel ({pct_missing}% missing — new entity signal)"
        else:
            fill_val = float(median_val) if not np.isnan(median_val) else 0.0
            strategy = f"fill median = {round(fill_val, 2)}"
 
        df[col]        = df[col].fillna(fill_val)
        col_stats[col] = {
            "missing"    : n_missing,
            "pct_missing": pct_missing,
            "strategy"   : strategy
        }
 
    report["D_columns"] = {
        "count"   : len(present),
        "strategy": "median if <50% missing; -1 sentinel if >50% missing",
        "details" : col_stats
    }
    return df
 
 
# ── Step 4: dist1, dist2 ──────────────────────────────────────────────────────
def clean_dist_cols(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    """
    Both from train_transaction.csv (NOT identity table).
 
    dist1 (~60% missing):
      - Fill median → log1p (values reach thousands) → drop raw.
 
    dist2 (~93% missing):
      - Too sparse to impute meaningfully (93% would be same fill value).
      - Drop raw dist2.
      - Create has_dist2 binary flag instead.
      - WHY: having identity distance data at all means the transaction
        was successfully linked to an identity record — that linkage
        pattern is itself a fraud signal.
    """
    print("[6/9] Cleaning dist1, dist2...")
    col_stats = {}
 
    # ── dist1
    if "dist1" in df.columns:
        n_missing   = int(df["dist1"].isna().sum())
        pct_missing = round(n_missing / len(df) * 100, 1)
        med         = df["dist1"].median()
        fill_val    = float(med) if not np.isnan(med) else 0.0
 
        df["dist1"]     = df["dist1"].fillna(fill_val)
        df["log_dist1"] = np.log1p(df["dist1"].clip(lower=0)).astype(np.float32)
        df.drop(columns=["dist1"], inplace=True)   # raw dropped, log kept
 
        col_stats["dist1"] = {
            "pct_missing"    : pct_missing,
            "strategy"       : f"fill median={round(fill_val,2)} → log1p → drop raw",
            "log_col_created": "log_dist1"
        }
 
    # ── dist2
    if "dist2" in df.columns:
        n_missing   = int(df["dist2"].isna().sum())
        pct_missing = round(n_missing / len(df) * 100, 1)
 
        # Binary flag: 1 = had a real dist2 value, 0 = missing
        df["has_dist2"] = df["dist2"].notna().astype(np.int8)
        df.drop(columns=["dist2"], inplace=True)   # raw dropped, flag kept
 
        col_stats["dist2"] = {
            "pct_missing"   : pct_missing,
            "strategy"      : "DROPPED raw — 93% missing too sparse to impute",
            "replacement"   : "has_dist2 binary flag (1=had value, 0=missing)",
            "why_flag_works": "linked identity record = fraud signal"
        }
 
    report["dist_columns"] = {
        "note"   : "dist1 and dist2 are from train_transaction.csv, not identity",
        "details": col_stats
    }
    return df
 
 
# ── Step 5: M columns ─────────────────────────────────────────────────────────
def clean_m_cols(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    """
    M cols are match flags: T / F / NaN / unexpected (e.g. "M2" in real data).
    Encoding: T=1, F=0, NaN=-1 (preserves missingness signal), other=2.
    NaN→-1 is intentional: missing match flag behaves differently from a
    confirmed mismatch (F=0) — keeping them separate helps the model.
    """
    print("[7/9] Cleaning M columns (T=1, F=0, NaN=-1, other=2)...")
    present   = [c for c in M_COLS if c in df.columns]
    col_stats = {}
 
    def encode_m(val):
        if pd.isna(val): return -1
        v = str(val).strip().upper()
        if v == "T":     return 1
        if v == "F":     return 0
        return 2
 
    for col in present:
        n_missing  = int(df[col].isna().sum())
        unexpected = int((~df[col].isin(["T", "F"]) & df[col].notna()).sum())
        df[col]    = df[col].apply(encode_m).astype(np.int8)
        col_stats[col] = {
            "missing"   : n_missing,
            "unexpected": unexpected,
            "strategy"  : "T=1, F=0, NaN=-1, other=2"
        }
 
    report["M_columns"] = {
        "count"   : len(present),
        "strategy": "T=1  F=0  NaN=-1(signal preserved)  other=2",
        "details" : col_stats
    }
    return df
 
 
# ── Step 6: V columns ─────────────────────────────────────────────────────────
def clean_v_cols(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    """
    V cols are Vesta proprietary risk features.
    Drop if >90% missing (pure noise).
    Fill median for remaining missing values.
    """
    print("[8a/9] Cleaning V columns (drop >90% missing, median fill rest)...")
    present = [c for c in V_COLS if c in df.columns]
    dropped = []
    imputed = []
 
    for col in present:
        n_missing   = int(df[col].isna().sum())
        pct_missing = n_missing / len(df)
        if pct_missing > 0.90:
            df.drop(columns=[col], inplace=True)
            dropped.append(col)
        elif n_missing > 0:
            med = df[col].median()
            df[col] = df[col].fillna(med if not np.isnan(med) else 0)
            imputed.append(col)
 
    report["V_columns"] = {
        "original_count"         : len(present),
        "dropped_gt90pct_missing": len(dropped),
        "imputed_with_median"    : len(imputed),
        "strategy"               : "drop if >90% missing; fill median otherwise"
    }
    print(f"    Kept {len(present)-len(dropped)} V cols | Dropped {len(dropped)}")
    return df
 
 
# ── Step 7: V column outlier treatment ───────────────────────────────────────
def treat_v_outliers(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    """
    V cols have 3 types:
      - Binary (max <= 1)     → skip, log1p would distort meaning
      - Low range (max <= 100) → skip, no meaningful outlier problem
      - High range with skew   → log1p transform
 
    Criteria for log1p:
      max > 100 AND (p99 / median) > 5  →  real outlier problem
    Raw V col is DROPPED after log version created (same logic as C cols).
    """
    print("[8b/9] Treating V column outliers (selective log1p)...")
    present  = [c for c in df.columns if c.startswith("V")]
    logged   = []
    skipped  = []
 
    for col in present:
        col_max    = float(df[col].max())
        col_median = float(df[col].median())
        col_p99    = float(df[col].quantile(0.99))
 
        # Binary cols — skip entirely
        if col_max <= 1:
            skipped.append(col)
            continue
 
        # Check outlier severity
        ratio = (col_p99 / col_median) if col_median > 0 else 0
 
        if col_max > 100 and ratio > 5:
            log_col     = f"log_{col}"
            df[log_col] = np.log1p(df[col].clip(lower=0)).astype(np.float32)
            df.drop(columns=[col], inplace=True)   # drop raw, keep log
            logged.append(col)
        else:
            skipped.append(col)
 
    report["V_outlier_treatment"] = {
        "log1p_applied_and_raw_dropped": len(logged),
        "skipped_binary_or_low_range"  : len(skipped),
        "strategy" : "log1p + drop raw only where max>100 AND p99/median>5",
        "logged_cols": logged[:20]
    }
    print(f"    log1p applied to {len(logged)} V cols | {len(skipped)} untouched")
    return df
 
 
# ── Step 8: Categorical columns ───────────────────────────────────────────────
def clean_categorical(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    print("[8c/9] Cleaning categorical columns...")
    cat_report = {}
 
    # ProductCD, card4, card6, email domains → fill unknown → label encode
    for col in CAT_TX:
        if col not in df.columns:
            continue
        n_missing = int(df[col].isna().sum())
        df[col]   = df[col].fillna("unknown")
        codes, _  = pd.factorize(df[col])
        df[col]   = codes
        cat_report[col] = {
            "missing" : n_missing,
            "strategy": "fill unknown → label encode"
        }
 
    # addr1, addr2 — numeric postal codes → fill median
    for col in ADDR_COLS:
        if col not in df.columns:
            continue
        n_missing = int(df[col].isna().sum())
        med       = df[col].median()
        df[col]   = df[col].fillna(med if not np.isnan(med) else 0)
        cat_report[col] = {
            "missing" : n_missing,
            "strategy": f"fill median={round(float(med),2)}"
        }
 
    # card1,2,3,5 — numeric → fill median
    for col in CARD_NUM:
        if col not in df.columns:
            continue
        n_missing = int(df[col].isna().sum())
        med       = df[col].median()
        df[col]   = df[col].fillna(med if not np.isnan(med) else 0)
        cat_report[col] = {
            "missing" : n_missing,
            "strategy": f"fill median={round(float(med),2)}"
        }
 
    # DeviceType: mobile=0, desktop=1, unknown=2
    if "DeviceType" in df.columns:
        n_missing        = int(df["DeviceType"].isna().sum())
        df["DeviceType"] = df["DeviceType"].fillna("unknown")
        device_map       = {"mobile": 0, "desktop": 1, "unknown": 2}
        df["DeviceType"] = (df["DeviceType"].str.lower()
                             .map(device_map).fillna(2).astype(np.int8))
        cat_report["DeviceType"] = {
            "missing" : n_missing,
            "strategy": "mobile=0, desktop=1, unknown/NaN=2"
        }
 
    # DeviceInfo: extract OS family → label encode
    if "DeviceInfo" in df.columns:
        n_missing = int(df["DeviceInfo"].isna().sum())
        def extract_os(val):
            if pd.isna(val): return "unknown"
            v = str(val).lower()
            if "windows" in v:                              return "windows"
            if "ios" in v or "iphone" in v or "ipad" in v: return "ios"
            if "android" in v:                              return "android"
            if "mac" in v:                                  return "mac"
            if "linux" in v:                                return "linux"
            return "other"
        df["DeviceInfo"]         = df["DeviceInfo"].apply(extract_os)
        codes, _                 = pd.factorize(df["DeviceInfo"])
        df["DeviceInfo"]         = codes
        cat_report["DeviceInfo"] = {
            "missing" : n_missing,
            "strategy": "extract OS family → label encode"
        }
 
    # id_12–id_38: high-cardinality categoricals
    for col in ID_CAT_COLS:
        if col not in df.columns:
            continue
        n_missing   = int(df[col].isna().sum())
        pct_missing = n_missing / len(df)
        if pct_missing > 0.90:
            df.drop(columns=[col], inplace=True)
            cat_report[col] = {"missing": n_missing, "strategy": "DROPPED >90% missing"}
            continue
        df[col]  = df[col].fillna("unknown")
        codes, _ = pd.factorize(df[col])
        df[col]  = codes
        cat_report[col] = {
            "missing" : n_missing,
            "strategy": "fill unknown → label encode"
        }
 
    # id_01–id_11: numeric identity cols
    for col in ID_NUM_COLS:
        if col not in df.columns:
            continue
        n_missing   = int(df[col].isna().sum())
        pct_missing = n_missing / len(df)
        if pct_missing > 0.90:
            df.drop(columns=[col], inplace=True)
            cat_report[col] = {"missing": n_missing, "strategy": "DROPPED >90% missing"}
            continue
        med     = df[col].median()
        df[col] = df[col].fillna(med if not np.isnan(med) else 0)
        cat_report[col] = {
            "missing" : n_missing,
            "strategy": f"fill median={round(float(med),2)}"
        }
 
    report["categorical_columns"] = cat_report
    return df
 
 
# ── Step 9: TransactionAmt outlier clip ───────────────────────────────────────
def treat_outliers(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    """
    TransactionAmt: clip at 99.9th percentile.
    WHY clip here vs log1p for C cols:
      C cols → high count IS the fraud signal, preserve it via log1p.
      TransactionAmt → extreme values (>$10k) are more likely data entry
      errors than genuine signals. Clipping is safer here.
    """
    cap   = df["TransactionAmt"].quantile(0.999)
    n_cap = int((df["TransactionAmt"] > cap).sum())
    df["TransactionAmt"] = df["TransactionAmt"].clip(upper=cap)
    report["outliers"] = {
        "TransactionAmt": {
            "strategy"   : "clip at 99.9th percentile",
            "cap_value"  : round(float(cap), 2),
            "rows_capped": n_cap
        },
        "C_columns_note": "C cols NOT clipped — log1p used + raw dropped"
    }
    print(f"    TransactionAmt clipped at {cap:.2f} ({n_cap} rows affected)")
    return df
 
 
# ── Step 10: Memory optimisation ──────────────────────────────────────────────
def optimise_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes("float64").columns:
        df[col] = df[col].astype(np.float32)
    for col in df.select_dtypes("int64").columns:
        if df[col].max() < 32767:
            df[col] = df[col].astype(np.int16)
    return df
 
 
# ── Main runner ───────────────────────────────────────────────────────────────
def run_pipeline():
    report = {}
 
    df = load_and_merge(report)
    df = clean_c_cols(df, report)        # fill 0 → log1p → drop raw
    df = clean_d_cols(df, report)        # median or -1 sentinel
    df = clean_dist_cols(df, report)     # dist1: median+log1p+drop | dist2: binary flag
    df = clean_m_cols(df, report)        # T=1, F=0, NaN=-1, other=2
    df = clean_v_cols(df, report)        # drop >90% missing, median fill rest
    df = treat_v_outliers(df, report)    # selective log1p on high-skew V cols
    df = clean_categorical(df, report)   # label encode, fill medians
    df = treat_outliers(df, report)      # clip TransactionAmt
    df = optimise_dtypes(df)             # float32, int16 to save memory
 
    # ── Save
    df.to_csv(OUT_DIR / "clean_data.csv", index=False)
 
    report["final_shape"]    = list(df.shape)
    report["null_remaining"] = int(df.isnull().sum().sum())
 
    # Column inventory — useful for feature engineering
    report["final_columns_by_type"] = {
        "log_C_cols"  : [c for c in df.columns if c.startswith("log_C")],
        "D_cols"      : [c for c in df.columns if c.startswith("D")],
        "M_cols"      : [c for c in df.columns if c.startswith("M")],
        "V_cols_raw"  : [c for c in df.columns if c.startswith("V")],
        "log_V_cols"  : [c for c in df.columns if c.startswith("log_V")],
        "dist_cols"   : [c for c in df.columns if "dist" in c],
        "id_cols"     : [c for c in df.columns if c.startswith("id_")],
        "other"       : [c for c in df.columns
                         if not any(c.startswith(p) for p in
                                    ["log_", "D", "M", "V", "id_"])]
    }
 
    with open(OUT_DIR / "cleaning_report.json", "w") as f:
        json.dump(report, f, indent=2)
 
    print("\n" + "=" * 60)
    print("  CLEANING COMPLETE")
    print(f"  Original shape   : {report['original_shape']}")
    print(f"  Final shape      : {report['final_shape']}")
    print(f"  Nulls remaining  : {report['null_remaining']}")
    print(f"  Output  →  data/clean/clean_data.csv")
    print(f"  Report  →  data/clean/cleaning_report.json")
    print("=" * 60)
 
 
if __name__ == "__main__":
    run_pipeline()
