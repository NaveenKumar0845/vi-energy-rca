from __future__ import annotations
import pandas as pd


def monitoring_summary(df: pd.DataFrame) -> dict:
    rows = max(1, len(df))
    auto = int(df.get("Prediction Status", pd.Series(dtype=str)).eq("Auto-predicted").sum()) if "Prediction Status" in df else 0
    manual = int(df.get("Predicted Dispute Type", pd.Series(dtype=str)).eq("Manual Review Required").sum()) if "Predicted Dispute Type" in df else 0
    anomalies = int(df.get("Is Anomaly", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if "Is Anomaly" in df else 0
    confidence = pd.to_numeric(df.get("Prediction Confidence", pd.Series(dtype=float)), errors="coerce")
    return {
        "rows": int(len(df)),
        "sites": int(df["IP Site ID"].nunique()) if "IP Site ID" in df else 0,
        "prediction_coverage": auto / rows,
        "manual_review_rate": manual / rows,
        "anomaly_rate": anomalies / rows,
        "mean_prediction_confidence": float(confidence.mean()) if confidence.notna().any() else None,
        "p10_prediction_confidence": float(confidence.quantile(0.10)) if confidence.notna().any() else None,
        "p90_prediction_confidence": float(confidence.quantile(0.90)) if confidence.notna().any() else None,
    }


def segment_health(df: pd.DataFrame) -> pd.DataFrame:
    if "Expense Nature" not in df:
        return pd.DataFrame()
    rows = []
    for head, part in df.groupby(df["Expense Nature"].astype(str).str.upper(), dropna=False):
        summary = monitoring_summary(part)
        summary["Expense Nature"] = head
        rows.append(summary)
    return pd.DataFrame(rows)


def critical_missingness(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "IP Site ID",
        "Month-Year",
        "Expense Nature",
        "Prorated Billed Amount (Excl GST)",
        "Prorated Debit Amount (Excl tax)",
        "Site Historical Average",
        "Prediction Confidence",
    ]
    rows = []
    for column in columns:
        if column not in df:
            rows.append({"Field": column, "Missing %": 100.0, "Status": "Missing column"})
            continue
        missing = float(df[column].isna().mean() * 100)
        status = "OK" if missing < 5 else "Review"
        if column in {"Site Historical Average", "Prediction Confidence"}:
            # Some nulls are expected for first observations or manual-review paths.
            status = "Expected/Review" if missing >= 5 else "OK"
        rows.append({"Field": column, "Missing %": round(missing, 2), "Status": status})
    return pd.DataFrame(rows)
