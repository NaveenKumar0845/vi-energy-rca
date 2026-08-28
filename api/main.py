from __future__ import annotations
import json
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException

from src.data_pipeline import prepare
from src.demo_data import demo_raw, demo_kpis
from src.io_utils import read_path
from src.models import DisputeClassifier, add_anomalies, prepare_taxonomy

ROOT = Path(__file__).resolve().parents[1]
TAX_CSV = ROOT / "data/reference/dispute_reasons.csv"

app = FastAPI(
    title="Telecom Energy & Billing Dispute RCA API",
    version="1.0.0",
    description=(
        "Interview-safe API surface over the synthetic pilot reconstruction. "
        "No proprietary Vi data is required."
    ),
)


@lru_cache(maxsize=1)
def _demo_analysis():
    taxonomy = prepare_taxonomy(read_path(TAX_CSV))
    prepared = prepare(demo_raw(), "raw", demo_kpis())
    model = DisputeClassifier(0.60).fit(prepared.analytical, taxonomy)
    analyzed = add_anomalies(model.predict(prepared.analytical), 0.10)
    return analyzed, model


@app.get("/health")
def health():
    return {"status": "ok", "data_mode": "synthetic_demo"}


@app.get("/demo/summary")
def demo_summary():
    analyzed, model = _demo_analysis()
    return {
        "rows": int(len(analyzed)),
        "sites": int(analyzed["IP Site ID"].nunique()),
        "auto_predictions": int(analyzed["Prediction Status"].eq("Auto-predicted").sum()),
        "manual_review": int(analyzed["Predicted Dispute Type"].eq("Manual Review Required").sum()),
        "anomalies": int(analyzed["Is Anomaly"].sum()),
        "trained_heads": sorted(model.models.keys()),
        "label_stats": model.label_stats,
    }


@app.get("/demo/sites/{site_id}")
def demo_site(site_id: str):
    analyzed, _ = _demo_analysis()
    rows = analyzed[analyzed["IP Site ID"].astype(str).eq(site_id)]
    if rows.empty:
        raise HTTPException(status_code=404, detail="Synthetic demo site not found")
    cols = [
        c
        for c in [
            "IP Site ID",
            "Month-Year",
            "Expense Nature",
            "Prorated Billed Amount (Excl GST)",
            "Prorated Debit Amount (Excl tax)",
            "Predicted Dispute Type",
            "Prediction Confidence",
            "Prediction Status",
            "Is Anomaly",
            "Anomaly Percentile",
            "Anomaly Evidence",
        ]
        if c in rows.columns
    ]
    return json.loads(rows[cols].to_json(orient="records", date_format="iso"))
