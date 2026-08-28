from __future__ import annotations
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_pipeline import prepare, prorate_raw
from src.demo_data import demo_raw, demo_kpis
from src.io_utils import read_path
from src.models import DisputeClassifier, add_anomalies, evaluate_time_split, prepare_taxonomy

RAW_DIR = ROOT / "data/raw"
PROCESSED = ROOT / "data/processed/filtered_data.xlsx"
TAX_XLSX = ROOT / "data/reference/dispute_reasons.xlsx"
TAX_CSV = ROOT / "data/reference/dispute_reasons.csv"


def find_private_raw():
    for pattern in ("*.xlsb", "*.xlsx", "*.xls", "*.csv"):
        files = sorted(RAW_DIR.glob(pattern))
        if files:
            return files[0]
    return None


def load_taxonomy():
    path = TAX_XLSX if TAX_XLSX.exists() else TAX_CSV
    return prepare_taxonomy(read_path(path)), path


def validate_dataset(label, raw, taxonomy, network=None):
    sample = raw.head(min(len(raw), 250)).copy()
    prorated, rejected = prorate_raw(sample)
    print(f"[{label}] raw_rows={len(raw)} smoke_prorated_rows={len(prorated)} rejected={len(rejected)}")

    prepared = prepare(raw, "raw", network)
    analytical = prepared.analytical
    model = DisputeClassifier(0.60).fit(analytical, taxonomy)
    analyzed = add_anomalies(model.predict(analytical), 0.10)
    metrics, _ = evaluate_time_split(analytical, taxonomy, 0.60)

    print(f"[{label}] analytical_rows={len(analytical)} sites={analytical['IP Site ID'].nunique()}")
    print(f"[{label}] trained_heads={sorted(model.models.keys())}")
    print(f"[{label}] label_stats={json.dumps(model.label_stats, default=str, sort_keys=True)}")
    print(f"[{label}] anomalies={int(analyzed['Is Anomaly'].sum())}")
    print(f"[{label}] evaluation={json.dumps(metrics, default=str, sort_keys=True)}")

    assert set(model.models) == {"EB", "DG"}, f"{label}: both EB and DG demo classifiers must train"
    assert len(analyzed) == len(analytical)
    assert "Predicted Dispute Type" in analyzed
    assert "Is Anomaly" in analyzed


def main():
    taxonomy, taxonomy_path = load_taxonomy()
    print(f"[taxonomy] file={taxonomy_path.relative_to(ROOT)} rows={len(taxonomy)}")

    # Public CI always validates a fully synthetic, interview-safe end-to-end path.
    validate_dataset("synthetic_demo", demo_raw(), taxonomy, demo_kpis())

    # Optional: when a developer has private/untracked files locally, validate them too.
    private_raw = find_private_raw()
    if private_raw is not None:
        print(f"[private_data] optional raw file detected: {private_raw.name}")
        private = read_path(private_raw)
        prepared = prepare(private, "raw")
        print(f"[private_data] analytical_rows={len(prepared.analytical)} rejected={len(prepared.rejected)}")
    else:
        print("[private_data] not present; expected for the public repository")

    if PROCESSED.exists():
        processed = read_path(PROCESSED)
        print(f"[private_processed] optional rows={len(processed)}")
    else:
        print("[private_processed] not present; expected for the public repository")

    print("Public synthetic architecture validation completed successfully.")


if __name__ == "__main__":
    main()
