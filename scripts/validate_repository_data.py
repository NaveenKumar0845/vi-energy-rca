from __future__ import annotations
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_pipeline import is_model_ready, prepare, prorate_raw
from src.io_utils import read_path
from src.models import (
    DisputeClassifier,
    add_anomalies,
    evaluate_time_split,
    prepare_taxonomy,
    valid_reasons,
)

RAW_DIR = ROOT / "data/raw"
PROCESSED = ROOT / "data/processed/filtered_data.xlsx"
TAXONOMY = ROOT / "data/reference/dispute_reasons.xlsx"


def find_raw() -> Path:
    preferred = RAW_DIR / "Dispute_Head_UPE_Indus_Jan23toDec23.xlsb"
    if preferred.exists():
        return preferred
    for pattern in ("*.xlsb", "*.xlsx", "*.xls", "*.csv"):
        files = sorted(RAW_DIR.glob(pattern))
        if files:
            return files[0]
    raise FileNotFoundError("No raw workbook found under data/raw")


def print_schema(label, path, df):
    print(f"[{label}] file={path.relative_to(ROOT)}")
    print(f"[{label}] shape={df.shape}")
    print(f"[{label}] columns={list(df.columns)}")


def compact_counts(series, limit=30):
    counts = series.dropna().astype(str).str.strip().value_counts().head(limit)
    return [(str(label), int(count)) for label, count in counts.items()]


def candidate_label_diagnostics(analytical, taxonomy, column):
    if column not in analytical:
        print(f"[candidate_label:{column}] missing")
        return
    total_matched = 0
    for head in ("EB", "DG"):
        allowed = set(valid_reasons(taxonomy, head))
        rows = analytical[analytical["Expense Nature"].astype(str).str.upper().eq(head)]
        labelled = rows[rows[column].notna()]
        matched = labelled[labelled[column].astype(str).str.strip().isin(allowed)]
        unmatched = labelled[~labelled[column].astype(str).str.strip().isin(allowed)]
        total_matched += len(matched)
        print(
            f"[candidate_label:{column}:{head}] labelled={len(labelled)} raw_classes={labelled[column].astype(str).str.strip().nunique()} "
            f"taxonomy_matched={len(matched)} matched_classes={matched[column].astype(str).str.strip().nunique() if not matched.empty else 0} "
            f"unmatched_rows={len(unmatched)} unmatched_classes={unmatched[column].astype(str).str.strip().nunique() if not unmatched.empty else 0}"
        )
        print(f"[candidate_counts:{column}:{head}] {json.dumps(compact_counts(labelled[column]), ensure_ascii=True)}")
        print(f"[candidate_unmatched:{column}:{head}] {json.dumps(compact_counts(unmatched[column]), ensure_ascii=True)}")
    print(f"[candidate_label:{column}] taxonomy_matched_total={total_matched}")


def main():
    raw_path = find_raw()
    raw = read_path(raw_path)
    print_schema("raw", raw_path, raw)

    raw_required = {
        "IP Site ID",
        "Expense Nature",
        "Actual Bill From Date",
        "Actual Bill To Date",
        "Billed Amount (Excl GST)",
    }
    raw_missing = sorted(raw_required - set(raw.columns))
    print(f"[raw] missing_required={raw_missing}")
    print(f"[raw] has_dispute_type={'Dispute Type' in raw.columns}")
    if "Dispute Type" in raw.columns:
        print(f"[raw] labelled_rows={int(raw['Dispute Type'].notna().sum())}")
    if raw_missing:
        raise AssertionError(f"Raw workbook missing required canonical columns: {raw_missing}")

    sample = raw.head(min(len(raw), 250)).copy()
    prorated, rejected = prorate_raw(sample)
    print(f"[raw] smoke_proration_input_rows={len(sample)}")
    print(f"[raw] smoke_proration_output_rows={len(prorated)}")
    print(f"[raw] smoke_proration_rejected_rows={len(rejected)}")

    taxonomy_raw = read_path(TAXONOMY)
    print_schema("taxonomy", TAXONOMY, taxonomy_raw)
    taxonomy = prepare_taxonomy(taxonomy_raw)
    print(f"[taxonomy] heads={sorted(taxonomy['Dispute Head'].dropna().astype(str).unique().tolist())}")
    print(f"[taxonomy] reasons={len(taxonomy)}")

    processed = read_path(PROCESSED)
    print_schema("processed", PROCESSED, processed)
    print(f"[processed] model_ready={is_model_ready(processed)}")
    print(f"[processed] has_dispute_type={'Dispute Type' in processed.columns}")
    if not is_model_ready(processed):
        raise AssertionError("filtered_data.xlsx does not satisfy the model-ready schema")

    raw_sites = set(raw["IP Site ID"].dropna().astype(str))
    processed_sites = set(processed["IP Site ID"].dropna().astype(str))
    overlap = len(raw_sites & processed_sites)
    denominator = max(len(processed_sites), 1)
    print(f"[crosscheck] raw_sites={len(raw_sites)} processed_sites={len(processed_sites)} overlap_sites={overlap} overlap_pct={overlap/denominator:.1%}")

    # Full real-data pipeline validation. Output only aggregate metrics; never print row-level data.
    prepared = prepare(raw, "raw")
    analytical = prepared.analytical
    print(f"[full_pipeline] analytical_rows={len(analytical)}")
    print(f"[full_pipeline] rejected_rows={len(prepared.rejected)}")
    print(f"[full_pipeline] months={analytical['Month-Year'].nunique()}")

    eligible_total = 0
    for head in ("EB", "DG"):
        allowed = set(valid_reasons(taxonomy, head))
        head_rows = analytical[analytical["Expense Nature"].astype(str).str.upper().eq(head)]
        labelled = head_rows[head_rows["Dispute Type"].notna()] if "Dispute Type" in head_rows else head_rows.iloc[0:0]
        eligible = labelled[labelled["Dispute Type"].isin(allowed)]
        unmatched = labelled[~labelled["Dispute Type"].isin(allowed)]
        eligible_total += len(eligible)
        print(
            f"[label_coverage:{head}] rows={len(head_rows)} labelled={len(labelled)} "
            f"taxonomy_matched={len(eligible)} matched_classes={eligible['Dispute Type'].nunique() if not eligible.empty else 0} "
            f"raw_classes={labelled['Dispute Type'].nunique() if not labelled.empty else 0} unmatched_rows={len(unmatched)} "
            f"unmatched_classes={unmatched['Dispute Type'].nunique() if not unmatched.empty else 0}"
        )
        print(f"[label_counts:{head}] {json.dumps(compact_counts(labelled['Dispute Type']), ensure_ascii=True)}")
        print(f"[unmatched_counts:{head}] {json.dumps(compact_counts(unmatched['Dispute Type']), ensure_ascii=True)}")
        print(f"[taxonomy_values:{head}] {json.dumps(sorted(allowed), ensure_ascii=True)}")
    print(f"[label_coverage] taxonomy_matched_total={eligible_total} analytical_rows={len(analytical)}")

    # Check whether the legacy reason/category field is actually a better supervised target.
    for candidate in ("Dispute Type", "Reason for Dispute Categoary", "Reason for Dispute Category"):
        candidate_label_diagnostics(analytical, taxonomy, candidate)

    model = DisputeClassifier(0.60).fit(analytical, taxonomy)
    analyzed = add_anomalies(model.predict(analytical), 0.10)
    status_counts = analyzed["Prediction Status"].value_counts(dropna=False).to_dict()
    print(f"[model] trained_heads={sorted(model.models.keys())}")
    print(f"[model] prediction_status={status_counts}")
    print(f"[model] anomalies={int(analyzed['Is Anomaly'].sum())}")

    try:
        metrics, _ = evaluate_time_split(analytical, taxonomy, 0.60)
        print(f"[evaluation] {json.dumps(metrics, default=str, sort_keys=True)}")
    except Exception as exc:
        print(f"[evaluation] unavailable={type(exc).__name__}: {exc}")

    # Verify the small processed file can be scored using the labelled raw history.
    processed_analytical = prepare(processed, "model-ready").analytical
    processed_scored = add_anomalies(model.predict(processed_analytical), 0.10)
    processed_status = processed_scored["Prediction Status"].value_counts(dropna=False).to_dict()
    print(f"[processed_scoring] rows={len(processed_scored)} prediction_status={processed_status} anomalies={int(processed_scored['Is Anomaly'].sum())}")

    print("Repository data and end-to-end ML validation completed successfully.")


if __name__ == "__main__":
    main()
