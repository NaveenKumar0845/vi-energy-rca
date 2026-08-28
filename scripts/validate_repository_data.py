from __future__ import annotations
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_pipeline import is_model_ready, prorate_raw
from src.io_utils import read_path
from src.models import prepare_taxonomy

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

    taxonomy = read_path(TAXONOMY)
    print_schema("taxonomy", TAXONOMY, taxonomy)
    taxonomy = prepare_taxonomy(taxonomy)
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
    print("Repository data validation completed successfully.")


if __name__ == "__main__":
    main()
