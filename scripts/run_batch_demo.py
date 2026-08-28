from __future__ import annotations
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_pipeline import prepare
from src.demo_data import demo_raw, demo_kpis
from src.io_utils import excel_bytes, read_path
from src.models import DisputeClassifier, add_anomalies, prepare_taxonomy


def main():
    taxonomy = prepare_taxonomy(read_path(ROOT / "data/reference/dispute_reasons.csv"))
    prepared = prepare(demo_raw(), "raw", demo_kpis())
    model = DisputeClassifier(0.60).fit(prepared.analytical, taxonomy)
    analyzed = add_anomalies(model.predict(prepared.analytical), 0.10)

    output_dir = ROOT / "data/outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "demo_analysis.xlsx"
    path.write_bytes(excel_bytes({"Analysis": analyzed, "Rejected Rows": prepared.rejected}))
    print(f"Wrote {len(analyzed)} analyzed rows to {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
