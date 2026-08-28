from __future__ import annotations
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.demo_data import demo_raw, demo_kpis


def main():
    output_dir = ROOT / "data/outputs/demo_inputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "synthetic_billing_disputes.csv"
    kpi_path = output_dir / "synthetic_network_kpis.csv"
    demo_raw().to_csv(raw_path, index=False)
    demo_kpis().to_csv(kpi_path, index=False)
    print(f"Wrote {raw_path.relative_to(ROOT)}")
    print(f"Wrote {kpi_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
