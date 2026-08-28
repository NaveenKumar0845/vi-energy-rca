from __future__ import annotations
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_pipeline import prepare
from src.demo_data import demo_raw, demo_kpis
from src.io_utils import read_path
from src.model_store import save_model
from src.models import DisputeClassifier, prepare_taxonomy


def main():
    taxonomy = prepare_taxonomy(read_path(ROOT / "data/reference/dispute_reasons.csv"))
    analytical = prepare(demo_raw(), "raw", demo_kpis()).analytical
    model = DisputeClassifier(0.60).fit(analytical, taxonomy)
    target = save_model(model, ROOT / "models/demo_dispute_classifier.joblib")
    print(f"Saved model to {target.relative_to(ROOT)}")
    print(f"Trained heads: {sorted(model.models.keys())}")
    print(f"Label stats: {model.label_stats}")


if __name__ == "__main__":
    main()
