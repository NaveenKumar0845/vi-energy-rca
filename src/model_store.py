from __future__ import annotations
from pathlib import Path
import joblib


def save_model(model, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, target)
    return target


def load_model(path: str | Path):
    return joblib.load(Path(path))
