from __future__ import annotations
import io
from pathlib import Path
import pandas as pd
from .data_pipeline import clean_data

def read_path(path: Path):
    if not path.exists(): raise FileNotFoundError(str(path))
    return clean_data(pd.read_csv(path) if path.suffix.lower()==".csv" else pd.read_excel(path))

def read_upload(upload):
    return clean_data(pd.read_csv(upload) if upload.name.lower().endswith(".csv") else pd.read_excel(upload))

def excel_bytes(sheets: dict[str,pd.DataFrame]):
    b=io.BytesIO()
    with pd.ExcelWriter(b,engine="openpyxl") as w:
        for name,df in sheets.items():
            if df is not None: df.to_excel(w,sheet_name=name[:31],index=False)
    return b.getvalue()
