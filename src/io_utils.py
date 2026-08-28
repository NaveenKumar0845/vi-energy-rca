from __future__ import annotations
import io
import re
from pathlib import Path
from typing import BinaryIO, Union

import pandas as pd

from .data_pipeline import clean_data

PathLike = Union[str, Path]


def _norm(value) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _excel_engine(name: str):
    suffix = Path(name).suffix.lower()
    if suffix == ".xlsb":
        return "pyxlsb"
    if suffix == ".xls":
        return "xlrd"
    return None


def _header_score(row) -> int:
    cells = [_norm(v) for v in row if pd.notna(v)]
    checks = [
        "ip site id",
        "expense nature",
        "actual bill from date",
        "actual bill to date",
        "billed amount",
        "month year",
        "dispute type",
        "dispute head",
        "dispute sub category",
    ]
    return sum(any(key in cell for cell in cells) for key in checks)


def _best_sheet_and_header(excel: pd.ExcelFile) -> tuple[str, int]:
    best_sheet = excel.sheet_names[0]
    best_header = 0
    best_score = -1

    for sheet in excel.sheet_names:
        try:
            preview = pd.read_excel(excel, sheet_name=sheet, header=None, nrows=25)
        except Exception:
            continue
        for idx, row in preview.iterrows():
            score = _header_score(row.tolist())
            if score > best_score:
                best_score = score
                best_sheet = sheet
                best_header = int(idx)

    return best_sheet, best_header


def _read_excel(source, name: str) -> pd.DataFrame:
    engine = _excel_engine(name)
    excel = pd.ExcelFile(source, engine=engine)
    sheet, header_row = _best_sheet_and_header(excel)
    return pd.read_excel(excel, sheet_name=sheet, header=header_row)


def read_path(path: PathLike) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    if path.suffix.lower() == ".csv":
        return clean_data(pd.read_csv(path))
    return clean_data(_read_excel(path, path.name))


def read_upload(upload) -> pd.DataFrame:
    name = getattr(upload, "name", "uploaded.xlsx")
    if name.lower().endswith(".csv"):
        return clean_data(pd.read_csv(upload))
    try:
        upload.seek(0)
    except Exception:
        pass
    return clean_data(_read_excel(upload, name))


def excel_bytes(sheets: dict[str, pd.DataFrame]):
    b = io.BytesIO()
    with pd.ExcelWriter(b, engine="openpyxl") as w:
        for name, df in sheets.items():
            if df is not None:
                df.to_excel(w, sheet_name=name[:31], index=False)
    return b.getvalue()
