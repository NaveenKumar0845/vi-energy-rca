from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

# Phrases are matched by specificity. This avoids mapping
# "Prorated Billed Amount" to the more generic "Billed Amount" field.
ALIASES = {
    "IP Site ID": ["ip site id"],
    "Expense Nature": ["expense nature rental eb dg tax de loading", "expense nature"],
    "Month-Year": ["month year"],
    "Actual Bill From Date": ["actual bill from date"],
    "Actual Bill To Date": ["actual bill to date"],
    "Invoice Date": ["invoice date"],
    "Prorated Billed Amount (Excl GST)": ["prorated billed amount excl gst"],
    "Prorated Debit Amount (Excl tax)": ["prorated debit amount excl tax"],
    "Prorated Debit Amount (Incl tax)": ["prorated debit amount incl tax"],
    "Billed Amount (Excl GST)": ["billed amount excl gst"],
    "Debit Amount (Excl tax)": ["debit amount excl tax"],
    "Debit Amount (Incl tax)": ["debit amount incl tax"],
    "Dispute Type": ["dispute type"],
    "Dispute Head": ["dispute head"],
    "Dispute Sub-Category": ["dispute sub category"],
    "Circle Code": ["circle code"],
    "Vendor Code": ["vendor code"],
}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename = {}
    occupied = set(out.columns)

    for column in out.columns:
        n = _norm(column)
        candidates = []
        for canonical, phrases in ALIASES.items():
            for phrase in phrases:
                p = _norm(phrase)
                if n == p:
                    candidates.append((10_000 + len(p), canonical))
                elif p and p in n:
                    candidates.append((len(p), canonical))
        if candidates:
            _, best = max(candidates, key=lambda item: item[0])
            if column != best and best not in occupied:
                rename[column] = best
                occupied.add(best)

    return out.rename(columns=rename)


def _numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    s = series.astype("string").str.strip()
    negative = s.str.match(r"^\(.*\)$", na=False)
    s = s.str.replace(",", "", regex=False).str.replace("₹", "", regex=False)
    s = s.str.replace(r"[()]", "", regex=True)
    out = pd.to_numeric(s, errors="coerce")
    out.loc[negative & out.notna()] *= -1
    return out


def _date_series(series: pd.Series) -> pd.Series:
    """Parse ordinary dates plus Excel serial dates emitted by XLSB readers."""
    result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    numeric = pd.to_numeric(series, errors="coerce")
    excel_mask = numeric.between(20_000, 80_000, inclusive="both")
    if excel_mask.any():
        result.loc[excel_mask] = pd.to_datetime(
            numeric.loc[excel_mask], unit="D", origin="1899-12-30", errors="coerce"
        )

    remaining = result.isna() & series.notna()
    if remaining.any():
        result.loc[remaining] = pd.to_datetime(
            series.loc[remaining].astype("string").str.strip(),
            errors="coerce",
            format="mixed",
            dayfirst=False,
        )
    return result


def _month_year(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip()
    parsed = pd.to_datetime(s, format="%b-%y", errors="coerce")
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(s.loc[missing], errors="coerce", format="mixed")
    return parsed.dt.to_period("M").astype("string")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = canonicalize_columns(df)
    for c in ["Actual Bill From Date", "Actual Bill To Date", "Invoice Date"]:
        if c in df:
            df[c] = _date_series(df[c])
    if "Month-Year" in df:
        df["Month-Year"] = _month_year(df["Month-Year"])
    for c in [
        "Billed Amount (Excl GST)",
        "Debit Amount (Excl tax)",
        "Debit Amount (Incl tax)",
        "Prorated Billed Amount (Excl GST)",
        "Prorated Debit Amount (Excl tax)",
        "Prorated Debit Amount (Incl tax)",
    ]:
        if c in df:
            df[c] = _numeric(df[c])
    if "Expense Nature" in df:
        df["Expense Nature"] = df["Expense Nature"].astype("string").str.strip().str.upper()
    if "Dispute Type" in df:
        df["Dispute Type"] = df["Dispute Type"].astype("string").str.strip()
    return df


def _month_ranges(start: pd.Timestamp, end: pd.Timestamp):
    cur = start.to_period("M")
    last = end.to_period("M")
    while cur <= last:
        yield cur
        cur += 1


def prorate_raw(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = clean_data(df)
    required = ["IP Site ID", "Expense Nature", "Actual Bill From Date", "Actual Bill To Date", "Billed Amount (Excl GST)"]
    missing = [c for c in required if c not in df]
    if missing:
        raise ValueError(f"Raw workbook missing columns: {missing}")

    rows, rejected = [], []
    amounts = [
        ("Billed Amount (Excl GST)", "Prorated Billed Amount (Excl GST)"),
        ("Debit Amount (Excl tax)", "Prorated Debit Amount (Excl tax)"),
        ("Debit Amount (Incl tax)", "Prorated Debit Amount (Incl tax)"),
    ]

    for idx, r in df.iterrows():
        start, end = r["Actual Bill From Date"], r["Actual Bill To Date"]
        if pd.isna(start) or pd.isna(end) or end < start:
            rejected.append({"Source Row": idx, "Reason": "Invalid billing period"})
            continue

        total_days = (end.normalize() - start.normalize()).days + 1
        for period in _month_ranges(start, end):
            ms, me = period.start_time.normalize(), period.end_time.normalize()
            overlap_start, overlap_end = max(start.normalize(), ms), min(end.normalize(), me)
            days = (overlap_end - overlap_start).days + 1
            if days <= 0:
                continue

            rec = r.to_dict()
            rec["Month-Year"] = str(period)
            rec["Proration Days"] = days
            rec["Bill Duration Days"] = total_days
            for src, dst in amounts:
                value = r.get(src, np.nan)
                rec[dst] = float(value) * days / total_days if pd.notna(value) else np.nan
            rows.append(rec)

    return clean_data(pd.DataFrame(rows)), pd.DataFrame(rejected)


def is_model_ready(df: pd.DataFrame) -> bool:
    cols = set(canonicalize_columns(df).columns)
    return {"IP Site ID", "Expense Nature", "Month-Year", "Prorated Billed Amount (Excl GST)"}.issubset(cols)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    x = clean_data(df).copy()
    x["_month"] = pd.to_datetime(x["Month-Year"], errors="coerce")
    x = x.sort_values(["IP Site ID", "Expense Nature", "_month"]).reset_index(drop=True)

    if "Bill Duration Days" not in x and {"Actual Bill From Date", "Actual Bill To Date"}.issubset(x.columns):
        x["Bill Duration Days"] = (
            x["Actual Bill To Date"].dt.normalize() - x["Actual Bill From Date"].dt.normalize()
        ).dt.days + 1

    if "Invoice Date" in x:
        invoice = pd.to_datetime(x["Invoice Date"], errors="coerce")
        x["Invoice Month"] = invoice.dt.month
        x["Invoice Quarter"] = invoice.dt.quarter

    g = x.groupby(["IP Site ID", "Expense Nature"], dropna=False)
    bill = "Prorated Billed Amount (Excl GST)"
    prev = g[bill].shift(1)
    x["Previous Month Billing"] = prev
    x["Site Historical Average"] = g[bill].transform(lambda s: s.shift(1).expanding().mean())
    x["Site Historical Std"] = g[bill].transform(lambda s: s.shift(1).expanding().std())
    x["Rolling 3M Average"] = g[bill].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    x["MoM Billing Change %"] = (x[bill] - prev) / prev.replace(0, np.nan) * 100
    x["Difference From Site Average %"] = (x[bill] - x["Site Historical Average"]) / x["Site Historical Average"].replace(0, np.nan) * 100
    debit = "Prorated Debit Amount (Excl tax)"
    debit_values = x[debit] if debit in x else pd.Series(0.0, index=x.index)
    x["Debit/Billed Ratio"] = debit_values / x[bill].replace(0, np.nan)
    x["Month"] = x["_month"].dt.month
    x["Quarter"] = x["_month"].dt.quarter
    return x.drop(columns=["_month"])


def merge_network(df: pd.DataFrame, kpi: Optional[pd.DataFrame]) -> pd.DataFrame:
    if kpi is None or kpi.empty:
        return df
    k = clean_data(kpi)
    if not {"IP Site ID", "Month-Year"}.issubset(k.columns):
        return df
    cols = [c for c in k.columns if c not in {"Expense Nature", "Dispute Type"}]
    return df.merge(k[cols].drop_duplicates(["IP Site ID", "Month-Year"]), on=["IP Site ID", "Month-Year"], how="left")


@dataclass
class PreparedData:
    analytical: pd.DataFrame
    rejected: pd.DataFrame
    mode: str


def prepare(df: pd.DataFrame, mode="auto", network: Optional[pd.DataFrame] = None) -> PreparedData:
    x = clean_data(df)
    detected = "model-ready" if is_model_ready(x) else "raw"
    mode = detected if mode == "auto" else mode
    if mode == "raw":
        x, rejected = prorate_raw(x)
    else:
        rejected = pd.DataFrame()
    x = merge_network(add_features(x), network)
    return PreparedData(x, rejected, mode)
