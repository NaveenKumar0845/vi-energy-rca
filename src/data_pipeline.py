from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd

ALIASES = {
    "IP Site ID": ["ip site id"],
    "Expense Nature": ["expense nature", "rental/eb/dg/tax/de-loading"],
    "Month-Year": ["month-year", "month year"],
    "Actual Bill From Date": ["actual bill from date"],
    "Actual Bill To Date": ["actual bill to date"],
    "Invoice Date": ["invoice date"],
    "Billed Amount (Excl GST)": ["billed amount", "excl gst"],
    "Debit Amount (Excl tax)": ["debit amount", "excl tax"],
    "Debit Amount (Incl tax)": ["debit amount", "incl tax"],
    "Prorated Billed Amount (Excl GST)": ["prorated billed amount", "excl gst"],
    "Prorated Debit Amount (Excl tax)": ["prorated debit amount", "excl tax"],
    "Prorated Debit Amount (Incl tax)": ["prorated debit amount", "incl tax"],
    "Dispute Type": ["dispute type"],
    "Circle Code": ["circle code"],
    "Vendor Code": ["vendor code"],
}

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()

def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    norms = {c: _norm(c) for c in out.columns}
    rename = {}
    for canonical, tokens in ALIASES.items():
        if canonical in out.columns:
            continue
        for c, n in norms.items():
            if all(_norm(t) in n for t in tokens):
                rename[c] = canonical
                break
    return out.rename(columns=rename)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = canonicalize_columns(df)
    for c in ["Actual Bill From Date", "Actual Bill To Date", "Invoice Date"]:
        if c in df: df[c] = pd.to_datetime(df[c], errors="coerce")
    if "Month-Year" in df:
        df["Month-Year"] = pd.to_datetime(df["Month-Year"].astype(str), errors="coerce").dt.to_period("M").astype(str)
    for c in ["Billed Amount (Excl GST)", "Debit Amount (Excl tax)", "Debit Amount (Incl tax)",
              "Prorated Billed Amount (Excl GST)", "Prorated Debit Amount (Excl tax)", "Prorated Debit Amount (Incl tax)"]:
        if c in df: df[c] = pd.to_numeric(df[c], errors="coerce")
    if "Expense Nature" in df: df["Expense Nature"] = df["Expense Nature"].astype(str).str.strip().str.upper()
    if "Dispute Type" in df: df["Dispute Type"] = df["Dispute Type"].astype("string").str.strip()
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
    if missing: raise ValueError(f"Raw workbook missing columns: {missing}")
    rows, rejected = [], []
    amounts = [("Billed Amount (Excl GST)", "Prorated Billed Amount (Excl GST)"),
               ("Debit Amount (Excl tax)", "Prorated Debit Amount (Excl tax)"),
               ("Debit Amount (Incl tax)", "Prorated Debit Amount (Incl tax)")]
    for idx, r in df.iterrows():
        start, end = r["Actual Bill From Date"], r["Actual Bill To Date"]
        if pd.isna(start) or pd.isna(end) or end < start:
            rejected.append({"Source Row": idx, "Reason": "Invalid billing period"}); continue
        total_days = (end.normalize() - start.normalize()).days + 1
        for period in _month_ranges(start, end):
            ms, me = period.start_time.normalize(), period.end_time.normalize()
            overlap_start, overlap_end = max(start.normalize(), ms), min(end.normalize(), me)
            days = (overlap_end - overlap_start).days + 1
            if days <= 0: continue
            rec = r.to_dict(); rec["Month-Year"] = str(period); rec["Proration Days"] = days
            for src, dst in amounts:
                rec[dst] = (float(r.get(src, 0) or 0) * days / total_days) if pd.notna(r.get(src, np.nan)) else np.nan
            rows.append(rec)
    return clean_data(pd.DataFrame(rows)), pd.DataFrame(rejected)

def is_model_ready(df: pd.DataFrame) -> bool:
    cols = set(canonicalize_columns(df).columns)
    return {"IP Site ID", "Expense Nature", "Month-Year", "Prorated Billed Amount (Excl GST)"}.issubset(cols)

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    x = clean_data(df).copy()
    x["_month"] = pd.to_datetime(x["Month-Year"], errors="coerce")
    x = x.sort_values(["IP Site ID", "Expense Nature", "_month"]).reset_index(drop=True)
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
    x["Debit/Billed Ratio"] = x.get(debit, 0) / x[bill].replace(0, np.nan)
    x["Month"] = x["_month"].dt.month; x["Quarter"] = x["_month"].dt.quarter
    return x.drop(columns=["_month"])

def merge_network(df: pd.DataFrame, kpi: Optional[pd.DataFrame]) -> pd.DataFrame:
    if kpi is None or kpi.empty: return df
    k = clean_data(kpi)
    if not {"IP Site ID", "Month-Year"}.issubset(k.columns): return df
    cols = [c for c in k.columns if c not in {"Expense Nature", "Dispute Type"}]
    return df.merge(k[cols].drop_duplicates(["IP Site ID", "Month-Year"]), on=["IP Site ID", "Month-Year"], how="left")

@dataclass
class PreparedData:
    analytical: pd.DataFrame
    rejected: pd.DataFrame
    mode: str

def prepare(df: pd.DataFrame, mode="auto", network: Optional[pd.DataFrame]=None) -> PreparedData:
    x = clean_data(df)
    detected = "model-ready" if is_model_ready(x) else "raw"
    mode = detected if mode == "auto" else mode
    if mode == "raw": x, rejected = prorate_raw(x)
    else: rejected = pd.DataFrame()
    x = merge_network(add_features(x), network)
    return PreparedData(x, rejected, mode)
