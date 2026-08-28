from __future__ import annotations
import numpy as np
import pandas as pd

EB_REASONS = [
    "High EB Consumption",
    "Incorrect EB Tariffs",
    "Incorrect / High DC ratio",
    "Duplicate/Retro Billing",
    "Site Locked / switch-off Billing",
    "Not part of BCC",
    "Non Active Non Radiating site",
    "Other (Please provide comment under separate Column)",
]
DG_REASONS = [
    "High DG Consumption",
    "Incorrect DG Rate",
    "Incorrect / negative / High DG run hour",
    "Incorrect / High DC ratio",
    "Duplicate/Retro Billing",
    "Not part of BCC",
    "Non Active Non Radiating site",
    "Other (Please provide comment under separate Column)",
]


def _billing_profile(expense: str, reason: str, base: float, rng) -> tuple[float, float]:
    billed = base + rng.normal(0, base * 0.035)
    debit_ratio = 0.06
    if reason.startswith("High EB") or reason.startswith("High DG"):
        billed *= 1.35
        debit_ratio = 0.14
    elif "Tariff" in reason or "DG Rate" in reason:
        billed *= 1.16
        debit_ratio = 0.40
    elif "run hour" in reason:
        billed *= 1.22
        debit_ratio = 0.38
    elif "DC ratio" in reason:
        billed *= 1.18
        debit_ratio = 0.33
    elif reason == "Duplicate/Retro Billing":
        billed *= 1.32
        debit_ratio = 0.72
    elif reason == "Site Locked / switch-off Billing":
        billed *= 0.82
        debit_ratio = 0.88
    elif reason == "Not part of BCC":
        debit_ratio = 0.96
    elif reason == "Non Active Non Radiating site":
        billed *= 0.58
        debit_ratio = 0.90
    elif reason.startswith("Other"):
        debit_ratio = 0.24
    debit = max(0.0, billed * debit_ratio + rng.normal(0, max(35, billed * 0.006)))
    return round(float(billed), 2), round(float(debit), 2)


def demo_raw(seed: int = 42, sites: int = 40, months: int = 12) -> pd.DataFrame:
    """Synthetic, interview-safe analogue of the telecom billing/dispute data.

    It intentionally mirrors the *shape and concepts* of the pilot without
    containing any Vodafone Idea customer, vendor, invoice or site data.
    """
    rng = np.random.default_rng(seed)
    rows = []
    circles = ["DL", "MH", "KA", "GJ", "UPW", "RJ", "TN", "WB"]
    vendors = ["INFRA-A", "INFRA-B", "INFRA-C"]
    ip_categories = ["MACRO", "MICRO", "IBS"]

    for s in range(1, sites + 1):
        site = f"DEMO-{100000 + s}"
        circle = circles[(s - 1) % len(circles)]
        vendor = vendors[(s - 1) % len(vendors)]
        category = ip_categories[(s - 1) % len(ip_categories)]
        site_factor = 1 + ((s % 7) - 3) * 0.025

        for month in range(1, months + 1):
            start = pd.Timestamp(2023, month, 1)
            end = start + pd.offsets.MonthEnd(0)
            record_month = start.strftime("%b-%y")
            for exp in ["EB", "DG"]:
                reasons = EB_REASONS if exp == "EB" else DG_REASONS
                reason = reasons[(s * 3 + month * 2 + (1 if exp == "DG" else 0)) % len(reasons)]
                seasonal = 1 + 0.08 * np.sin((month - 1) / 12 * 2 * np.pi)
                base = (22000 if exp == "EB" else 10500) * site_factor * seasonal
                billed, debit = _billing_profile(exp, reason, base, rng)

                rows.append({
                    "IP Category": category,
                    "Month": record_month,
                    "Circle Code": circle,
                    "Circle Name": f"Demo {circle}",
                    "IP Name": vendor,
                    "Vendor Code": vendor,
                    "IP Site ID": site,
                    "Invoice No": f"DEMO-INV-{s:04d}-{month:02d}-{exp}",
                    "Invoice Date": end,
                    "Actual Bill From Date": start,
                    "Actual Bill To Date": end,
                    "Expense Nature": exp,
                    "Billed Amount (Excl GST)": billed,
                    "Debit Amount (Excl tax)": debit,
                    "Debit Amount (Incl tax)": round(debit * 1.18, 2),
                    "Reason for Dispute Category": reason,
                    "Dispute Type": reason,
                    "Remarks (If any)": "Synthetic demonstration record",
                })
    return pd.DataFrame(rows)


def demo_kpis(seed: int = 7, sites: int = 40, months: int = 12) -> pd.DataFrame:
    """Synthetic network KPI evidence used only to demonstrate optional RCA joins."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(1, sites + 1):
        site = f"DEMO-{100000 + s}"
        for month in range(1, months + 1):
            stress = (s % 6) / 6 + (month - 1) / 24
            rows.append({
                "IP Site ID": site,
                "Month-Year": f"2023-{month:02d}",
                "RSRP (dBm)": round(-94 - stress * 10 + rng.normal(0, 1.8), 2),
                "RSRQ (dB)": round(-9 - stress * 4 + rng.normal(0, 0.8), 2),
                "SINR (dB)": round(17 - stress * 10 + rng.normal(0, 1.2), 2),
                "DL Throughput (Mbps)": round(58 - stress * 20 + rng.normal(0, 4), 2),
                "UL Throughput (Mbps)": round(19 - stress * 6 + rng.normal(0, 2), 2),
                "PRB Utilization (%)": round(45 + stress * 38 + rng.normal(0, 3), 2),
                "Availability (%)": round(99.8 - stress * 0.9 + rng.normal(0, 0.05), 3),
                "Alarm Count": int(rng.poisson(0.4 + stress * 1.8)),
            })
    return pd.DataFrame(rows)
