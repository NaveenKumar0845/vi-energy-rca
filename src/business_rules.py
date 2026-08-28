from __future__ import annotations
import math
import pandas as pd


def _num(row: pd.Series, key: str):
    value = row.get(key)
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except Exception:
        return None


def derive_business_signals(row: pd.Series) -> list[dict]:
    """Generate deterministic evidence signals before the LLM reasons over a row.

    These are intentionally simple pilot rules. In a production implementation,
    thresholds should come from Vi-approved business/network policy rather than
    being treated as universal engineering truth.
    """
    signals: list[dict] = []

    mom = _num(row, "MoM Billing Change %")
    if mom is not None and abs(mom) >= 30:
        signals.append({
            "signal": "Large month-on-month billing movement",
            "value": round(mom, 2),
            "severity": "high" if abs(mom) >= 60 else "medium",
            "type": "observed_billing_signal",
        })

    site_delta = _num(row, "Difference From Site Average %")
    if site_delta is not None and abs(site_delta) >= 35:
        signals.append({
            "signal": "Billing materially differs from prior site baseline",
            "value": round(site_delta, 2),
            "severity": "high" if abs(site_delta) >= 75 else "medium",
            "type": "observed_billing_signal",
        })

    debit_ratio = _num(row, "Debit/Billed Ratio")
    if debit_ratio is not None and debit_ratio >= 0.40:
        signals.append({
            "signal": "High disputed/debited share of billed amount",
            "value": round(debit_ratio, 3),
            "severity": "high" if debit_ratio >= 0.75 else "medium",
            "type": "observed_financial_signal",
        })

    if bool(row.get("Is Anomaly", False)):
        signals.append({
            "signal": "Multivariate anomaly detector flagged the record",
            "value": _num(row, "Anomaly Percentile"),
            "severity": "high",
            "type": "model_signal",
        })

    confidence = _num(row, "Prediction Confidence")
    prediction = row.get("Predicted Dispute Type")
    if prediction and str(prediction) != "Manual Review Required":
        signals.append({
            "signal": "ML dispute prediction",
            "value": str(prediction),
            "confidence": round(confidence, 3) if confidence is not None else None,
            "severity": "informational",
            "type": "model_signal",
        })

    # Optional network signals. These must not be interpreted as causal by
    # themselves; they simply add evidence when network KPI data exists.
    rsrp = _num(row, "RSRP (dBm)")
    sinr = _num(row, "SINR (dB)")
    prb = _num(row, "PRB Utilization (%)")
    availability = _num(row, "Availability (%)")
    alarms = _num(row, "Alarm Count")

    if rsrp is not None and rsrp < -105:
        signals.append({"signal": "Weak RSRP observation", "value": rsrp, "severity": "medium", "type": "network_observation"})
    if sinr is not None and sinr < 5:
        signals.append({"signal": "Low SINR observation", "value": sinr, "severity": "medium", "type": "network_observation"})
    if prb is not None and prb > 85:
        signals.append({"signal": "High PRB utilization observation", "value": prb, "severity": "medium", "type": "network_observation"})
    if availability is not None and availability < 99.0:
        signals.append({"signal": "Lower availability observation", "value": availability, "severity": "medium", "type": "network_observation"})
    if alarms is not None and alarms >= 3:
        signals.append({"signal": "Elevated alarm count observation", "value": int(alarms), "severity": "medium", "type": "network_observation"})

    return signals
