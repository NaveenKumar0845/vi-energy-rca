from __future__ import annotations
import json
import os
import re
import urllib.request
import urllib.error

import pandas as pd

from src.business_rules import derive_business_signals

SYSTEM_RULES = """You are a telecom energy and billing RCA assistant.
Use only the supplied structured evidence and deterministic signals.
Do not invent meter readings, tariffs, alarms, contracts, site visits, vendor actions, or network conditions.
Separate observed facts from model inference and recommended investigation.
A model prediction is not a confirmed root cause.
Never claim a 5G/network cause unless network KPI evidence is supplied.
If evidence is insufficient, explicitly say that the root cause is not established.
Return concise JSON with barrier_analysis, root_causes, evidence, corrective_actions, risk_level, confidence, additional_data_required."""


def _gemini(prompt: str, model: str = "gemini-2.5-flash", json_mode: bool = True) -> str:
    from google import genai
    from google.genai import types

    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not configured")
    client = genai.Client(api_key=key)
    kwargs = {"temperature": 0.15}
    if json_mode:
        kwargs["response_mime_type"] = "application/json"
    cfg = types.GenerateContentConfig(**kwargs)
    response = client.models.generate_content(model=model, contents=prompt, config=cfg)
    return response.text


def _ollama(prompt: str, model: str = "phi3:mini", json_mode: bool = True) -> str:
    base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    body = {"model": model, "prompt": prompt, "stream": False}
    if json_mode:
        body["format"] = "json"
    payload = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        f"{base}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ollama is unavailable at {base}: {exc}") from exc
    text = result.get("response")
    if not text:
        raise RuntimeError("Ollama returned an empty response")
    return text


def generate_text(
    prompt: str,
    model: str,
    provider: str = "Gemini",
    json_mode: bool = True,
) -> str:
    provider = str(provider).strip().lower()
    if provider.startswith("ollama"):
        return _ollama(prompt, model, json_mode=json_mode)
    return _gemini(prompt, model, json_mode=json_mode)


def _safe(v):
    if pd.isna(v):
        return None
    if isinstance(v, pd.Timestamp):
        return str(v)
    try:
        if hasattr(v, "item"):
            return v.item()
    except Exception:
        pass
    return v


def evidence_package(row: pd.Series) -> dict:
    keep = [
        "IP Site ID",
        "Circle Code",
        "Vendor Code",
        "Month-Year",
        "Record Month",
        "Expense Nature",
        "Prorated Billed Amount (Excl GST)",
        "Prorated Debit Amount (Excl tax)",
        "Billed Amount (Excl GST)",
        "Debit Amount (Excl tax)",
        "Bill Duration Days",
        "Site Historical Average",
        "Site Historical Std",
        "Rolling 3M Average",
        "MoM Billing Change %",
        "Difference From Site Average %",
        "Debit/Billed Ratio",
        "Predicted Dispute Type",
        "Prediction Confidence",
        "Second Candidate",
        "Second Candidate Confidence",
        "Prediction Status",
        "Is Anomaly",
        "Anomaly Percentile",
        "Anomaly Evidence",
        "RSRP (dBm)",
        "RSRQ (dB)",
        "SINR (dB)",
        "DL Throughput (Mbps)",
        "UL Throughput (Mbps)",
        "PRB Utilization (%)",
        "Availability (%)",
        "Alarm Count",
    ]
    evidence = {k: _safe(row.get(k)) for k in keep if k in row.index and pd.notna(row.get(k))}
    evidence["deterministic_signals"] = derive_business_signals(row)
    evidence["network_kpi_supplied"] = any(
        k in evidence
        for k in [
            "RSRP (dBm)",
            "RSRQ (dB)",
            "SINR (dB)",
            "DL Throughput (Mbps)",
            "UL Throughput (Mbps)",
            "PRB Utilization (%)",
            "Availability (%)",
            "Alarm Count",
        ]
    )
    return evidence


def rca_for_row(
    row: pd.Series,
    taxonomy: pd.DataFrame,
    model: str = "gemini-2.5-flash",
    provider: str = "Gemini",
) -> dict:
    evidence = evidence_package(row)
    head = str(row.get("Expense Nature", "")).upper()
    allowed = taxonomy.loc[
        taxonomy["Dispute Head"].eq(head), "Dispute Sub-Category"
    ].tolist()
    prompt = (
        f"{SYSTEM_RULES}\n\n"
        f"Approved {head} dispute taxonomy: {allowed}\n"
        "Analyze the evidence as a pilot RCA assistant. The prediction is a hypothesis, not a fact.\n"
        f"Evidence:\n{json.dumps(evidence, default=str)}"
    )
    try:
        parsed = json.loads(generate_text(prompt, model, provider, json_mode=True))
    except Exception as exc:
        parsed = {
            "barrier_analysis": [],
            "root_causes": ["GenAI RCA unavailable; root cause not established"],
            "evidence": [str(exc)],
            "corrective_actions": ["Review the structured evidence and deterministic signals manually"],
            "risk_level": "Unknown",
            "confidence": 0,
            "additional_data_required": [
                "Relevant meter/energy evidence",
                "Tariff/contract evidence where applicable",
                "Site or network evidence if a network cause is being investigated",
            ],
        }
    return parsed


def extract_filters(question: str, df: pd.DataFrame):
    q = question.lower()
    mask = pd.Series(True, index=df.index)
    notes = []
    if "IP Site ID" in df:
        for site in df["IP Site ID"].dropna().astype(str).unique():
            if site.lower() in q:
                mask &= df["IP Site ID"].astype(str).eq(site)
                notes.append(site)
                break
    if "Expense Nature" in df:
        for exp in ["EB", "DG", "RENTAL", "TAX"]:
            if re.search(rf"\b{exp.lower()}\b", q):
                mask &= df["Expense Nature"].astype(str).str.upper().eq(exp)
                notes.append(exp)
                break
    if "Month-Year" in df:
        for m in df["Month-Year"].dropna().astype(str).unique():
            if m.lower() in q:
                mask &= df["Month-Year"].astype(str).eq(m)
                notes.append(m)
                break
    return df[mask].copy(), notes


def answer_query(
    question: str,
    df: pd.DataFrame,
    model: str = "gemini-2.5-flash",
    provider: str = "Gemini",
) -> str:
    relevant, notes = extract_filters(question, df)
    if relevant.empty:
        relevant = (
            df.sort_values("Anomaly Percentile", ascending=False).head(12)
            if "Anomaly Percentile" in df
            else df.head(12)
        )
    else:
        relevant = relevant.sort_values("Month-Year").tail(18)
    cols = [
        c
        for c in [
            "IP Site ID",
            "Month-Year",
            "Expense Nature",
            "Prorated Billed Amount (Excl GST)",
            "Prorated Debit Amount (Excl tax)",
            "Site Historical Average",
            "MoM Billing Change %",
            "Difference From Site Average %",
            "Predicted Dispute Type",
            "Prediction Confidence",
            "Prediction Status",
            "Is Anomaly",
            "Anomaly Evidence",
            "RSRP (dBm)",
            "SINR (dB)",
            "PRB Utilization (%)",
            "Availability (%)",
            "Alarm Count",
        ]
        if c in relevant
    ]
    records = relevant[cols].to_dict(orient="records")
    for record in records:
        record["deterministic_signals"] = derive_business_signals(pd.Series(record))
    prompt = (
        "Answer the user's question using only the supplied structured telecom energy/billing evidence. "
        "Do not convert correlation into causation. If the evidence is insufficient, say so. "
        "Keep the answer concise and separate observations from interpretation.\n"
        f"Question: {question}\n"
        f"Detected filters: {notes}\n"
        f"Rows: {json.dumps(records, default=str)}"
    )
    try:
        return generate_text(prompt, model, provider, json_mode=False)
    except Exception as exc:
        return f"GenAI unavailable: {exc}"
