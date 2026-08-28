from __future__ import annotations
import json, os, re
import pandas as pd

SYSTEM_RULES = """You are a telecom billing RCA assistant. Use only supplied evidence. Do not invent meter readings, tariffs, alarms, contracts, or network conditions. Separate observed facts from inference. If evidence is insufficient, say so. Never claim a 5G/network cause unless KPI evidence is supplied. Return concise JSON with barrier_analysis, root_causes, evidence, corrective_actions, risk_level, confidence, additional_data_required."""

def _gemini(prompt: str, model="gemini-2.5-flash") -> str:
    from google import genai
    from google.genai import types
    key=os.getenv("GEMINI_API_KEY")
    if not key: raise RuntimeError("GEMINI_API_KEY is not configured in Streamlit Secrets")
    client=genai.Client(api_key=key)
    cfg=types.GenerateContentConfig(temperature=0.15,response_mime_type="application/json")
    return client.models.generate_content(model=model,contents=prompt,config=cfg).text

def _safe(v):
    if pd.isna(v): return None
    if isinstance(v,(pd.Timestamp,)): return str(v)
    try:
        if hasattr(v,"item"): return v.item()
    except Exception: pass
    return v

def evidence_package(row: pd.Series) -> dict:
    keep=["IP Site ID","Month-Year","Expense Nature","Prorated Billed Amount (Excl GST)","Prorated Debit Amount (Excl tax)",
          "Site Historical Average","Rolling 3M Average","MoM Billing Change %","Difference From Site Average %","Debit/Billed Ratio",
          "Predicted Dispute Type","Prediction Confidence","Second Candidate","Is Anomaly","Anomaly Percentile","Anomaly Evidence",
          "RSRP (dBm)","RSRQ (dB)","SINR (dB)","DL Throughput (Mbps)","UL Throughput (Mbps)","PRB Utilization (%)","Availability (%)","Alarm Count"]
    return {k:_safe(row.get(k)) for k in keep if k in row.index and pd.notna(row.get(k))}

def rca_for_row(row: pd.Series, taxonomy: pd.DataFrame, model="gemini-2.5-flash") -> dict:
    evidence=evidence_package(row)
    head=str(row.get("Expense Nature","")).upper()
    allowed=taxonomy.loc[taxonomy["Dispute Head"].eq(head),"Dispute Sub-Category"].tolist()
    prompt=f"{SYSTEM_RULES}\n\nAllowed {head} dispute reasons: {allowed}\nEvidence:\n{json.dumps(evidence,default=str)}"
    try:
        parsed=json.loads(_gemini(prompt,model))
    except Exception as exc:
        parsed={"barrier_analysis":[],"root_causes":["GenAI unavailable"],"evidence":[str(exc)],"corrective_actions":["Review structured evidence manually"],"risk_level":"Unknown","confidence":0,"additional_data_required":[]}
    return parsed

def extract_filters(question: str, df: pd.DataFrame):
    q=question.lower(); mask=pd.Series(True,index=df.index); notes=[]
    if "IP Site ID" in df:
        for site in df["IP Site ID"].dropna().astype(str).unique():
            if site.lower() in q: mask &= df["IP Site ID"].astype(str).eq(site); notes.append(site); break
    if "Expense Nature" in df:
        for exp in ["EB","DG","RENTAL","TAX"]:
            if re.search(rf"\b{exp.lower()}\b",q): mask &= df["Expense Nature"].astype(str).str.upper().eq(exp); notes.append(exp); break
    if "Month-Year" in df:
        for m in df["Month-Year"].dropna().astype(str).unique():
            if m.lower() in q: mask &= df["Month-Year"].astype(str).eq(m); notes.append(m); break
    return df[mask].copy(),notes

def answer_query(question: str, df: pd.DataFrame, model="gemini-2.5-flash") -> str:
    relevant,notes=extract_filters(question,df)
    if relevant.empty: relevant=df.sort_values("Anomaly Percentile",ascending=False).head(12) if "Anomaly Percentile" in df else df.head(12)
    else: relevant=relevant.sort_values("Month-Year").tail(18)
    cols=[c for c in ["IP Site ID","Month-Year","Expense Nature","Prorated Billed Amount (Excl GST)","Site Historical Average","MoM Billing Change %","Predicted Dispute Type","Prediction Confidence","Is Anomaly","Anomaly Evidence","RSRP (dBm)","SINR (dB)","PRB Utilization (%)","Availability (%)"] if c in relevant]
    prompt=f"Answer the user's question using only this structured telecom billing evidence. If insufficient, say so.\nQuestion: {question}\nRows:\n{relevant[cols].to_json(orient='records')}"
    try:
        from google import genai
        key=os.getenv("GEMINI_API_KEY")
        if not key: return "Gemini is not configured. Add GEMINI_API_KEY in Streamlit Secrets."
        return genai.Client(api_key=key).models.generate_content(model=model,contents=prompt).text
    except Exception as exc: return f"GenAI unavailable: {exc}"
