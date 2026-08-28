from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
import streamlit as st

from src.data_pipeline import prepare
from src.demo_data import demo_raw, demo_kpis
from src.genai import answer_query, rca_for_row
from src.io_utils import read_path, read_upload, excel_bytes
from src.models import DisputeClassifier, add_anomalies, evaluate_time_split, prepare_taxonomy
from src.visuals import monthly_billing, disputes, anomalies

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data/raw"
FILTERED = ROOT / "data/processed/filtered_data.xlsx"
TAX_XLSX = ROOT / "data/reference/dispute_reasons.xlsx"
TAX_CSV = ROOT / "data/reference/dispute_reasons.csv"
KPI_XLSX = ROOT / "data/network/network_kpis.xlsx"


def discover_raw_file() -> Path:
    preferred = [
        RAW_DIR / "Dispute_Head_UPE_Indus_Jan23toDec23.xlsb",
        RAW_DIR / "Dispute_Head_UPE_Indus_jan23toDec23.xlsx",
        RAW_DIR / "Dispute_Head_UPE_Indus_Jan23toDec23.xlsx",
    ]
    for path in preferred:
        if path.exists():
            return path
    candidates = []
    for pattern in ("*.xlsb", "*.xlsx", "*.xls", "*.csv"):
        candidates.extend(sorted(RAW_DIR.glob(pattern)))
    if candidates:
        return candidates[0]
    return preferred[0]


RAW = discover_raw_file()

st.set_page_config(page_title="Vi Energy & Billing Dispute RCA", page_icon="⚡", layout="wide")
st.markdown("<style>h1,h2,h3{color:#e60000}.stButton>button{background:#e60000;color:white}</style>", unsafe_allow_html=True)


def get_secret():
    try:
        key = st.secrets.get("GEMINI_API_KEY")
        if key:
            os.environ["GEMINI_API_KEY"] = str(key)
    except Exception:
        pass


def load_tax(upload=None):
    if upload is not None:
        return prepare_taxonomy(read_upload(upload))
    if TAX_XLSX.exists():
        return prepare_taxonomy(read_path(TAX_XLSX))
    return prepare_taxonomy(read_path(TAX_CSV))


def main():
    get_secret()
    st.title("⚡ Vi Energy, Billing Dispute & GenAI RCA")
    st.caption("Raw billing → monthly proration → leakage-aware features → EB/DG dispute ML → multivariate anomaly detection → evidence-grounded RCA")

    with st.sidebar:
        st.header("Run configuration")
        source = st.radio("Data source", ["Synthetic demo", "Repository data", "Upload files"])
        repo_file = st.selectbox("Repository billing file", ["Raw dispute history", "Existing filtered_data"], disabled=source != "Repository data")
        conf = st.slider("Prediction confidence", 0.30, 0.95, 0.60, 0.05)
        contamination = st.slider("Expected anomaly rate", 0.01, 0.30, 0.10, 0.01)
        model_name = st.text_input("Gemini model", "gemini-2.5-flash")
        uploaded = training_upload = tax_upload = kpi_upload = None
        if source == "Upload files":
            uploaded = st.file_uploader("Billing/dispute file", type=["xlsx", "xls", "xlsb", "csv"])
            training_upload = st.file_uploader("Optional separate labelled training file", type=["xlsx", "xls", "xlsb", "csv"])
            tax_upload = st.file_uploader("Optional dispute reasons master", type=["xlsx", "xls", "xlsb", "csv"])
            kpi_upload = st.file_uploader("Optional network KPI file", type=["xlsx", "xls", "xlsb", "csv"])

    if source == "Upload files" and uploaded is None:
        st.info("Upload a billing file, or switch to Synthetic demo.")
        return

    try:
        training = None
        if source == "Synthetic demo":
            raw = demo_raw()
            network = demo_kpis()
            mode = "raw"
        elif source == "Repository data":
            network = read_path(KPI_XLSX) if KPI_XLSX.exists() else None
            if repo_file == "Raw dispute history":
                if not RAW.exists():
                    raise FileNotFoundError(f"No raw billing workbook found in {RAW_DIR}")
                raw = read_path(RAW)
                mode = "raw"
            else:
                raw = read_path(FILTERED)
                mode = "model-ready"
                if RAW.exists():
                    training = prepare(read_path(RAW), "raw", network).analytical
        else:
            raw = read_upload(uploaded)
            network = read_upload(kpi_upload) if kpi_upload else None
            mode = "auto"
            if training_upload:
                training = prepare(read_upload(training_upload), "auto", network).analytical

        taxonomy = load_tax(tax_upload)
        p = prepare(raw, mode, network)
        df = p.analytical
        train = training if training is not None else df
        clf = DisputeClassifier(conf).fit(train, taxonomy)
        analyzed = add_anomalies(clf.predict(df), contamination)
    except Exception as exc:
        st.error(f"Pipeline failed: {exc}")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(analyzed):,}")
    c2.metric("Sites", analyzed["IP Site ID"].nunique())
    c3.metric("Auto predictions", int(analyzed["Prediction Status"].eq("Auto-predicted").sum()))
    c4.metric("Anomalies", int(analyzed["Is Anomaly"].sum()))

    tabs = st.tabs(["📊 Data", "🎯 Prediction", "🚨 Anomalies", "📈 Evaluation", "🧠 RCA", "💬 Chat"])
    with tabs[0]:
        st.write(f"Detected input mode: **{p.mode}**")
        if source == "Repository data":
            st.caption(f"Repository file: `{RAW.relative_to(ROOT) if repo_file == 'Raw dispute history' else FILTERED.relative_to(ROOT)}`")
        if not p.rejected.empty:
            st.warning(f"Rejected raw rows: {len(p.rejected)}")
        st.plotly_chart(monthly_billing(analyzed), use_container_width=True)
        st.dataframe(analyzed.head(250), use_container_width=True)
    with tabs[1]:
        cols = [c for c in ["IP Site ID", "Month-Year", "Expense Nature", "Prorated Billed Amount (Excl GST)", "Predicted Dispute Type", "Prediction Confidence", "Second Candidate", "Prediction Status"] if c in analyzed]
        st.plotly_chart(disputes(analyzed), use_container_width=True)
        st.dataframe(analyzed[cols], use_container_width=True)
        st.caption("Low-confidence predictions are routed to Manual Review Required; invalid predictions are never replaced randomly.")
    with tabs[2]:
        st.plotly_chart(anomalies(analyzed), use_container_width=True)
        a = analyzed[analyzed["Is Anomaly"]].sort_values("Anomaly Percentile", ascending=False)
        st.dataframe(a[[c for c in ["IP Site ID", "Month-Year", "Expense Nature", "Prorated Billed Amount (Excl GST)", "Anomaly Percentile", "Anomaly Evidence", "Predicted Dispute Type"] if c in a]], use_container_width=True)
    with tabs[3]:
        if "Dispute Type" in df and df["Dispute Type"].notna().any():
            if st.button("Run chronological evaluation"):
                metrics, pred = evaluate_time_split(df, taxonomy, conf)
                st.json(metrics)
                st.dataframe(pred.head(100), use_container_width=True)
        else:
            st.info("Actual Dispute Type labels are required for evaluation.")
    with tabs[4]:
        idx = st.selectbox("Select row", analyzed.index.tolist(), format_func=lambda i: f"{analyzed.loc[i, 'IP Site ID']} | {analyzed.loc[i, 'Month-Year']} | {analyzed.loc[i, 'Expense Nature']}")
        st.json({k: str(v) for k, v in analyzed.loc[idx].to_dict().items() if pd.notna(v)})
        if st.button("Generate grounded RCA"):
            with st.spinner("Generating RCA..."):
                st.json(rca_for_row(analyzed.loc[idx], taxonomy, model_name))
    with tabs[5]:
        q = st.text_input("Ask about a site, month, EB/DG dispute or anomaly")
        if st.button("Ask") and q:
            with st.spinner("Retrieving relevant structured rows..."):
                st.write(answer_query(q, analyzed, model_name))

    st.divider()
    st.download_button(
        "Download complete analysis workbook",
        excel_bytes({"Analysis": analyzed, "Rejected Rows": p.rejected}),
        "vi_energy_rca_analysis.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


if __name__ == "__main__":
    main()
