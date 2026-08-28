# Vi Energy, Billing Dispute & GenAI RCA — Complete Pilot

This repository contains the consolidated Vi Energy/Billing Dispute RCA pilot: validation, monthly proration, leakage-aware feature engineering, hierarchical dispute classification, multivariate anomaly detection, deterministic evidence generation, and grounded GenAI RCA.

> **Data safety:** The repository is currently public. Do **not** upload proprietary Vodafone Idea workbooks here until you change the repository visibility to **private**. The bundled demo data is synthetic and safe for a public portfolio.

## Main data paths

When the repository is private, place the real workbooks at:

```text
data/raw/Dispute_Head_UPE_Indus_jan23toDec23.xlsx
data/reference/dispute_reasons.xlsx
data/processed/filtered_data.xlsx
data/network/network_kpis.xlsx       # optional
```

## Streamlit Community Cloud

Deploy with:

- Repository: `NaveenKumar0845/vi-energy-rca`
- Branch: `main`
- Main file: `streamlit_app.py`

Add `GEMINI_API_KEY` in Streamlit App Settings → Secrets. Never commit a real API key.

## Local run

```bash
python -m venv .venv
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The app starts with bundled synthetic demo data, so the complete ML/anomaly workflow can be tested without proprietary Vi files.
