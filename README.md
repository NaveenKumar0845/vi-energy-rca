# Vi Energy, Billing Dispute & GenAI RCA

A cloud-ready Streamlit pilot for telecom site billing dispute prediction, anomaly detection and evidence-grounded root-cause analysis.

> **Important data-safety note:** this GitHub repository is currently **public**. The code and synthetic demo are safe to publish, but do **not** upload real Vodafone Idea workbooks until you change the repository to **private**.

## Architecture

```text
Raw billing/dispute workbook
        ↓
Header/date/amount validation
        ↓
Inclusive-day monthly proration
        ↓
Site × month feature engineering
        ↓
┌────────────────────────────┐
│ Hierarchical Random Forest │  EB → only EB reasons
│ Isolation Forest           │  DG → only DG reasons
└─────────────┬──────────────┘
              ↓
      Structured evidence
              ↓
        Gemini GenAI RCA
              ↓
Barrier → Root Cause → Corrective Action
              ↓
        Streamlit dashboard
```

The solution does **not** claim a 5G/network root cause unless an optional network KPI file is supplied.

## Upload these files after making the repository private

| File | Exact GitHub path | Purpose |
|---|---|---|
| `Dispute_Head_UPE_Indus_jan23toDec23.xlsx` | `data/raw/Dispute_Head_UPE_Indus_jan23toDec23.xlsx` | Raw labelled historical billing/dispute data |
| `Dispute reasons.xlsx` | `data/reference/dispute_reasons.xlsx` | EB/DG dispute taxonomy |
| `filtered_data.xlsx` | `data/processed/filtered_data.xlsx` | Existing model-ready/prorated dataset |
| `network_kpis.xlsx` | `data/network/network_kpis.xlsx` | Optional network/5G KPI evidence |

The repository already contains a CSV copy of the EB/DG taxonomy for the demo.

## What was fixed from the original prototype

- implemented the missing raw → monthly proration transformation;
- preserved `Dispute Type` as the supervised target;
- engineered prior-history features instead of treating site IDs as meaningful numbers;
- separated EB and DG candidate reason spaces;
- removed random prediction fallback and added `Manual Review Required` for low confidence;
- moved anomaly detection from a single billing amount to multivariate billing/history features;
- added chronological evaluation to reduce leakage;
- replaced five-random-row chat context with query-specific structured retrieval;
- constrained GenAI RCA to supplied evidence and forbids unsupported 5G/network claims;
- supports Gemini API for Streamlit Community Cloud without committing secrets.

## Deploy on Streamlit Community Cloud

Use:

- Repository: `NaveenKumar0845/vi-energy-rca`
- Branch: `main`
- Main file path: `streamlit_app.py`

In **App settings → Secrets**, add:

```toml
GEMINI_API_KEY="your-key"
```

Never add the real key to GitHub.

## Local run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The default **Synthetic demo** runs without proprietary data. The ML/anomaly tabs work without an API key; the GenAI RCA/chat tabs require `GEMINI_API_KEY`.

## Input notes

### Raw billing workbook
Required minimum fields are `IP Site ID`, bill from/to dates, expense nature, and billed amount. Keeping the real `Dispute Type` is required for supervised training/evaluation.

### Processed workbook
Expected minimum fields are `IP Site ID`, `Expense Nature`, `Month-Year`, and `Prorated Billed Amount (Excl GST)`.

### Network KPI workbook
If supplied, it can include RSRP, RSRQ, SINR, DL/UL throughput, PRB utilization, availability and alarm count. Engineering thresholds should be Vi-approved before any production claim.

## Tests

```bash
pytest -q
```

The tests cover cross-month proration/reconciliation, leakage-aware historical features, EB/DG prediction guardrails, and anomaly-output creation.
