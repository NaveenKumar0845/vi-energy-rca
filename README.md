# Vi Energy, Billing Dispute & GenAI RCA

Interview-safe reconstruction of a Vodafone Idea telecom energy and billing dispute RCA pilot. The repository contains **code, architecture documentation and synthetic demo data only**; proprietary Vi workbooks are not required for the public demo.

## What the project solves

Telecom site bills can contain EB/DG disputes caused by abnormal consumption, incorrect rates or ratios, duplicate/retro billing, locked or non-radiating sites and other business exceptions. The pilot turns site billing history into a consistent site-month analytical layer, predicts likely dispute reasons, flags unusual billing behavior and gives an LLM a grounded evidence package for barrier analysis, root-cause hypotheses and corrective actions.

## End-to-end flow

```text
Billing / dispute data
        +
Approved EB/DG taxonomy
        +
Optional network KPI evidence
        |
        v
Data validation & normalization
        |
        v
Monthly proration
        |
        v
Leakage-aware feature engineering
        |
   +----+-------------------+
   |                        |
   v                        v
EB/DG Random Forest    Isolation Forest
   |                        |
   +------------+-----------+
                |
                v
      Deterministic evidence
                |
                v
         GenAI RCA layer
     Gemini cloud / Ollama local
                |
                v
       Streamlit dashboard
                |
                v
       Human review / export
```

## Key engineering decisions

- raw `.xlsx`, `.xls`, `.xlsb` and `.csv` ingestion;
- inclusive-day monthly proration of billed and debit amounts;
- site-history features using only prior observations;
- canonical historical-label mapping to the approved dispute taxonomy;
- separate EB and DG classifiers so predictions cannot cross business taxonomies;
- confidence threshold with `Manual Review Required` instead of forced predictions;
- multivariate Isolation Forest anomaly detection;
- deterministic billing/network evidence signals before LLM reasoning;
- evidence-grounded RCA with explicit hallucination guardrails;
- query-specific structured retrieval instead of sending random rows to the LLM;
- optional local Ollama support to mirror the original pilot style and Gemini support for Streamlit Community Cloud;
- optional FastAPI service for a production-oriented serving pattern.

## Public repository data policy

Real customer/company workbooks are intentionally excluded. `.gitignore` blocks raw, processed and network data files from being committed.

Use one of three modes in the Streamlit app:

1. **Synthetic demo** — default and safe for GitHub/Streamlit Cloud.
2. **Upload files** — analyze a file for the current Streamlit session.
3. **Private/local data** — use untracked workbooks on a local clone.

The public synthetic generator mirrors the *schema and analytical behavior* of the pilot without containing Vi site IDs, invoices, vendors or financial records.

## Project documentation

- `docs/ARCHITECTURE.md` — pilot architecture, data sources, APIs and production reference design.
- `docs/DATA_CONTRACT.md` — source schemas, joins, data quality and privacy rules.
- `docs/INTERVIEW_GUIDE.md` — 30-second, 90-second and deep-dive interview explanations plus difficult Q&A.

## Repository structure

```text
vi-energy-rca/
├── streamlit_app.py
├── api/
│   └── main.py
├── src/
│   ├── data_pipeline.py
│   ├── demo_data.py
│   ├── models.py
│   ├── business_rules.py
│   ├── genai.py
│   ├── io_utils.py
│   └── visuals.py
├── data/
│   ├── raw/          # private/untracked
│   ├── processed/    # private/untracked
│   ├── network/      # private/untracked
│   └── reference/    # safe taxonomy CSV
├── docs/
├── scripts/
├── tests/
└── requirements.txt
```

## Run the Streamlit app

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The ML and anomaly views run without an API key.

### Gemini on Streamlit Community Cloud

Deploy with:

- Repository: `NaveenKumar0845/vi-energy-rca`
- Branch: `main`
- Main file path: `streamlit_app.py`

Add only in **App settings → Secrets**:

```toml
GEMINI_API_KEY="your-key"
```

### Ollama for local pilot-style testing

```bash
ollama pull phi3:mini
ollama serve
streamlit run streamlit_app.py
```

Then choose **Ollama (local pilot)** in the sidebar.

## Optional API service

```bash
uvicorn api.main:app --reload
```

Representative endpoints:

```text
GET /health
GET /demo/summary
GET /demo/sites/{site_id}
```

The API uses synthetic data by design. In a production implementation it would sit behind authentication and connect to governed data/model services.

## Tests and CI

```bash
pytest -q
python scripts/validate_repository_data.py
```

CI validates the synthetic end-to-end path so the project remains runnable without proprietary data.

## Important scope distinction

The implemented/reconstructed pilot is fundamentally an **energy and billing dispute RCA platform**. Network KPI evidence can be joined as an extension, but the billing dataset alone should not be presented as a complete 5G RF-performance optimization system.

A good interview title is:

**GenAI-Powered Telecom Energy & Billing Dispute RCA Platform**
