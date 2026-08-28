# Deployment Guide

## Streamlit Community Cloud

The repository is designed to run publicly using only synthetic data.

Configuration:

```text
Repository: NaveenKumar0845/vi-energy-rca
Branch: main
Main file: streamlit_app.py
```

The ML, anomaly, monitoring and data views run without any secret.

For Gemini-powered RCA/chat, add this only in Streamlit **App settings → Secrets**:

```toml
GEMINI_API_KEY="your-key"
```

Never commit the key to GitHub.

## Local run

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Local Ollama mode

To mirror the local-LLM style used in the pilot:

```bash
ollama pull phi3:mini
ollama serve
streamlit run streamlit_app.py
```

In the sidebar choose:

```text
GenAI runtime = Ollama (local pilot)
LLM model = phi3:mini
```

Optional custom Ollama endpoint:

```bash
export OLLAMA_BASE_URL=http://localhost:11434
```

## FastAPI service

```bash
uvicorn api.main:app --reload
```

Then open the automatically generated API docs at the `/docs` route of your local server.

Current public-demo endpoints:

```text
GET /health
GET /demo/summary
GET /demo/sites/{site_id}
```

## Private data usage

Do not commit company workbooks. If you have authorized local data, place it only in ignored local paths:

```text
data/raw/
data/processed/
data/network/
```

Or use the Streamlit **Upload files** mode so the file is used only for the running session.

## Production deployment pattern

A real enterprise deployment should separate the UI from the data/model services:

```text
Frontend / Streamlit / internal portal
                |
                v
        authenticated API layer
                |
      +---------+----------+
      |                    |
      v                    v
 ML scoring services    RCA service
      |                    |
      +---------+----------+
                |
                v
 governed warehouse / feature data / approved document store
```

Recommended production controls:
- SSO/RBAC;
- private networking;
- secret manager;
- encrypted data stores;
- API authentication and rate limits;
- audit logs;
- model and prompt versioning;
- human approval before financial actions;
- drift/coverage/latency monitoring;
- no raw proprietary files in source control.
