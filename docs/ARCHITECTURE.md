# Architecture — Telecom Energy, Billing Dispute & GenAI RCA

This repository is an interview-safe reconstruction of the Vi pilot. It contains code and synthetic data only. The architecture below deliberately separates **what the pilot implemented** from the **production-scale architecture you can describe as the natural enterprise evolution**. Do not present the production extensions as already deployed unless you actually implemented them.

## 1. Business objective

Telecom tower/site energy bills can contain disputes caused by abnormal consumption, tariff/rate issues, incorrect ratios, duplicate or retro billing, locked/non-radiating sites, tenancy issues, unsupported billing and other business exceptions. Manually reviewing site history, invoice periods, debit amounts and prior disputes is slow.

The pilot converts site billing/dispute data into a site-month analytical layer, predicts the likely dispute reason, identifies unusual billing behavior, and gives an LLM a structured evidence package so it can generate barrier analysis, root-cause hypotheses and corrective actions without inventing unsupported facts.

## 2. Pilot implementation architecture

```text
Historical billing / dispute export
        +
Approved EB/DG dispute taxonomy
        +
Optional network KPI file
        |
        v
+----------------------------+
| Data ingestion & cleaning  |
| pandas / Excel / CSV       |
| header/date normalization  |
+-------------+--------------+
              |
              v
+----------------------------+
| Monthly proration engine   |
| bill-from / bill-to dates  |
| billed + debit allocation  |
+-------------+--------------+
              |
              v
+----------------------------+
| Feature engineering        |
| site history               |
| rolling averages           |
| MoM change                 |
| debit/billed ratio         |
| circle/vendor/time         |
+-------------+--------------+
              |
       +------+------+
       |             |
       v             v
+-------------+  +------------------+
| Hierarchical|  | Isolation Forest |
| RF model    |  | anomaly detector |
| EB / DG     |  | multivariate     |
+------+------+  +--------+---------+
       |                  |
       +---------+--------+
                 |
                 v
+----------------------------+
| Deterministic evidence     |
| rules + model confidence   |
+-------------+--------------+
              |
              v
+----------------------------+
| GenAI RCA                  |
| original pilot: local LLM  |
| reconstructed cloud demo:  |
| Gemini; optional Ollama    |
+-------------+--------------+
              |
              v
+----------------------------+
| Streamlit UI               |
| data / predictions         |
| anomalies / evaluation     |
| RCA / conversational Q&A   |
+----------------------------+
```

## 3. Data sources

### Data directly represented by the original pilot material

**Historical billing and dispute data**
- IP Site ID
- circle / vendor / infrastructure metadata
- invoice number and invoice date
- actual bill from/to dates
- expense nature: EB, DG, rental, tax, etc.
- billed amount
- debit amount excluding and including tax
- dispute reason/category
- dispute type
- provision, debit-note and SAP-document fields

**Dispute taxonomy**
- Dispute Head: EB or DG
- approved Dispute Sub-Category values

**Prorated analytical data**
- IP Site ID
- Month-Year
- Expense Nature
- Prorated Billed Amount
- Prorated Debit Amount excluding tax
- Prorated Debit Amount including tax

### Production data sources that would strengthen the solution

These are architectural extensions, not claims about the original prototype:
- tower/site master and tenancy information;
- electricity meter readings and DG run-hour data;
- tariff/rate master and contract/SLA data;
- SAP/finance posting status and debit-note lifecycle;
- OSS/NMS network KPIs and alarms (RSRP, RSRQ, SINR, throughput, PRB utilization, availability, alarms);
- site lock/switch-off/non-radiating status;
- historical dispute-resolution notes and corrective-action outcomes.

## 4. Ingestion layer

### Pilot

The pilot accepts `.xlsx`, `.xls`, `.xlsb` and `.csv` files. The loader detects headers, normalizes column names, parses ordinary dates and Excel serial dates, standardizes expense nature and numeric amounts, and rejects invalid billing periods.

### Production reference design

```text
Billing system / tower partner files ----\
SAP finance -----------------------------+--> secure ingestion --> landing zone
Energy / meter platform -----------------+
OSS/NMS KPIs & alarms -------------------+
Contract / tariff master ----------------/
```

Typical enterprise ingestion choices would be scheduled batch files, database views or authenticated REST/API connectors. The pilot code does not claim a specific Vi production API.

## 5. Canonical site-month data model

The analytical grain is primarily:

```text
IP Site ID + Expense Nature + Month-Year
```

For bills spanning multiple months, inclusive billing days are calculated and amounts are allocated proportionally:

```text
Prorated amount
= original amount x overlap days / total bill-period days
```

This is applied to billed amount and debit amounts so different billing periods can be compared consistently.

## 6. Feature engineering

The model does not treat an arbitrary site identifier as a meaningful number. It uses behavioral and contextual features such as:
- current prorated billed/debit amounts;
- debit-to-billed ratio;
- bill duration and proration days;
- previous-month billing;
- expanding historical site average and standard deviation;
- rolling 3-month average;
- month-on-month billing change;
- difference from historical site average;
- month / quarter / invoice timing;
- circle, vendor and site category;
- optional network KPIs when available.

Historical features are shifted so the current row does not leak its value into its own historical baseline.

## 7. Dispute classification

The classifier is hierarchical by expense head:

```text
Expense Nature
   |
   +-- EB --> only EB dispute reasons
   |
   +-- DG --> only DG dispute reasons
```

A Random Forest is used because the pilot data is tabular, mixes numeric and categorical features, supports non-linear interactions, and gives probability estimates. Categorical variables are one-hot encoded and numeric fields are median-imputed.

The model returns:
- top predicted dispute reason;
- prediction confidence;
- second candidate and confidence;
- `Manual Review Required` when confidence is below threshold.

There is no random fallback.

## 8. Label quality layer

Historical data can contain legacy labels such as `Other`, spacing variants or older naming. The repository creates a canonical target by:
1. preferring the more taxonomy-aligned reason-category field when present;
2. falling back to Dispute Type;
3. normalizing only deterministic text/spacing variants;
4. mapping a small set of explicit legacy aliases to approved taxonomy values;
5. excluding extremely rare classes from automated training when support is insufficient.

No fuzzy LLM label guessing is used in the supervised target.

## 9. Anomaly detection

Isolation Forest detects unusual records using multiple billing/history variables rather than a single amount. The output contains:
- anomaly flag;
- anomaly percentile/score;
- human-readable evidence such as MoM movement, deviation from site baseline and debit ratio.

The anomaly result is a prioritization signal, not proof of fraud or an incorrect bill.

## 10. Deterministic business-rule layer

Before the LLM receives evidence, deterministic signals are generated for conditions such as:
- large month-on-month bill movement;
- large deviation from site baseline;
- high debit/billed ratio;
- anomaly-detector flag;
- ML dispute prediction and confidence;
- optional weak network KPI observations.

This makes the LLM consume evidence rather than raw tables alone.

## 11. GenAI RCA layer

The LLM receives a compact evidence package containing site/month/expense context, billing metrics, historical statistics, ML prediction, anomaly result, deterministic signals and optional network KPIs.

Guardrails instruct the LLM to:
- use only supplied evidence;
- separate facts, inference and recommended investigation;
- treat the ML result as a hypothesis, not a confirmed cause;
- avoid inventing tariffs, meter readings, contracts, alarms or site conditions;
- never claim a 5G/network cause without KPI evidence;
- return `insufficient evidence` where appropriate.

Structured RCA output:

```json
{
  "barrier_analysis": [],
  "root_causes": [],
  "evidence": [],
  "corrective_actions": [],
  "risk_level": "",
  "confidence": 0,
  "additional_data_required": []
}
```

## 12. Conversational query flow

The chat layer does not send five random records to the LLM. It performs deterministic retrieval first:

```text
User question
   |
   v
extract site / expense / month filters
   |
   v
retrieve matching structured rows
   |
   v
compute / attach evidence
   |
   v
LLM answer grounded only in retrieved rows
```

For this structured use case, dataframe/SQL-style retrieval is preferable to adding a vector database unnecessarily. RAG becomes useful later for unstructured contracts, tariff circulars, SLAs and historical resolution notes.

## 13. Serving layer

### Streamlit

The dashboard contains six logical views:
1. data overview and monthly billing;
2. dispute prediction;
3. anomaly inspection;
4. chronological evaluation;
5. evidence-grounded RCA;
6. conversational Q&A.

### API service — production-oriented extension

The repository also contains an optional FastAPI service for health, model summary and demo site analysis. In an enterprise design, Streamlit/another frontend would call authenticated APIs instead of directly loading source data.

Representative endpoints:

```text
GET  /health
GET  /demo/summary
GET  /demo/sites/{site_id}
```

A production deployment would add authenticated endpoints for ingestion, scoring, RCA generation, feedback and monitoring.

## 14. Production reference architecture

```text
                         ENTERPRISE SOURCES
 Billing / SAP / Tower partner / Energy meter / Site master / OSS-NMS
                              |
                              v
                    Secure ingestion layer
                 Batch + DB/API connectors
                              |
                              v
                  Data lake / warehouse zones
              Raw -> validated -> curated site-month
                              |
             +----------------+----------------+
             |                                 |
             v                                 v
      Feature pipeline                  Document pipeline
 billing/site/network history        contracts/SLA/tariffs/SOPs
             |                                 |
             v                                 v
    ML model services                  RAG index (optional)
 RF/GBM classifier + anomaly                  |
             |                                 |
             +---------------+-----------------+
                             |
                             v
                       RCA orchestrator
                  rules + ML + GenAI + policy
                             |
              +--------------+--------------+
              |              |              |
              v              v              v
         Dashboard       REST APIs       Alerts/cases
              |
              v
     Human review & feedback loop
              |
              v
 model monitoring / drift / audit / retraining
```

## 15. Security and governance

For a production telecom deployment:
- keep customer/site-level data inside approved private storage;
- do not commit source workbooks or API keys to Git;
- use role-based access and least privilege;
- encrypt data in transit and at rest;
- maintain model/prompt versions and audit logs;
- mask or minimize sensitive identifiers for LLM prompts;
- use human approval for financial dispute actions;
- monitor prediction coverage, class drift, anomaly rates, latency and LLM failure rate.

The public GitHub repository intentionally runs on synthetic data and ignores private raw/processed/network workbooks.

## 16. What to call the project

Most accurate title for the implemented pilot:

**GenAI-Powered Telecom Energy & Billing Dispute RCA Platform**

If network KPI integration is included in the discussion, say it was an **extension path / integrated evidence layer** unless you actually had live OSS/NMS data. Avoid calling the current dataset alone a full 5G performance optimization system because the billing data by itself does not contain RF/network-performance measurements.
