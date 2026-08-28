# Data Contract — Interview-Safe Schema

No proprietary Vodafone Idea rows are required in this repository. The application can run entirely on synthetic data. If private data is used locally, it should follow the contracts below and remain outside Git.

## A. Raw billing/dispute dataset

Minimum required fields:

| Field | Type | Purpose |
|---|---|---|
| IP Site ID | string | Site identifier |
| Expense Nature | string | EB, DG, Rental, Tax, etc. |
| Actual Bill From Date | date | Start of bill service period |
| Actual Bill To Date | date | End of bill service period |
| Billed Amount (Excl GST) | numeric | Original billed amount |

Recommended fields:

| Field | Type | Purpose |
|---|---|---|
| Month / Record Month | month | Operational record month used for time-based evaluation |
| Circle Code / Circle Name | string | Geography/context |
| Vendor Code / IP Name | string | Infrastructure partner context |
| IP Category | string | Site type/category |
| Invoice No | string | Invoice traceability |
| Invoice Date | date | Invoice timing |
| Debit Amount (Excl tax) | numeric | Disputed/debited value |
| Debit Amount (Incl tax) | numeric | Tax-inclusive debit value |
| Reason for Dispute Category | string | Preferred historical supervised label |
| Dispute Type | string | Legacy/fallback dispute label |
| Provision fields | numeric | Financial provisioning context |
| Debit note | string/date | Dispute-finance lifecycle |
| SAP Document Number | string | Finance posting traceability |

## B. Dispute taxonomy

| Field | Type | Purpose |
|---|---|---|
| Dispute Head | string | EB or DG |
| Dispute Sub-Category | string | Approved candidate dispute reason |

The classifier must never return a DG-only reason for an EB record or vice versa.

## C. Model-ready / prorated dataset

Minimum fields:

| Field | Type |
|---|---|
| IP Site ID | string |
| Expense Nature | string |
| Month-Year | YYYY-MM |
| Prorated Billed Amount (Excl GST) | numeric |
| Prorated Debit Amount (Excl tax) | numeric |
| Prorated Debit Amount (Incl tax) | numeric |

Optional historical labels and source metadata can be retained for training/evaluation.

## D. Optional network KPI dataset

Join grain:

```text
IP Site ID + Month-Year
```

Representative fields:

| Field | Type |
|---|---|
| RSRP (dBm) | numeric |
| RSRQ (dB) | numeric |
| SINR (dB) | numeric |
| DL Throughput (Mbps) | numeric |
| UL Throughput (Mbps) | numeric |
| PRB Utilization (%) | numeric |
| Availability (%) | numeric |
| Alarm Count | numeric |

These fields are optional evidence. Their presence does not automatically prove that a billing dispute was caused by network performance.

## E. Future production-source contracts

A production implementation would normally define governed contracts for:
- site master and tenancy status;
- electricity meter readings;
- DG run hours and fuel records;
- tariff/rate master;
- SLA / contract documents;
- SAP posting/debit-note status;
- site lock/switch-off/radiating status;
- network KPIs and alarm events;
- dispute resolution and reviewer feedback.

## F. Data quality checks

Recommended validation rules:
- IP Site ID cannot be null for scoring records;
- Actual Bill To Date must be on or after Actual Bill From Date;
- billed/debit amounts must parse as numeric;
- Expense Nature must be standardized;
- Month-Year must parse to a valid month;
- historical dispute labels must map to the approved EB/DG taxonomy before supervised training;
- duplicated invoice/site/period records should be investigated before aggregation;
- site-month network KPI joins should be checked for one-to-many duplication;
- train/test chronology should use an operational record period when available rather than retroactive service-period dates.

## G. Data privacy rule for this repository

The following paths are intentionally ignored by Git:

```text
data/raw/*
data/processed/*
data/network/*
data/reference/*.xlsx
data/reference/*.xls
data/reference/*.xlsb
```

Only README/schema files and synthetic/reference-safe assets should be committed.
