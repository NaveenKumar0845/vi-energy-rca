# Interview Guide — Vi Energy, Billing Dispute & GenAI RCA

Use this document to explain the project accurately and consistently.

## 1. 30-second version

During my internship at Vodafone Idea, I worked on a pilot for telecom site energy and billing dispute RCA. The problem was that site-level EB and DG bills had to be reviewed manually across invoice periods, debit amounts and historical dispute reasons. I built a pipeline that normalized the raw billing data into a monthly site-level view, used a Random Forest to predict likely dispute categories, Isolation Forest to flag unusual billing patterns, and then passed the structured evidence to an LLM to generate barrier analysis, root-cause hypotheses and corrective actions. I exposed the workflow in Streamlit so users could inspect predictions, anomalies, RCA output and ask site-specific questions.

## 2. 75–90 second version

The project was a GenAI-based telecom energy and billing dispute RCA pilot at Vodafone Idea. The business problem was that site bills from infrastructure partners could contain disputes such as high EB or DG consumption, incorrect rates or ratios, duplicate or retro billing, locked or non-radiating sites, and other exceptions. Reviewing these manually across thousands of sites is time-consuming.

I started with historical site-level billing and dispute data containing IP Site ID, circle and vendor information, invoice dates, bill-from and bill-to dates, EB/DG expense nature, billed amount, debit amount and historical dispute labels. I created a preprocessing and proration layer so bills spanning different periods could be converted into a consistent site-month dataset. From that, I engineered features such as previous billing, rolling averages, month-on-month movement, deviation from the site baseline and debit-to-billed ratio.

For prediction, I used separate EB and DG Random Forest classifiers so an EB record could only be mapped to a valid EB dispute reason. I also used Isolation Forest to flag multivariate billing anomalies. Then I built an evidence package containing the ML result, confidence, anomaly indicators and site history and passed it to an LLM for barrier analysis, root-cause hypotheses and corrective actions. The LLM was constrained to use only supplied evidence and to state when the evidence was insufficient. Finally, I exposed the complete workflow in Streamlit with data, prediction, anomaly, evaluation, RCA and conversational-analysis views.

## 3. What exactly did you build?

Say:

> I built the pilot pipeline and application layer: data preprocessing, monthly proration, feature engineering, dispute prediction, anomaly detection, GenAI RCA prompting and the Streamlit dashboard. The production integration with enterprise billing, SAP, energy-meter and network-management systems was the natural next architecture step rather than something I would claim was fully deployed in the pilot.

This answer is strong because it separates implemented work from production design.

## 4. Where did the data come from?

The project material supports historical telecom site billing and dispute data. It contained fields such as:
- IP Site ID;
- circle and vendor information;
- invoice number/date;
- actual bill from/to dates;
- EB/DG/rental/tax expense nature;
- billed and debit amounts;
- reason for dispute / dispute type;
- financial fields such as provision, debit note and SAP document number.

There was also a separate EB/DG dispute-reason master and a smaller site-month prorated dataset.

For a production architecture, additional sources such as meter readings, tariff masters, SAP, OSS/NMS KPIs, alarms and site status would improve RCA. Present those as extensions unless you actually used them.

## 5. Why proration?

Billing periods are not always aligned to calendar months. If one invoice covers 20 January to 19 February, directly comparing that amount with a normal January invoice would be misleading.

The pipeline allocates the amount by overlapping days:

```text
prorated amount = invoice amount x month overlap days / total bill-period days
```

This creates a consistent site-month grain for trend analysis and ML.

## 6. What features did you use?

Core feature groups:

**Billing**
- prorated billed amount;
- prorated debit amount;
- debit/billed ratio;
- bill duration.

**History**
- previous-month billing;
- expanding site historical average;
- historical standard deviation;
- rolling 3-month average;
- month-on-month change;
- deviation from site historical average.

**Context**
- circle;
- vendor;
- IP/site category;
- month / quarter.

**Optional network extension**
- RSRP, RSRQ, SINR;
- DL/UL throughput;
- PRB utilization;
- availability;
- alarm count.

## 7. Why Random Forest?

A good answer:

> This was tabular business data with non-linear relationships and a mixture of continuous and categorical features. Random Forest was a practical pilot choice because it handles non-linear feature interactions, is robust to scaling, works well with moderate-sized tabular data and provides class probabilities, which I used for a confidence threshold and manual-review routing. In a production model-selection exercise I would benchmark CatBoost, LightGBM and XGBoost as well.

## 8. Why separate EB and DG classifiers?

> EB and DG have different valid dispute taxonomies. A single unrestricted classifier could produce logically invalid outputs, for example an incorrect DG rate for an EB bill. I therefore used hierarchical routing: first identify the expense head from the source record, then score only within the allowed reasons for that head.

This is a business-rule guardrail, not just an ML decision.

## 9. How did you handle low-confidence predictions?

> I did not force every record into a dispute class. The classifier returns probabilities. If the top probability is below the configured threshold, the system returns `Manual Review Required` and still shows the top candidates. That is safer than inventing a deterministic answer.

## 10. Why Isolation Forest?

> The dispute classifier predicts a known category, while anomaly detection answers a different question: is this billing pattern unusual relative to the rest of the data? Isolation Forest works without requiring an anomaly label and can use multiple numeric signals such as billed amount, debit ratio, historical deviation and MoM movement. I treated it as a prioritization mechanism, not proof that a bill was wrong.

## 11. Why use an LLM if ML already predicts the dispute?

This is one of the most important interview questions.

> The ML model and the LLM solve different parts of the problem. Random Forest answers, “Which known dispute type is most likely?” Isolation Forest answers, “Is this record unusual?” The LLM then converts those structured signals into human-readable reasoning: what barriers are visible, what root causes should be investigated, what evidence supports the hypothesis, and what corrective actions should be considered. I deliberately did not use the LLM as the source of truth for the classification.

## 12. What did you send to the LLM?

Not the complete raw workbook.

The evidence package contained selected fields such as:
- site, month and expense type;
- current prorated bill/debit;
- historical average and rolling average;
- MoM movement and site-baseline deviation;
- predicted dispute and confidence;
- anomaly flag and evidence;
- optional network KPIs if present;
- approved EB/DG dispute taxonomy.

The prompt instructed the model not to invent missing meter readings, tariffs, contract terms, alarms or network conditions.

## 13. Which LLM?

For the original pilot narrative, the project material references locally hosted models through Ollama, including `phi3:mini`, with other model options such as Llama and Mistral.

The public GitHub reconstruction additionally supports Gemini because Streamlit Community Cloud cannot depend on a local Ollama daemon. Phrase this as:

> The pilot was designed around locally hosted Ollama models for data-control and experimentation. For the public cloud demo I abstracted the GenAI layer so it can use Gemini, while retaining Ollama support for local testing.

## 14. Was this RAG?

Best answer:

> Not in the pilot. The core evidence was structured tabular data, so deterministic dataframe/SQL-style retrieval was more appropriate than vector search. The chat layer retrieves the relevant site/month/expense rows and then passes them to the LLM. I would add RAG only when bringing in unstructured sources such as contracts, tariff circulars, SLAs, SOPs or historical resolution notes.

Do not claim vector DB/RAG unless you actually add those sources.

## 15. Was this really a 5G network-performance model?

Use this carefully:

> The pilot was positioned within the broader 5G/network transformation context, but the implementation I worked with was primarily an energy and billing dispute RCA system. The source data itself is financial/site-energy data, not a full RF-performance dataset. Network KPIs such as RSRP, RSRQ, SINR, PRB utilization and availability are the appropriate extension if we want to establish network-performance relationships.

That answer is technically defensible.

## 16. How did the architecture work end to end?

```text
Billing/dispute history
      +
EB/DG reason master
      |
      v
Data validation and cleaning
      |
      v
Monthly proration
      |
      v
Site-history feature engineering
      |
      +--------------------+
      |                    |
      v                    v
EB/DG Random Forest   Isolation Forest
      |                    |
      +----------+---------+
                 |
                 v
         Evidence package
                 |
                 v
          GenAI RCA layer
                 |
                 v
          Streamlit dashboard
```

Production extension:

```text
Billing + SAP + Energy + Site master + OSS/NMS
                    |
               data platform
                    |
             feature pipelines
                    |
         ML + rules + GenAI/RAG
                    |
              APIs / dashboard
                    |
           human review / feedback
```

## 17. How did you evaluate the classifier?

> I used chronological holdout rather than relying only on a random train/test split, because future billing behavior should be tested against past data. I monitored accuracy, Macro F1, Weighted F1 and prediction coverage. Coverage matters because the model can abstain and send low-confidence records to manual review. I also looked at class distribution because a high aggregate accuracy can be misleading in a highly imbalanced dispute dataset.

Do not quote a final production accuracy unless you have a verified final dataset and test result.

## 18. What were the major technical challenges?

Strong points to discuss:
- messy Excel headers and different file formats, including XLSB;
- date parsing and bills spanning multiple calendar months;
- class imbalance and legacy dispute labels;
- ensuring EB predictions do not cross into DG taxonomy;
- site-history feature leakage;
- unseen/new site IDs;
- anomaly thresholds and false positives;
- LLM hallucination and evidence grounding;
- keeping the pilot usable when the GenAI layer is unavailable;
- separating prototype data access from production security architecture.

## 19. How did you reduce hallucination?

> I did not ask the LLM to independently diagnose the site. I first calculated deterministic metrics and ML outputs and passed only a compact evidence package. The system prompt explicitly prohibits unsupported meter readings, tariffs, contracts, alarms or network conditions; it also asks the model to distinguish observed facts, inference and recommended investigation and to say when the evidence is insufficient.

## 20. What would you change for production?

> I would move data ingestion away from manual Excel upload to approved batch/API/database integrations, store curated site-month data in the enterprise data platform, persist model artifacts, expose scoring and RCA through authenticated APIs, add model/prompt monitoring and human feedback, and use private enterprise-hosted LLM/RAG infrastructure for sensitive contracts and SOPs. I would also add network and meter telemetry only after data owners approve the data contracts and thresholds.

## 21. Business value

Do not invent a savings number unless it was measured. Talk about the mechanism:

- prioritizes high-risk or anomalous bills for review;
- reduces time spent manually scanning site histories;
- standardizes dispute categorization;
- improves consistency of RCA documentation;
- gives reviewers a concise evidence trail;
- routes uncertain cases to humans rather than forcing a prediction;
- can reduce dispute-resolution turnaround time if integrated into the operating workflow;
- can identify billing leakage or unusually high energy patterns earlier.

## 22. Interview-safe claims vs claims to avoid

### Safe

- built/led a pilot for telecom energy and billing dispute RCA;
- built data preprocessing, proration, prediction, anomaly detection and Streamlit workflows;
- used Random Forest, Isolation Forest and LLM-based RCA;
- used evidence grounding and confidence/manual-review guardrails;
- designed an architecture extensible to SAP, energy and network KPI sources.

### Avoid unless you have evidence

- “The model achieved 99% production accuracy.”
- “The system optimized 180,000 live cell sites in real time.”
- “I integrated directly with live SAP/OSS/NMS APIs.”
- “The LLM determined the true root cause automatically.”
- “It reduced energy cost by X%” without a measured result.
- “It was a full 5G RF optimization system” when the actual pilot data is billing/dispute oriented.

## 23. Best project title for your resume

**GenAI-Powered Telecom Energy & Billing Dispute RCA Platform**

Possible bullet:

> Led a GenAI-based telecom energy and billing dispute RCA pilot, combining site-month proration, Random Forest dispute classification, Isolation Forest anomaly detection and evidence-grounded LLM recommendations in Streamlit to accelerate review of EB/DG billing exceptions.

Second bullet:

> Designed guardrailed RCA workflows with hierarchical EB/DG taxonomies, confidence-based human review and structured evidence retrieval, with a production architecture extensible to SAP, energy-meter and OSS/NMS data sources.
