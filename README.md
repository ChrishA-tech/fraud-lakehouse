# Fraud Lakehouse on Databricks

End-to-end fraud analytics on **Databricks**: a Lakeflow pipeline builds bronze, silver and gold **Delta** tables
with data-quality expectations; **Unity Catalog** governs PII with column masks, tags and grants; analytical
**SQL** produces KPIs, cohorts and concentration analysis; an **XGBoost** model explains every decision with
**SHAP** and is tracked and registered with **MLflow**; an **A/B simulation** tests whether a stricter fraud rule
reduces losses; and the results become an **Excel** report (with a Power Query script) and a one-page **business
memo** with the impact in GBP. One **Job** runs it all.

```mermaid
flowchart LR
  CSV[PaySim CSV in UC volume] -->|Auto Loader| B[bronze_transactions]
  B -->|expectations: drop/warn| S[silver_transactions<br/>synthetic PII]
  S --> G[gold_transactions<br/>features, no PII]
  S --> CU[customers<br/>column masks + PII tags]
  G --> M[XGBoost + SHAP<br/>MLflow, UC registry]
  M --> SC[scored_test] --> AB[A/B simulation<br/>power, Welch, bootstrap CI]
  G --> K[KPI SQL<br/>windows, cohorts, Pareto]
  SC --> K
  K --> X[Excel + Power Query]
  AB --> MEMO[Business memo, GBP]
  M --> MEMO
```

## Results

<!-- RESULTS:START -->
Run 2026-11-16 on Databricks serverless (trial workspace); test period = last 16 days (never used for training or threshold choice).

Data: 6,362,620 gold rows; 2,770,409 TRANSFER/CASH_OUT rows modelled; time split at step 323 / 377 (train / validation / test = 1,951,895 / 410,437 / 408,077).

| Model (test period) | PR-AUC | Precision | Recall | Fraud value stopped | False declines | Net saving |
|---|---|---|---|---|---|---|
| XGBoost, all features | 0.9995 | 1.000 | 0.999 | 99.6% | 0 | £6,296,153,556 |
| XGBoost, without balance-error artefact | 0.9936 | 0.915 | 0.994 | 100.0% | 372 | £6,322,778,249 |

**Stricter rule (offline A/B simulation):** mean cost per transaction changed by £-74.8714 (95% CI £-204.7116 to £+9.9025, p = 0.205); 155,797,938 transactions per arm needed for a 10% effect (not met); stricter rule wrongly declined 100.0% of genuine payments (guardrail 1%). Decision: reject the stricter rule: it wrongly declines 100.0% of genuine payments (guardrail 1%).

PaySim is simulated: balance fields reveal fraud almost perfectly (even without the two balance-error features, old/new balances and `orig_emptied` remain). Treat these scores as an upper bound, not an estimate for real payments; the GBP figures are PaySim amounts relabelled for illustration.

**Top SHAP drivers:** error_balance_orig (0.186), old_balance_orig (0.0597), amount (0.0304).
<!-- RESULTS:END -->

Memo: [`results/memo.md`](results/memo.md) · Excel: [`results/fraud_report.xlsx`](results/fraud_report.xlsx) ·
Power Query: [`results/power_query_daily_kpis.m`](results/power_query_daily_kpis.m)

## What each part shows

| Skill | Where | What it does |
|---|---|---|
| Databricks: Delta, Lakeflow, Jobs | `pipeline/fraud_pipeline.py`, `databricks.yml` | Auto Loader ingestion; bronze, silver and gold Delta tables; one Job runs setup, pipeline, governance, training, A/B test and report |
| Data quality | `fraudlake/transforms.py` (`CRITICAL`, `WARN`) | Lakeflow expectations: critical failures dropped, warnings counted; failure counts read from the pipeline event log into the results |
| Data governance | `notebooks/02_governance.py` | PII isolated in silver; `customers` table with column masks (`pii_readers` see values, others see `***`), PII column tags, grants; lineage in Catalog Explorer |
| Analytical SQL | `fraudlake/kpis.py` | Window functions (7-day rolling sums, `LAG`, `RANK`, `NTILE`, running Pareto share), receiving-account cohorts, model performance by day |
| XGBoost + SHAP | `fraudlake/model.py` | Time-based split, cost-based threshold, top-3 reasons for every scored transaction, global importance; a second model without PaySim's balance-error artefact |
| MLflow | `notebooks/03_train_ab.py` | Params, metrics and models for both feature sets; the main model registered in Unity Catalog (`workspace.fraud.fraud_xgb`) |
| Statistics, A/B testing | `fraudlake/abtest.py` | Power analysis (sample size for a 10% effect), stable random assignment, Welch t-test, bootstrap CI, decision rule |
| Excel / Power Query | `fraudlake/excel.py` | Workbook with KPI sheets and a chart; Power Query M script that rebuilds the daily KPIs from the CSV exports |
| Business "so what" | `fraudlake/memo.py` | One-page memo generated from the results: GBP impact, recommendation, caveats |

The same transforms, SQL and model code run locally with PySpark (`python -m fraudlake.run_local --data <csv>`)
and are covered by tests on PaySim-like synthetic data (`pytest`, needs Java).

## Run it on Databricks Free Edition

1. Sign up at **databricks.com/learn/free-edition** (free, no card).
2. Download PaySim from Kaggle (`ealaxi/paysim1`, the `PS_2017...log.csv` file).
3. In the workspace: **Workspace > Create > Git folder**, URL `https://github.com/ChrishA-tech/fraud-lakehouse`.
   Run `notebooks/01_setup` once (creates the schema and volumes), then upload the CSV to
   **Catalog > workspace > fraud > Volumes > raw**.
4. Install the Databricks CLI, `databricks auth login --host <workspace-url>`, then `./deploy.sh`: it builds the
   `fraudlake` wheel, uploads it to the `raw` volume (notebooks and the pipeline install it from there), deploys
   the bundle and runs the job.
5. Download `exports/results_bundle.zip` from the volume and run `python3 publish.py` to add the results here.

## Honest limits

- **PaySim is simulated** (Lopez-Rojas et al., 2016; Kaggle, CC BY-SA 4.0). Amounts are simulated currency, shown
  as GBP for illustration. The balance-error features nearly separate fraud because of how the simulator books it,
  so the model is also reported without them.
- **The A/B test is an offline simulation** on known outcomes, not a live experiment; the memo says so.
- **PII is synthetic**: fake names, `example.com` emails and Ofcom drama-range phone numbers, added to demonstrate
  masking. PaySim contains no personal data.
- Free Edition limits (one SQL warehouse, one pipeline per type, five concurrent job tasks) shape the design;
  the job fits within them.
