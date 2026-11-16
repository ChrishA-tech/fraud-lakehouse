# Databricks notebook source
# MAGIC %md
# MAGIC # 4. Analytical SQL KPIs, Excel report, business memo
# MAGIC Saves KPI tables, then writes `results_bundle.zip` (results.json, memo.md, fraud_report.xlsx, CSVs, Power
# MAGIC Query script) to the `exports` volume. Download it from Catalog Explorer for the GitHub repo.

# COMMAND ----------

# MAGIC %pip install -q openpyxl
# MAGIC %pip install -q --no-deps --force-reinstall /Volumes/workspace/fraud/raw/libs/fraud_lakehouse-0.1.0-py3-none-any.whl
# MAGIC %restart_python

# COMMAND ----------

import io, json, os, shutil, sys, tempfile, zipfile
from datetime import datetime, timezone
from pathlib import Path
from fraudlake import config as C, excel, kpis, memo

S = f"{C.CATALOG}.{C.SCHEMA}"
tables = {}
for name, args in (("daily_kpis", {"gold": f"{S}.gold_transactions"}), ("dest_cohorts", {"gold": f"{S}.gold_transactions"}),
                   ("mule_concentration", {"gold": f"{S}.gold_transactions"}), ("model_daily", {"scored": f"{S}.scored_test"})):
    df = kpis.run(spark, name, **args)
    df.write.mode("overwrite").saveAsTable(f"{S}.kpi_{name}")
    tables[name] = (df.limit(5000) if name == "mule_concentration" else df).toPandas()
tables["top_reasons"] = spark.table(f"{S}.fraud_explanations").orderBy("score", ascending=False).limit(500).toPandas()

try:   # data-quality results recorded by the pipeline's expectations
    dq = {r["name"]: int(r["failed"] or 0) for r in spark.sql(f"""
      SELECT e.name, sum(e.failed_records) AS failed FROM (
        SELECT explode(from_json(details:flow_progress.data_quality.expectations,
               'array<struct<name:string,dataset:string,passed_records:long,failed_records:long>>')) AS e
        FROM event_log(TABLE({S}.silver_transactions))
        WHERE event_type = 'flow_progress' AND origin.update_id = (      -- latest pipeline update only
          SELECT origin.update_id FROM event_log(TABLE({S}.silver_transactions))
          WHERE event_type = 'create_update' ORDER BY timestamp DESC LIMIT 1)) GROUP BY e.name""").collect()}
except Exception as e:
    dq = {"unavailable": str(e).split(chr(10))[0][:200]}
res = json.loads(dbutils.fs.head(f"{C.EXPORT_DIR}/model_ab.json", 10_000_000))
res.update({"run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "platform": "Databricks serverless (trial workspace)",
            "expectation_failures": dq, "data": "PaySim (Kaggle, CC BY-SA 4.0)"})

tmp = Path(tempfile.mkdtemp()) / "results"
tmp.mkdir()
(tmp / "results.json").write_text(json.dumps(res, indent=2, default=str))
(tmp / "memo.md").write_text(memo.render(res))
excel.write(tables, tmp)
zpath = Path(tempfile.mkdtemp()) / "results_bundle.zip"
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
    for p in tmp.rglob("*"):
        z.write(p, p.relative_to(tmp.parent))
shutil.copy(zpath, f"{C.EXPORT_DIR}/results_bundle.zip")
print(memo.render(res))
