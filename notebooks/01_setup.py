# Databricks notebook source
# MAGIC %md
# MAGIC # 1. Setup: catalog objects and raw data check
# MAGIC Creates the `workspace.fraud` schema and two Unity Catalog volumes. Upload the PaySim CSV
# MAGIC (`PS_20174392719_1491204439457_log.csv`) to the `raw` volume before the pipeline runs:
# MAGIC **Catalog > workspace > fraud > Volumes > raw > Upload to this volume**.
# MAGIC
# MAGIC The `fraudlake` package is installed as a wheel from the `raw` volume (`deploy.sh` puts it there): importing
# MAGIC modules from bundle-deployed workspace files can fail with `Errno 5 Input/output error` on some workspaces.

# COMMAND ----------

# MAGIC %pip install -q --no-deps --force-reinstall /Volumes/workspace/fraud/raw/libs/fraud_lakehouse-0.1.0-py3-none-any.whl

# COMMAND ----------

from fraudlake import config as C

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {C.CATALOG}.{C.SCHEMA} COMMENT 'PaySim fraud lakehouse'")
for v in (C.VOLUME, "exports"):
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {C.CATALOG}.{C.SCHEMA}.{v}")

files = [f for f in dbutils.fs.ls(C.RAW_DIR) if f.name.endswith(".csv")]
if not files:
    raise FileNotFoundError(f"No CSV in {C.RAW_DIR}. Upload the PaySim CSV to the 'raw' volume, then re-run.")
print("Raw files:", [(f.name, round(f.size / 1e6, 1)) for f in files], "MB")

# Data contract: fail fast on a truncated file instead of training on a fraction of the data.
n = spark.read.option("header", True).csv(C.RAW_DIR).count()
print(f"Raw rows: {n:,} (expected {C.EXPECTED_RAW_ROWS:,})")
if n + 1 == C.EXCEL_ROW_LIMIT:
    raise ValueError(f"The CSV has {n:,} rows: it was cut at Excel's 1,048,576-row limit. Re-download and unzip it "
                     "in Terminal without opening it in Excel or Numbers.")
if abs(n - C.EXPECTED_RAW_ROWS) > 0.01 * C.EXPECTED_RAW_ROWS:
    raise ValueError(f"Expected about {C.EXPECTED_RAW_ROWS:,} rows, found {n:,}: the raw file is incomplete or not PaySim.")
