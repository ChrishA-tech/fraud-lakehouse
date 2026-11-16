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
