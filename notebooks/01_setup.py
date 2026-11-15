# Databricks notebook source
# MAGIC %md
# MAGIC # 1. Setup: catalog objects and raw data check
# MAGIC Creates the `workspace.fraud` schema and two Unity Catalog volumes. Upload the PaySim CSV
# MAGIC (`PS_20174392719_1491204439457_log.csv`) to the `raw` volume before the pipeline runs:
# MAGIC **Catalog > workspace > fraud > Volumes > raw > Upload to this volume**.

# COMMAND ----------

import os, sys
sys.path.append(os.path.dirname(os.getcwd()))
from fraudlake import config as C

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {C.CATALOG}.{C.SCHEMA} COMMENT 'PaySim fraud lakehouse'")
for v in (C.VOLUME, "exports"):
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {C.CATALOG}.{C.SCHEMA}.{v}")

files = [f for f in dbutils.fs.ls(C.RAW_DIR) if f.name.endswith(".csv")]
if not files:
    raise FileNotFoundError(f"No CSV in {C.RAW_DIR}. Upload the PaySim CSV to the 'raw' volume, then re-run.")
print("Raw files:", [(f.name, round(f.size / 1e6, 1)) for f in files], "MB")
