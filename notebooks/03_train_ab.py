# Databricks notebook source
# MAGIC %md
# MAGIC # 3. XGBoost + SHAP with MLflow, and the stricter-rule A/B simulation

# COMMAND ----------

# MAGIC %pip install -q xgboost shap openpyxl
# MAGIC %pip install -q --no-deps --force-reinstall /Volumes/workspace/fraud/raw/libs/fraud_lakehouse-0.1.0-py3-none-any.whl
# MAGIC %restart_python

# COMMAND ----------

import json, os, sys
import mlflow
from fraudlake import abtest, config as C, model

S = f"{C.CATALOG}.{C.SCHEMA}"
me = spark.sql("SELECT current_user()").first()[0]
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(f"/Users/{me}/fraud-lakehouse")

cols = ["txn_id", "step", "day", "type", "amount", "is_fraud", "name_dest"] + sorted(set(model.FEATURE_SETS["full"]) - {"amount"})
pdf = spark.table(f"{S}.gold_transactions").where(f"type IN {tuple(C.MODEL_TYPES)}").select(*cols).toPandas()
print(f"{len(pdf):,} TRANSFER/CASH_OUT rows")

fit = model.fit_and_score(pdf, shap_rows=20000, log_mlflow=True, registered_name=f"{S}.fraud_xgb")
spark.createDataFrame(fit["scored"]).write.mode("overwrite").saveAsTable(f"{S}.scored_test")
spark.createDataFrame(fit["explained"]).write.mode("overwrite").saveAsTable(f"{S}.fraud_explanations")

s = fit["scored"]
ab = abtest.simulate(s.txn_id, s.score, s.is_fraud, s.amount, fit["threshold"])
out = {"model": fit["results"], "ab_test": ab, "test_days": int(s.day.nunique()), "gold_rows": len(pdf)}
dbutils.fs.put(f"{C.EXPORT_DIR}/model_ab.json", json.dumps(out, indent=2, default=str), overwrite=True)
print(json.dumps({"full": fit["results"]["full"]["test"], "ab": ab["decision"]}, indent=2))
