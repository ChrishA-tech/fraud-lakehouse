"""
Run the whole pipeline locally with PySpark (same transforms, expectations and SQL as on Databricks).

    python -m fraudlake.run_local --data ~/Downloads/PS_20174392719_1491204439457_log.csv
    python -m fraudlake.run_local --data <csv> --sample 0.2      # faster

Writes results/results.json, results/memo.md, results/fraud_report.xlsx, results/csv/*.csv.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from . import abtest, config as C, excel, kpis, memo, model, transforms as T

ROOT = Path(__file__).resolve().parent.parent


def spark_session():
    from pyspark.sql import SparkSession
    return (SparkSession.builder.master("local[*]").appName("fraud-lakehouse")
            .config("spark.ui.enabled", "false").config("spark.sql.shuffle.partitions", "16")
            .config("spark.driver.memory", "6g").getOrCreate())


def run(data: str, sample: float = 1.0, out: Path = ROOT / "results", shap_rows: int = 20000, log=print):
    spark = spark_session()
    spark.sparkContext.setLogLevel("ERROR")
    raw = T.read_raw(spark, data)
    if sample < 1:
        raw = raw.sample(fraction=sample, seed=C.SEED)
    silver, dq = T.apply_expectations(T.to_silver(raw))
    gold = T.to_gold(silver).cache()
    gold.createOrReplaceTempView("gold")
    log(f"gold rows: {gold.count():,}; expectation failures: {dq}")
    tables = {"daily_kpis": kpis.run(spark, "daily_kpis", gold="gold").toPandas(),
              "dest_cohorts": kpis.run(spark, "dest_cohorts", gold="gold").toPandas(),
              "mule_concentration": kpis.run(spark, "mule_concentration", gold="gold").limit(5000).toPandas()}
    cols = ["txn_id", "step", "day", "type", "amount", "is_fraud", "name_dest"] + \
           sorted(set(model.FEATURE_SETS["full"]) - {"amount"})
    pdf = gold.where(gold.type.isin(*C.MODEL_TYPES)).select(*cols).toPandas()
    log(f"training on {len(pdf):,} TRANSFER/CASH_OUT rows ...")
    fit = model.fit_and_score(pdf, shap_rows=shap_rows)
    scored = fit["scored"]
    spark.createDataFrame(scored).createOrReplaceTempView("scored")
    tables["model_daily"] = kpis.run(spark, "model_daily", scored="scored").toPandas()
    ab = abtest.simulate(scored.txn_id, scored.score, scored.is_fraud, scored.amount, fit["threshold"])
    tables["top_reasons"] = fit["explained"].head(500)
    results = {"run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "data": Path(data).name,
               "sample_fraction": sample, "gold_rows": int(gold.count()), "expectation_failures": dq,
               "test_days": int(scored.day.nunique()), "model": fit["results"], "ab_test": ab}
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2, default=str))
    (out / "memo.md").write_text(memo.render(results))
    excel.write(tables, out)
    spark.stop()
    return results


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--sample", type=float, default=1.0)
    a = ap.parse_args(argv)
    r = run(a.data, a.sample)
    print(json.dumps({"full": r["model"]["full"]["test"], "ab_decision": r["ab_test"]["decision"]}, indent=2))


if __name__ == "__main__":
    main()
