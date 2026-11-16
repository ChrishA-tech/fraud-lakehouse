import json

import numpy as np
import pandas as pd
import pytest

from fraudlake import abtest, config as C, excel, kpis, memo, model, transforms as T


@pytest.fixture(scope="module")
def gold(spark, csv_path):
    silver, dq = T.apply_expectations(T.to_silver(T.read_raw(spark, csv_path)))
    g = T.to_gold(silver).cache()
    g.createOrReplaceTempView("gold")
    return g, dq


def test_silver_ids_pii_and_expectations(spark, csv_path):
    raw = T.read_raw(spark, csv_path)
    bad = spark.createDataFrame([(1, "WIRE", 10.0, "C1", 5.0, 0.0, "C2", 0.0, 0.0, 0, 0),
                                 (1, "TRANSFER", -5.0, "C1", 5.0, 0.0, "C2", 0.0, 0.0, 0, 0),
                                 (900, "PAYMENT", 5.0, "C3", 5.0, 0.0, "M4", 0.0, 0.0, 0, 0)], T.RAW_SCHEMA)
    s = T.to_silver(raw.unionByName(bad))
    clean, dq = T.apply_expectations(s)
    assert dq["known_type"] == 1 and dq["amount_valid"] == 1 and dq["step_in_month"] == 1
    assert clean.count() == raw.count() + 1                       # 2 critical failures dropped, warning kept
    r = clean.select("txn_id", "customer_email", "customer_phone", "day", "hour_of_day").first()
    assert r.customer_email.endswith("@example.com") and r.customer_phone.startswith("+44 7700 900")
    assert clean.select("txn_id").distinct().count() == clean.count()
    again = T.to_silver(raw).select("txn_id").orderBy("txn_id").limit(5).collect()
    assert again == T.to_silver(raw).select("txn_id").orderBy("txn_id").limit(5).collect()   # deterministic


def test_gold_window_uses_only_the_past(spark):
    rows = [(s, "TRANSFER", 100.0, f"C{s}", 100.0, 0.0, "C9", 0.0, 100.0, 0, 0) for s in (10, 20, 34, 34, 60)]
    g = T.to_gold(T.to_silver(spark.createDataFrame(rows, T.RAW_SCHEMA))).orderBy("step").collect()
    assert [r.dest_txn_count_24h for r in g] == [0, 1, 2, 2, 0]   # window [step-24, step-1]: same-hour rows and later rows excluded
    assert all(r.dest_first_day == 1 for r in g)


def test_kpi_sql(spark, gold):
    g, _ = gold
    daily = kpis.run(spark, "daily_kpis", gold="gold").toPandas()
    p = g.toPandas()
    p["fv"] = p.amount * p.is_fraud
    ref = p.groupby("day").fv.sum().rolling(7, min_periods=1).sum().round(2).to_numpy()
    assert np.allclose(daily.fraud_value_7d.to_numpy(), ref, atol=0.05 * len(ref))
    assert daily.fraud_value_change_dod.isna().iloc[0]
    mule = kpis.run(spark, "mule_concentration", gold="gold").toPandas()
    assert abs(mule.cumulative_share.iloc[-1] - 1.0) < 1e-6 and mule.value_rank.iloc[0] == 1
    coh = kpis.run(spark, "dest_cohorts", gold="gold").toPandas()
    assert (coh.days_since_first_seen >= 0).all()


def test_daily_kpis_one_row_per_day_even_with_tied_types(spark):
    spark.sql("""CREATE OR REPLACE TEMP VIEW tie_gold AS SELECT * FROM VALUES
                 (1, 'TRANSFER', 100.0D, 1), (1, 'CASH_OUT', 100.0D, 1), (2, 'TRANSFER', 50.0D, 1), (2, 'PAYMENT', 5.0D, 0)
                 AS t(day, type, amount, is_fraud)""")
    d = kpis.run(spark, "daily_kpis", gold="tie_gold").toPandas()
    assert d.day.tolist() == [1, 2] and d.fraud_value_7d.tolist() == [200.0, 250.0]
    assert d.top_fraud_type.tolist() == ["CASH_OUT", "TRANSFER"]          # tie broken alphabetically


def test_model_threshold_shap_and_ab(spark, gold):
    g, _ = gold
    cols = ["txn_id", "step", "day", "type", "amount", "is_fraud", "name_dest"] + sorted(set(model.FEATURE_SETS["full"]) - {"amount"})
    pdf = g.where(g.type.isin(*C.MODEL_TYPES)).select(*cols).toPandas()
    fit = model.fit_and_score(pdf, shap_rows=300)
    full = fit["results"]["full"]["test"]
    assert full["pr_auc"] > 0.8 and full["net_saving_gbp"] > 0
    assert full["fraud_value_caught"] + full["fraud_value_missed"] == pytest.approx(full["fraud_value_lost_without_model"], rel=1e-6)
    assert set(fit["results"]["shap_global_importance"]) == set(model.FEATURE_SETS["full"])
    assert fit["explained"].reason_1.str.contains(r"\(\+|\(-").all()
    s = fit["scored"]
    spark.createDataFrame(s).createOrReplaceTempView("scored")
    md = kpis.run(spark, "model_daily", scored="scored").toPandas()
    assert md.fraud_value_caught.sum() == pytest.approx(full["fraud_value_caught"], rel=1e-3)
    ab = abtest.simulate(s.txn_id, s.score, s.is_fraud, s.amount, fit["threshold"])
    assert ab["design"]["simulation"] and ab["control"]["n"] + ab["treatment"]["n"] == len(s)
    assert ab["ci95"][0] <= ab["difference_mean_cost_gbp"] <= ab["ci95"][1]
    results = {"model": fit["results"], "ab_test": ab, "test_days": int(s.day.nunique())}
    text = memo.render(results)
    assert "Net saving" in text and "£" in text and "simulation" in text


def test_costs_and_sample_size():
    p, y, a = np.array([0.9, 0.2, 0.8, 0.1]), np.array([1, 1, 0, 0]), np.array([100.0, 50.0, 30.0, 20.0])
    c = model.costs(p, y, a, 0.5)
    assert c == {"fraud_value_caught": 100.0, "fraud_value_missed": 50.0, "false_declines": 1, "alerts": 2,
                 "net_saving_gbp": 100.0 - C.FALSE_DECLINE_COST_GBP}
    # textbook: sd=1, delta=0.1 (mde 10% of mean 1), alpha .05, power .8 -> ~1570 per arm
    assert abtest.sample_size_per_arm(1.0, 1.0) == pytest.approx(1570, abs=2)
    arms = abtest.assign([f"t{i}" for i in range(10000)])
    assert 0.47 < arms.mean() < 0.53 and (arms == abtest.assign([f"t{i}" for i in range(10000)])).all()


def test_ab_detects_a_real_difference():
    rng = np.random.default_rng(0)
    n = 40000
    y = (rng.random(n) < 0.02).astype(int)
    score = np.clip(np.where(y == 1, rng.normal(0.45, 0.15, n), rng.normal(0.1, 0.08, n)), 0, 1)
    amount = np.full(n, 1000.0)
    r = abtest.simulate([f"t{i}" for i in range(n)], score, y, amount, thr_control=0.6, factor=0.5)
    assert r["p_value"] < 0.05 and r["difference_mean_cost_gbp"] < 0 and r["decision"].startswith("ship")


def test_excel_and_power_query(tmp_path):
    t = {"daily_kpis": pd.DataFrame({"day": [1, 2, 3], "fraud_value": [1.0, 2.0, 3.0], "fraud_value_7d": [1.0, 3.0, 6.0]}),
         "top_reasons": pd.DataFrame({"txn_id": ["a"], "reason_1": ["amount = 1.00 (+0.50)"]})}
    path = excel.write(t, tmp_path)
    from openpyxl import load_workbook
    wb = load_workbook(path)
    assert wb.sheetnames == ["daily_kpis", "top_reasons"] and wb["daily_kpis"]._charts
    assert "Csv.Document" in (tmp_path / "power_query_daily_kpis.m").read_text()
    assert (tmp_path / "csv" / "daily_kpis.csv").exists()


def test_run_local_end_to_end(csv_path, tmp_path):
    from fraudlake.run_local import run
    r = run(csv_path, out=tmp_path, shap_rows=200, log=lambda *a: None)
    assert (tmp_path / "memo.md").exists() and (tmp_path / "fraud_report.xlsx").exists()
    assert json.loads((tmp_path / "results.json").read_text())["model"]["full"]["test"]["pr_auc"] > 0.8


def test_time_split_by_row_share_handles_front_loaded_steps():
    import numpy as np
    import pandas as pd
    from fraudlake import model
    rng = np.random.default_rng(1)
    step = np.concatenate([rng.integers(1, 400, 9000), rng.integers(400, 744, 300)])   # PaySim-like: dense early
    df = pd.DataFrame({"step": step, "type": "TRANSFER", "is_fraud": (rng.random(len(step)) < 0.05).astype(int)})
    out = model.model_frame(df)
    share = out.split.value_counts(normalize=True)
    assert set(share.index) == {"train", "valid", "test"} and share["train"] > 0.6 and share["test"] > 0.08
    assert out.groupby("split").step.max()["train"] < out.groupby("split").step.min()["valid"]   # time-ordered


def test_ab_guardrail_rejects_rule_that_declines_genuine_payments():
    rng = np.random.default_rng(1)
    n = 20000
    y = (rng.random(n) < 0.01).astype(int)
    score = np.where(y == 1, 0.9, rng.uniform(0.3, 0.5, n))       # genuine payments score between the two thresholds
    r = abtest.simulate([f"t{i}" for i in range(n)], score, y, np.full(n, 5000.0), thr_control=0.6, factor=0.5)
    assert r["guardrail"]["breached"] and r["decision"].startswith("reject the stricter rule")
    assert r["treatment"]["false_decline_rate"] > 0.9 and r["control"]["false_decline_rate"] == 0
