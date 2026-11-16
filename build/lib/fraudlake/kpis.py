"""
Analytical SQL (Spark SQL / Databricks SQL). Table names are placeholders filled at run time, so the same queries
run on Databricks ({gold} = workspace.fraud.gold_transactions) and locally (temporary views).

daily_kpis        daily volume, fraud rate, fraud value; 7-day rolling sums (ROWS window), day-over-day change (LAG),
                  and which payment type lost most each day (RANK)
dest_cohorts      receiving accounts grouped by the day they first appear; fraud value and active accounts by
                  days since first seen (cohort matrix)
mule_concentration  receiving accounts ranked by fraud value received; cumulative share (Pareto) and NTILE decile
model_daily       daily model performance on the test period: fraud caught vs missed, alerts, rolling 7-day precision
"""
from __future__ import annotations

from typing import Dict

QUERIES: Dict[str, str] = {
    "daily_kpis": """
WITH d AS (
  SELECT day, COUNT(*) AS txns, SUM(is_fraud) AS fraud_txns, ROUND(SUM(amount), 2) AS value,
         ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 2) AS fraud_value
  FROM {gold} GROUP BY day),
t AS (
  SELECT day, type, SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END) AS fv,
         RANK() OVER (PARTITION BY day ORDER BY SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END) DESC) AS r
  FROM {gold} GROUP BY day, type)
SELECT d.day, d.txns, d.fraud_txns, ROUND(d.fraud_txns / d.txns, 6) AS fraud_rate, d.value, d.fraud_value,
       SUM(d.fraud_value) OVER (ORDER BY d.day ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS fraud_value_7d,
       ROUND(SUM(d.fraud_txns) OVER (ORDER BY d.day ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)
             / SUM(d.txns) OVER (ORDER BY d.day ROWS BETWEEN 6 PRECEDING AND CURRENT ROW), 6) AS fraud_rate_7d,
       ROUND(d.fraud_value - LAG(d.fraud_value) OVER (ORDER BY d.day), 2) AS fraud_value_change_dod,
       t.type AS top_fraud_type
FROM d LEFT JOIN t ON d.day = t.day AND t.r = 1 AND t.fv > 0
ORDER BY d.day""",
    "dest_cohorts": """
SELECT dest_first_day AS cohort_day, day - dest_first_day AS days_since_first_seen,
       COUNT(DISTINCT name_dest) AS active_accounts, COUNT(*) AS txns,
       ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 2) AS fraud_value,
       ROUND(AVG(is_fraud), 6) AS fraud_rate
FROM {gold}
WHERE dest_is_merchant = 0
GROUP BY dest_first_day, day - dest_first_day
ORDER BY cohort_day, days_since_first_seen""",
    "mule_concentration": """
WITH r AS (
  SELECT name_dest, COUNT(*) AS fraud_txns, ROUND(SUM(amount), 2) AS fraud_value
  FROM {gold} WHERE is_fraud = 1 GROUP BY name_dest)
SELECT name_dest, fraud_txns, fraud_value,
       DENSE_RANK() OVER (ORDER BY fraud_value DESC) AS value_rank,
       NTILE(10) OVER (ORDER BY fraud_value DESC) AS value_decile,
       ROUND(SUM(fraud_value) OVER (ORDER BY fraud_value DESC, name_dest ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
             / SUM(fraud_value) OVER (), 4) AS cumulative_share
FROM r ORDER BY value_rank, name_dest""",
    "model_daily": """
WITH d AS (
  SELECT day, SUM(flagged) AS alerts, SUM(CASE WHEN flagged = 1 AND is_fraud = 1 THEN 1 ELSE 0 END) AS true_alerts,
         ROUND(SUM(CASE WHEN flagged = 1 AND is_fraud = 1 THEN amount ELSE 0 END), 2) AS fraud_value_caught,
         ROUND(SUM(CASE WHEN flagged = 0 AND is_fraud = 1 THEN amount ELSE 0 END), 2) AS fraud_value_missed
  FROM {scored} GROUP BY day)
SELECT day, alerts, true_alerts, fraud_value_caught, fraud_value_missed,
       ROUND(SUM(true_alerts) OVER w / NULLIF(SUM(alerts) OVER w, 0), 4) AS precision_7d,
       ROUND(SUM(fraud_value_caught) OVER w / NULLIF(SUM(fraud_value_caught + fraud_value_missed) OVER w, 0), 4)
         AS value_recall_7d
FROM d WINDOW w AS (ORDER BY day ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)
ORDER BY day""",
}


def run(spark, name: str, **tables):
    return spark.sql(QUERIES[name].format(**tables))
