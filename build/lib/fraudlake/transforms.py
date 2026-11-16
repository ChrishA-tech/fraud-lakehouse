"""
PySpark transformations shared by the Lakeflow pipeline (pipeline/fraud_pipeline.py) and the local runner.

bronze  raw PaySim CSV, typed
silver  snake_case columns, deterministic transaction id, day/hour, balance-error features, merchant flags,
        synthetic PII (for the governance demo; PaySim has none), data-quality expectations applied
gold    per-transaction features with 24-hour rolling activity of the receiving account (window functions)
"""
from __future__ import annotations

from typing import Dict, Tuple

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType

RAW_SCHEMA = StructType([
    StructField("step", IntegerType()), StructField("type", StringType()), StructField("amount", DoubleType()),
    StructField("nameOrig", StringType()), StructField("oldbalanceOrg", DoubleType()),
    StructField("newbalanceOrig", DoubleType()), StructField("nameDest", StringType()),
    StructField("oldbalanceDest", DoubleType()), StructField("newbalanceDest", DoubleType()),
    StructField("isFraud", IntegerType()), StructField("isFlaggedFraud", IntegerType())])

# Lakeflow expectations: name -> SQL condition. CRITICAL rows are dropped; WARN rows are kept and counted.
CRITICAL = {"has_account_ids": "name_orig IS NOT NULL AND name_dest IS NOT NULL",
            "amount_valid": "amount IS NOT NULL AND amount >= 0",
            "known_type": "type IN ('PAYMENT','TRANSFER','CASH_OUT','DEBIT','CASH_IN')",
            "label_binary": "is_fraud IN (0, 1)"}
WARN = {"step_in_month": "step BETWEEN 1 AND 744",
        "balances_non_negative": "old_balance_orig >= 0 AND old_balance_dest >= 0"}

_FIRST = ["Olivia", "Amelia", "Isla", "Ava", "Noah", "Oliver", "George", "Arthur", "Priya", "Aarav", "Mia", "Leo",
          "Freya", "Hamza", "Zara", "Ethan"]
_LAST = ["Smith", "Jones", "Taylor", "Brown", "Williams", "Wilson", "Patel", "Khan", "Evans", "Thomas", "Roberts",
         "Walker", "Wright", "Hughes", "Edwards", "Green"]


def read_raw(spark, path: str) -> DataFrame:
    return spark.read.option("header", True).schema(RAW_SCHEMA).csv(path)


def to_silver(df: DataFrame) -> DataFrame:
    s = (df.select(F.col("step"), F.col("type"), F.col("amount"),
                   F.col("nameOrig").alias("name_orig"), F.col("oldbalanceOrg").alias("old_balance_orig"),
                   F.col("newbalanceOrig").alias("new_balance_orig"), F.col("nameDest").alias("name_dest"),
                   F.col("oldbalanceDest").alias("old_balance_dest"), F.col("newbalanceDest").alias("new_balance_dest"),
                   F.col("isFraud").alias("is_fraud"), F.col("isFlaggedFraud").alias("is_flagged_by_rule"))
         .withColumn("txn_id", F.sha2(F.concat_ws("|", "step", "type", "amount", "name_orig", "name_dest",
                                                  "old_balance_orig", "old_balance_dest"), 256).substr(1, 20))
         .withColumn("day", F.floor((F.col("step") - 1) / 24).cast("int") + 1)
         .withColumn("hour_of_day", ((F.col("step") - 1) % 24).cast("int"))
         .withColumn("error_balance_orig", F.col("new_balance_orig") + F.col("amount") - F.col("old_balance_orig"))
         .withColumn("error_balance_dest", F.col("old_balance_dest") + F.col("amount") - F.col("new_balance_dest"))
         .withColumn("dest_is_merchant", F.col("name_dest").startswith("M").cast("int")))
    return add_synthetic_pii(s)


def add_synthetic_pii(df: DataFrame) -> DataFrame:
    """Deterministic FAKE name/email/phone per sender, so masking can be demonstrated. Emails use the reserved
    example.com domain and phones the Ofcom drama range (07700 900000-900999): none can belong to a real person."""
    h = F.abs(F.xxhash64("name_orig"))
    first = F.element_at(F.array(*[F.lit(x) for x in _FIRST]), (h % len(_FIRST)).cast("int") + 1)
    last = F.element_at(F.array(*[F.lit(x) for x in _LAST]), ((h / len(_FIRST)) % len(_LAST)).cast("int") + 1)
    num = (h % 1000).cast("int")
    return (df.withColumn("customer_name", F.concat_ws(" ", first, last))
              .withColumn("customer_email", F.lower(F.concat(first, F.lit("."), last, num.cast("string"), F.lit("@example.com"))))
              .withColumn("customer_phone", F.concat(F.lit("+44 7700 900"), F.lpad(num.cast("string"), 3, "0"))))


def apply_expectations(df: DataFrame) -> Tuple[DataFrame, Dict[str, int]]:
    """Local equivalent of Lakeflow @expect_all_or_drop(CRITICAL) + @expect_all(WARN)."""
    aggs = [F.sum(F.when(~F.expr(c), 1).otherwise(0)).alias(n) for n, c in {**CRITICAL, **WARN}.items()]
    counts = {k: int(v or 0) for k, v in df.agg(*aggs).first().asDict().items()}
    keep = " AND ".join(f"({c})" for c in CRITICAL.values())
    return df.filter(F.expr(keep)), counts


PII_COLUMNS = ("customer_name", "customer_email", "customer_phone")


def to_gold(silver: DataFrame) -> DataFrame:
    """Transaction features plus the receiving account's activity in the previous 24 hours (no future leakage:
    the window ends one hour before the transaction's step)."""
    w = Window.partitionBy("name_dest").orderBy("step").rangeBetween(-24, -1)
    w_first = Window.partitionBy("name_dest")
    return (silver.drop(*PII_COLUMNS)                    # PII stays in silver; analysts use gold
            .withColumn("dest_txn_count_24h", F.count("*").over(w))
            .withColumn("dest_amount_24h", F.coalesce(F.sum("amount").over(w), F.lit(0.0)))
            .withColumn("dest_first_day", F.min("day").over(w_first))
            .withColumn("amount_to_orig_balance", F.when(F.col("old_balance_orig") > 0,
                                                         F.col("amount") / F.col("old_balance_orig")).otherwise(F.lit(None)))
            .withColumn("orig_emptied", ((F.col("old_balance_orig") > 0) & (F.col("new_balance_orig") == 0)).cast("int"))
            .withColumn("is_transfer", (F.col("type") == "TRANSFER").cast("int")))
