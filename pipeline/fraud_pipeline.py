# Lakeflow Declarative Pipeline (Delta): bronze -> silver (data-quality expectations) -> gold features.
# Transform logic is shared with the local runner and unit tests (fraudlake/transforms.py).
import sys

sys.path.append(spark.conf.get("bundle.sourcePath", "."))  # noqa: F821  (spark is provided by the pipeline)

import dlt  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

from fraudlake import config as C, transforms as T  # noqa: E402


@dlt.table(name="bronze_transactions", comment="Raw PaySim CSV, ingested incrementally with Auto Loader.",
           table_properties={"quality": "bronze"})
def bronze_transactions():
    return (spark.readStream.format("cloudFiles")  # noqa: F821
            .option("cloudFiles.format", "csv").option("header", "true")
            .schema(T.RAW_SCHEMA).load(C.RAW_DIR + "/")
            .withColumn("_source_file", F.col("_metadata.file_path"))
            .withColumn("_ingested_at", F.current_timestamp()))


@dlt.table(name="silver_transactions",
           comment="Cleaned transactions with synthetic PII (governed in Unity Catalog). Critical rules drop rows.",
           table_properties={"quality": "silver", "contains_pii": "true"})
@dlt.expect_all_or_drop(T.CRITICAL)
@dlt.expect_all(T.WARN)
def silver_transactions():
    return T.to_silver(dlt.read_stream("bronze_transactions"))


@dlt.table(name="gold_transactions",
           comment="Analyst-facing features (no PII) with 24-hour receiving-account activity.",
           table_properties={"quality": "gold"})
def gold_transactions():
    return T.to_gold(dlt.read("silver_transactions"))
