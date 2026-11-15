# Databricks notebook source
# MAGIC %md
# MAGIC # 2. Governance with Unity Catalog
# MAGIC - PII lives only in `silver_transactions`; analysts use `gold_transactions` (no PII).
# MAGIC - A separate `customers` table holds the synthetic PII with **column masks**: only members of the
# MAGIC   `pii_readers` group see real values; everyone else sees masked ones.
# MAGIC - Columns are **tagged** as PII; table **grants** separate analysts from PII readers; lineage is recorded
# MAGIC   automatically (Catalog Explorer > table > Lineage).

# COMMAND ----------

S = "workspace.fraud"

def run(sql):
    try:
        spark.sql(sql)
        print("ok  ", sql.split("\n")[0][:110])
    except Exception as e:  # e.g. a group that does not exist on Free Edition: report, keep going
        print("skip", sql.split("\n")[0][:110], "->", str(e).split("\n")[0][:160])

run(f"""CREATE OR REPLACE TABLE {S}.customers AS
SELECT name_orig AS customer_id, first(customer_name) AS customer_name, first(customer_email) AS customer_email,
       first(customer_phone) AS customer_phone, count(*) AS txns, round(sum(amount), 2) AS total_sent
FROM {S}.silver_transactions GROUP BY name_orig""")

run(f"""CREATE OR REPLACE FUNCTION {S}.mask_email(v STRING) RETURNS STRING
RETURN CASE WHEN is_account_group_member('pii_readers') THEN v ELSE regexp_replace(v, '^[^@]+', '***') END""")
run(f"""CREATE OR REPLACE FUNCTION {S}.mask_name(v STRING) RETURNS STRING
RETURN CASE WHEN is_account_group_member('pii_readers') THEN v ELSE concat(left(v, 1), '*** ', left(split(v, ' ')[1], 1), '***') END""")
run(f"""CREATE OR REPLACE FUNCTION {S}.mask_phone(v STRING) RETURNS STRING
RETURN CASE WHEN is_account_group_member('pii_readers') THEN v ELSE concat('*** **** ***', right(v, 3)) END""")

for col, fn in (("customer_email", "mask_email"), ("customer_name", "mask_name"), ("customer_phone", "mask_phone")):
    run(f"ALTER TABLE {S}.customers ALTER COLUMN {col} SET MASK {S}.{fn}")
    run(f"ALTER TABLE {S}.customers ALTER COLUMN {col} SET TAGS ('pii' = '{col.split('_')[1]}')")

run(f"COMMENT ON TABLE {S}.customers IS 'Synthetic PII (fake names, example.com emails, Ofcom drama-range phones). Masked unless in pii_readers.'")
run(f"GRANT SELECT ON TABLE {S}.gold_transactions TO `account users`")
run(f"GRANT SELECT ON TABLE {S}.customers TO `account users`")      # masked view of PII for everyone else
run(f"GRANT SELECT ON TABLE {S}.silver_transactions TO `pii_readers`")

# COMMAND ----------

display(spark.sql(f"SELECT customer_id, customer_name, customer_email, customer_phone FROM {S}.customers LIMIT 5"))
