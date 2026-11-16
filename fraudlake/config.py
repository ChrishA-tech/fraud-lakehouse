"""Every assumption that changes a reported number lives here."""
CATALOG, SCHEMA, VOLUME = "workspace", "fraud", "raw"
RAW_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"          # upload the PaySim CSV here on Databricks
EXPORT_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/exports"

TYPES = ("PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN")
MODEL_TYPES = ("TRANSFER", "CASH_OUT")      # PaySim only contains fraud in these two types
# Time split by row share, in step (hour) order: earliest 70% train, next 15% validate, latest 15% test.
# PaySim rows are concentrated in the early steps, so fixed step cut-offs (e.g. 500/600) leave splits empty.
TRAIN_SHARE, VALID_SHARE = 0.70, 0.15
SEED = 7

# Business assumptions (illustrative; PaySim amounts are simulated currency units, reported here as GBP)
FALSE_DECLINE_COST_GBP = 10.0               # cost of wrongly declining a genuine payment (review + customer friction)
AB_STRICTER_FACTOR = 0.5                    # treatment rule: decline at half the control score threshold
AB_MDE = 0.10                               # minimum detectable effect: 10% change in cost per transaction
AB_ALPHA, AB_POWER = 0.05, 0.80
