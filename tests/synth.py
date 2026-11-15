"""PaySim-like synthetic transactions for tests (same columns; fraud = TRANSFER/CASH_OUT that empty the sender)."""
import numpy as np
import pandas as pd

TYPES = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]


def make(n: int = 30000, fraud_rate: float = 0.02, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    step = rng.integers(1, 744, n)
    typ = rng.choice(TYPES, n, p=[0.34, 0.08, 0.35, 0.01, 0.22])
    old_o = np.round(rng.lognormal(10, 1.2, n), 2)
    amount = np.round(np.minimum(rng.lognormal(9, 1.3, n), old_o * rng.uniform(0.05, 0.9, n)), 2)
    fraud = (np.isin(typ, ["TRANSFER", "CASH_OUT"]) & (rng.random(n) < fraud_rate / 0.43)).astype(int)
    amount = np.where(fraud == 1, old_o, amount)                      # fraud empties the account
    new_o = np.round(np.where(fraud == 1, 0.0, old_o - amount), 2)
    dests = np.array([f"C{rng.integers(1e8, 1e9)}" for _ in range(n // 20)])
    mules = dests[: max(len(dests) // 50, 3)]
    name_dest = np.where(fraud == 1, rng.choice(mules, n),
                         np.where(typ == "PAYMENT", [f"M{x}" for x in rng.integers(1e8, 1e9, n)], rng.choice(dests, n)))
    old_d = np.round(rng.lognormal(10, 1.5, n), 2) * (typ != "PAYMENT")
    new_d = np.round(np.where(fraud == 1, old_d, old_d + amount * (typ != "PAYMENT")), 2)  # fraud not booked to dest
    return pd.DataFrame({"step": step, "type": typ, "amount": amount, "nameOrig": [f"C{x}" for x in rng.integers(1e8, 1e9, n)],
                         "oldbalanceOrg": old_o, "newbalanceOrig": new_o, "nameDest": name_dest, "oldbalanceDest": old_d,
                         "newbalanceDest": new_d, "isFraud": fraud, "isFlaggedFraud": 0})
