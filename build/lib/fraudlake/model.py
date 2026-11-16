"""
Fraud model: XGBoost on TRANSFER and CASH_OUT (the only PaySim types with fraud), time-based split, decision
threshold chosen on validation to maximise net savings in GBP, SHAP reasons for every scored transaction.

Two feature sets are compared on purpose: PaySim's balance-error features almost perfectly separate fraud (an
artefact of how the simulator books fraudulent transfers), so the model is also trained without them to show how
much of the score depends on that artefact.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import config as C

BASE = ["amount", "old_balance_orig", "new_balance_orig", "old_balance_dest", "new_balance_dest", "hour_of_day",
        "is_transfer", "dest_txn_count_24h", "dest_amount_24h", "amount_to_orig_balance", "orig_emptied"]
FEATURE_SETS = {"full": BASE + ["error_balance_orig", "error_balance_dest"], "no_balance_errors": BASE}


def model_frame(gold_pdf: pd.DataFrame) -> pd.DataFrame:
    df = gold_pdf[gold_pdf["type"].isin(C.MODEL_TYPES)].copy()
    df["split"] = np.where(df["step"] <= C.TRAIN_MAX_STEP, "train", np.where(df["step"] <= C.VALID_MAX_STEP, "valid", "test"))
    return df


def train(df: pd.DataFrame, features: List[str], seed: int = C.SEED):
    from xgboost import XGBClassifier
    tr, va = df[df.split == "train"], df[df.split == "valid"]
    pos = max(int(tr.is_fraud.sum()), 1)
    m = XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                      min_child_weight=5, scale_pos_weight=(len(tr) - pos) / pos, eval_metric="aucpr",
                      early_stopping_rounds=40, tree_method="hist", random_state=seed, n_jobs=-1)
    m.fit(tr[features], tr.is_fraud, eval_set=[(va[features], va.is_fraud)], verbose=False)
    return m


def costs(p: np.ndarray, y: np.ndarray, amount: np.ndarray, thr: float) -> Dict[str, float]:
    flag = p >= thr
    caught = float(amount[flag & (y == 1)].sum())
    missed = float(amount[~flag & (y == 1)].sum())
    false_declines = int((flag & (y == 0)).sum())
    return {"fraud_value_caught": round(caught, 2), "fraud_value_missed": round(missed, 2),
            "false_declines": false_declines, "alerts": int(flag.sum()),
            "net_saving_gbp": round(caught - C.FALSE_DECLINE_COST_GBP * false_declines, 2)}


def choose_threshold(p: np.ndarray, y: np.ndarray, amount: np.ndarray) -> float:
    grid = np.unique(np.concatenate([np.linspace(0.01, 0.99, 99), np.quantile(p, np.linspace(0.90, 0.9999, 60))]))
    return float(max(grid, key=lambda t: costs(p, y, amount, t)["net_saving_gbp"]))


def evaluate(p: np.ndarray, y: np.ndarray, amount: np.ndarray, thr: float) -> Dict[str, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score
    flag = p >= thr
    tp = int((flag & (y == 1)).sum())
    out = {"n": int(len(y)), "fraud_cases": int(y.sum()), "threshold": round(thr, 4),
           "pr_auc": round(float(average_precision_score(y, p)), 4), "roc_auc": round(float(roc_auc_score(y, p)), 4),
           "precision": round(tp / max(int(flag.sum()), 1), 4), "recall": round(tp / max(int(y.sum()), 1), 4),
           "value_recall": round(float(amount[flag & (y == 1)].sum() / max(amount[y == 1].sum(), 1e-9)), 4)}
    out.update(costs(p, y, amount, thr))
    return out


def shap_reasons(model, X: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """Top-k features pushing each prediction towards (+) or away from (-) fraud, as readable text."""
    import shap
    sv = shap.TreeExplainer(model).shap_values(X)
    sv = sv[1] if isinstance(sv, list) else sv
    cols = np.array(X.columns)
    order = np.argsort(-np.abs(sv), axis=1)[:, :k]
    rows = []
    for i in range(len(X)):
        r = {}
        for j, f in enumerate(order[i], 1):
            r[f"reason_{j}"] = f"{cols[f]} = {X.iat[i, f]:,.2f} ({sv[i, f]:+.2f})"
        rows.append(r)
    out = pd.DataFrame(rows, index=X.index)
    global_imp = pd.Series(np.abs(sv).mean(0), index=cols).sort_values(ascending=False)
    return out, global_imp


def fit_and_score(gold_pdf: pd.DataFrame, shap_rows: int = 20000, log_mlflow: bool = False,
                  registered_name: Optional[str] = None) -> Dict:
    df = model_frame(gold_pdf)
    results, models = {"split_sizes": df.split.value_counts().to_dict(),
                       "fraud_by_split": df.groupby("split").is_fraud.sum().astype(int).to_dict()}, {}
    for name, feats in FEATURE_SETS.items():
        m = train(df, feats)
        va, te = df[df.split == "valid"], df[df.split == "test"]
        pv = m.predict_proba(va[feats])[:, 1]
        thr = choose_threshold(pv, va.is_fraud.to_numpy(), va.amount.to_numpy())
        pt = m.predict_proba(te[feats])[:, 1]
        results[name] = {"features": feats, "best_iteration": int(getattr(m, "best_iteration", -1) or -1),
                         "test": evaluate(pt, te.is_fraud.to_numpy(), te.amount.to_numpy(), thr)}
        models[name] = (m, feats, thr, pt)
        if log_mlflow:
            _log_mlflow(name, m, feats, results[name], te[feats].head(5), registered_name)
    m, feats, thr, pt = models["full"]
    te = df[df.split == "test"].copy()
    te["score"], te["flagged"] = pt, (pt >= thr).astype(int)
    sample = te.sort_values("score", ascending=False).head(shap_rows)
    reasons, imp = shap_reasons(m, sample[feats])
    results["shap_global_importance"] = {k: round(float(v), 4) for k, v in imp.items()}
    scored = te[["txn_id", "step", "day", "type", "amount", "is_fraud", "score", "flagged", "name_dest"]]
    explained = sample[["txn_id", "score", "is_fraud"]].join(reasons)
    no_model = {"fraud_value_lost_without_model": round(float(te.loc[te.is_fraud == 1, "amount"].sum()), 2)}
    results["full"]["test"].update(no_model)
    return {"results": results, "scored": scored, "explained": explained, "threshold": thr}


def _log_mlflow(name, model, feats, res, example, registered_name):
    import mlflow
    with mlflow.start_run(run_name=f"xgboost_{name}"):
        mlflow.log_params({"feature_set": name, "n_features": len(feats), "threshold": res["test"]["threshold"],
                           "false_decline_cost_gbp": C.FALSE_DECLINE_COST_GBP})
        mlflow.log_metrics({k: float(v) for k, v in res["test"].items() if isinstance(v, (int, float))})
        reg = registered_name if (registered_name and name == "full") else None
        try:                                    # MLflow 3 signature
            mlflow.xgboost.log_model(model, name="model", input_example=example, registered_model_name=reg)
        except TypeError:                       # MLflow 2.x
            mlflow.xgboost.log_model(model, artifact_path="model", input_example=example, registered_model_name=reg)
