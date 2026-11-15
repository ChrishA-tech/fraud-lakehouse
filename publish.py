#!/usr/bin/env python3
"""
Publish fraud-lakehouse to GitHub, with the results from the Databricks Job if you have them.

    python3 publish.py                         # code only (first push, so Databricks can clone it)
    python3 publish.py --results ~/Downloads/results_bundle.zip

Unzips the Job's results into results/, writes the README results section from results/results.json,
commits, creates the public repo if needed and pushes. Standard library only (works on Python 3.9).
"""
import base64
import getpass
import json
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = "fraud-lakehouse"
DESC = ("Fraud analytics on Databricks: Lakeflow Delta pipeline, Unity Catalog PII masking, analytical SQL, "
        "XGBoost + SHAP, MLflow, A/B test simulation, Excel/Power Query, GBP business memo.")
TOPICS = ["databricks", "delta-lake", "lakehouse", "unity-catalog", "mlflow", "xgboost", "shap", "fraud-detection",
          "ab-testing", "pyspark", "sql"]
AUTHOR = ["-c", "user.name=Chrish Pep Felix A", "-c", "user.email=chrishpepfelixa@gmail.com"]
TRAILER = ("\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n"
           "Claude-Session: https://claude.ai/code/session_01DVp71MJqKBK4sk34js4Jfb")
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def git(*a, check=True):
    r = subprocess.run(["git", *a], cwd=HERE, text=True, capture_output=True)
    if check and r.returncode:
        raise SystemExit(f"git {a[0]} failed: {r.stderr or r.stdout}")
    return r


def api(method, path, token, body=None):
    req = urllib.request.Request("https://api.github.com" + path, method=method,
                                 data=json.dumps(body).encode() if body is not None else None)
    for k, v in {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                 "Content-Type": "application/json", "User-Agent": "fraud-lakehouse-publish"}.items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def gbp(x):
    return f"£{x:,.0f}"


def results_md(r):
    m, nb, ab = r["model"]["full"]["test"], r["model"]["no_balance_errors"]["test"], r["ab_test"]
    L = [f"Run {r.get('run_at', '')[:10]} on {r.get('platform', 'Databricks')}; test period = last "
         f"{r.get('test_days', '?')} days (never used for training or threshold choice).", "",
         "| Model (test period) | PR-AUC | Precision | Recall | Fraud value stopped | False declines | Net saving |",
         "|---|---|---|---|---|---|---|"]
    for name, t in (("XGBoost, all features", m), ("XGBoost, without balance-error artefact", nb)):
        L.append(f"| {name} | {t['pr_auc']} | {t['precision']:.3f} | {t['recall']:.3f} | {t['value_recall']:.1%} | "
                 f"{t['false_declines']:,} | {gbp(t['net_saving_gbp'])} |")
    L += ["", f"**Stricter rule (offline A/B simulation):** mean cost per transaction changed by "
              f"£{ab['difference_mean_cost_gbp']:+.4f} (95% CI £{ab['ci95'][0]:+.4f} to £{ab['ci95'][1]:+.4f}, "
              f"p {'< 0.001' if ab['p_value'] < 0.001 else '= ' + format(ab['p_value'], '.3f')}); "
              f"{format(ab['design']['required_n_per_arm'], ',') if ab['design']['required_n_per_arm'] is not None else 'n/a'} transactions per arm needed for a "
              f"{ab['design']['mde']:.0%} effect ({'met' if ab['design']['powered'] else 'not met'}). Decision: "
              f"{ab['decision']}."]
    dq = r.get("expectation_failures") or {}
    if dq and "unavailable" not in dq:
        L += ["", "**Data quality (Lakeflow expectations):** " + ", ".join(f"{k} {v:,}" for k, v in dq.items()) + "."]
    top = list((r["model"].get("shap_global_importance") or {}).items())[:3]
    if top:
        L += ["", "**Top SHAP drivers:** " + ", ".join(f"{k} ({v})" for k, v in top) + "."]
    return "\n".join(L)


args = sys.argv[1:]
zip_path = Path(args[args.index("--results") + 1]).expanduser() if "--results" in args else None
if zip_path:
    if not zip_path.exists():
        raise SystemExit(f"{zip_path} not found")
    shutil.rmtree(HERE / "results", ignore_errors=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(HERE)
    r = json.loads((HERE / "results" / "results.json").read_text())
    s = (HERE / "README.md").read_text()
    (HERE / "README.md").write_text(s.split(START)[0] + START + "\n" + results_md(r) + "\n" + END + s.split(END)[1])
    print("results added to README")

if not (HERE / ".git").exists():
    git("init", "-q", "-b", "main")
git("add", "-A")
if git("diff", "--cached", "--quiet", check=False).returncode:
    git(*AUTHOR, "commit", "-q", "-m", ("Databricks Job results" if zip_path else "Fraud lakehouse on Databricks") + TRAILER)

token = getpass.getpass("Paste GitHub token (hidden): ").strip()
s, user = api("GET", "/user", token)
if s != 200:
    raise SystemExit(f"Token rejected ({s}): {user.get('message')}")
login = user["login"]
print(f"Authenticated as {login}")
if input(f"Create/update PUBLIC repo {login}/{REPO} and push? [y/N]: ").strip().lower() != "y":
    raise SystemExit("Stopped. Nothing was pushed.")
s, _ = api("GET", f"/repos/{login}/{REPO}", token)
if s == 404:
    s, info = api("POST", "/user/repos", token, {"name": REPO, "private": False, "description": DESC})
    if s not in (200, 201):
        raise SystemExit(f"Could not create repo ({s}): {info.get('message')}")
api("PUT", f"/repos/{login}/{REPO}/topics", token, {"names": TOPICS})
hdr = base64.b64encode(f"x-access-token:{token}".encode()).decode()
r = subprocess.run(["git", "-c", f"http.https://github.com/.extraheader=AUTHORIZATION: basic {hdr}", "push", "-q",
                    f"https://github.com/{login}/{REPO}.git", "main"], cwd=HERE, text=True, capture_output=True)
if r.returncode:
    raise SystemExit("git push failed:\n" + (r.stderr or r.stdout).replace(token, "<hidden>"))
print(f"pushed https://github.com/{login}/{REPO}\nDelete the token at github.com/settings/tokens.")
