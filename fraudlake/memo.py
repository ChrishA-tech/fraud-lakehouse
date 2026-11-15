"""One-page business memo with the GBP impact, generated from the measured results (never hand-written)."""
from __future__ import annotations

from typing import Dict

from . import config as C


def _gbp(x: float) -> str:
    return f"£{x:,.0f}" if abs(x) >= 100 else f"£{x:,.2f}"


def _p(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def _n(n) -> str:
    return f"{n:,}" if n is not None else "n/a (the current rule already costs nothing)"


def render(results: Dict) -> str:
    m = results["model"]["full"]["test"]
    nb = results["model"]["no_balance_errors"]["test"]
    ab = results["ab_test"]
    days = results.get("test_days", "?")
    lost = m["fraud_value_lost_without_model"]
    return f"""# Fraud model: what it is worth

**To:** Head of Payments Risk | **Data:** PaySim simulated mobile-money transactions (amounts shown as GBP for
illustration) | **Period scored:** last {days} days, never used in training

## Recommendation
Deploy the model at the validated threshold ({m['threshold']}). {ab['decision'].capitalize()}.

## Impact in the test period
| | Without model | With model |
|---|---|---|
| Fraud value lost | {_gbp(lost)} | {_gbp(m['fraud_value_missed'])} |
| Fraud value stopped | £0 | {_gbp(m['fraud_value_caught'])} ({m['value_recall']:.1%} of fraud value) |
| Genuine payments wrongly declined | 0 | {m['false_declines']:,} (cost {_gbp(m['false_declines'] * C.FALSE_DECLINE_COST_GBP)} at {_gbp(C.FALSE_DECLINE_COST_GBP)} each) |
| **Net saving** | | **{_gbp(m['net_saving_gbp'])}** |

Alerts raised: {m['alerts']:,}; {m['precision']:.1%} of them were fraud (precision), and {m['recall']:.1%} of fraud
cases were caught (recall). PR-AUC {m['pr_auc']}.

## Stricter rule test (offline A/B simulation)
Transactions were randomly split between the current threshold ({ab['control']['threshold']}) and a stricter one
({ab['treatment']['threshold']}). Mean cost per transaction changed by {_gbp(ab['difference_mean_cost_gbp'])}
(95% CI {_gbp(ab['ci95'][0])} to {_gbp(ab['ci95'][1])}, p = {_p(ab['p_value'])}). The test needed
{_n(ab['design']['required_n_per_arm'])} transactions per arm to detect a {ab['design']['mde']:.0%} change and
{'had enough' if ab['design']['powered'] else 'did NOT have enough'}. The stricter rule raised
{ab['treatment']['false_declines']:,} false declines against {ab['control']['false_declines']:,}.

## Caveats
- PaySim is simulated data. Two balance features nearly separate fraud because of how the simulator books it;
  without them the model's PR-AUC is {nb['pr_auc']} (net saving {_gbp(nb['net_saving_gbp'])}). Expect the
  lower figure on real data until proven otherwise.
- The A/B result is a simulation on known outcomes, not a live experiment. A live test should follow, with the
  same design and sample size.
- The false-decline cost ({_gbp(C.FALSE_DECLINE_COST_GBP)}) is an assumption; the threshold should be re-tuned
  with the real figure.
"""
