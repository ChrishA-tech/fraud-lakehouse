# Fraud model: what it is worth

**To:** Head of Payments Risk | **Data:** PaySim simulated mobile-money transactions (amounts shown as GBP for
illustration) | **Period scored:** last 16 days, never used in training

## Recommendation
Deploy the model at the validated threshold (0.52). Reject the stricter rule: it wrongly declines 100.0% of genuine payments (guardrail 1%).

## Impact in the test period
| | Without model | With model |
|---|---|---|
| Fraud value lost | £6,323,807,770 | £27,654,215 |
| Fraud value stopped | £0 | £6,296,153,556 (99.6% of fraud value) |
| Genuine payments wrongly declined | 0 | 0 (cost £0.00 at £10.00 each) |
| **Net saving** | | **£6,296,153,556** |

Alerts raised: 4,005; 100.0% of them were fraud (precision), and 99.9% of fraud
cases were caught (recall). PR-AUC 0.9995.

## Stricter rule test (offline A/B simulation)
Transactions were randomly split between the current threshold (0.52) and a stricter one
(0.26). Mean cost per transaction changed by £-74.87
(95% CI £-205 to £9.90, p = 0.205). The test needed
155,797,938 transactions per arm to detect a 10% change and
did NOT have enough. The stricter rule raised
202,005 false declines against 0
(100.0% of genuine payments; guardrail 1%).

## Caveats
- PaySim is simulated data. Two balance features nearly separate fraud because of how the simulator books it;
  without them the model's PR-AUC is 0.9936 (net saving £6,322,778,249). Expect the
  lower figure on real data until proven otherwise.
- The A/B result is a simulation on known outcomes, not a live experiment. A live test should follow, with the
  same design and sample size.
- The false-decline cost (£10.00) is an assumption; the threshold should be re-tuned
  with the real figure.
