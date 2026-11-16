# Fraud model: what it is worth

**To:** Head of Payments Risk | **Data:** PaySim simulated mobile-money transactions (amounts shown as GBP for
illustration) | **Period scored:** last 3 days, never used in training

## Recommendation
Deploy the model at the validated threshold (0.67). No significant difference: keep the current rule.

## Impact in the test period
| | Without model | With model |
|---|---|---|
| Fraud value lost | £872,023,127 | £1,598,829 |
| Fraud value stopped | £0 | £870,424,298 (99.8% of fraud value) |
| Genuine payments wrongly declined | 0 | 4 (cost £40.00 at £10.00 each) |
| **Net saving** | | **£870,424,258** |

Alerts raised: 646; 99.4% of them were fraud (precision), and 99.7% of fraud
cases were caught (recall). PR-AUC 0.997.

## Stricter rule test (offline A/B simulation)
Transactions were randomly split between the current threshold (0.67) and a stricter one
(0.335). Mean cost per transaction changed by £-53.75
(95% CI £-161 to £0.01, p = 0.257). The test needed
36,340,332 transactions per arm to detect a 10% change and
did NOT have enough. The stricter rule raised
14 false declines against 1.

## Caveats
- PaySim is simulated data. Two balance features nearly separate fraud because of how the simulator books it;
  without them the model's PR-AUC is 0.9928 (net saving £870,423,868). Expect the
  lower figure on real data until proven otherwise.
- The A/B result is a simulation on known outcomes, not a live experiment. A live test should follow, with the
  same design and sample size.
- The false-decline cost (£10.00) is an assumption; the threshold should be re-tuned
  with the real figure.
