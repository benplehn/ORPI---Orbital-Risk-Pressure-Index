# ORPI Scientific Validation Report

Generated: 2026-02-06 16:44 UTC
Latest batch: `20260206_164441`
Batches analyzed: 4

## 1. Distinguishability Checks

Target requirement: model should separate 53deg vs 97deg and 550km vs 800km.

| Pair | Delta percentile | Status |
|---|---:|---|
| 550/53 vs 550/97 | 14.74 | PASS |
| 550/53 vs 800/53 | 25.19 | PASS |

### Reference Orbit Scores (latest batch)

| Orbit | ORPI | Percentile | Pressure Pctl | Volatility Pctl | Growth Pctl |
|---|---:|---:|---:|---:|---:|
| A_550_53 | 62.90 | 81.11 | 82.05 | 0.37 | 49.87 |
| B_550_97 | 70.64 | 95.84 | 94.60 | 0.13 | 49.87 |
| C_800_53 | 52.51 | 55.92 | 48.10 | 76.52 | 49.87 |
| D_800_97 | 50.10 | 50.24 | 59.26 | 9.88 | 49.87 |

## 2. Temporal Stability

| Orbit | Samples | ORPI std dev | ORPI range | Last ORPI | Status |
|---|---:|---:|---:|---:|---|
| A_550_53 | 4 | 0.03 | 0.08 | 62.90 | PASS |
| B_550_97 | 4 | 0.01 | 0.03 | 70.64 | PASS |
| C_800_53 | 4 | 0.87 | 1.79 | 52.51 | PASS |

## 3. Rolling Backtest (3-6 snapshots)

| Window size | Windows | Pass rate | Mean ORPI std dev | Mean ORPI range | Worst ORPI range | Worst sep 550/53 vs 550/97 | Worst sep 550/53 vs 800/53 | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 3 | 2 | 100.00% | 0.23 | 0.50 | 1.79 | 14.67 | 20.76 | PASS |
| 4 | 1 | 100.00% | 0.24 | 0.50 | 1.79 | 14.67 | 20.76 | PASS |

### Rolling windows detail

| Window | Batches | Max ORPI range | Min sep 550/53 vs 550/97 | Min sep 550/53 vs 800/53 | Status |
|---|---|---:|---:|---:|---|
| W3 | `20260205_212939, 20260205_221225, 20260206_163319` | 1.79 | 14.67 | 20.76 | PASS |
| W3 | `20260205_221225, 20260206_163319, 20260206_164441` | 1.79 | 14.67 | 20.76 | PASS |
| W4 | `20260205_212939, 20260205_221225, 20260206_163319, 20260206_164441` | 1.79 | 14.67 | 20.76 | PASS |

## 4. Catalog Freshness (TLE)

| Metric | Value |
|---|---:|
| TLE count | 29855 |
| p50 age (days) | 0.54 |
| p90 age (days) | 1.69 |
| max age (days) | 20.97 |
| Freshness gate p90 <= 30d | PASS |

## 5. Gate Summary

| Gate | Value | Threshold | Status |
|---|---:|---:|---|
| Separability 550km/53deg vs 550km/97deg (delta percentile >= 10) | 14.74 | >= 10.00 | PASS |
| Separability 550km/53deg vs 800km/53deg (delta percentile >= 10) | 25.19 | >= 10.00 | PASS |
| TLE freshness p90 <= 30 days | 1.69 | <= 30.00 | PASS |
| Temporal stability A_550_53 ORPI range <= 20 | 0.08 | <= 20.00 | PASS |
| Temporal stability B_550_97 ORPI range <= 20 | 0.03 | <= 20.00 | PASS |
| Temporal stability C_800_53 ORPI range <= 20 | 1.79 | <= 20.00 | PASS |
| Rolling backtest W3 pass rate >= 80% | 100.00 | >= 80.00 | PASS |
| Rolling backtest W4 pass rate >= 80% | 100.00 | >= 80.00 | PASS |

## 6. Notes

- ORPI v1 remains a comparative index, not a conjunction event predictor.
- Growth block depends on declarative scenarios and should be tracked with explicit scenario versioning.
- Future model upgrades should add covariance and maneuver uncertainty models.

