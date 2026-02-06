# ORPI Validation Card

- Model name: ORPI
- Version: v1.1.0
- Date: 2026-02-06
- Validation script: `src/validate_science.py`
- Latest report: `docs/validation_reports/latest.md`

## 1) Validation scope
This card defines the current automated acceptance protocol for ORPI v1.1.0.

Validation dimensions:
- Distinguishability (reference orbit separation)
- Temporal stability (reference-cell drift)
- Rolling backtest stability (windows of 3 to 6 snapshots when available)
- Catalog freshness (TLE epoch age)

## 2) Acceptance gates
- Distinguishability gate A:
  - `|Percentile(550/53) - Percentile(550/97)| >= 10`
- Distinguishability gate B:
  - `|Percentile(550/53) - Percentile(800/53)| >= 10`
- Freshness gate:
  - `p90(TLE age days) <= 30`
- Reference stability gates:
  - For selected cells, ORPI range over recent batches `<= 20`
- Rolling backtest gates:
  - For each rolling window size `W` in `[3..6]` (if enough snapshots),
    pass-rate of windows satisfying:
    - `max cell ORPI range <= 20`
    - `min separability A >= 10`
    - `min separability B >= 10`
  - Acceptance: `rolling_pass_rate(W) >= 80%`

## 3) Current status (latest report)
- Distinguishability gate A: PASS
- Distinguishability gate B: PASS
- Freshness gate: PASS
- Reference stability gates: PASS
- Rolling backtest gates: PASS (when window available)

Source of truth:
- `docs/validation_reports/latest.md`

## 4) Method notes
- Validation is deterministic given a fixed input snapshot.
- Scores are comparative (percentile-based), robust to outliers.
- Confidence scoring is validated indirectly through freshness/coverage/stability signals.
- Rolling backtests are computed on consecutive historical batches in chronological order.

## 5) Residual risks
- Validation set is currently synthetic/reference-cell based, not event-labeled.
- No calibration against conjunction truth data yet.
- Scenario sensitivity depends on declarative scenario quality.
- Batch count may be limited early in project life, reducing backtest coverage.

## 6) Planned upgrades
- Expand validation set to larger orbit grids and time windows.
- Add out-of-time backtesting against historical snapshots.
- Add drift monitoring dashboards for ORPI + confidence components.
- Add calibration against conjunction and claims-related proxy labels.
