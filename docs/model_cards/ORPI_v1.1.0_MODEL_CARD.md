# ORPI Model Card

- Model name: ORPI
- Version: v1.1.0
- Release date: 2026-02-06
- Lifecycle status: Active (comparative model)
- Scope: LEO/MEO/GEO orbital cell ranking

## 1) Purpose and intended users
ORPI provides a comparative orbital pressure score to support insurance underwriting and portfolio triage.
It is designed for ranking and explainability, not event-level conjunction prediction.

Primary users:
- Underwriters and portfolio managers
- Risk engineering analysts
- Product and model governance reviewers

Primary outputs:
- ORPI score (0-100)
- Percentile (0-100)
- Component and subcomponent diagnostics
- Confidence score (0-100)

## 2) Model boundary
In scope:
- Zone-level comparative pressure across altitude/inclination cells
- Stable ranking for decision support
- Scenario-aware trend contribution (declarative)

Out of scope:
- Collision event prediction against named objects
- Maneuver-aware forecasting
- Covariance-level uncertainty propagation

## 3) Inputs and data dependencies
- TLE snapshot: `data/tle/latest.txt`
- Propagated samples: `data/samples/*.npz`
- Feature store: `data/orpi.db`
- Scenario metadata: `data/scenarios/*.yaml`

Minimum data quality assumptions:
- TLE catalog freshness monitored via p90 age.
- Cell coverage sufficient for percentile comparison.
- Latest batch contains complete feature columns.

## 4) Feature definition (cell level)
- `N_eff`: effective occupancy proxy.
- `Vrel_proxy`: relative geometry/velocity proxy.
- `pressure_mean = sum(N_eff * Vrel_proxy)`.
- `risk_sigma`: variability proxy.
- `trend_annual`: annualized scenario delta.

Derived transforms:
- `exposure_raw = log1p(pressure_mean)`
- `congestion_raw = log1p(N_eff)`
- `geometry_raw = Vrel_proxy`
- `volatility_raw = risk_sigma / (pressure_mean + eps)`

## 5) Scoring specification
Top-level score:
`ORPI = w_p*PressureBlock + w_v*VolatilityBlock + w_g*GrowthBlock`

Default weights:
- `w_p = 0.62`
- `w_v = 0.14`
- `w_g = 0.24`

Pressure block:
`PressureBlock = w_exp*Exposure + w_cong*Congestion + w_geo*Geometry`

Pressure sub-weights:
- `w_exp = 0.50`
- `w_cong = 0.10`
- `w_geo = 0.40`

Percentile normalization:
- Each block is converted to percentile on the latest batch distribution.
- Final ORPI is clipped to `[0,100]`.

## 6) Confidence layer (v1.1.0)
Confidence score:
`Confidence = 0.40*Freshness + 0.35*Coverage + 0.25*Stability`

Component definitions:
- Freshness: score from TLE p90 age (best <=3 days, worst >=30 days).
- Coverage: percentile rank of `sample_count_sum` for selected cell.
- Stability: recent inter-batch spread and persistence for the same cell.

Confidence is reported with each ORPI score and must be interpreted with the score.

## 7) Outputs and explainability contract
API response fields:
- `orpi_score`, `percentile`, `rating`
- `components`, `subcomponents`, `drivers`
- `confidence_score`, `confidence.freshness|coverage|stability`
- `underwriting_stance`, `justification`

Explainability requirement:
- Any score must be traceable to a cell, feature values, and component percentiles.

## 8) Validation and monitoring hooks
Validation source:
- `src/validate_science.py`
- `docs/validation_reports/latest.md`
- `docs/validation_cards/ORPI_v1.1.0_VALIDATION_CARD.md`

Core gates monitored:
- Distinguishability across reference orbit pairs.
- Temporal stability on reference cells.
- Rolling mini-backtest across 3-6 snapshots.
- TLE freshness.

## 9) Versioning and governance
- Include `batch_id`, model version, and scenario metadata in all outputs.
- Keep deterministic reruns through `make full-pipeline-offline`.
- Track model changes in versioned model/validation cards.
- Keep validation report history for auditability.

## 10) Known limitations
- No covariance propagation.
- No maneuver modeling.
- No event-level conjunction prediction.
- Scenario growth is declarative and non-guaranteed.
