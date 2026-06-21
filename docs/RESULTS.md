# Holdout results

The model was trained using target windows before 1 April 2024 and evaluated on 96 rolling two-hour windows from 1–8 April 2024 across 466 active zones.

## Volume model

| Metric | ClearLane | Same window last week | Seven-day average |
|---|---:|---:|---:|
| MAE | 0.436 | 0.478 | 0.523 |
| Rank correlation | 0.314 | 0.250 | 0.216 |
| Weighted capture@10 | 37.6% | 24.8% | 32.8% |
| Weighted capture@25 | 53.0% | 35.3% | 48.5% |
| Weighted capture@50 | 64.6% | 44.3% | 61.3% |

## Impact model

| Metric | ClearLane | Same window last week | Seven-day average |
|---|---:|---:|---:|
| MAE | 0.553 | 0.783 | 0.864 |
| Rank correlation | 0.312 | 0.250 | 0.214 |
| Weighted capture@10 | 35.3% | 22.2% | 28.9% |
| Weighted capture@25 | 51.0% | 33.1% | 44.2% |
| Weighted capture@50 | 62.3% | 41.9% | 57.4% |

## Interpretation

The models improve both aggregate error and operational ranking. With ten teams, the impact model's top ten zones contain 35.3% of observed holdout obstruction units, compared with 22.2% for the same-window-last-week baseline.

These are historical holdout results, not evidence that enforcement itself reduced congestion. A production trial must measure patrol exposure and traffic conditions before and after intervention.
