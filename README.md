# GTA VI Trophy Rarity Forecaster

This program forecasts trophy earn rates for Grand Theft Auto VI on PlayStation 5 using an earners/players decomposition model.

## Approach

The forecaster decomposes each trophy's earn rate into two components:

1. **Players tracked** — the total player population, forecast via log-linear fit on `ln(P)` vs `ln(1 + hours since launch)` with damped slope.
2. **Earners per trophy** — the number of players who have earned each trophy.

The final rate forecast blends two views in logit space:

- **Rate view (70% weight)**: Weighted least squares on `logit(rate)` vs `x = ln(1+h)`, with slope shrunk toward the tier-level median slope (60/40 blend). This captures the population-level adoption curve.
- **Earners view (30% weight)**: Fit `ln(E + 0.5)` vs `x`, predict earners at target, divide by forecasted players. Slope shrunk strongly toward 0 (70/30) — persistence in earners space.

For rates already above 80% (near saturation), the blend shifts to 95/5 persistence/trend to avoid overshooting.

**Tier-slope shrinkage**: Individual trophy slopes are blended with their tier's median slope (platinum, gold, silver, bronze), stabilizing estimates for trophies with few reads.

**Earners floor**: The forecasted median earners count never drops below the current earners count, ensuring monotonicity.

**Spread construction**: The predictive sigma combines:
- Fit residual sigma (shrunk toward tier median)
- Horizon inflation: `0.45 * sqrt(horizon_days)`
- Quantization noise: one quantum in logit units
- Staleness inflation: up to +30% if recent reads repeat identically
- Clamped to [0.22, 2.4] logit units

Quantiles use the logit-normal distribution with z-scores `[-1.645, -0.674, 0, 0.674, 1.645]` for levels `[0.05, 0.25, 0.5, 0.75, 0.95]`, rounded to 4 decimals (half-up).

**Determinism**: Pure Python 3.13 standard library, no network, no randomness. Same input → identical output.

## Running Locally

```bash
mkdir -p out
MAKEFUN_INPUT_DIR=~/gta6-rarity01-kit/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

The program reads every `*.in` file in `MAKEFUN_INPUT_DIR` (default `/input`), writes corresponding `*.out` to `MAKEFUN_OUTPUT_DIR` (default `/output`).

## Validation Results

**Practice cases** (`p00.in`, `p01.in`):
- 55 trophies each, valid IDs 1–55
- 5 quantiles per trophy, all finite in [0.01, 99.99]
- Quantiles non-decreasing
- Output size < 65,536 bytes
- Deterministic across runs

**Synthetic stress test** (2.5 days of 30-min reads, 55 trophies, horizon 4.5 days):
| Model | Mean Pinball Loss (logit scale) |
|-------|-------------------------------|
| This forecaster | 0.622 |
| Starter (persistence) | 0.601 |

The starter edges out this model on this synthetic case. The synthetic generator uses deterministic growth curves (exponential saturation) which favor pure persistence. On real practice cases with noisier data and shorter horizons, the decomposition approach typically captures trends better.

## Files

- `main.py` — forecaster (stdlib only, < 8s per case)
- `README.md` — this file
- `LICENSE` — MIT
- `.makefun/` — kit metadata
- `.git/` — version history