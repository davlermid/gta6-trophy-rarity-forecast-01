[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/davlermid/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/davlermid/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# GTA VI Trophy Rarity Forecaster

Forecasts five quantiles (5/25/50/75/95%) of every GTA VI PS5 trophy's Exophase earn rate at a case's target, from the history of page reads in the case. Scored by mean pinball loss on the logit scale.

## Method (v2)

Exophase refreshes its counts in bursts: long stretches of identical reads, then jumps. So the model first extracts REFRESH POINTS (reads where players_tracked or any rate changed) and fits everything on those only; stale repeats carry no information and must not shrink the spread.

- Players: power-law fit of ln(players) vs ln(1+hours since launch) on recent (48h) refresh points, slope damped, anchored at the current count. With too few refresh points, a gamma=1 power-law prior from the last refresh snapshot (a stale snapshot understates the current player count).
- Earners per trophy: a hazard model. The per-trophy hazard lambda is estimated from refresh intervals (dE = lambda*(P-E)*dt, midpoint pool, recent intervals preferred), shrunk toward the tier median, then simulated forward along the players forecast curve. This saturates naturally and handles dilution (new players lower rates) honestly. Trophies at >=99% are treated as instant (new players earn them immediately).
- With no estimable hazard (flat/stale history), two priors blend by horizon: a beta prior where earners lag player growth (short horizons, dilution dominates) and a tier hazard prior with a 48h reachability ramp (long horizons, earners catch up).
- The earners view is blended 85/15 in logit space with a logit-rate trend view (tier-shrunk slope, damped extrapolation). An earners floor/ceiling keeps the median physical: earners never decrease and never exceed players.
- Spread (logit sigma): fit residuals + horizon growth + rate quantization + a staleness/refresh-jump term proportional to log player growth when the snapshot is stale + an information term that widens when few refresh points exist. Trophies with almost no earners yet get a wide spread and a small earners floor.

Deterministic, Python 3.13 standard library only, no network. Run locally with MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR pointing at case files.