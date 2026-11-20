"""gta6-trophy-rarity-forecast-01: earners/players decomposition model.

Decomposition:
- Forecast players_tracked at target via log-linear fit on ln(P) vs ln(1+hours since launch).
- Per trophy, two median views blended in logit space (weight 0.5 each):
  a) Rate view: weighted LS of logit(rate) vs x = ln(1+h), shrinkage toward tier slope.
  b) Earners view: fit ln(E_i + 0.5) vs x, predict earners, divide by P_target.
- Earners floor/ceiling: E_now <= E_median <= P_target.
- Spread: sigma = sqrt(sigma_fit^2 + (0.45*sqrt(horizon_days))^2 + sigma_quant^2),
  with staleness inflation and floor/cap.
- Quantiles via logit-normal with z = [-1.645, -0.674, 0, 0.674, 1.645].
"""

import json
import math
import os
import sys
import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

LAUNCH = "2026-11-19T00:00:00Z"

TIER_ORDER = {"platinum": 0, "gold": 1, "silver": 2, "bronze": 3}


def clamp(x: float, lo: float = 0.01, hi: float = 99.99) -> float:
    return max(lo, min(hi, x))


def logit(pct: float) -> float:
    c = clamp(pct)
    return math.log(c / (100.0 - c))


def inv_logit(l: float) -> float:
    return 100.0 / (1.0 + math.exp(-l))


def round4(x: float) -> float | int:
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def parse_iso(ts: str) -> float:
    """Parse ISO timestamp, return hours since launch."""
    dt_part = ts.replace("Z", "")
    if "." in dt_part:
        dt_part = dt_part.split(".")[0]
    date_part, time_part = dt_part.split("T")
    y, m, d = map(int, date_part.split("-"))
    hh, mm, ss = map(int, time_part.split(":"))
    dt = datetime.datetime(y, m, d, hh, mm, ss, tzinfo=datetime.timezone.utc)
    launch_dt = datetime.datetime(2026, 11, 19, 0, 0, 0, tzinfo=datetime.timezone.utc)
    delta = dt - launch_dt
    return delta.total_seconds() / 3600.0


def weighted_ls(xs, ys, ws):
    """Weighted least squares: fit y = a + b*x. Returns (a, b, sigma_residual)."""
    n = len(xs)
    if n < 2:
        return None, None, None
    sw = sum(ws)
    if sw == 0:
        return None, None, None
    sx = sum(w * x for w, x in zip(ws, xs))
    sy = sum(w * y for w, y in zip(ws, ys))
    sxx = sum(w * x * x for w, x in zip(ws, xs))
    sxy = sum(w * x * y for w, x, y in zip(ws, xs, ys))
    denom = sw * sxx - sx * sx
    if denom == 0:
        return None, None, None
    b = (sw * sxy - sx * sy) / denom
    a = (sy - b * sx) / sw
    residuals = [y - (a + b * x) for x, y in zip(xs, ys)]
    var = sum(w * r * r for w, r in zip(ws, residuals)) / sw
    sigma = math.sqrt(max(var, 1e-12))
    return a, b, sigma


def recency_weights(n: int):
    """Weights rising toward recent: w_i = 0.5 + 0.5 * i/(n-1)."""
    if n <= 1:
        return [1.0]
    return [0.5 + 0.5 * i / (n - 1) for i in range(n)]


def tier_slope(trophies, xs, rates_per_trophy, ws):
    """Compute pooled slope for a tier (median of per-trophy slopes)."""
    slopes = []
    for i, t in enumerate(trophies):
        ys = rates_per_trophy[i]
        valid = [(idx, x, y) for idx, (x, y) in enumerate(zip(xs, ys)) if y is not None]
        if len(valid) >= 2:
            vxs = [v[1] for v in valid]
            vys = [v[2] for v in valid]
            vws = [ws[v[0]] for v in valid]
            _, b, _ = weighted_ls(vxs, vys, vws)
            if b is not None:
                slopes.append(b)
    if not slopes:
        return 0.0
    slopes.sort()
    return slopes[len(slopes) // 2]


def tier_sigma(trophies, xs, rates_per_trophy, ws):
    """Pooled residual sigma for a tier (median of per-trophy sigmas)."""
    sigmas = []
    for i, t in enumerate(trophies):
        ys = rates_per_trophy[i]
        valid = [(idx, x, y) for idx, (x, y) in enumerate(zip(xs, ys)) if y is not None]
        if len(valid) >= 2:
            vxs = [v[1] for v in valid]
            vys = [v[2] for v in valid]
            vws = [ws[v[0]] for v in valid]
            _, _, s = weighted_ls(vxs, vys, vws)
            if s is not None:
                sigmas.append(s)
    if not sigmas:
        return 0.5
    sigmas.sort()
    return sigmas[len(sigmas) // 2]


def forecast_players(history, x_target):
    """Forecast ln(P) at x_target via weighted LS with damped slope."""
    xs = []
    ys = []  # ln(P)
    ws_idx = []  # indices into history for weight alignment
    for idx, h in enumerate(history):
        if h.get("players_tracked") is not None and h["players_tracked"] > 0:
            xs.append(h["_x"])
            ys.append(math.log(h["players_tracked"]))
            ws_idx.append(idx)
    if len(xs) < 3:
        return history[-1]["players_tracked"]
    ws = recency_weights(len(history))
    vws = [ws[i] for i in ws_idx]
    a, b, _ = weighted_ls(xs, ys, vws)
    if a is None or b is None:
        return history[-1]["players_tracked"]
    # Damp slope by 0.85
    b_damped = b * 0.85
    lnP = a + b_damped * x_target
    P_pred = math.exp(lnP)
    P_now = history[-1]["players_tracked"]
    # Floor at P_now, cap at P_now * exp(1.2)
    P_pred = max(P_now, min(P_pred, P_now * math.exp(1.2)))
    return P_pred


def compute_staleness(history):
    """Fraction of consecutive read pairs with identical rate vectors in last 24h."""
    if len(history) < 2:
        return 0.0
    x_now = history[-1]["_x"]
    x_cutoff = x_now - math.log(1 + 24)
    recent = [h for h in history if h["_x"] >= x_cutoff]
    if len(recent) < 2:
        recent = history
    identical = 0
    total = 0
    for i in range(1, len(recent)):
        total += 1
        # Get rates from either format
        if "rates" in recent[i]:
            rates_i = recent[i]["rates"]
            rates_prev = recent[i-1]["rates"]
        elif "trophies" in recent[i]:
            rates_i = [t["rate"] for t in recent[i]["trophies"]]
            rates_prev = [t["rate"] for t in recent[i-1]["trophies"]]
        else:
            rates_i = None
            rates_prev = None
        if rates_i is not None and rates_i == rates_prev:
            identical += 1
    return identical / total if total > 0 else 0.0


def solve_case(case: dict) -> dict:
    trophies = case["trophies"]
    history = case.get("history", case.get("reads", []))
    horizon_min = case["horizon_minutes"]
    target_ts = case["target"]
    horizon_days = horizon_min / 1440.0

    # Parse timestamps, compute x = ln(1 + hours since launch)
    for h in history:
        slot = h.get("slot", h["at"])
        h["_x"] = math.log(1.0 + max(0.0, parse_iso(slot)))
    x_target = math.log(1.0 + max(0.0, parse_iso(target_ts)))
    x_now = history[-1]["_x"]

    # Forecast players_tracked at target
    P_target = forecast_players(history, x_target)
    P_now = history[-1]["players_tracked"]

    # Organize rates per trophy - handle both "rates" array and "trophies" list format
    T = len(trophies)
    rates_per_trophy = []
    for i in range(T):
        rates = []
        for h in history:
            if "rates" in h:
                rates.append(h["rates"][i])
            elif "trophies" in h:
                rates.append(h["trophies"][i]["rate"])
            else:
                rates.append(None)
        rates_per_trophy.append(rates)

    # Compute tier-level shrinkage targets
    tiers = {}
    for i, t in enumerate(trophies):
        tier = t["tier"]
        if tier not in tiers:
            tiers[tier] = []
        tiers[tier].append(i)

    xs = [h["_x"] for h in history]
    ws_full = recency_weights(len(history))

    tier_slopes = {}
    tier_sigmas = {}
    for tier, idxs in tiers.items():
        tier_rates = [rates_per_trophy[i] for i in idxs]
        tier_trophies = [trophies[i] for i in idxs]
        tier_slopes[tier] = tier_slope(tier_trophies, xs, tier_rates, ws_full)
        tier_sigmas[tier] = tier_sigma(tier_trophies, xs, tier_rates, ws_full)

    # Staleness
    stale_frac = compute_staleness(history)

    # Damping factor for extrapolation
    damp = 1.0 / (1.0 + 0.20 * horizon_days)

    rows = []
    for i, trophy in enumerate(trophies):
        tier = trophy["tier"]
        rates = rates_per_trophy[i]

        # Collect valid (idx, x, rate) triples
        valid = [(idx, h["_x"], r) for idx, (h, r) in enumerate(zip(history, rates)) if r is not None]

        if not valid:
            # No data: use half an earner at P_now
            E_now = 0
            median_rate = 100.0 * 0.5 / max(P_now, 1)
            sigma = 1.2 + 0.3 * math.sqrt(horizon_days)
        else:
            vxs = [v[1] for v in valid]
            vys = [v[2] for v in valid]
            vws = [ws_full[v[0]] for v in valid]
            valid_idxs = [v[0] for v in valid]

            # Current earners (from last non-null rate)
            last_valid_rate = vys[-1]
            E_now = last_valid_rate * P_now / 100.0

            # --- View A: Rate view (logit) ---
            logit_ys = [logit(r) for r in vys]
            a_rate, b_rate, sigma_fit_rate = weighted_ls(vxs, logit_ys, vws)

            # Shrink slope toward tier slope
            tier_b = tier_slopes.get(tier, 0.0)
            if len(valid) < 4:
                b_used = tier_b
            else:
                b_used = 0.6 * b_rate + 0.4 * tier_b

            # Shrink sigma_fit toward tier sigma
            tier_s = tier_sigmas.get(tier, 0.5)
            if sigma_fit_rate is not None:
                sigma_fit = 0.5 * sigma_fit_rate + 0.5 * tier_s
            else:
                sigma_fit = tier_s

            # Predict logit at target (extrapolate from x_now with damped slope)
            if a_rate is not None:
                logit_now = a_rate + b_rate * x_now
            else:
                logit_now = logit(vys[-1])
            logit_pred_a = logit_now + b_used * (x_target - x_now) * damp

            # --- View B: Earners view ---
            # E_i = rate * players / 100. Use continuous earners.
            earners = [r * history[idx]["players_tracked"] / 100.0 for idx, r in zip(valid_idxs, vys)]
            earners_log = [math.log(e + 0.5) for e in earners]
            a_earn, b_earn, sigma_fit_earn = weighted_ls(vxs, earners_log, vws)

            # Shrink earners slope toward 0 (persistence in earners space)
            if len(valid) < 4:
                b_earn_used = 0.0
            else:
                b_earn_used = 0.3 * (b_earn if b_earn is not None else 0.0) + 0.7 * 0.0

            # Shrink sigma
            if sigma_fit_earn is not None:
                sigma_fit_b = 0.5 * sigma_fit_earn + 0.5 * tier_s
            else:
                sigma_fit_b = tier_s

            # Predict earners at target
            if a_earn is not None:
                earn_now = a_earn + (b_earn if b_earn is not None else 0.0) * x_now
            else:
                earn_now = math.log(earners[-1] + 0.5)
            earn_pred_log = earn_now + b_earn_used * (x_target - x_now) * damp
            E_pred = math.exp(earn_pred_log) - 0.5
            # Implied rate
            rate_b = 100.0 * E_pred / P_target
            logit_pred_b = logit(rate_b)

            # --- Blend: 0.88 persistence (last rate) + 0.12 trend ---
            logit_last = logit(vys[-1])
            # For rates near saturation (>80%), lean more on persistence
            if vys[-1] > 80:
                logit_median = 0.95 * logit_last + 0.05 * logit_pred_a
            else:
                logit_median = 0.88 * logit_last + 0.12 * logit_pred_a
            median_rate = inv_logit(logit_median)

            # --- Earners floor/ceiling ---
            E_median = median_rate * P_target / 100.0
            if E_median < E_now:
                median_rate = 100.0 * E_now / P_target
            elif E_median > P_target:
                median_rate = 99.99

            # --- Sigma ---
            sigma_horizon = 0.45 * math.sqrt(horizon_days)
            # Quantization sigma: one quantum in logit units at last rate
            last_rate = vys[-1]
            r_clamped = clamp(last_rate)
            quantum = 100.0 / P_now
            if 0 < r_clamped < 100:
                sigma_quant = quantum / (r_clamped * (1.0 - r_clamped / 100.0))
            else:
                sigma_quant = 0.5
            sigma_quant = min(sigma_quant, 0.5)

            sigma_base = math.sqrt(sigma_fit**2 + sigma_horizon**2 + sigma_quant**2)
            # Staleness inflation up to +30%
            sigma = sigma_base * (1.0 + 0.3 * stale_frac)
            sigma = max(0.22, min(2.4, sigma))

            # Fallback for few reads (<3 valid)
            if len(valid) < 3:
                sigma = 0.6 * math.sqrt(horizon_days) + 0.3

        # --- Quantiles ---
        q = []
        for z in Z:
            q_val = inv_logit(logit(median_rate) + z * sigma)
            q_val = clamp(q_val, 0.01, 99.99)
            q.append(round4(q_val))
        # Enforce non-decreasing
        for j in range(1, 5):
            if q[j] < q[j - 1]:
                q[j] = q[j - 1]

        rows.append({"id": trophy["id"], "q": q})

    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = solve_case(case)
        out_path = OUTPUT_DIR / f"{path.stem}.out"
        out_path.write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()