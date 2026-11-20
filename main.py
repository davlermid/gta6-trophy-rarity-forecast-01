"""GTA VI Trophy Rarity Forecaster v2: refresh-aware hazard earners/players forecaster."""

import json
import math
import os
import sys
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Constants per spec
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]
BETA0 = 0.07
P_PRIOR_GAMMA = 1.0
P_DAMP = 0.60
P_MAX_GROWTH = 8.0
P_WINDOW_H = 48.0
LAMBDA_DECAY = 0.02
BLEND_EARN = 0.85
C_HORIZON = 0.50
C_STALE = 0.45
C_INFO = 0.4
SIGMA_FLOOR = 0.25
SIGMA_CAP = 2.6

# Tier hazard priors
TIER_LAM_MAX = {
    "platinum": 0.0004,
    "gold": 0.002,
    "silver": 0.006,
}


def clamp(x: float, lo: float = 0.01, hi: float = 99.99) -> float:
    return min(hi, max(lo, x))


def logit(pct: float) -> float:
    p = clamp(pct)
    return math.log(p / (100 - p))


def inv_logit(l: float) -> float:
    return 100.0 / (1.0 + math.exp(-l))


def round4(x: float) -> float | int:
    """Round to 4 decimals, half up. Return int when whole."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value.is_integer() else value


def parse_iso(ts: str) -> float:
    """Parse ISO timestamp, return hours since 2026-11-19T00:00:00Z."""
    # Strip Z, drop fractional seconds, split T
    ts_clean = ts.rstrip('Z')
    if '.' in ts_clean:
        ts_clean = ts_clean.split('.')[0]
    date_part, time_part = ts_clean.split('T')
    year, month, day = map(int, date_part.split('-'))
    hour, minute, second = map(int, time_part.split(':'))
    
    # Reference: 2026-11-19T00:00:00Z
    ref_year, ref_month, ref_day = 2026, 11, 19
    
    # Calculate days difference using datetime would be cleaner but we'll do manual
    # Since it's only a few days around launch, use simple day count
    # Days in months for 2026 (non-leap)
    month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    
    def days_since_epoch(y, m, d):
        days = 0
        for yr in range(2026, y):
            days += 365 + (1 if yr % 4 == 0 and (yr % 100 != 0 or yr % 400 == 0) else 0)
        for mo in range(1, m):
            days += month_days[mo - 1]
        days += d - 1
        return days
    
    ref_days = days_since_epoch(ref_year, ref_month, ref_day)
    target_days = days_since_epoch(year, month, day)
    day_diff = target_days - ref_days
    
    return day_diff * 24.0 + hour + minute / 60.0 + second / 3600.0


def weighted_ls(xs: list[float], ys: list[float], ws: list[float]):
    """Weighted least squares. Returns (a, b, sigma_residual) or (None, None, None)."""
    n = len(xs)
    if n < 2:
        return None, None, None
    sum_w = sum(ws)
    if sum_w == 0:
        return None, None, None
    sum_wx = sum(w * x for w, x in zip(ws, xs))
    sum_wy = sum(w * y for w, y in zip(ws, ys))
    sum_wxx = sum(w * x * x for w, x in zip(ws, xs))
    sum_wxy = sum(w * x * y for w, x, y in zip(ws, xs, ys))
    
    denom = sum_w * sum_wxx - sum_wx * sum_wx
    if denom == 0:
        return None, None, None
    
    b = (sum_w * sum_wxy - sum_wx * sum_wy) / denom
    a = (sum_wy - b * sum_wx) / sum_w
    
    # Residual standard deviation
    residuals = [y - (a + b * x) for x, y in zip(xs, ys)]
    sigma_res = math.sqrt(sum(w * r * r for w, r in zip(ws, residuals)) / sum_w) if sum_w > 0 else 0.0
    
    return a, b, sigma_res


def recency_weights(n: int) -> list[float]:
    if n <= 1:
        return [1.0]
    return [0.5 + 0.5 * i / (n - 1) for i in range(n)]


def median(vals: list[float], default: float = 0.0) -> float:
    if not vals:
        return default
    sorted_vals = sorted(vals)
    return sorted_vals[len(sorted_vals) // 2]


def wmedian(pairs: list[tuple[float, float]]) -> float:
    """Weighted median. pairs = [(value, weight), ...]"""
    if not pairs:
        return 0.0
    pairs = sorted(pairs, key=lambda p: p[0])
    total_w = sum(w for _, w in pairs)
    if total_w == 0:
        return pairs[0][0]
    half = total_w / 2.0
    cum = 0.0
    for val, w in pairs:
        cum += w
        if cum >= half:
            return val
    return pairs[-1][0]


def solve_case(case: dict) -> dict:
    trophies = case["trophies"]
    history = case.get("history", case.get("reads", []))
    horizon_minutes = case["horizon_minutes"]
    horizon_days = horizon_minutes / 1440.0
    horizon_hours = horizon_minutes / 60.0
    
    if not history:
        raise ValueError("No history")
    
    # Step 2: compute h, x for each read; get target and now
    for read in history:
        slot = read.get("slot", read.get("at"))
        read["_h"] = parse_iso(slot)
        read["_x"] = math.log(1.0 + max(0.0, read["_h"]))
    
    h_tgt = parse_iso(case["target"])
    x_tgt = math.log(1.0 + max(0.0, h_tgt))
    h_now = history[-1]["_h"]
    x_now = history[-1]["_x"]
    
    # P_now: last read's players_tracked, else last non-null scanning backwards, else 1
    P_now = history[-1].get("players_tracked")
    if P_now is None:
        for read in reversed(history):
            if read.get("players_tracked") is not None:
                P_now = read["players_tracked"]
                break
    if P_now is None:
        P_now = 1.0
    
    # Step 3: read_rates
    def read_rates(h):
        if "rates" in h:
            return h["rates"]
        if "trophies" in h:
            return [t.get("rate") for t in h["trophies"]]
        return [None] * len(trophies)
    
    # Step 4: REFRESH POINTS
    refresh_indices = [0]
    for i in range(1, len(history)):
        prev_rates = read_rates(history[i-1])
        curr_rates = read_rates(history[i])
        prev_p = history[i-1].get("players_tracked")
        curr_p = history[i].get("players_tracked")
        if prev_rates != curr_rates or prev_p != curr_p:
            refresh_indices.append(i)
    
    n_refresh = len(refresh_indices)
    last_refresh_h = history[refresh_indices[-1]]["_h"]
    stale_hours = max(0.0, h_now - last_refresh_h)
    
    # Step 5: PLAYERS FORECAST
    rp_all = []
    for idx in refresh_indices:
        h = history[idx]
        if h.get("players_tracked") is not None:
            rp_all.append((h["_h"], h["_x"], float(h["players_tracked"])))
    
    # Try window first: points with h >= h_now - 48
    rp_window = [(h, x, p) for h, x, p in rp_all if h >= h_now - P_WINDOW_H]
    
    P_tgt = None
    for rp in (rp_window, rp_all):
        if len(rp) >= 3:
            distinct_P = len(set(p for _, _, p in rp))
            if distinct_P >= 3:
                xs = [x for _, x, _ in rp]
                ys = [math.log(p) for _, _, p in rp]
                ws = recency_weights(len(rp))
                a, b, sigma_res = weighted_ls(xs, ys, ws)
                if a is not None:
                    lnP_tgt = math.log(P_now) + b * P_DAMP * (x_tgt - x_now)
                    P_tgt = math.exp(lnP_tgt)
                    break
    
    if P_tgt is None:
        # Prior F = ((1+max(h_tgt,0))/(1+max(last_refresh_h,-0.99)))^1.0
        denom = 1.0 + max(last_refresh_h, -0.99)
        F = ((1.0 + max(h_tgt, 0.0)) / denom) ** P_PRIOR_GAMMA
        P_tgt = P_now * F
    
    # Clamp
    max_P = max(P_now * P_MAX_GROWTH, 1500000.0)
    P_tgt = max(P_now, min(P_tgt, max_P))
    F_P = P_tgt / P_now
    
    # Step 6: stale_weight
    stale_weight = max(1.0 / (1.0 + 0.35 * max(0, n_refresh - 1)), min(1.0, stale_hours / 48.0))
    
    # Step 7: P_at(h)
    def P_at(h):
        if h <= h_now:
            return P_now
        x_h = math.log(1.0 + max(0.0, h))
        # Linear interpolation in log space between (x_now, ln P_now) and (x_tgt, ln P_tgt)
        if x_tgt == x_now:
            return P_now
        t = (x_h - x_now) / (x_tgt - x_now)
        t = clamp(t, 0.0, 1.0)
        lnP = math.log(P_now) + t * (math.log(P_tgt) - math.log(P_now))
        return math.exp(lnP)
    
    # Step 8: Per-trophy refresh series
    T = len(trophies)
    trophy_refresh_series = []
    for ti in range(T):
        series = []
        for idx in refresh_indices:
            h = history[idx]
            rates = read_rates(h)
            rate = rates[ti] if ti < len(rates) else None
            P = h.get("players_tracked")
            if rate is not None and P is not None:
                series.append((h["_h"], h["_x"], float(rate), float(P)))
        trophy_refresh_series.append(series)
    
    # Step 9: HAZARD lambda per trophy
    lambda_i_list = [None] * T
    tier_lambdas = {}  # tier -> list of lambdas
    
    for ti in range(T):
        series = trophy_refresh_series[ti]
        if len(series) < 2:
            continue
        pairs = []
        # Prefer pairs with end h >= h_now - 48
        recent_pairs = []
        all_pairs = []
        for j in range(len(series) - 1):
            h0, x0, r0, P0 = series[j]
            h1, x1, r1, P1 = series[j+1]
            dt = h1 - h0
            if dt < 0.25:
                continue
            E0 = r0 * P0 / 100.0
            E1 = r1 * P1 / 100.0
            pool = (P0 + P1) / 2.0 - (E0 + E1) / 2.0
            if pool <= 0:
                continue
            lam = (E1 - E0) / (pool * dt)
            if lam > 0:
                weight = dt
                all_pairs.append((lam, weight))
                if h1 >= h_now - 48.0:
                    recent_pairs.append((lam, weight))
        chosen = recent_pairs if len(recent_pairs) >= 2 else all_pairs
        if chosen:
            lambda_i_list[ti] = wmedian(chosen)
            tier = trophies[ti]["tier"]
            if tier not in tier_lambdas:
                tier_lambdas[tier] = []
            tier_lambdas[tier].append(lambda_i_list[ti])
    
    # Tier lambda = median of lambda_i in tier
    tier_lambda_median = {}
    for tier, lambdas in tier_lambdas.items():
        tier_lambda_median[tier] = median(lambdas)
    
    # Step 10: Rate fit per trophy
    trophy_rate_fit = [None] * T
    tier_slopes = {}  # tier -> list of b_r
    tier_sigmas = {}  # tier -> list of s_r
    
    for ti in range(T):
        series = trophy_refresh_series[ti]
        if len(series) < 2:
            continue
        xs = [x for _, x, _, _ in series]
        ys = [logit(r) for _, _, r, _ in series]
        ws = recency_weights(len(series))
        a_r, b_r, s_r = weighted_ls(xs, ys, ws)
        if a_r is not None:
            trophy_rate_fit[ti] = (a_r, b_r, s_r, len(series))
            tier = trophies[ti]["tier"]
            if tier not in tier_slopes:
                tier_slopes[tier] = []
                tier_sigmas[tier] = []
            tier_slopes[tier].append(b_r)
            tier_sigmas[tier].append(s_r)
    
    tier_slope_median = {}
    tier_sigma_median = {}
    for tier in tier_slopes:
        tier_slope_median[tier] = median(tier_slopes[tier], 0.0)
        tier_sigma_median[tier] = median(tier_sigmas[tier], 0.5)
    
    # Step 11: Per trophy forecast
    results = []
    
    for ti in range(T):
        trophy = trophies[ti]
        tier = trophy["tier"]
        
        # 11a: last_rate
        last_rate = None
        for read in reversed(history):
            rates = read_rates(read)
            if ti < len(rates) and rates[ti] is not None:
                last_rate = rates[ti]
                break
        
        if last_rate is None:
            median_rate = 100.0 * 0.5 / max(P_now, 1.0)
            sigma = 1.2 + 0.3 * math.sqrt(horizon_days)
            logit_med = logit(median_rate)
            # Build quantiles directly
            q = [round4(clamp(inv_logit(logit_med + z * sigma))) for z in Z]
            # Ensure non-decreasing
            for i in range(1, 5):
                if q[i] < q[i-1]:
                    q[i] = q[i-1]
            results.append({"id": trophy["id"], "q": q})
            continue
        
        # 11b: E_now
        E_now = last_rate * P_now / 100.0
        
        # 11c: lambda
        lambda_i = lambda_i_list[ti]
        tier_lambda = tier_lambda_median.get(tier)
        
        if lambda_i is not None and tier_lambda is not None:
            lambda_final = math.exp(0.75 * math.log(lambda_i) + 0.25 * math.log(tier_lambda))
        elif lambda_i is not None:
            lambda_final = lambda_i
        elif tier_lambda is not None:
            lambda_final = tier_lambda * 0.7
        else:
            lambda_final = None
        
        # 11d: EARNERS VIEW
        if last_rate >= 99.0:
            # Instant trophy
            E_tgt = P_tgt * last_rate / 100.0
        elif lambda_final is not None:
            # Simulate with hazard
            lam = clamp(lambda_final, 1e-6, 0.5)
            E = E_now
            steps = 24 if horizon_hours <= 48 else 48
            dt_step = horizon_hours / steps if steps else horizon_hours
            for s in range(steps):
                h_mid = h_now + (s + 0.5) * dt_step
                P_mid = P_at(h_mid)
                lam_now = lam * math.exp(-LAMBDA_DECAY * (h_mid - h_now) / 24.0)
                pool = max(P_mid - E, 0.0)
                E += lam_now * pool * dt_step
                if E > P_mid:
                    E = P_mid
            E_tgt = E
        else:
            # No lambda: blend beta prior and tier hazard prior
            beta_i = BETA0 + 0.93 * math.sqrt(last_rate / 100.0)
            E_beta = max(E_now, 0.5) * (F_P ** beta_i)
            
            lam_max = TIER_LAM_MAX.get(tier, 0.015)
            E = E_now
            for s in range(24):
                dt_step = horizon_hours / 24.0
                h_mid = h_now + (s + 0.5) * dt_step
                P_mid = P_at(h_mid)
                lam_r = lam_max * (1.0 - math.exp(-max(h_mid, 0.0) / 48.0))
                pool = max(P_mid - E, 0.0)
                E += lam_r * pool * dt_step
                if E > P_mid:
                    E = P_mid
            E_haz = E
            
            w_haz = 1.0 - math.exp(-horizon_days / 2.0)
            E_tgt = math.exp(w_haz * math.log(max(E_haz, 0.01)) + (1.0 - w_haz) * math.log(max(E_beta, 0.01)))
        
        # Floor for very new trophies
        if E_now < 1.0:
            E_tgt = max(E_tgt, 0.0015 * P_tgt * math.sqrt(horizon_days))
        
        # Clamp E_tgt
        E_tgt = min(max(E_tgt, E_now), P_tgt)
        logit_earn = logit(100.0 * E_tgt / P_tgt)
        
        # 11e: RATE VIEW
        fit = trophy_rate_fit[ti]
        if fit is not None and fit[3] >= 3:
            a_r, b_r, s_r, n_pts = fit
            tier_slope = tier_slope_median.get(tier, 0.0)
            b_used = 0.6 * b_r + 0.4 * tier_slope
            damp = 1.0 / (1.0 + 0.15 * horizon_days)
            logit_rate = a_r + b_r * x_now + b_used * (x_tgt - x_now) * damp
            w_fit = min(1.0, (n_pts - 2) / 6.0)
            logit_rate = w_fit * logit_rate + (1.0 - w_fit) * logit(last_rate)
        else:
            logit_rate = logit(last_rate)
        
        # 11f: BLEND
        logit_med = BLEND_EARN * logit_earn + (1.0 - BLEND_EARN) * logit_rate
        
        # Earners floor/ceiling
        E_med = inv_logit(logit_med) * P_tgt / 100.0
        if E_med < E_now:
            logit_med = logit(100.0 * E_now / P_tgt)
        elif E_med > P_tgt:
            logit_med = logit(99.99)
        
        # 11g: SIGMA
        if fit is not None:
            _, _, s_r, _ = fit
            tier_sigma = tier_sigma_median.get(tier, 0.5)
            sigma_fit = 0.5 * s_r + 0.5 * tier_sigma
            if fit[3] < 3:
                sigma_fit = max(sigma_fit, 0.35)
        else:
            sigma_fit = tier_sigma_median.get(tier, 0.5)
        
        quantum = 100.0 / max(P_now, 1.0)
        rc = clamp(last_rate)
        if 0.0 < rc < 100.0:
            sigma_quant = min(quantum / (rc * (1.0 - rc / 100.0)), 0.5)
        else:
            sigma_quant = 0.5
        
        sigma_horizon = C_HORIZON * math.sqrt(horizon_days)
        sigma_stale = C_STALE * abs(math.log(max(F_P, 1.0))) * stale_weight
        sigma_info = C_INFO / math.sqrt(max(n_refresh, 1))
        
        sigma = math.sqrt(
            sigma_fit**2 + sigma_quant**2 + sigma_horizon**2 +
            sigma_stale**2 + sigma_info**2
        )
        
        if E_now < 1.0:
            sigma = max(sigma, 1.5)
        
        sigma = clamp(sigma, SIGMA_FLOOR, SIGMA_CAP)
        
        # 11h: Quantiles
        q = [round4(clamp(inv_logit(logit_med + z * sigma))) for z in Z]
        
        # Enforce non-decreasing
        for i in range(1, 5):
            if q[i] < q[i-1]:
                q[i] = q[i-1]
        
        results.append({"id": trophy["id"], "q": q})
    
    return {"trophies": results}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = solve_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8"
        )


if __name__ == "__main__":
    main()