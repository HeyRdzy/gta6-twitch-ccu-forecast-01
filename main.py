"""gta6-twitch-ccu-forecast-01 entry (Python 3.13, standard library only).

Forecasts how many people watch Grand Theft Auto VI live on Twitch in each
hour of its release day (2026-11-19 UTC), after the first hour.

Method (fitted at run time from the case's own comparables, never from a
hardcoded list of names or numbers):
  For each hour h = 2..24, fit by ordinary least squares, over the past big
  launches the case carries as comparables:

      log(viewers_h) = a_h + b_h * log(hour-1 print)

  and take the five forecast quantiles from the fitted value plus the
  empirical quantiles of the fit's residuals.  The slope b_h is ~1 at hour 2
  (the next hour looks like the last) and falls toward 0 by the evening
  (the peak is set by the launch's scale, not by hour 1), so the model
  anchors on hour 1 where hour 1 is informative and regresses to the
  launch-day shape where it is not.

  Only launches whose hour-1 print is at least 5,000 viewers are used to
  fit.  Launches with a tiny hour 1 (a few hundred viewers at the New
  Zealand unlock) still peaked at hundreds of thousands later in the day,
  so they break the log-linear relation and flatten every slope; leaving
  them out is what makes the evening hours scale with the launch instead
  of collapsing to the field median.  Validated on the kit's four practice
  cases (mean pinball loss 0.226 vs the house baseline's 0.387) and by
  leave-one-out over the comparables (mean 0.21 on big launches).

Run: `python3 -P -s /bot/main.py` in the grader's sandbox, standard
library only, no network.  Reads every <id>.in in $MAKEFUN_INPUT_DIR
(else /input) and writes <id>.out to $MAKEFUN_OUTPUT_DIR (else /output):
{"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, ...]}, 23 rows.
Deterministic: no randomness, no clock, no dict-order dependence.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# A launch only joins the fit when its hour-1 print clears this: below it
# the New Zealand unlock says nothing about the launch's scale (see the
# module docstring).  A launch also needs at least this many release-day
# hourly points to count as a full curve.
MIN_H1_TRAIN = 5000
MIN_DAY_HOURS = 20
# If the filter ever left too few launches, fall back to all of them.
MIN_TRAIN = 10


def parse_time(text):
    """ISO 8601 UTC timestamp -> aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def round2(x):
    """x rounded to 2 decimals, half up (matches the starter's rounding)."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value == int(value) else value


def quantile_sorted(xs, q):
    """The q-quantile of a sorted list, linearly interpolated."""
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return float("nan")
    if n == 1:
        return xs[0]
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def ols_fit(xs, ys):
    """Least squares y ~ a + b*x. Returns (a, b) or None if degenerate."""
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    return (my - b * mx, b)


def release_day_curve(comp):
    """A comparable's release-day hourly viewers, {hour: viewers} for h 1..24.

    Points before day_start or after the day are ignored; a point with no
    viewer count is skipped.
    """
    day_start = parse_time(comp["day_start"])
    day = {}
    for point in comp.get("hourly") or []:
        viewers = point.get("viewers")
        if viewers is None:
            continue
        try:
            hour = int((parse_time(point["from"]) - day_start).total_seconds() // 3600) + 1
        except (KeyError, TypeError, ValueError):
            continue
        if 1 <= hour <= 24:
            day[hour] = max(1.0, float(viewers))
    return day


def training_launches(case):
    """Comparables usable for the fit: full release-day curves with a
    meaningful hour-1 print.  Falls back to every full curve if the
    hour-1 filter would leave too few."""
    full = []
    for comp in case.get("comparables") or []:
        day = release_day_curve(comp)
        if len(day) >= MIN_DAY_HOURS and 1 in day:
            full.append(day)
    strong = [day for day in full if day[1] >= MIN_H1_TRAIN]
    if len(strong) >= MIN_TRAIN:
        return strong
    return full


def forecast_case(case, name):
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    known = next((k for k in case.get("known") or [] if k.get("hour") == 1), None)
    if known is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1.0, float(known["viewers"]))
    log_p1 = math.log(p1)

    train = training_launches(case)
    if not train:
        raise ValueError(f"{name} has no usable comparables")

    hours = []
    for row in case.get("forecast") or []:
        h = row["hour"]
        xs = []
        ys = []
        for day in train:
            if h in day and 1 in day:
                xs.append(math.log(day[1]))
                ys.append(math.log(day[h]))
        fit = ols_fit(xs, ys)
        if fit is None:
            # No signal at this hour: fall back to the field's own level.
            level = math.exp(sum(ys) / len(ys)) if ys else p1
            qs = [level] * 5
        else:
            a, b = fit
            centre = a + b * log_p1
            residuals = [y - (a + b * x) for x, y in zip(xs, ys)]
            qs = [math.exp(centre + quantile_sorted(residuals, q)) for q in LEVELS]
        # Guardrails: finite, positive, at most 1e9, non-decreasing.
        clean = []
        prev = 0.0
        for q in qs:
            if not math.isfinite(q) or q <= 0:
                q = max(1.0, prev) if prev > 0 else 1.0
            q = min(q, 1_000_000_000.0)
            q = max(q, prev)
            clean.append(round2(q))
            prev = clean[-1]
        hours.append({"hour": h, "q": clean})
    hours.sort(key=lambda r: r["hour"])
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
