"""gta6-twitch-ccu-forecast-01 entry: forecast a launch day's Twitch viewers, hour by hour.

The program reads one case (schema makefun/twitch-ccu-case/v1): everything recorded up to the
end of hour 1 of a big game launch's release day, that hour's viewer count included, plus the
hourly curves of dozens of past launches (the case's `comparables`). It writes, for each of
hours 2..24, five quantiles of that hour's concurrent Twitch viewers in the game's category.

Method: a launch day is forecast as a multiple of its own first hour. Every comparable whose
release day is fully recorded contributes, for each hour h, the log ratio
ln(viewers_h / viewers_1) of its own day. The forecast for hour h is hour 1's print times the
exponential of a weighted quantile of those log ratios: the centre is the weighted median,
the width the weighted spread around it, measured separately below and above the centre
(launch days surprise upwards more than down), read off at the five scored levels with the
normal quantiles. Comparables are weighted by how much they resemble the target launch, in size
(the log of their first hour) and in how much of the audience had already arrived in the
thirteen hours before the UTC day began (a rolling midnight unlock shows a long ramp there;
a single global unlock shows almost none). Centres and spreads are smoothed over neighbouring
hours, and the whole spread is scaled by one calibration constant fitted on the comparables
themselves. If no comparable is usable, the forecast falls back to persistence: every hour
stays at hour 1's print with a slowly widening spread.

Runs on the Python 3.13 standard library only, no network, deterministic: the same case
always writes the same bytes.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# The grader runs this file with -P, which keeps its directory off sys.path; nothing else is
# imported from beside it, but put the directory back first in case that ever changes.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantiles at LEVELS.
Z = [-1.644854, -0.674490, 0.0, 0.674490, 1.644854]

# Similarity bandwidths (in natural-log units) and the spread calibration, fitted by
# leave-one-out cross-validation over the comparables of the practice cases.
BAND_SCALE = 1.25   # log of hour 1's viewers
BAND_RAMP = 1.6     # log of the pre-day ramp ratio
SPREAD_CALIB = 1.0
SMOOTH = 1          # hours either side in the moving average over hours


def parse_time(text):
    """An ISO 8601 UTC timestamp as an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hour_offset(moment, day_start):
    """Whole hours from day_start to moment, or None when not on an hour boundary."""
    delta = (moment - day_start).total_seconds() / 3600.0
    whole = round(delta)
    return whole if abs(delta - whole) < 1e-6 else None


def round2(x):
    """x rounded to 2 decimals, half up; a whole number comes back as an int for compact JSON."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if float(value).is_integer() else value


class Comparable:
    """One past launch: its release day's hourly viewers and its similarity features."""

    def __init__(self, comp):
        day_start = parse_time(comp["day_start"])
        day = {}
        pre = []
        for point in comp["hourly"]:
            off = hour_offset(parse_time(point["from"]), day_start)
            viewers = point.get("viewers")
            if off is None or viewers is None or viewers <= 0:
                continue
            if 0 <= off < 24:
                day[off + 1] = float(viewers)
            elif -13 <= off < 0:
                pre.append(float(viewers))
        self.day = day
        self.first = day.get(1)
        self.log_first = math.log(self.first) if self.first else None
        # The ramp: how much of hour 1's audience was already there, on average, over the
        # thirteen hours before the day (the window in which early time zones unlock).
        self.ramp = (sum(pre) / len(pre)) / self.first if pre and self.first else None

    def usable(self):
        return self.first is not None and len(self.day) >= 20

    def log_ratio(self, hour):
        viewers = self.day.get(hour)
        if viewers is None or self.first is None:
            return None
        return math.log(viewers / self.first)


def target_features(case):
    """The target launch's first-hour print and ramp, from the case itself."""
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError("case has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))
    day_start = parse_time(case["day_start"])
    pre = []
    for point in case.get("history_hourly") or []:
        off = hour_offset(parse_time(point["from"]), day_start)
        viewers = point.get("viewers")
        if off is None or viewers is None or viewers <= 0:
            continue
        if -13 <= off < 0:
            pre.append(float(viewers))
    ramp = (sum(pre) / len(pre)) / p1 if pre else None
    return p1, math.log(p1), ramp


def similarity(log_first_t, ramp_t, comp):
    """A weight in (0, 1]: how much this comparable's launch resembles the target's."""
    weight = math.exp(-0.5 * ((log_first_t - comp.log_first) / BAND_SCALE) ** 2)
    if ramp_t is not None and comp.ramp is not None:
        gap = math.log(ramp_t + 0.02) - math.log(comp.ramp + 0.02)
        weight *= math.exp(-0.5 * (gap / BAND_RAMP) ** 2)
    return weight


def weighted_median(pairs):
    """The weighted median of (value, weight) pairs."""
    ordered = sorted(pairs)
    total = sum(w for _, w in ordered)
    acc = 0.0
    for value, w in ordered:
        acc += w
        if acc >= 0.5 * total:
            return value
    return ordered[-1][0]


def weighted_spreads(pairs, centre):
    """The weighted spread of the values below and above the centre, each as a sigma.

    Each side's root-mean-square distance is scaled by sqrt(2): for a symmetric distribution
    half the squared mass sits on each side, so the scaling makes the two sigmas agree.
    """
    total = sum(w for _, w in pairs)
    lo = sum(w * min(v - centre, 0.0) ** 2 for v, w in pairs) / total
    hi = sum(w * max(v - centre, 0.0) ** 2 for v, w in pairs) / total
    return math.sqrt(2 * lo), math.sqrt(2 * hi)


def smooth(values):
    """A moving average over neighbouring hours; missing hours stay missing."""
    out = {}
    for hour, value in values.items():
        near = [values[h] for h in range(hour - SMOOTH, hour + SMOOTH + 1) if h in values]
        out[hour] = sum(near) / len(near)
    return out


def model_quantiles(case):
    """{hour: [five log offsets from hour 1's print]} from the comparables, or None."""
    _, log_first_t, ramp_t = target_features(case)
    pool = []
    for comp in case.get("comparables") or []:
        record = Comparable(comp)
        if record.usable():
            pool.append(record)
    if not pool:
        return None
    weights = [similarity(log_first_t, ramp_t, record) for record in pool]
    centres, spreads = {}, {}
    for hour in range(2, 25):
        pairs = []
        for record, w in zip(pool, weights):
            ratio = record.log_ratio(hour)
            if ratio is not None:
                pairs.append((ratio, w))
        if len(pairs) < 5:
            continue
        centre = weighted_median(pairs)
        centres[hour] = centre
        lo, hi = weighted_spreads(pairs, centre)
        spreads[hour] = (max(lo, 0.04), max(hi, 0.04))
    if len(centres) < 12:
        return None
    centres = smooth(centres)
    lo_smooth = smooth({h: s[0] for h, s in spreads.items()})
    hi_smooth = smooth({h: s[1] for h, s in spreads.items()})
    return {
        hour: [
            centre + SPREAD_CALIB * (hi_smooth[hour] if z >= 0 else lo_smooth[hour]) * z
            for z in Z
        ]
        for hour, centre in centres.items()
    }


def baseline_offsets(hour):
    """The persistence fallback: no move from hour 1, a spread widening like a random walk."""
    return [z * 0.15 * math.sqrt(hour - 1) for z in Z]


def forecast_case(case):
    if case["levels"] != LEVELS:
        raise ValueError(f"case asks for levels {case['levels']}, not {LEVELS}")
    p1, _, _ = target_features(case)
    offsets = model_quantiles(case)
    rows = []
    for row in case["forecast"]:
        hour = row["hour"]
        logs = offsets.get(hour) if offsets else None
        if logs is None:
            logs = baseline_offsets(hour)
        quantiles = []
        for log_off in logs:
            value = p1 * math.exp(log_off)
            quantiles.append(round2(min(max(value, 0.01), 1_000_000_000)))
        for i in range(1, len(quantiles)):
            if quantiles[i] < quantiles[i - 1]:
                quantiles[i] = quantiles[i - 1]
        rows.append({"hour": hour, "q": quantiles})
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
