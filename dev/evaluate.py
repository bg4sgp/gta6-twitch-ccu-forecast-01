"""Leave-one-out cross-validation of the entry's forecaster over the practice comparables.

Each comparable with a fully recorded release day takes a turn as the target: a pseudo-case is
built from its own data (its hour-1 print, its pre-day history, the other comparables as the
pool), the entry's model forecasts it, and the forecast is scored with the contest's pinball
loss on the log scale against its real hours 2..24. The house baseline is scored the same way.
Run from the repository root: python3 dev/evaluate.py
"""

import importlib.util
import itertools
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KIT_INPUT = Path("/home/hatch/workspace/makefun/gta6-twitch-ccu-forecast-01-kit/input")

spec = importlib.util.spec_from_file_location("entry_main", ROOT / "main.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)

LEVELS = entry.LEVELS


def pinball(y, q, tau):
    d = math.log(max(y, 1)) - math.log(q)
    return 2 * max(tau * d, (tau - 1) * d)


def load_pool():
    case = json.loads((KIT_INPUT / "p00.in").read_text(encoding="utf-8"))
    return case["comparables"]


def day_view(comp):
    """{hour: viewers} of the release day and {offset: viewers} around it, from a comparable."""
    from datetime import datetime

    def parse(ts):
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))

    start = parse(comp["day_start"])
    hours = {}
    for point in comp["hourly"]:
        off = (parse(point["from"]) - start).total_seconds() / 3600.0
        if off == int(off):
            hours[int(off)] = (point["viewers"], point.get("channels"))
    return hours


def pseudo_case(comp, pool):
    hours = day_view(comp)
    history = [
        {"from": p["from"], "viewers": p["viewers"], "channels": p.get("channels")}
        for p in comp["hourly"]
    ]
    return {
        "levels": LEVELS,
        "day_start": comp["day_start"],
        "known": [
            {
                "hour": 1,
                "viewers": hours[0][0],
                "channels": hours[0][1],
                "from": comp["day_start"],
                "until": comp["day_start"],
                "reads": 1,
            }
        ],
        "forecast": [{"hour": h} for h in range(2, 25)],
        "history_hourly": history,
        "history": history,
        "comparables": pool,
    }


def score(forecast, comp):
    hours = day_view(comp)
    total, count = 0.0, 0
    rows = {row["hour"]: row["q"] for row in forecast["hours"]}
    for h in range(2, 25):
        if h - 1 not in hours:
            continue
        y = hours[h - 1][0]
        for tau, q in zip(LEVELS, rows[h]):
            total += pinball(y, q, tau)
            count += 1
    return total / count


def baseline_forecast(case):
    p1 = max(1.0, float(case["known"][0]["viewers"]))
    return {
        "hours": [
            {"hour": h, "q": [p1 * math.exp(off) for off in entry.baseline_offsets(h)]}
            for h in range(2, 25)
        ]
    }


def is_clean_rolling(comp):
    text = comp["unlock"].lower()
    return "midnight" in text and "early access" not in text and "early-access" not in text


def run(params=None):
    if params:
        entry.BAND_SCALE, entry.BAND_RAMP, entry.SPREAD_CALIB = params
    comps = load_pool()
    targets = []
    for comp in comps:
        hours = day_view(comp)
        if 0 in hours and len([o for o in hours if 0 <= o < 24]) >= 20:
            targets.append(comp)
    results = {}
    for comp in targets:
        pool = [c for c in comps if c["name"] != comp["name"]]
        case = pseudo_case(comp, pool)
        model_loss = score(entry.forecast_case(case), comp)
        base_loss = score(baseline_forecast(case), comp)
        results[comp["name"]] = (model_loss, base_loss, is_clean_rolling(comp))
    return results


def summarise(results, label):
    model = [v[0] for v in results.values()]
    base = [v[1] for v in results.values()]
    clean = [v[0] for v in results.values() if v[2]]
    clean_base = [v[1] for v in results.values() if v[2]]
    mean = lambda xs: sum(xs) / len(xs)
    print(
        f"{label:34s} model {mean(model):.4f} | baseline {mean(base):.4f} "
        f"| clean-rolling model {mean(clean):.4f} vs base {mean(clean_base):.4f} "
        f"({len(clean)} targets)"
    )
    return mean(model), mean(clean)


if __name__ == "__main__":
    grid = "--grid" in sys.argv
    if not grid:
        results = run()
        summarise(results, "default params")
        worst = sorted(results.items(), key=lambda kv: -kv[1][0])[:6]
        for name, (m, b, clean) in worst:
            print(f"  worst: {name[:40]:42s} model {m:.3f} base {b:.3f} clean={clean}")
    else:
        best = []
        for band_scale, band_ramp, calib in itertools.product(
            [0.9, 1.25, 1.75], [0.9, 1.6, 99.0], [0.85, 1.0, 1.15]
        ):
            results = run((band_scale, band_ramp, calib))
            overall, clean = summarise(
                results, f"s={band_scale} r={band_ramp} c={calib}"
            )
            best.append((clean, overall, (band_scale, band_ramp, calib)))
        best.sort()
        print("\nTop by clean-rolling loss:")
        for clean, overall, params in best[:8]:
            print(f"  clean {clean:.4f} overall {overall:.4f} params {params}")
