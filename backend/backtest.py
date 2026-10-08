"""Check the xP model against gameweeks that have already been played.

For each gameweek, the model only sees matches before it; its predictions are then compared
with what players actually scored.

    python backtest.py                     # this season so far
    python backtest.py --season 2025-26    # a full past season (see history.py)
    python backtest.py --season 2025-26 --tune          # tune PARAMS: fit on odd GWs, check on even GWs
    python backtest.py --season 2025-26 --tune-starts   # fit the start calibration the same way
"""
import argparse
import math

from fpl import get_bootstrap_data, get_fixtures, get_all_element_summaries, get_current_gameweek
from model import Predictor, PARAMS, previous_season_name


class LiveSeason:
    """This season so far, from the live API (same interface as history.Season)."""

    def __init__(self):
        self.bootstrap = get_bootstrap_data()
        self.fixtures = get_fixtures()
        self.summaries = get_all_element_summaries([p["id"] for p in self.bootstrap["elements"]])
        self.gws = list(range(1, get_current_gameweek(self.bootstrap["events"])))

    def bootstrap_for(self, gw):
        return self.bootstrap

    def actual_points(self, gw):
        points = {}
        for pid, s in self.summaries.items():
            rows = [r for r in s["history"] if r["round"] == gw]
            if rows:
                points[pid] = sum(r["total_points"] for r in rows)
        return points


def load(season=None):
    if season is None:
        return LiveSeason()
    import history
    return history.Season(season, get_bootstrap_data())


# ---------- scoring ----------

def baselines(season, gw):
    form, ppg = {}, {}
    for pid, s in season.summaries.items():
        before = [r for r in s["history"] if r["round"] < gw]
        recent = [r for r in before if r["round"] >= gw - 4]
        form[pid] = sum(r["total_points"] for r in recent) / len(recent) if recent else 0.0
        played = [r for r in before if r["minutes"] > 0]
        ppg[pid] = sum(r["total_points"] for r in played) / len(played) if played else 0.0
    # (The dataset's "xP" column isn't used: it was recorded after team news/matches, so it leaks
    # results. FPL's real pre-deadline number is form, which is compared here.)
    return {"form": form, "points per game": ppg}


def newcomer_ids(season):
    """Players with no Premier League minutes last season."""
    prev = previous_season_name(season.bootstrap)
    return {pid for pid, s in season.summaries.items()
            if not any(x["season_name"] == prev and x["minutes"] > 0 for x in s["history_past"])}


def score(pred, actual, only=None):
    ids = [pid for pid in actual if pid in pred and (only is None or pid in only)]
    errors = [pred[i] - actual[i] for i in ids]
    mae = sum(abs(e) for e in errors) / len(errors)
    mse = sum(e * e for e in errors) / len(errors)
    bias = sum(errors) / len(errors)
    mp = sum(pred[i] for i in ids) / len(ids)
    ma = sum(actual[i] for i in ids) / len(ids)
    cov = sum((pred[i] - mp) * (actual[i] - ma) for i in ids)
    sp = math.sqrt(sum((pred[i] - mp) ** 2 for i in ids))
    sa = math.sqrt(sum((actual[i] - ma) ** 2 for i in ids))
    corr = cov / (sp * sa) if sp and sa else 0.0
    top = sorted(ids, key=lambda i: -pred[i])[:min(50, len(ids) // 4)]
    top50 = sum(actual[i] for i in top) / len(top)
    return {"mae": mae, "mse": mse, "corr": corr, "top50": top50, "bias": bias}


def average(results):
    return {k: sum(r[k] for r in results) / len(results) for k in results[0]}


def predictors(season, gws, params):
    return {gw: Predictor(season.bootstrap_for(gw), season.fixtures, season.summaries, gw, params, use_news=False)
            for gw in gws}


def evaluate(season, gws, params, only=None, built=None):
    results = []
    for gw in gws:
        predictor = built[gw] if built else Predictor(season.bootstrap_for(gw), season.fixtures,
                                                      season.summaries, gw, params, use_news=False)
        pred = {pid: predictor.xp(pid, gw) for pid in predictor.players}
        results.append(score(pred, season.actual_points(gw), only))
    return average(results)


def fmt(name, r):
    print(f"  {name:<22} MAE {r['mae']:.3f}  MSE {r['mse']:.3f}  corr {r['corr']:.3f}  "
          f"top-50 avg actual {r['top50']:.2f}  bias {r['bias']:+.2f}")


# ---------- start chances ----------

def start_brier(built, season, gws, params):
    total = n = 0
    for gw in gws:
        predictor = built[gw]
        predictor.params = params
        predictor._start_cache = {}
        predictor._cache = {}
        for pid in predictor.players:
            rows = [r for r in season.summaries[pid]["history"] if r["round"] == gw]
            if len(rows) == 1:
                total += (predictor.start_chance(pid, gw) - rows[0]["starts"]) ** 2
                n += 1
    return total / n


def fit_start_calibration(built, season, gws):
    best = None
    for a in [x / 4 for x in range(-14, 9)]:
        for b in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
            params = dict(PARAMS, start_calibration=(a, b))
            s = start_brier(built, season, gws, params)
            if best is None or s < best[0]:
                best = (s, (a, b))
    return best


def last_game_rule(season, gws):
    """Baseline: whoever started his team's last match starts again."""
    total = n = 0
    for gw in gws:
        for s in season.summaries.values():
            rows = [r for r in s["history"] if r["round"] == gw]
            prev = [r for r in s["history"] if r["round"] < gw]
            if len(rows) == 1 and prev:
                p = 0.9 if prev[-1]["starts"] else 0.05
                total += (p - rows[0]["starts"]) ** 2
                n += 1
    return total / n


def tune_starts(season, train, test):
    built = predictors(season, sorted(set(train + test)), PARAMS)
    print("\nStart chance accuracy (Brier score, lower is better)")
    print(f"  'started last game' rule, test GWs: {last_game_rule(season, test):.4f}")
    print(f"  current {PARAMS['start_calibration']}: train {start_brier(built, season, train, PARAMS):.4f}, "
          f"test {start_brier(built, season, test, PARAMS):.4f}")
    _, cal = fit_start_calibration(built, season, train)
    params = dict(PARAMS, start_calibration=cal)
    print(f"  fitted on train {cal}: test {start_brier(built, season, test, params):.4f}")


# ---------- tuning ----------

# Tune on squared error: this is an expected-points model, and absolute error rewards predicting
# the typical outcome (shrinking players with big upside towards average), which under-rated
# premiums, regular starters and defensive-contribution points.
TUNE_METRIC = "mse"

TUNE_GRID = {
    "prior_minutes": [450, 900, 1800, 3600],
    "newcomer_prior_minutes": [90, 180, 360, 720],
    "decay": [0.8, 0.9, 0.95, 1.0],
    "minutes_decay": [0.25, 0.4, 0.55, 0.7],
    "team_prior_matches": [2, 4, 8, 16],
    "team_prior_scale": [0.1, 0.18, 0.3],
    "goal_weight": [0.0, 0.2, 0.4],
    "conceded_shape": [3, 6, 12, 1000],
    "finishing_shrink": [20, 40, 80, 1e9],
    "assist_shrink": [3, 7, 15, 1e9],
    "penalty_mode": ["none", "prob"],
    "match_cap": [{}, {"xg": 1.5, "xa": 1.0}, {"xg": 1.0, "xa": 0.7}],
    "newcomer_factor": [0.7, 0.8, 0.9, 1.0],
    "sub_prior_matches": [0.5, 2, 5],
    "dc_dispersion": [1.5, 2.0, 2.5],
}


def tune(season, train, test):
    """Coordinate search: change one setting at a time, keep it if the train GWs improve."""
    params = dict(PARAMS)
    best = evaluate(season, train, params)[TUNE_METRIC]
    print(f"\nTuning on GW{train[0]}..{train[-1]} (odd), start {TUNE_METRIC} {best:.4f}")
    for _ in range(2):
        for key, values in TUNE_GRID.items():
            for v in values:
                if v == params[key]:
                    continue
                trial = dict(params, **{key: v})
                m = evaluate(season, train, trial)[TUNE_METRIC]
                if m < best - 1e-4:
                    best, params = m, trial
                    print(f"  {key} = {v}: train {TUNE_METRIC} {m:.4f}", flush=True)
    changed = {k: params[k] for k in params if params[k] != PARAMS[k]}
    print(f"\nChanged settings: {changed or 'none'}")
    fmt("current, test GWs", evaluate(season, test, PARAMS))
    fmt("tuned, test GWs", evaluate(season, test, params))
    print(f"\nTuned settings: {params}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", help="past season like 2025-26 (default: this season so far)")
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--tune-starts", action="store_true")
    args = parser.parse_args()

    season = load(args.season)
    gws = [gw for gw in season.gws if gw >= 3]
    print(f"Backtesting {args.season or 'this season'}, gameweeks {gws[0]}-{gws[-1]}\n")

    base = {}
    for gw in gws:
        actual = season.actual_points(gw)
        for name, pred in baselines(season, gw).items():
            base.setdefault(name, []).append(score(pred, actual))
    for name, results in base.items():
        fmt(name, average(results))
    fmt("xP model", evaluate(season, gws, PARAMS))

    newcomers = newcomer_ids(season)
    print(f"\nPlayers new to the Premier League ({len(newcomers)}):")
    fmt("form", average([score(baselines(season, gw)["form"], season.actual_points(gw), newcomers) for gw in gws]))
    fmt("xP model", evaluate(season, gws, PARAMS, newcomers))

    train = [gw for gw in gws if gw % 2 == 1]
    test = [gw for gw in gws if gw % 2 == 0]
    if args.tune_starts:
        tune_starts(season, train, test)
    if args.tune:
        tune(season, train, test)


if __name__ == "__main__":
    main()
