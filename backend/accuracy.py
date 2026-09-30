"""How accurate have the predictions been? Compares predictions with what players actually scored.

Every time the app builds its model before a deadline, `save_snapshot` records that gameweek's
predictions for every player ("live": exactly what the app showed). Gameweeks with no snapshot
are rebuilt the way backtest.py does, from matches before that deadline ("rebuilt": no injury
news from the time, so slightly less informed than the live app).
"""
import json
import os
from datetime import datetime, timezone

from backtest import score
from model import Predictor

SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), "predictions")
POSITION_NAMES = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def season_id(bootstrap):
    year = int(bootstrap["events"][0]["deadline_time"][:4])
    return f"{year}-{str(year + 1)[2:]}"


def snapshot_path(bootstrap, gw):
    return os.path.join(SNAPSHOT_DIR, season_id(bootstrap), f"gw{gw}.json")


def save_snapshot(predictor):
    """Record the upcoming gameweek's predictions (the latest pre-deadline version wins)."""
    gw = predictor.before_gw
    if datetime.now(timezone.utc) >= predictor.deadlines[gw]:
        return
    path = snapshot_path(predictor.bootstrap, gw)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "source": "live",
        "saved": datetime.now(timezone.utc).isoformat(),
        "xp": {str(pid): round(predictor.xp(pid, gw), 3) for pid in predictor.players},
    }
    with open(path, "w") as f:
        json.dump(data, f)


def load_predictions(bootstrap, fixtures, summaries, gw):
    """Predictions for a finished gameweek: the saved live snapshot, or rebuilt and then saved."""
    path = snapshot_path(bootstrap, gw)
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        return data["source"], {int(pid): xp for pid, xp in data["xp"].items()}

    predictor = Predictor(bootstrap, fixtures, summaries, gw, use_news=False)
    xp = {pid: predictor.xp(pid, gw) for pid in predictor.players}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"source": "rebuilt", "saved": datetime.now(timezone.utc).isoformat(),
                   "xp": {str(pid): round(v, 3) for pid, v in xp.items()}}, f)
    return "rebuilt", xp


def actual_points(summaries, gw):
    points = {}
    for pid, s in summaries.items():
        rows = [r for r in s["history"] if r["round"] == gw]
        if rows:
            points[pid] = sum(r["total_points"] for r in rows)
    return points


def form_before(summaries, gw):
    """FPL's own number going into the gameweek: average points over the previous 4 gameweeks."""
    form = {}
    for pid, s in summaries.items():
        recent = [r for r in s["history"] if gw - 4 <= r["round"] < gw]
        form[pid] = sum(r["total_points"] for r in recent) / len(recent) if recent else 0.0
    return form


def accuracy_report(bootstrap, fixtures, summaries, team_id=None, get_picks=None):
    elements = {p["id"]: p for p in bootstrap["elements"]}
    teams = {t["id"]: t["short_name"] for t in bootstrap["teams"]}
    finished = [e["id"] for e in bootstrap["events"] if e["finished"] and e["id"] >= 3]

    def player(pid):
        p = elements[pid]
        return {"name": p["web_name"], "team": teams[p["team"]], "position": POSITION_NAMES[p["element_type"]]}

    weeks = []
    for gw in finished:
        source, xp = load_predictions(bootstrap, fixtures, summaries, gw)
        actual = actual_points(summaries, gw)
        ours = score(xp, actual)
        form = score(form_before(summaries, gw), actual)
        weeks.append({"gw": gw, "source": source, "xp": xp, "actual": actual,
                      "model": {k: round(v, 3) for k, v in ours.items()},
                      "form": {k: round(v, 3) for k, v in form.items()}})

    if not weeks:
        return {"gameweeks": [], "season": None, "latest": None}

    def season_average(key):
        return {k: round(sum(w[key][k] for w in weeks) / len(weeks), 3) for k in ("mae", "corr", "top50", "bias")}

    latest = weeks[-1]
    xp, actual = latest["xp"], latest["actual"]
    played = [pid for pid in actual if pid in xp]
    top = sorted(played, key=lambda p: -xp[p])[:10]
    surprise = sorted(played, key=lambda p: actual[p] - xp[p])
    report_latest = {
        "gw": latest["gw"],
        "source": latest["source"],
        "top_predicted": [player(p) | {"xp": round(xp[p], 1), "actual": actual[p]} for p in top],
        "over": [player(p) | {"xp": round(xp[p], 1), "actual": actual[p]} for p in reversed(surprise[-5:])],
        "under": [player(p) | {"xp": round(xp[p], 1), "actual": actual[p]} for p in surprise[:5]],
        "squad": None,
    }

    if team_id and get_picks:
        picks = get_picks(team_id, latest["gw"])
        squad = []
        for pick in picks["picks"]:
            pid, mult = pick["element"], pick["multiplier"]
            squad.append(player(pid) | {
                "xp": round(xp.get(pid, 0.0) * mult, 1),
                "actual": actual.get(pid, 0) * mult,
                "starting": mult > 0,
                "captain": pick.get("is_captain", False),
            })
        report_latest["squad"] = squad
        report_latest["squad_xp"] = round(sum(p["xp"] for p in squad), 1)
        report_latest["squad_actual"] = sum(p["actual"] for p in squad)
        report_latest["chip"] = picks.get("active_chip")

    return {
        "gameweeks": [{k: w[k] for k in ("gw", "source", "model", "form")} for w in weeks],
        "season": {"model": season_average("model"), "form": season_average("form"),
                   "from_gw": weeks[0]["gw"], "to_gw": latest["gw"]},
        "latest": report_latest,
    }
