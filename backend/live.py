"""Live gameweek points for a manager's team: points so far, provisional bonus, projected
auto-subs and captaincy, and how each player compares with what the model predicted."""
import json
import os
from datetime import datetime, timezone

from accuracy import snapshot_path
from fpl import cached_get, get_user_squad

POSITION_NAMES = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
XI_MIN = {1: 1, 2: 3, 3: 2, 4: 1}
XI_MAX = {1: 1, 2: 5, 3: 5, 4: 3}
DC_THRESHOLD = {2: 10, 3: 12, 4: 12}


def describe(stats, element_type):
    """What a player's points so far came from, in words."""
    def count(n, one, many):
        return f"{n} {one if n == 1 else many}"
    parts = []
    if stats.get("goals_scored"):
        parts.append(count(stats["goals_scored"], "goal", "goals"))
    if stats.get("assists"):
        parts.append(count(stats["assists"], "assist", "assists"))
    if stats.get("clean_sheets") and element_type < 4:
        parts.append("clean sheet")
    if stats.get("saves", 0) >= 3:
        parts.append(f"{stats['saves']} saves")
    if stats.get("penalties_saved"):
        parts.append(count(stats["penalties_saved"], "penalty save", "penalty saves"))
    if stats.get("penalties_missed"):
        parts.append(count(stats["penalties_missed"], "missed penalty", "missed penalties"))
    threshold = DC_THRESHOLD.get(element_type)
    if threshold and stats.get("defensive_contribution", 0) >= threshold:
        parts.append("defensive contribution")
    if stats.get("yellow_cards"):
        parts.append("yellow card")
    if stats.get("red_cards"):
        parts.append("red card")
    if stats.get("own_goals"):
        parts.append(count(stats["own_goals"], "own goal", "own goals"))
    return parts


def provisional_bonus(fixture):
    """Bonus points a fixture's BPS table would give (FPL's rules: 3-2-1, ties share)."""
    bps = {s["identifier"]: s for s in fixture.get("stats", [])}.get("bps")
    if not bps:
        return {}
    scores = sorted(((x["value"], x["element"]) for side in ("h", "a") for x in bps[side]), reverse=True)
    bonus, awarded, i = {}, 0, 0
    while i < len(scores) and awarded < 3:
        tied = [p for v, p in scores if v == scores[i][0]]
        points = 3 - awarded
        for p in tied:
            bonus[p] = points
        awarded += len(tied)
        i += len(tied)
    return bonus


def fixture_state(fixture):
    if fixture["finished_provisional"] or fixture["finished"]:
        return "finished"
    return "live" if fixture["started"] else "upcoming"


def live_team(team_id, bootstrap):
    events = bootstrap["events"]
    event = next((e for e in events if e["is_current"]), None) or max(
        (e for e in events if e["finished"]), key=lambda e: e["id"])
    gw = event["id"]
    elements = {p["id"]: p for p in bootstrap["elements"]}
    teams = {t["id"]: t for t in bootstrap["teams"]}

    picks = get_user_squad(team_id, gw)
    live = {e["id"]: e for e in cached_get(f"/event/{gw}/live/", ttl=60)["elements"]}
    fixtures = cached_get(f"/fixtures/?event={gw}", ttl=60)
    by_team = {}
    for f in fixtures:
        by_team.setdefault(f["team_h"], []).append(f)
        by_team.setdefault(f["team_a"], []).append(f)

    # Bonus FPL hasn't added yet: from the live BPS table of started matches without confirmed bonus
    pending_bonus = {}
    for f in fixtures:
        if not f["started"]:
            continue
        confirmed = {s["identifier"]: s for s in f.get("stats", [])}.get("bonus")
        if confirmed and (confirmed["h"] or confirmed["a"]):
            continue
        for pid, b in provisional_bonus(f).items():
            pending_bonus[pid] = pending_bonus.get(pid, 0) + b

    predicted = {}
    path = snapshot_path(bootstrap, gw)
    if os.path.exists(path):
        with open(path) as fh:
            predicted = {int(k): v for k, v in json.load(fh)["xp"].items()}

    chip = picks.get("active_chip")
    rows = []
    for pick in picks["picks"]:
        pid = pick["element"]
        p = elements[pid]
        stats = live.get(pid, {}).get("stats", {})
        team_fixtures = by_team.get(p["team"], [])
        states = [fixture_state(f) for f in team_fixtures]
        events_text = describe(stats, p["element_type"])
        bonus_now = stats.get("bonus", 0)
        provisional = pending_bonus.get(pid, 0) if not bonus_now else 0
        if bonus_now:
            events_text.append(f"{bonus_now} bonus")
        elif provisional:
            events_text.append(f"{provisional} bonus (provisional)")
        rows.append({
            "id": pid,
            "name": p["web_name"],
            "team": teams[p["team"]]["short_name"],
            "position": POSITION_NAMES[p["element_type"]],
            "type": p["element_type"],
            "slot": pick["position"],
            "multiplier": pick["multiplier"],
            "captain": pick["is_captain"],
            "vice": pick["is_vice_captain"],
            "minutes": stats.get("minutes", 0),
            "points": stats.get("total_points", 0),
            "provisional_bonus": provisional,
            "events": events_text,
            "predicted": round(predicted[pid], 1) if pid in predicted else None,
            "fixtures": [{
                "opponent": teams[f["team_a"] if f["team_h"] == p["team"] else f["team_h"]]["short_name"],
                "home": f["team_h"] == p["team"],
                "state": fixture_state(f),
                "minute": f["minutes"],
                "kickoff": f["kickoff_time"],
                "score": f"{f['team_h_score']}-{f['team_a_score']}" if f["started"] else None,
            } for f in team_fixtures],
            "done": bool(states) and all(s == "finished" for s in states),
            "blank": not team_fixtures,
        })

    projected = project_autosubs(rows, chip)
    hits = picks["entry_history"]["event_transfers_cost"]
    live_total = sum((r["points"] + r["provisional_bonus"]) * r["multiplier"] for r in rows) - hits
    projected_total = sum((r["points"] + r["provisional_bonus"]) * r["projected_multiplier"] for r in rows) - hits
    return {
        "gameweek": gw,
        "finished": event["finished"],
        "chip": chip,
        "hits": hits,
        "live_total": live_total,
        "projected_total": projected_total,
        "autosubs": projected,
        "average": event.get("average_entry_score"),
        "highest": event.get("highest_score"),
        "matches": {s: sum(1 for f in fixtures if fixture_state(f) == s) for s in ("finished", "live", "upcoming")},
        "players": sorted(rows, key=lambda r: r["slot"]),
        "updated": datetime.now(timezone.utc).isoformat(),
    }


def project_autosubs(rows, chip):
    """FPL's end-of-gameweek auto-subs and captaincy, applied to what's happened so far.

    A starter who didn't play once all his matches are finished is replaced by the first bench
    player (in bench order) who played, keeping a valid formation; keepers only replace keepers.
    If the captain didn't play, the vice-captain gets the armband. Bench Boost: everyone counts.
    """
    for r in rows:
        r["projected_multiplier"] = r["multiplier"]
    if chip == "bboost":
        return []
    starters = [r for r in rows if r["slot"] <= 11]
    bench = sorted([r for r in rows if r["slot"] > 11], key=lambda r: r["slot"])
    out = [r for r in starters if (r["done"] and r["minutes"] == 0) or r["blank"]]
    subs = []
    for gone in out:
        counts = {t: sum(1 for r in rows if r["type"] == t and r["projected_multiplier"] > 0) for t in XI_MIN}
        for b in bench:
            if b["projected_multiplier"] > 0 or b["minutes"] == 0:
                continue
            if (gone["type"] == 1) != (b["type"] == 1):
                continue
            after = dict(counts)
            after[gone["type"]] -= 1
            after[b["type"]] += 1
            if all(XI_MIN[t] <= after[t] <= XI_MAX[t] for t in XI_MIN):
                gone["projected_multiplier"] = 0
                b["projected_multiplier"] = 1
                subs.append({"out": gone["name"], "in": b["name"]})
                break
    captain = next((r for r in rows if r["captain"]), None)
    vice = next((r for r in rows if r["vice"]), None)
    # The vice-captain takes the armband whenever the captain doesn't play (subbed off the team or not)
    captain_missed = captain and ((captain["done"] and captain["minutes"] == 0) or captain["blank"])
    if captain_missed and vice and vice["projected_multiplier"] > 0 and vice["minutes"] > 0:
        boost = 3 if chip == "3xc" else 2
        captain["projected_multiplier"] = min(captain["projected_multiplier"], 1)
        vice["projected_multiplier"] = boost
        subs.append({"captain": vice["name"]})
    return subs
