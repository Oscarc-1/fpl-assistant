"""Past FPL seasons for backtesting, from the public vaastav/Fantasy-Premier-League dataset.

Each season is converted into the same shapes the live API gives the model (bootstrap,
fixtures, per-player summaries), so `Predictor` runs on it unchanged. Files are downloaded
once into backend/.cache/history/.

Please cite the dataset if you publish anything built on it:
https://github.com/vaastav/Fantasy-Premier-League
"""
import copy
import csv
import io
import os
from datetime import datetime, timedelta

import requests

DATA_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache", "history")

INT_FIELDS = [
    "minutes", "starts", "goals_scored", "assists", "clean_sheets", "goals_conceded", "own_goals",
    "penalties_saved", "penalties_missed", "yellow_cards", "red_cards", "saves", "bonus", "bps",
    "total_points", "defensive_contribution", "round", "fixture", "opponent_team", "element",
]


def read_csv(season, name):
    path = os.path.join(CACHE_DIR, season, name)
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        response = requests.get(f"{DATA_URL}/{season}/{name}", timeout=60)
        response.raise_for_status()
        with open(path, "wb") as f:
            f.write(response.content)
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(io.StringIO(f.read())))


def season_name(season):
    """'2025-26' -> '2025/26' (the format element-summary uses)."""
    return season.replace("-", "/")


def previous(season, n=1):
    start = int(season[:4]) - n
    return f"{start}-{str(start + 1)[2:]}"


def as_int(value):
    if value in ("", "None", None):
        return None
    return int(float(value))


class Season:
    """One past season, ready for the model."""

    def __init__(self, season, live_bootstrap):
        self.season = season
        players = [p for p in read_csv(season, "players_raw.csv") if as_int(p["element_type"]) in (1, 2, 3, 4)]
        teams = read_csv(season, "teams.csv")
        rows = read_csv(season, "gws/merged_gw.csv")
        team_ids = {t["name"]: int(t["id"]) for t in teams}

        # Fixtures
        self.fixtures = []
        for f in read_csv(season, "fixtures.csv"):
            self.fixtures.append({
                "id": int(f["id"]),
                "event": as_int(f["event"]),
                "team_h": int(f["team_h"]),
                "team_a": int(f["team_a"]),
                "team_h_score": as_int(f["team_h_score"]),
                "team_a_score": as_int(f["team_a_score"]),
                "finished": f["finished"] == "True",
                "kickoff_time": f["kickoff_time"],
            })

        # Gameweek deadlines: 90 minutes before each gameweek's first kickoff
        first_kickoff = {}
        for f in self.fixtures:
            if f["event"] and f["kickoff_time"]:
                first_kickoff[f["event"]] = min(first_kickoff.get(f["event"], f["kickoff_time"]), f["kickoff_time"])
        events = []
        for gw in sorted(first_kickoff):
            kickoff = datetime.fromisoformat(first_kickoff[gw].replace("Z", "+00:00"))
            events.append({"id": gw, "deadline_time": (kickoff - timedelta(minutes=90)).isoformat().replace("+00:00", "Z")})
        self.gws = [e["id"] for e in events]

        # Per-match rows, typed like the live API
        ids = {int(p["id"]) for p in players}
        self.history = {pid: [] for pid in ids}
        self.team_by_gw = {}     # (player, gw) -> club that gameweek (handles January transfers)
        for r in rows:
            pid = int(r["element"])
            if pid not in ids:
                continue
            row = dict(r)
            for field in INT_FIELDS:
                row[field] = as_int(r.get(field)) or 0
            row["was_home"] = r["was_home"] == "True"
            row.setdefault("defensive_contribution", 0)
            self.history[pid].append(row)
            if r.get("team") in team_ids:
                self.team_by_gw[(pid, row["round"])] = team_ids[r["team"]]
        for h in self.history.values():
            h.sort(key=lambda r: (r["round"], r["kickoff_time"]))

        # Previous seasons' totals, linked by the player's permanent code
        past_by_code = {}
        for n in (1, 2, 3):
            past = previous(season, n)
            try:
                past_players = read_csv(past, "players_raw.csv")
            except requests.HTTPError:
                continue
            has_dc = int(past[:4]) >= 2025  # defensive contributions were first recorded in 2025/26
            for p in past_players:
                if as_int(p["minutes"]) and as_int(p["element_type"]) in (1, 2, 3, 4):
                    entry = {
                        "season_name": season_name(past),
                        "minutes": as_int(p["minutes"]),
                        "starts": as_int(p.get("starts")) or 0,
                        "goals_scored": as_int(p["goals_scored"]),
                        "assists": as_int(p["assists"]),
                        "expected_goals": p.get("expected_goals") or "0",
                        "expected_assists": p.get("expected_assists") or "0",
                        "bonus": as_int(p["bonus"]),
                        "saves": as_int(p["saves"]),
                        "yellow_cards": as_int(p["yellow_cards"]),
                        "red_cards": as_int(p["red_cards"]),
                        "own_goals": as_int(p["own_goals"]),
                        "penalties_missed": as_int(p["penalties_missed"]),
                        "total_points": as_int(p["total_points"]),
                    }
                    if has_dc:
                        entry["defensive_contribution"] = as_int(p.get("defensive_contribution")) or 0
                    past_by_code.setdefault(p["code"], []).append(entry)
        self.summaries = {}
        for p in players:
            pid = int(p["id"])
            past = sorted(past_by_code.get(p["code"], []), key=lambda s: s["season_name"])
            self.summaries[pid] = {"history": self.history[pid], "history_past": past}

        # Team strength: rescale FPL's 1000-1400 numbers onto the 2-5 scale used now
        strengths = [int(t[k]) for t in teams for k in ("strength_overall_home", "strength_overall_away")]
        lo, hi = min(strengths), max(strengths)
        scale = (lambda v: 2 + 3 * (int(v) - lo) / (hi - lo)) if hi > 10 else (lambda v: int(v))

        self.bootstrap = {
            "events": events,
            "teams": [{
                "id": int(t["id"]),
                "name": t["name"],
                "short_name": t["short_name"],
                "strength_overall_home": scale(t["strength_overall_home"]),
                "strength_overall_away": scale(t["strength_overall_away"]),
            } for t in teams],
            "elements": [{
                "id": int(p["id"]),
                "code": p["code"],
                "web_name": p["web_name"],
                "element_type": int(p["element_type"]),
                "team": int(p["team"]),
                "status": "a",
                "chance_of_playing_next_round": None,
                "news": "",
                "news_added": None,
                "now_cost": int(p["now_cost"]),
                "cost_change_start": int(p["cost_change_start"]),
                # End-of-season snapshot: the best available, but may differ from earlier in the season
                "penalties_order": as_int(p.get("penalties_order")),
                "yellow_cards": 0,
            } for p in players],
            "game_config": copy.deepcopy(live_bootstrap["game_config"]),
        }
        if int(season[:4]) < 2025:
            # Defensive contribution points didn't exist before 2025/26
            for pos in self.bootstrap["game_config"]["scoring"]["defensive_contribution"]:
                self.bootstrap["game_config"]["scoring"]["defensive_contribution"][pos] = 0

    def bootstrap_for(self, gw):
        """The season as it looked going into `gw`: players at the club they're at that week."""
        boot = copy.copy(self.bootstrap)
        elements = []
        for p in self.bootstrap["elements"]:
            team = self.team_by_gw.get((p["id"], gw))
            if team is None:
                earlier = [g for g in range(gw - 1, 0, -1) if (p["id"], g) in self.team_by_gw]
                team = self.team_by_gw[(p["id"], earlier[0])] if earlier else p["team"]
            elements.append(dict(p, team=team))
        boot["elements"] = elements
        return boot

    def actual_points(self, gw):
        points = {}
        for pid, rows in self.history.items():
            gw_rows = [r for r in rows if r["round"] == gw]
            if gw_rows:
                points[pid] = sum(r["total_points"] for r in gw_rows)
        return points
