import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import requests

FPL_BASE_URL = "https://fantasy.premierleague.com/api"
CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")

session = requests.Session()
session.headers["User-Agent"] = "fpl-assistant"


def cached_get(path, ttl):
    """GET an FPL API path, reusing a copy saved on disk if it's newer than `ttl` seconds."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, path.strip("/").replace("/", "_") + ".json")
    if os.path.exists(cache_file) and time.time() - os.path.getmtime(cache_file) < ttl:
        with open(cache_file) as f:
            return json.load(f)

    response = session.get(f"{FPL_BASE_URL}{path}", timeout=20)
    response.raise_for_status()
    data = response.json()
    with open(cache_file, "w") as f:
        json.dump(data, f)
    return data


def get_bootstrap_data():
    return cached_get("/bootstrap-static/", ttl=300)

def get_fixtures():
    return cached_get("/fixtures/", ttl=600)

def get_element_summary(player_id):
    """Per-match history this season plus past season totals for one player."""
    return cached_get(f"/element-summary/{player_id}/", ttl=3600)

def get_all_element_summaries(player_ids):
    with ThreadPoolExecutor(max_workers=8) as pool:
        return dict(zip(player_ids, pool.map(get_element_summary, player_ids)))

def get_entry_history(team_id):
    return cached_get(f"/entry/{team_id}/history/", ttl=300)

def get_entry_transfers(team_id):
    return cached_get(f"/entry/{team_id}/transfers/", ttl=300)

def get_user_squad(team_id, gameweek):
    response = requests.get(f"{FPL_BASE_URL}/entry/{team_id}/event/{gameweek}/picks/")
    if response.status_code != 200:
        raise ValueError(
            f"Team ID {team_id} not found for gameweek {gameweek}. "
            "FPL team IDs change each season - check your ID in the Points page URL."
        )
    return response.json()

def get_active_squad(team_id, gameweek):
    """Squad the manager will have going into `gameweek`, based on their last picks."""
    squad_data = get_user_squad(team_id, gameweek - 1)
    # A Free Hit squad only lasts one gameweek, then reverts to the squad from before it
    if squad_data.get("active_chip") == "freehit" and gameweek - 2 >= 1:
        squad_data = get_user_squad(team_id, gameweek - 2)
    return squad_data

def get_current_gameweek(events):
    for event in events:
        if event["is_next"]:
            return event["id"]
    # No upcoming gameweek (e.g. end of season) - fall back to the current one
    for event in events:
        if event["is_current"]:
            return event["id"]
    raise ValueError("Could not determine the current gameweek from the FPL API")


if __name__ == "__main__":
    data = get_bootstrap_data()
    print(f"Current gameweek: {get_current_gameweek(data['events'])}")
