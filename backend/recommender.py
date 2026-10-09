"""Builds everything the frontend shows for a manager, using the xP model and planner."""
import os
import threading
import time
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from fpl import (get_bootstrap_data, get_fixtures, get_all_element_summaries, get_current_gameweek,
                 get_active_squad, get_entry_history, get_entry_transfers, cached_get)
from accuracy import save_snapshot
from model import AdjustedPredictor, Overrides, Predictor
from chips import chip_plan
from optimizer import best_single_transfers, plan_transfers
from planner import (FT_VALUE, FUTURE_WEIGHTS, HIT_COST, HORIZON, MIN_GAIN_PER_HIT, MIN_GAIN_PER_MOVE,
                     best_lineup, free_transfers, selling_prices)

POSITION_NAMES = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

# Settings from the environment (see render.yaml for the public deployment)
EXAMPLE_TEAM_ID = int(os.environ.get("EXAMPLE_TEAM_ID", "408324"))
MODEL_TTL = int(os.environ.get("MODEL_TTL_SECONDS", "300"))       # how long to reuse the built model
WARM_EXAMPLE = os.environ.get("WARM_EXAMPLE", "0") == "1"          # precompute the example team

_predictor_cache = {"built": 0, "predictor": None}
_predictor_lock = threading.Lock()
_chip_context = {}  # team id -> inputs for the chip planner from the last analysis
_results = {}       # (kind, team id) -> (model build time, result)
_team_locks = defaultdict(threading.Lock)


def get_predictor():
    """Building the model takes a few seconds, so reuse it for MODEL_TTL seconds."""
    rebuilt = False
    with _predictor_lock:
        if time.time() - _predictor_cache["built"] > MODEL_TTL:
            bootstrap = get_bootstrap_data()
            fixtures = get_fixtures()
            summaries = get_all_element_summaries([p["id"] for p in bootstrap["elements"]])
            gw = get_current_gameweek(bootstrap["events"])
            _predictor_cache["predictor"] = Predictor(bootstrap, fixtures, summaries, gw)
            _predictor_cache["built"] = time.time()
            save_snapshot(_predictor_cache["predictor"])
            rebuilt = True
    if rebuilt and WARM_EXAMPLE:
        warm_example()
    return _predictor_cache["predictor"]


def cached_result(kind, team_id, compute, overrides=None):
    """Reuse a team's results until the model is rebuilt (the planning is slow), and never compute
    the same team (with the same overrides) twice at once (e.g. the background warm-up and a visitor)."""
    overrides = overrides or Overrides()
    key = (kind, team_id, overrides.key)
    with _team_locks[key]:
        get_predictor()
        built = _predictor_cache["built"]
        hit = _results.get(key)
        if hit and hit[0] == built:
            return hit[1]
        result = compute(team_id, overrides)
        _results[key] = (built, result)
        if len(_results) > 100:  # keep memory bounded: drop the oldest entries
            for key in sorted(_results, key=lambda k: _results[k][0])[:20]:
                _results.pop(key, None)
        return result


def warm_example():
    """Work out the example team in the background so visitors trying it get results straight away."""
    def run():
        try:
            cached_result("analysis", EXAMPLE_TEAM_ID, analyse_team)
            cached_result("chips", EXAMPLE_TEAM_ID, plan_chips)
        except Exception as e:  # a failed warm-up just means the first visitor computes it
            print(f"Example team warm-up failed: {e}")
    threading.Thread(target=run, daemon=True).start()


def analyse_team(team_id, overrides=None):
    overrides = overrides or Overrides()
    predictor = get_predictor()
    if overrides:
        predictor = AdjustedPredictor(predictor, overrides)
    bootstrap = predictor.bootstrap
    gw = predictor.before_gw
    last_gw = max(e["id"] for e in bootstrap["events"])
    gws = list(range(gw, min(gw + HORIZON, last_gw + 1)))

    elements = {p["id"]: p for p in bootstrap["elements"]}
    positions = {pid: p["element_type"] for pid, p in elements.items()}
    teams = {t["id"]: t for t in bootstrap["teams"]}

    squad_data = get_active_squad(team_id, gw)
    squad = [pick["element"] for pick in squad_data["picks"]]
    history = get_entry_history(team_id)
    entry = cached_get(f"/entry/{team_id}/", ttl=300)
    transfers = get_entry_transfers(team_id)

    bank = squad_data["entry_history"]["bank"] / 10
    squad_value = squad_data["entry_history"]["value"] / 10
    max_banked = 1 + bootstrap["game_config"]["rules"]["max_extra_free_transfers"]
    free = free_transfers(history, entry["started_event"], gw, max_banked)
    sell = selling_prices(squad, transfers, history, elements)

    def player_info(pid):
        p = elements[pid]
        return {
            "id": pid,
            "name": p["web_name"],
            "team": teams[p["team"]]["short_name"],
            "position": POSITION_NAMES[p["element_type"]],
            "price": p["now_cost"] / 10,
            "owned": float(p["selected_by_percent"]),
        }

    def fixture_labels(team_id, g):
        labels = []
        for f in predictor.fixtures_by_gw_team[(g, team_id)]:
            home = f["team_h"] == team_id
            opp = teams[f["team_a"] if home else f["team_h"]]["short_name"]
            labels.append(opp + (" (H)" if home else " (A)"))
        return labels

    # --- lineup for this gameweek ---
    xp_now = {pid: predictor.xp(pid, gw) for pid in elements}
    starters, bench = best_lineup(squad, positions, xp_now)
    ranked = sorted(starters, key=lambda p: -xp_now[p])
    captain, vice = ranked[0], ranked[1]

    # Who starts in each week of the horizon (the lineup can change week to week)
    starting_by_gw = {g: set(best_lineup(squad, positions, {p: predictor.xp(p, g) for p in squad})[0]) for g in gws}

    def headline(pid, g):
        """What the page shows: points if he starts, plus his chance of starting.
        Decisions (lineup, captain, transfers) still use the chance-adjusted `xp`."""
        return {"if_starts": round(predictor.points_if_starts(pid, g), 1),
                "start_chance": round(predictor.start_chance(pid, g), 2)}

    squad_rows = []
    for pid in squad:
        info = player_info(pid)
        info.update(headline(pid, gw))
        info.update({
            "xp": round(xp_now[pid], 1),
            "xp_by_gw": [{"gw": g, "xp": round(predictor.xp(pid, g), 1), **headline(pid, g),
                          "starting": pid in starting_by_gw[g],
                          "fixtures": fixture_labels(elements[pid]["team"], g)} for g in gws],
            "xp_horizon": round(sum(predictor.xp(pid, g) for g in gws), 1),
            "selling_price": sell[pid],
            "starting": pid in starters,
            "bench_order": bench.index(pid) + 1 if pid in bench else None,
            "captain": pid == captain,
            "vice": pid == vice,
            "breakdown": predictor.breakdown(pid, gw, if_starts=True),
        })
        squad_rows.append(info)
    squad_rows.sort(key=lambda r: (not r["starting"], r["bench_order"] or 0,
                                   list(POSITION_NAMES.values()).index(r["position"]), -r["xp"]))
    lineup_xp = sum(xp_now[p] for p in starters) + xp_now[captain]

    # Captain options: the top starters by predicted points, with their chance of a haul or a blank
    captain_options = []
    for pid in ranked[:5]:
        chances = predictor.upside(pid, gw)
        captain_options.append(player_info(pid) | headline(pid, gw) | {
            "xp": round(xp_now[pid], 1),
            "haul": round(chances["haul"], 3),
            "blank": round(chances["blank"], 3),
            "fixtures": fixture_labels(elements[pid]["team"], gw),
            "captain": pid == captain,
        })

    # --- transfers: the best plan for each number of moves, from the optimiser ---
    weights = FUTURE_WEIGHTS[:len(gws)]
    xp_by_gw = {g: {pid: predictor.xp(pid, g) for pid in elements} for g in gws}
    total = {pid: sum(xp_by_gw[g][pid] for g in gws) for pid in elements}

    def describe(plan):
        moves = []
        for out_id, in_id in zip(plan["out"], plan["in"]):
            moves.append({"out": player_info(out_id) | {"xp_horizon": round(total[out_id], 1), "sell": sell[out_id]},
                          "in": player_info(in_id) | {"xp_horizon": round(total[in_id], 1)}})
        return moves

    # Week-by-week plans: for each number of moves this week, the best plan for the following weeks.
    # Solved in parallel (the solver runs outside Python's lock).
    counts = list(range(0, min(free + 2, 5) + 1))
    with ThreadPoolExecutor(max_workers=len(counts)) as pool:
        solved = pool.map(lambda k: plan_transfers(xp_by_gw, gws, weights, elements, squad, sell, bank, free,
                                                   FT_VALUE, moves_now=k, keep=overrides.keep,
                                                   hit_margin=MIN_GAIN_PER_HIT), counts)
    plans = {k: plan for k, plan in zip(counts, solved) if plan}
    base = plans[0]["value"]
    options = []
    for k, plan in plans.items():
        this_week = plan["weeks"][0]
        later = [{"gw": wk["gw"], "moves": describe(wk), "hits": HIT_COST * wk["hits"]}
                 for wk in plan["weeks"][1:] if wk["in"]]
        options.append({
            "label": "Save your transfers" if k == 0 else f"{k} transfer{'s' if k > 1 else ''} now",
            "transfers": k,
            "moves": describe(this_week),
            "hit": HIT_COST * this_week["hits"],
            "net": round(plan["value"] - base, 1),
            "later": later,
            "banked_after": plan["weeks"][1]["free"] if len(plan["weeks"]) > 1 else None,
            "best": False,
        })
    # An extra move has to earn its place: MIN_GAIN_PER_MOVE if it's free, MIN_GAIN_PER_HIT more if it
    # costs a hit (on top of paying for the hit, which is already in `net`)
    def required(k):
        return sum(MIN_GAIN_PER_MOVE if i <= free else MIN_GAIN_PER_HIT for i in range(1, k + 1))
    recommended = options[0]
    for option in options[1:]:
        if option["net"] - required(option["transfers"]) > recommended["net"] - required(recommended["transfers"]):
            recommended = option
    recommended["best"] = True

    # Alternatives: the best single transfer for each player you could sell
    alternatives = [{"move": describe({"out": [o], "in": [i]})[0], "gain": round(g, 1)}
                    for g, o, i in best_single_transfers(xp_by_gw, gws, weights, elements, squad, sell, bank,
                                                         keep=overrides.keep)]

    # --- top players by position (this gameweek) ---
    top_players = {}
    for pos, name in POSITION_NAMES.items():
        n = 3 if pos == 1 else 5
        best = sorted([pid for pid in elements if positions[pid] == pos], key=lambda p: -xp_now[p])[:n]
        top_players[name] = [player_info(pid) | headline(pid, gw) | {
            "xp": round(xp_now[pid], 1),
            "xp_horizon": round(sum(predictor.xp(pid, g) for g in gws), 1),
            "fixtures": fixture_labels(elements[pid]["team"], gw),
            "in_squad": pid in squad,
        } for pid in best]

    # Everything the chip planner needs, kept so /chips doesn't redo the transfer plans
    _chip_context[(team_id, overrides.key)] = {
        "built": _predictor_cache["built"], "history": history, "squad": squad,
        "plan_weeks": plans[recommended["transfers"]]["weeks"], "sell": sell, "bank": bank, "free": free,
    }

    return {
        "team_id": team_id,
        "overrides": {"starts": sorted(overrides.starts), "out": sorted(overrides.out), "keep": sorted(overrides.keep)},
        "team_name": entry.get("name"),
        "gameweek": gw,
        "gameweeks": gws,
        "bank": bank,
        "squad_value": squad_value,
        "free_transfers": free,
        "lineup_xp": round(lineup_xp, 1),
        "captain_options": captain_options,
        "squad": squad_rows,
        "transfers": {"options": options, "alternatives": alternatives, "free": free},
        "top_players": top_players,
        "alerts": squad_alerts(squad, elements, predictor, gw, player_info),
    }


def squad_alerts(squad, elements, predictor, gw, player_info):
    alerts = []
    ban_limit = 5 if gw <= 19 else 10 if gw <= 32 else None
    for pid in squad:
        p = elements[pid]
        info = player_info(pid)
        if p["status"] != "a" or (p["chance_of_playing_next_round"] or 100) < 100:
            chance = p["chance_of_playing_next_round"]
            updated = p.get("news_added")
            age = (time.time() - datetime.fromisoformat(updated.replace("Z", "+00:00")).timestamp()) / 86400 if updated else None
            alerts.append({"player": info, "type": "injury",
                           "text": p["news"] or "Flagged by FPL",
                           "detail": f"{chance}% chance of playing" if chance is not None else None,
                           # FPL often leaves flags unchanged for weeks; say how old the news is
                           "updated": updated, "stale": age is not None and age > 14})
        projections = p.get("price_change_projections") or []
        if projections and not p.get("price_change_locked_until"):
            likelihood = projections[0]["likelihood"]
            if likelihood >= 3:
                alerts.append({"player": info, "type": "price_up",
                               "text": "Likely to rise in price at the next update"})
            elif likelihood <= -3:
                alerts.append({"player": info, "type": "price_down",
                               "text": "Likely to drop in price at the next update"})
        if ban_limit and p["yellow_cards"] == ban_limit - 1:
            alerts.append({"player": info, "type": "suspension",
                           "text": f"On {p['yellow_cards']} yellow cards, one more means a ban"})
        if predictor.fixture_count(p["team"], gw) == 0:
            alerts.append({"player": info, "type": "blank", "text": f"No fixture in GW{gw}"})
        elif predictor.fixture_count(p["team"], gw) > 1:
            alerts.append({"player": info, "type": "double", "text": f"Plays twice in GW{gw}"})
    return alerts


def plan_chips(team_id, overrides=None):
    """The chip plan for a team (slow: many optimiser runs), reusing the last analysis's transfer plan."""
    overrides = overrides or Overrides()
    predictor = get_predictor()
    if overrides:
        predictor = AdjustedPredictor(predictor, overrides)
    context = _chip_context.get((team_id, overrides.key))
    if not context or context["built"] != _predictor_cache["built"]:
        analyse_team(team_id, overrides)
        context = _chip_context[(team_id, overrides.key)]
    bootstrap = predictor.bootstrap
    elements = {p["id"]: p for p in bootstrap["elements"]}
    teams = {t["id"]: t["short_name"] for t in bootstrap["teams"]}
    result = chip_plan(predictor, bootstrap, context["history"], context["squad"], context["plan_weeks"],
                       context["sell"], context["bank"], predictor.before_gw, context["free"])
    for chip in result["chips"]:
        ids = chip.pop("squad_ids", None)
        if ids:
            gws = range(chip["planned_gw"], chip["planned_gw"] + HORIZON)
            chip["squad"] = [{
                "name": elements[p]["web_name"], "team": teams[elements[p]["team"]],
                "position": POSITION_NAMES[elements[p]["element_type"]], "price": elements[p]["now_cost"] / 10,
                "xp_horizon": round(sum(predictor.xp(p, g) for g in gws if g in predictor.deadlines), 1),
                "new": p not in (context["squad"] if chip["planned_gw"] == predictor.before_gw
                                 else _planned_squad(context, chip["planned_gw"] - 1)),
            } for p in sorted(ids, key=lambda p: (elements[p]["element_type"], -elements[p]["now_cost"]))]
    return result


def _planned_squad(context, gw):
    weeks = {wk["gw"]: wk for wk in context["plan_weeks"]}
    return weeks[gw]["squad"] if gw in weeks else context["plan_weeks"][-1]["squad"]
