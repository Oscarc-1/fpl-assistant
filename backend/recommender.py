"""Builds everything the frontend shows for a manager, using the xP model and planner."""
import time
from concurrent.futures import ThreadPoolExecutor
from statistics import median

from fpl import (get_bootstrap_data, get_fixtures, get_all_element_summaries, get_current_gameweek,
                 get_active_squad, get_entry_history, get_entry_transfers, cached_get)
from accuracy import save_snapshot
from model import Predictor
from optimizer import best_single_transfers, optimise
from planner import (FUTURE_WEIGHTS, HIT_COST, HORIZON, MIN_GAIN_PER_MOVE, ROLL_VALUE, WILDCARD_THRESHOLD,
                     available_chips, best_lineup, free_transfers, selling_prices)

POSITION_NAMES = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
CHIP_NAMES = {"wildcard": "Wildcard", "freehit": "Free Hit", "bboost": "Bench Boost", "3xc": "Triple Captain"}

_predictor_cache = {"built": 0, "predictor": None}


def get_predictor():
    """Building the model takes a few seconds, so reuse it for 5 minutes."""
    if time.time() - _predictor_cache["built"] > 300:
        bootstrap = get_bootstrap_data()
        fixtures = get_fixtures()
        summaries = get_all_element_summaries([p["id"] for p in bootstrap["elements"]])
        gw = get_current_gameweek(bootstrap["events"])
        _predictor_cache["predictor"] = Predictor(bootstrap, fixtures, summaries, gw)
        _predictor_cache["built"] = time.time()
        save_snapshot(_predictor_cache["predictor"])
    return _predictor_cache["predictor"]


def analyse_team(team_id):
    predictor = get_predictor()
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

    squad_rows = []
    for pid in squad:
        info = player_info(pid)
        info.update({
            "xp": round(xp_now[pid], 1),
            "xp_by_gw": [{"gw": g, "xp": round(predictor.xp(pid, g), 1),
                          "fixtures": fixture_labels(elements[pid]["team"], g)} for g in gws],
            "xp_horizon": round(sum(predictor.xp(pid, g) for g in gws), 1),
            "selling_price": sell[pid],
            "starting": pid in starters,
            "bench_order": bench.index(pid) + 1 if pid in bench else None,
            "captain": pid == captain,
            "vice": pid == vice,
            "breakdown": predictor.breakdown(pid, gw),
        })
        squad_rows.append(info)
    squad_rows.sort(key=lambda r: (not r["starting"], r["bench_order"] or 0,
                                   list(POSITION_NAMES.values()).index(r["position"]), -r["xp"]))
    lineup_xp = sum(xp_now[p] for p in starters) + xp_now[captain]

    # Captain options: the top starters by predicted points, with their chance of a haul or a blank
    captain_options = []
    for pid in ranked[:5]:
        chances = predictor.upside(pid, gw)
        captain_options.append(player_info(pid) | {
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

    # Solve the plans in parallel (the solver runs outside Python's lock)
    counts = range(0, min(free + 2, 6) + 1)
    with ThreadPoolExecutor(max_workers=len(counts)) as pool:
        solved = pool.map(lambda k: optimise(xp_by_gw, gws, weights, elements, squad, sell, bank,
                                             transfers=k, free=free), counts)
    plans = {k: plan for k, plan in zip(counts, solved) if plan}
    base = plans[0]["value"]
    options = []
    for k, plan in plans.items():
        gain = plan["value"] + HIT_COST * plan["hits"] - base  # before hits
        net = plan["value"] - base + (ROLL_VALUE if k == 0 and free < 5 else 0.0)
        options.append({
            "label": "Save your transfer" if k == 0 else f"{k} transfer{'s' if k > 1 else ''}",
            "transfers": k,
            "moves": describe(plan),
            "gain": round(gain, 1),
            "hit": HIT_COST * plan["hits"],
            "net": round(net, 1),
            "best": False,
        })
    # Each extra transfer has to earn its place: at least MIN_GAIN_PER_MOVE more than the plan before it
    recommended = options[0]
    for option in options[1:]:
        if option["net"] >= recommended["net"] + MIN_GAIN_PER_MOVE * (option["transfers"] - recommended["transfers"]):
            recommended = option
    recommended["best"] = True

    # Alternatives: the best single transfer for each player you could sell
    alternatives = [{"move": describe({"out": [o], "in": [i]})[0], "gain": round(g, 1)}
                    for g, o, i in best_single_transfers(xp_by_gw, gws, weights, elements, squad, sell, bank)]

    # --- top players by position (this gameweek) ---
    top_players = {}
    for pos, name in POSITION_NAMES.items():
        n = 3 if pos == 1 else 5
        best = sorted([pid for pid in elements if positions[pid] == pos], key=lambda p: -xp_now[p])[:n]
        top_players[name] = [player_info(pid) | {
            "xp": round(xp_now[pid], 1),
            "xp_horizon": round(sum(predictor.xp(pid, g) for g in gws), 1),
            "fixtures": fixture_labels(elements[pid]["team"], gw),
            "in_squad": pid in squad,
        } for pid in best]

    return {
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
        "chips": chip_advice(predictor, bootstrap, history, squad, sell, bank, gw, elements, positions,
                             plans, recommended, player_info),
        "alerts": squad_alerts(squad, elements, predictor, gw, player_info),
    }


def chip_advice(predictor, bootstrap, history, squad, sell, bank, gw, elements, positions,
                plans, best_option, player_info):
    chips = available_chips(bootstrap, history, gw)
    last_gw = max(e["id"] for e in bootstrap["events"])
    advice = []
    for chip in chips:
        name = CHIP_NAMES[chip["name"]]
        entry = {"chip": name, "expires": chip["stop_event"]}

        if chip["name"] == "wildcard":
            # The best squad with unlimited free transfers, against the best normal plan
            gws = list(range(gw, min(gw + HORIZON, last_gw + 1)))
            xp_by_gw = {g: {pid: predictor.xp(pid, g) for pid in elements} for g in gws}
            wildcard = optimise(xp_by_gw, gws, FUTURE_WEIGHTS[:len(gws)], elements, squad, sell, bank)
            normal = plans[best_option["transfers"]]["value"] - plans[0]["value"]
            gain = wildcard["value"] - plans[0]["value"]
            extra = gain - normal
            entry.update({
                "gain": round(gain, 1),
                "normal_gain": round(normal, 1),
                "changes": len(wildcard["in"]),
                "squad": [player_info(p) | {"xp_horizon": round(sum(xp_by_gw[g][p] for g in gws), 1),
                                            "new": p not in squad}
                          for p in sorted(wildcard["squad"], key=lambda p: (positions[p], -elements[p]["now_cost"]))],
            })
            entry["advice"] = (
                f"Worth playing now: the best Wildcard squad gains {entry['gain']} pts over the next "
                f"{len(gws)} gameweeks, {round(extra, 1)} more than your best normal transfers."
                if extra >= WILDCARD_THRESHOLD else
                f"Hold it. A Wildcard now gains {entry['gain']} pts over {len(gws)} gameweeks, only "
                f"{round(extra, 1)} more than your best normal transfers ({entry['normal_gain']})."
            )
            advice.append(entry)
            continue

        window = list(range(gw, chip["stop_event"] + 1))
        by_gw = {}
        for g in window:
            xp = {pid: predictor.xp(pid, g) for pid in elements}
            starters, bench = best_lineup(squad, positions, xp)
            own = sum(xp[p] for p in starters) + max(xp[p] for p in starters)
            if chip["name"] == "3xc":
                by_gw[g] = max(xp[p] for p in starters)
            elif chip["name"] == "bboost":
                by_gw[g] = sum(xp[p] for p in bench)
            elif chip["name"] == "freehit":
                # The best one-week squad within your team's value, against your own best XI
                free_hit = optimise({g: xp}, [g], [1.0], elements, squad, sell, bank)
                by_gw[g] = free_hit["lineups"][g]["points"] - own

        best_gw = max(by_gw, key=by_gw.get)
        typical = median(by_gw.values())
        doubles = [g for g in window if any(predictor.fixture_count(elements[p]["team"], g) > 1 for p in squad)]
        blanks = [g for g in window if sum(predictor.fixture_count(elements[p]["team"], g) == 0 for p in squad) >= 3]
        entry.update({
            "best_gw": best_gw,
            "best_value": round(by_gw[best_gw], 1),
            "this_week_value": round(by_gw[gw], 1),
            "typical_value": round(typical, 1),
            "double_gws": doubles,
            "blank_gws": blanks,
        })
        if best_gw == gw and by_gw[gw] > typical * 1.15:
            entry["advice"] = "This looks like the best week to play it before it expires."
        else:
            entry["advice"] = (f"Hold it. GW{best_gw} currently looks best "
                               f"({entry['best_value']} pts vs {entry['this_week_value']} this week).")
        advice.append(entry)
    return advice


def squad_alerts(squad, elements, predictor, gw, player_info):
    alerts = []
    ban_limit = 5 if gw <= 19 else 10 if gw <= 32 else None
    for pid in squad:
        p = elements[pid]
        info = player_info(pid)
        if p["status"] != "a" or (p["chance_of_playing_next_round"] or 100) < 100:
            chance = p["chance_of_playing_next_round"]
            alerts.append({"player": info, "type": "injury",
                           "text": p["news"] or "Flagged by FPL",
                           "detail": f"{chance}% chance of playing" if chance is not None else None})
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
