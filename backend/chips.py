"""Chip planner: when to play each chip before it expires.

Every chip is valued in every remaining gameweek of its window, using the squad the transfer plan
expects you to have that week. Then the chips are assigned to different weeks (one chip per
gameweek) to maximise the total. The plan is recomputed on every load, so an injury crisis or a
team collapsing moves a chip (usually the Wildcard) earlier on its own.

  Triple Captain: the best captain's points that week (the extra multiplier)
  Bench Boost:    the bench's points that week
  Free Hit:       the best one-week squad within your team value, minus your own best XI
  Wildcard:       the best squad over the next 6 weeks from that week, minus the best normal
                  transfer plan from that same week (the honest comparison: normal transfers
                  would also improve your team)
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from optimizer import optimise, plan_transfers
from planner import FT_VALUE, FUTURE_WEIGHTS, HORIZON, available_chips, best_lineup

CHIP_NAMES = {"wildcard": "Wildcard", "freehit": "Free Hit", "bboost": "Bench Boost", "3xc": "Triple Captain"}
BREAK_DAYS = 12  # a gap this long between deadlines means an international break


def deadline(bootstrap, gw):
    event = next(e for e in bootstrap["events"] if e["id"] == gw)
    return datetime.fromisoformat(event["deadline_time"].replace("Z", "+00:00"))


def gameweek_notes(predictor, bootstrap, squad_for, window):
    """Doubles, blanks and international breaks for each week of the window."""
    notes = {}
    for g in window:
        squad = squad_for(g)
        teams = {predictor.players[p].team for p in squad if p in predictor.players}
        doubles = sum(1 for p in squad if predictor.fixture_count(predictor.players[p].team, g) > 1)
        blanks = sum(1 for p in squad if predictor.fixture_count(predictor.players[p].team, g) == 0)
        league_doubles = sum(1 for t in bootstrap["teams"] if predictor.fixture_count(t["id"], g) > 1)
        league_blanks = sum(1 for t in bootstrap["teams"] if predictor.fixture_count(t["id"], g) == 0)
        after_break = g > 1 and (deadline(bootstrap, g) - deadline(bootstrap, g - 1)).days >= BREAK_DAYS
        notes[g] = {"gw": g, "your_doubles": doubles, "your_blanks": blanks, "doubles": league_doubles,
                    "blanks": league_blanks, "after_break": after_break, "teams": len(teams)}
    return notes


def chip_plan(predictor, bootstrap, history, current_squad, plan_weeks, sell, bank, gw, free):
    """Plan every available chip. `plan_weeks` is the recommended transfer plan's weeks."""
    elements = {p["id"]: p for p in bootstrap["elements"]}
    positions = {pid: p["element_type"] for pid, p in elements.items()}
    last_gw = max(e["id"] for e in bootstrap["events"])
    chips = available_chips(bootstrap, history, gw)
    if not chips:
        return {"chips": [], "timeline": []}

    stop = max(c["stop_event"] for c in chips)
    window = list(range(gw, stop + 1))
    planned = {wk["gw"]: wk for wk in plan_weeks}
    last_planned = plan_weeks[-1]

    def squad_for(g):
        """The squad the transfer plan expects you to have in week g (its last squad beyond the plan)."""
        return planned[g]["squad"] if g in planned else last_planned["squad"]

    def squad_before(g):
        """The squad going into week g, before that week's planned transfers (what a Wildcard replaces)."""
        return list(current_squad) if g == gw else squad_for(g - 1)

    def free_for(g):
        if g in planned:
            return planned[g]["free"]
        return min(5, last_planned["free"] + (g - last_planned["gw"]))

    sell_all = {p: sell.get(p, elements[p]["now_cost"] / 10) for p in elements}
    xp_cache = {}

    def xp(g):
        if g not in xp_cache:
            xp_cache[g] = {p: predictor.xp(p, g) for p in elements}
        return xp_cache[g]

    names = {c["name"] for c in chips}
    values = {name: {} for name in names}

    def week_values(g):
        pts = xp(g)
        squad = squad_for(g)
        starters, bench = best_lineup(squad, positions, pts)
        own = sum(pts[p] for p in starters) + max(pts[p] for p in starters)
        out = {}
        if "3xc" in names:
            out["3xc"] = max(pts[p] for p in starters)
        if "bboost" in names:
            out["bboost"] = sum(pts[p] for p in bench)
        if "freehit" in names:
            free_hit = optimise({g: pts}, [g], [1.0], elements, squad, sell_all, bank)
            out["freehit"] = free_hit["lineups"][g]["points"] - own if free_hit else 0.0
        if "wildcard" in names:
            gws = list(range(g, min(g + HORIZON, last_gw + 1)))
            weights = FUTURE_WEIGHTS[:len(gws)]
            horizon_xp = {h: xp(h) for h in gws}
            start = squad_before(g)
            wildcard = optimise(horizon_xp, gws, weights, elements, start, sell_all, bank)
            normal = plan_transfers(horizon_xp, gws, weights, elements, start, sell_all, bank, free_for(g), FT_VALUE)
            if wildcard and normal:
                # Free transfers are kept when a Wildcard is played, and keep building up after it
                banked = min(5, free_for(g) + len(gws) - 1)
                out["wildcard"] = wildcard["value"] + FT_VALUE * banked - normal["value"]
                out["wildcard_squad"] = wildcard["squad"]
        return g, out

    # Precompute predictions (the model isn't thread-safe while filling its caches), then solve in parallel
    for g in window:
        xp(g)
        for h in range(g, min(g + HORIZON, last_gw + 1)):
            xp(h)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = dict(pool.map(week_values, window))
    wildcard_squads = {}
    for g, out in results.items():
        for name in names:
            if name in out:
                values[name][g] = out[name]
        if "wildcard_squad" in out:
            wildcard_squads[g] = out["wildcard_squad"]

    # One chip per gameweek, each before its own deadline: pick the best combination of weeks
    chip_list = sorted(chips, key=lambda c: c["name"])
    best_total, best_weeks = None, None
    options = {c["name"]: [g for g in window if g <= c["stop_event"] and g in values[c["name"]]] for c in chip_list}
    for combo in _assignments(chip_list, options):
        total = sum(values[c["name"]][g] for c, g in zip(chip_list, combo))
        if best_total is None or total > best_total:
            best_total, best_weeks = total, combo

    notes = gameweek_notes(predictor, bootstrap, squad_for, window)
    plan = []
    for chip, week in zip(chip_list, best_weeks):
        name = chip["name"]
        by_week = values[name]
        ranked = sorted(by_week, key=by_week.get, reverse=True)
        entry = {
            "chip": CHIP_NAMES[name],
            "key": name,
            "expires": chip["stop_event"],
            "planned_gw": week,
            "planned_value": round(by_week[week], 1),
            "this_week_value": round(by_week.get(gw, 0.0), 1),
            "values": [{"gw": g, "value": round(by_week[g], 1)} for g in sorted(by_week)],
            "runner_up": next(({"gw": g, "value": round(by_week[g], 1)} for g in ranked if g != week), None),
            "reasons": _reasons(name, week, notes[week], by_week, gw),
        }
        entry["status"], entry["advice"] = advice(name, week, by_week[week], by_week, gw, chip["stop_event"])
        if name == "wildcard" and week in wildcard_squads:
            entry["squad_ids"] = wildcard_squads[week]
            entry["changes"] = len(set(wildcard_squads[week]) - set(squad_before(week)))
        plan.append(entry)
    plan.sort(key=lambda c: c["planned_gw"])
    return {"chips": plan, "timeline": [notes[g] for g in window]}


# A chip is only recommended for THIS week if it clearly stands out; otherwise hold it as insurance
PLAY_NOW = {"wildcard": 8.0, "freehit": 8.0}  # points gained (vs normal transfers / your own XI)
STANDOUT = 2.5                                 # Bench Boost / Triple Captain: points above a typical week


def advice(name, week, value, by_week, gw, expires):
    typical = sorted(by_week.values())[len(by_week) // 2]
    standout = value - typical >= STANDOUT and (name not in PLAY_NOW or value >= PLAY_NOW[name] / 2)
    if name in PLAY_NOW:
        strong = value >= PLAY_NOW[name]
    else:
        strong = standout
    if week == gw and strong:
        return "play", "Play it this week: this is clearly its best remaining week."
    if week != gw and (strong or standout):
        return "planned", f"Planned for GW{week}. Hold it until then unless injuries or form change the picture."
    return "hold", (f"Hold it. No week stands out yet (best is GW{week}, worth {value:.1f}). Keeping it gives you "
                    f"insurance against an injury crisis or a fixture swing, and the plan will update as double and "
                    f"blank gameweeks are announced. Use it by GW{expires}.")


def _assignments(chips, options):
    """Every way to give each chip a week from its options, with no two chips in the same week."""
    def rec(i, used):
        if i == len(chips):
            yield ()
            return
        for g in options[chips[i]["name"]]:
            if g not in used:
                for rest in rec(i + 1, used | {g}):
                    yield (g,) + rest
    yield from rec(0, frozenset())


def _reasons(name, week, note, by_week, gw):
    """Plain-English reasons a chip is planned for a week."""
    reasons = []
    typical = sorted(by_week.values())[len(by_week) // 2]
    if note["your_doubles"]:
        reasons.append(f"{note['your_doubles']} of your players have a double gameweek.")
    if note["your_blanks"]:
        reasons.append(f"{note['your_blanks']} of your players have no fixture.")
    if note["after_break"]:
        reasons.append("It's the first gameweek after an international break: wait for injury news before the deadline.")
    if name == "wildcard" and by_week[week] > typical:
        reasons.append("Your squad falls furthest behind the best possible squad around this week "
                       "(fixture swings, injuries or form), more than normal transfers can fix.")
    if name in ("3xc", "bboost") and by_week[week] > typical:
        reasons.append("Your players' fixtures are strongest this week.")
    return reasons
