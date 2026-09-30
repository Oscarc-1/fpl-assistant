"""Squad optimiser: the best 15-man squad under FPL's rules, solved exactly with PuLP + HiGHS.

One model covers every decision: transfers (with a set number of moves and hits), a Wildcard
(any number of moves, no hits) and a Free Hit (one gameweek, squad reverts afterwards).
It maximises predicted points of the best XI plus captain in each gameweek of the horizon,
weighted by how reliable predictions that far ahead are, plus a little for the bench
(which scores when a starter doesn't play).
"""
from collections import defaultdict

import pulp

SQUAD = {1: 2, 2: 5, 3: 5, 4: 3}
XI_MIN = {1: 1, 2: 3, 3: 2, 4: 1}
XI_MAX = {1: 1, 2: 5, 3: 5, 4: 3}
CLUB_LIMIT = 3
BENCH_WEIGHT = {1: 0.02, 2: 0.1, 3: 0.1, 4: 0.1}  # bench keepers almost never come on
CANDIDATES_PER_POSITION = 45


def candidates(xp, gws, weights, positions, elements, keep):
    """Players worth considering: the best by predicted points, the cheapest, and the current squad."""
    total = {p: sum(w * xp[g].get(p, 0.0) for g, w in zip(gws, weights)) for p in positions}
    pool = set(keep)
    for pos in SQUAD:
        players = [p for p in positions if positions[p] == pos and elements[p]["status"] != "u"
                   and elements[p].get("can_transact", True)]
        pool.update(sorted(players, key=lambda p: -total[p])[:CANDIDATES_PER_POSITION])
        pool.update(sorted(players, key=lambda p: elements[p]["now_cost"])[:6])  # budget fillers
    return sorted(pool)


def optimise(xp, gws, weights, elements, current, sell, bank, transfers=None, free=1, hit_cost=4):
    """Best squad over `gws`.

    xp: {gw: {player: predicted points}}; current: current squad (ids); sell: selling prices (£m)
    transfers: exact number of transfers to make, or None for unlimited (Wildcard / Free Hit)
    free: free transfers available (hits are charged beyond these when `transfers` is set)
    Returns dict with squad, per-gameweek lineup/captain, transfers in/out, value and cost.
    """
    positions = {p: e["element_type"] for p, e in elements.items()}
    pool = candidates(xp, gws, weights, positions, elements, current)
    current = set(current)
    cost = {p: sell[p] if p in current else elements[p]["now_cost"] / 10 for p in pool}
    budget = bank + sum(sell[p] for p in current)

    prob = pulp.LpProblem("fpl", pulp.LpMaximize)
    squad = {p: pulp.LpVariable(f"s_{p}", cat="Binary") for p in pool}
    xi = {(p, g): pulp.LpVariable(f"x_{p}_{g}", cat="Binary") for p in pool for g in gws}
    cap = {(p, g): pulp.LpVariable(f"c_{p}_{g}", cat="Binary") for p in pool for g in gws}

    objective = []
    for g, w in zip(gws, weights):
        for p in pool:
            pts = xp[g].get(p, 0.0)
            objective.append(w * pts * (xi[p, g] + cap[p, g]))
            objective.append(w * BENCH_WEIGHT[positions[p]] * pts * (squad[p] - xi[p, g]))

    moves = pulp.lpSum(squad[p] for p in pool if p not in current)
    hits = pulp.LpVariable("hits", lowBound=0, cat="Integer")
    if transfers is not None:
        prob += moves == transfers
        prob += hits >= transfers - free
        objective.append(-hit_cost * hits)
    else:
        prob += hits == 0
    prob += pulp.lpSum(objective)

    # Squad rules
    prob += pulp.lpSum(cost[p] * squad[p] for p in pool) <= budget + 1e-6
    for pos, n in SQUAD.items():
        prob += pulp.lpSum(squad[p] for p in pool if positions[p] == pos) == n
    by_club = defaultdict(list)
    for p in pool:
        by_club[elements[p]["team"]].append(p)
    for players in by_club.values():
        prob += pulp.lpSum(squad[p] for p in players) <= CLUB_LIMIT

    # Lineup rules each gameweek
    for g in gws:
        prob += pulp.lpSum(xi[p, g] for p in pool) == 11
        prob += pulp.lpSum(cap[p, g] for p in pool) == 1
        for pos in SQUAD:
            n = pulp.lpSum(xi[p, g] for p in pool if positions[p] == pos)
            prob += n >= XI_MIN[pos]
            prob += n <= XI_MAX[pos]
        for p in pool:
            prob += xi[p, g] <= squad[p]
            prob += cap[p, g] <= xi[p, g]

    prob.solve(pulp.HiGHS(msg=False, timeLimit=20))
    if pulp.LpStatus[prob.status] != "Optimal":
        return None

    chosen = [p for p in pool if squad[p].value() > 0.5]
    lineups = {}
    for g in gws:
        starters = [p for p in chosen if xi[p, g].value() > 0.5]
        captain = next(p for p in chosen if cap[p, g].value() > 0.5)
        lineups[g] = {"starters": starters, "captain": captain,
                      "points": sum(xp[g].get(p, 0.0) for p in starters) + xp[g].get(captain, 0.0)}
    n_moves = sum(1 for p in chosen if p not in current)
    return {
        "squad": chosen,
        "lineups": lineups,
        "in": sorted([p for p in chosen if p not in current], key=lambda p: positions[p]),
        "out": sorted([p for p in current if p not in chosen], key=lambda p: positions[p]),
        "hits": max(0, n_moves - free) if transfers is not None else 0,
        "value": pulp.value(prob.objective),
        "spend": sum(cost[p] for p in chosen),
        "budget": budget,
    }


def squad_value(squad, xp, gws, weights, positions):
    """The optimiser's objective for a fixed squad: best XI + captain each gameweek, plus a little
    for the bench (so values are comparable with `optimise(...)["value"]` before hits)."""
    from planner import best_lineup
    total = 0.0
    for g, w in zip(gws, weights):
        pts = {p: xp[g].get(p, 0.0) for p in squad}
        starters, bench = best_lineup(squad, positions, pts)
        total += w * (sum(pts[p] for p in starters) + max(pts[p] for p in starters)
                      + sum(BENCH_WEIGHT[positions[p]] * pts[p] for p in bench))
    return total


def best_single_transfers(xp, gws, weights, elements, current, sell, bank, limit=5):
    """The best single transfer for each player you could sell, best first (one per player sold)."""
    positions = {p: e["element_type"] for p, e in elements.items()}
    pool = candidates(xp, gws, weights, positions, elements, current)
    base = squad_value(current, xp, gws, weights, positions)
    clubs = defaultdict(int)
    for p in current:
        clubs[elements[p]["team"]] += 1
    options = []
    for out in current:
        best = None
        for inn in pool:
            if inn in current or positions[inn] != positions[out]:
                continue
            if elements[inn]["now_cost"] / 10 > bank + sell[out] + 1e-6:
                continue
            same_club = elements[inn]["team"] == elements[out]["team"]
            if clubs[elements[inn]["team"]] - same_club >= CLUB_LIMIT:
                continue
            new = [inn if p == out else p for p in current]
            gain = squad_value(new, xp, gws, weights, positions) - base
            if best is None or gain > best[0]:
                best = (gain, out, inn)
        if best and best[0] > 0:
            options.append(best)
    options.sort(key=lambda x: -x[0])
    return options[:limit]


def plan_transfers(xp, gws, weights, elements, current, sell, bank, free, ft_value,
                   moves_now=None, max_banked=5, hit_cost=4):
    """Plan transfers week by week over `gws` (the first entry is the upcoming gameweek).

    Tracks free transfers (+1 a week, up to `max_banked`), the bank, and hits week by week, so
    "make it now with a hit" is weighed against "make it next week for free". Free transfers still
    banked after the horizon are worth `ft_value` each (flexibility for injuries and later moves).
    `moves_now` fixes how many transfers are made this week (None = let the plan decide).
    """
    positions = {p: e["element_type"] for p, e in elements.items()}
    pool = candidates(xp, gws, weights, positions, elements, current)
    current = set(current)
    price = {p: elements[p]["now_cost"] / 10 for p in pool}
    sale = {p: sell[p] if p in current else price[p] for p in pool}  # price changes ignored ahead

    prob = pulp.LpProblem("fpl_plan", pulp.LpMaximize)
    squad, buy, sold, xi, cap = {}, {}, {}, {}, {}
    for g in gws:
        for p in pool:
            squad[p, g] = pulp.LpVariable(f"s_{p}_{g}", cat="Binary")
            buy[p, g] = pulp.LpVariable(f"b_{p}_{g}", cat="Binary")
            sold[p, g] = pulp.LpVariable(f"o_{p}_{g}", cat="Binary")
            xi[p, g] = pulp.LpVariable(f"x_{p}_{g}", cat="Binary")
            cap[p, g] = pulp.LpVariable(f"c_{p}_{g}", cat="Binary")
    ft = {g: pulp.LpVariable(f"ft_{g}", lowBound=0, upBound=max_banked, cat="Integer") for g in gws}
    ft_end = pulp.LpVariable("ft_end", lowBound=1, upBound=max_banked, cat="Integer")
    hits = {g: pulp.LpVariable(f"h_{g}", lowBound=0, cat="Integer") for g in gws}
    money = {g: pulp.LpVariable(f"m_{g}", lowBound=0) for g in gws}

    objective = []
    prev_money = bank
    for i, g in enumerate(gws):
        w = weights[i]
        n_moves = pulp.lpSum(buy[p, g] for p in pool)
        for p in pool:
            before = (1 if p in current else 0) if i == 0 else squad[p, gws[i - 1]]
            prob += squad[p, g] == before + buy[p, g] - sold[p, g]
            prob += buy[p, g] + sold[p, g] <= 1
            pts = xp[g].get(p, 0.0)
            objective.append(w * pts * (xi[p, g] + cap[p, g]))
            objective.append(w * BENCH_WEIGHT[positions[p]] * pts * (squad[p, g] - xi[p, g]))
        # Money: sales fund purchases, week by week
        prob += money[g] == prev_money + pulp.lpSum(sale[p] * sold[p, g] - price[p] * buy[p, g] for p in pool)
        prev_money = money[g]
        # Free transfers: hits for moves beyond them; unused ones roll over (+1 next week, capped)
        if i == 0:
            prob += ft[g] == free
        prob += hits[g] >= n_moves - ft[g]
        nxt = ft[gws[i + 1]] if i + 1 < len(gws) else ft_end
        prob += nxt <= ft[g] - n_moves + hits[g] + 1
        objective.append(-hit_cost * hits[g])
        # Squad and lineup rules
        for pos, count in SQUAD.items():
            prob += pulp.lpSum(squad[p, g] for p in pool if positions[p] == pos) == count
        by_club = defaultdict(list)
        for p in pool:
            by_club[elements[p]["team"]].append(p)
        for players in by_club.values():
            prob += pulp.lpSum(squad[p, g] for p in players) <= CLUB_LIMIT
        prob += pulp.lpSum(xi[p, g] for p in pool) == 11
        prob += pulp.lpSum(cap[p, g] for p in pool) == 1
        for pos in SQUAD:
            n = pulp.lpSum(xi[p, g] for p in pool if positions[p] == pos)
            prob += n >= XI_MIN[pos]
            prob += n <= XI_MAX[pos]
        for p in pool:
            prob += xi[p, g] <= squad[p, g]
            prob += cap[p, g] <= xi[p, g]
    if moves_now is not None:
        prob += pulp.lpSum(buy[p, gws[0]] for p in pool) == moves_now
    objective.append(ft_value * ft_end)
    prob += pulp.lpSum(objective)

    prob.solve(pulp.HiGHS(msg=False, timeLimit=30, mip_rel_gap=0.002))
    if pulp.LpStatus[prob.status] not in ("Optimal", "Not Solved") or prob.objective is None:
        return None

    weeks = []
    for g in gws:
        weeks.append({
            "gw": g,
            "in": sorted([p for p in pool if buy[p, g].value() > 0.5], key=lambda p: positions[p]),
            "out": sorted([p for p in pool if sold[p, g].value() > 0.5], key=lambda p: positions[p]),
            "free": round(ft[g].value()),
            "hits": round(hits[g].value()),
            "squad": [p for p in pool if squad[p, g].value() > 0.5],
        })
    return {"weeks": weeks, "value": pulp.value(prob.objective), "ft_end": round(ft_end.value())}
