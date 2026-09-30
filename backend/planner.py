"""Squad decisions built on the xP model: lineup, free transfers, selling prices and chips.
The squad-building decisions (transfers, Wildcard, Free Hit) are solved in optimizer.py."""
from collections import Counter

FORMATION_MIN = {1: 1, 2: 3, 3: 2, 4: 1}
FORMATION_MAX = {1: 1, 2: 5, 3: 5, 4: 3}
HORIZON = 6                 # this gameweek and the next 5
# How much each future gameweek's prediction counts: its accuracy relative to this week's,
# measured over 2025/26 (backtest correlation 0, 1, ... 5 weeks ahead)
FUTURE_WEIGHTS = [1.0, 0.96, 0.89, 0.88, 0.84, 0.82]
HIT_COST = 4
ROLL_VALUE = 1.0            # rough worth of banking a free transfer for later
MIN_GAIN_PER_MOVE = 1.0     # an extra transfer must add at least this much to be recommended
WILDCARD_THRESHOLD = 15.0   # extra points over normal transfers before a Wildcard is recommended


# ---------- the manager's current position ----------

def free_transfers(history, entry_started, current_gw, max_banked):
    """Free transfers available for `current_gw`, replaying past gameweeks."""
    chips = {c["event"]: c["name"] for c in history["chips"]}
    by_event = {h["event"]: h for h in history["current"]}
    ft = 0
    for gw in range(entry_started + 1, current_gw + 1):
        ft = min(ft + 1, max_banked)
        prev = by_event.get(gw)
        if gw == current_gw or prev is None:
            continue
        if chips.get(gw) in ("wildcard", "freehit"):
            continue  # transfers on these chips don't use free transfers
        ft = max(ft - prev["event_transfers"], 0)
    return max(ft, 1)


def selling_prices(squad_ids, transfers, history, elements):
    """What each squad player sells for: FPL keeps half of any price rise (rounded down)."""
    free_hit_gws = {c["event"] for c in history["chips"] if c["name"] == "freehit"}
    bought_at = {}
    for t in sorted(transfers, key=lambda t: t["time"]):
        if t["event"] not in free_hit_gws:
            bought_at[t["element_in"]] = t["element_in_cost"]

    prices = {}
    for pid in squad_ids:
        p = elements[pid]
        now = p["now_cost"]
        bought = bought_at.get(pid, now - p["cost_change_start"])
        prices[pid] = (now if now <= bought else bought + (now - bought) // 2) / 10
    return prices


def available_chips(bootstrap, history, current_gw):
    """Chips usable now, with the last gameweek each can be played in."""
    used = [(c["name"], c["event"]) for c in history["chips"]]
    chips = []
    for chip in bootstrap["chips"]:
        if not chip["start_event"] <= current_gw <= chip["stop_event"]:
            continue
        spent = any(name == chip["name"] and chip["start_event"] <= ev <= chip["stop_event"]
                    for name, ev in used)
        if not spent:
            chips.append({"name": chip["name"], "stop_event": chip["stop_event"]})
    return chips


# ---------- lineup ----------

def best_lineup(squad, positions, xp):
    """Highest-scoring legal XI from a squad. Returns (starters, bench) ordered for FPL."""
    by_pos = {pos: sorted([p for p in squad if positions[p] == pos], key=lambda p: -xp[p])
              for pos in FORMATION_MIN}
    starters = []
    for pos, n in FORMATION_MIN.items():
        starters += by_pos[pos][:n]
    rest = sorted([p for pos in (2, 3, 4) for p in by_pos[pos][FORMATION_MIN[pos]:]], key=lambda p: -xp[p])
    counts = Counter(positions[p] for p in starters)
    for p in rest:
        if len(starters) == 11:
            break
        if counts[positions[p]] < FORMATION_MAX[positions[p]]:
            starters.append(p)
            counts[positions[p]] += 1
    bench_outfield = sorted([p for p in squad if p not in starters and positions[p] != 1], key=lambda p: -xp[p])
    bench_gk = [p for p in squad if p not in starters and positions[p] == 1]
    return starters, bench_gk + bench_outfield
