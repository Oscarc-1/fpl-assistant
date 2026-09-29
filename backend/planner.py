"""Squad decisions built on the xP model: lineup, captain, transfers and chips."""
from collections import Counter
from itertools import combinations

FORMATION_MIN = {1: 1, 2: 3, 3: 2, 4: 1}
FORMATION_MAX = {1: 1, 2: 5, 3: 5, 4: 3}
HORIZON = 6                 # this gameweek and the next 5
# How much each future gameweek's prediction counts: its accuracy relative to this week's,
# measured over 2025/26 (backtest correlation 0, 1, ... 5 weeks ahead)
FUTURE_WEIGHTS = [1.0, 0.96, 0.89, 0.88, 0.84, 0.82]
HIT_COST = 4
ROLL_VALUE = 1.0            # rough worth of banking a free transfer for later


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


def lineup_points(squad, positions, xp):
    """xP of the best XI plus the captain's points counted twice."""
    starters, _ = best_lineup(squad, positions, xp)
    return sum(xp[p] for p in starters) + max(xp[p] for p in starters)


# ---------- transfers ----------

class TransferPlanner:
    def __init__(self, predictor, bootstrap, squad, bank, sell_prices, gws):
        self.predictor = predictor
        self.elements = {p["id"]: p for p in bootstrap["elements"]}
        self.positions = {pid: p["element_type"] for pid, p in self.elements.items()}
        self.squad = list(squad)
        self.bank = bank
        self.sell = sell_prices
        self.gws = gws
        self.weights = [FUTURE_WEIGHTS[min(i, len(FUTURE_WEIGHTS) - 1)] for i in range(len(gws))]
        self.xp = {gw: {pid: predictor.xp(pid, gw) for pid in self.elements} for gw in gws}
        self.total = {pid: sum(self.xp[gw][pid] for gw in gws) for pid in self.elements}

    def squad_value(self, squad):
        return sum(w * lineup_points(squad, self.positions, self.xp[gw]) for gw, w in zip(self.gws, self.weights))

    def _valid(self, squad, spend):
        teams = Counter(self.elements[p]["team"] for p in squad)
        return spend <= self.bank + 1e-9 and max(teams.values()) <= 3

    def _candidates(self, pos, limit=40):
        pool = [p for p in self.elements if self.positions[p] == pos and p not in self.squad
                and self.elements[p]["status"] != "u" and self.elements[p]["can_transact"]]
        return sorted(pool, key=lambda p: -self.total[p])[:limit]

    def price(self, pid):
        return self.elements[pid]["now_cost"] / 10

    def plan(self, free):
        base = self.squad_value(self.squad)
        singles = []
        for out in self.squad:
            pos = self.positions[out]
            for inn in self._candidates(pos):
                spend = self.price(inn) - self.sell[out]
                new = [inn if p == out else p for p in self.squad]
                if self._valid(new, spend):
                    singles.append((self.squad_value(new) - base, out, inn, spend))
        singles.sort(key=lambda s: -s[0])

        # Pairs: the best single moves, plus moves that free up money for an upgrade
        cash_freeing = sorted([s for s in singles if s[3] < 0], key=lambda s: s[3] / max(abs(s[0]), 0.1))[:15]
        pool = {(s[1], s[2]): s for s in singles[:30] + cash_freeing}.values()
        doubles = []
        for a, b in combinations(pool, 2):
            if a[1] == b[1] or a[2] == b[2]:
                continue
            new = [a[2] if p == a[1] else b[2] if p == b[1] else p for p in self.squad]
            spend = a[3] + b[3]
            if self._valid(new, spend):
                doubles.append((self.squad_value(new) - base, [(a[1], a[2]), (b[1], b[2])], spend))
        doubles.sort(key=lambda d: -d[0])

        options = [{"transfers": [], "gain": 0.0, "hit": 0,
                    "net": ROLL_VALUE if free < 5 else 0.0}]
        if singles:
            g, out, inn, spend = singles[0]
            hit = HIT_COST * max(0, 1 - free)
            options.append({"transfers": [(out, inn)], "gain": g, "hit": hit, "net": g - hit, "spend": spend})
        if doubles:
            g, moves, spend = doubles[0]
            hit = HIT_COST * max(0, 2 - free)
            # Using both free transfers also gives up banking one
            options.append({"transfers": moves, "gain": g, "hit": hit, "net": g - hit, "spend": spend})
        best = max(options, key=lambda o: o["net"])

        alternatives = []
        seen_out = set()
        for g, out, inn, spend in singles:
            if out in seen_out:
                continue
            seen_out.add(out)
            alternatives.append({"out": out, "in": inn, "gain": g, "spend": spend})
            if len(alternatives) == 5:
                break
        return {"options": options, "best": best, "alternatives": alternatives}


# ---------- free hit team ----------

def free_hit_points(predictor, elements, positions, budget, gw):
    """Rough best XI (plus captain) available on a Free Hit, within budget and max 3 per club."""
    xp = {pid: predictor.xp(pid, gw) for pid in elements}
    pool = sorted([p for p in elements if elements[p]["status"] != "u"], key=lambda p: -xp[p])
    bench_cost = 4.0 * 4  # four cheapest bench players
    budget -= bench_cost
    chosen, teams, counts, spent = [], Counter(), Counter(), 0.0
    for p in pool:
        if len(chosen) == 11:
            break
        pos, team, price = positions[p], elements[p]["team"], elements[p]["now_cost"] / 10
        if counts[pos] >= FORMATION_MAX[pos] or teams[team] >= 3:
            continue
        # Leave room (and money, at the cheapest price) for every slot still to fill
        slots_left = 11 - len(chosen) - 1
        still_needed = sum(max(FORMATION_MIN[q] - counts[q] - (q == pos), 0) for q in FORMATION_MIN)
        if still_needed > slots_left or spent + price + slots_left * 4.0 > budget:
            continue
        chosen.append(p)
        counts[pos] += 1
        teams[team] += 1
        spent += price
    if not chosen:
        return 0.0
    return sum(xp[p] for p in chosen) + max(xp[p] for p in chosen)
