"""Expected points (xP) model.

A player's xP for a fixture is the probability-weighted average of every way FPL
awards points (minutes, goals, assists, clean sheets, goals conceded, defensive
contributions, saves, bonus, cards). Each part is built from:

  * how likely the player is to be available, start, and last 60 minutes
  * per-90 rates (xG, xA, ...) that blend this season with previous seasons, where
    previous seasons are trusted less the more this season disagrees with them
    (e.g. a player whose role has changed)
  * team attack/defence ratings fitted from this season's xG, used to adjust for
    the opponent and for home/away

Constants in PARAMS are chosen by backtest.py rather than by hand.
"""
import math
import re
import threading
from collections import defaultdict
from datetime import datetime, timezone

POSITIONS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
PENALTY_XG = 0.76           # xG of one penalty
PENALTY_CONVERSION = 0.78   # share of penalties scored
PENALTY_TAKE = 0.9          # chance the designated taker takes one while on the pitch
PENALTY_SAVED = 0.17        # share of penalties the keeper saves (the rest of the misses go wide)
PENALTY_RATE = 0.14         # penalties per team per match (long-run Premier League average)
TAKERS = set()              # players on a club's penalty list (set by Predictor)
TYPICAL_XG = {}             # player id -> typical xG per match, to judge how unusual a big-xG game is
PENALTY_MODE = ["prob"]     # set from PARAMS["penalty_mode"] by the Predictor
DC_THRESHOLD = {"GKP": None, "DEF": 10, "MID": 12, "FWD": 12}

# Tuned on 2025/26 odd gameweeks (`backtest.py --season 2025-26 --tune`), kept only after
# also improving 2025/26 even gameweeks and all of 2024/25.
PARAMS = {
    "decay": 0.8,               # weight kept per gameweek further in the past
    "minutes_decay": 0.55,      # same, for who starts (lineups change faster than underlying stats)
    "prior_minutes": 450,       # how much past seasons count, in minutes of this-season evidence
    "team_prior_matches": 16,   # how much FPL's pre-season team strength counts, in matches
    "team_prior_scale": 0.3,    # how far apart FPL's 2-5 strength ratings put teams
    "goal_weight": 0.0,         # share of actual goals (vs xG) in team ratings
    "start_prior_matches": 0.5, # how much last season's start rate counts, in matches
    "picked_prior": 0.75,       # chance a player is picked when a place is open to him, before evidence
    "picked_prior_matches": 1,  # how much that prior counts, in matches
    # Unknown in past gameweeks (no historical injury news), so the backtest assumes every player
    # has the league's typical chance of being fit: 7.6% of regular starters are injured or
    # suspended at any one time (measured from FPL's flags)
    "backtest_availability": 0.924,# Chance of being picked when a place is open = logistic(a + b * log-odds of his record),
    # fitted on past gameweeks by `backtest.py --tune-starts`
    "start_calibration": (-0.5, 1.5),
    "sub_prior_matches": 0.5,   # how much the typical chance of coming off the bench counts
    # "none": penalties stay inside xG; "hard"/"prob": taken out of xG and predicted from the taker order
    "penalty_mode": "prob",
    "dc_dispersion": 1.0,       # game-to-game spread of defensive contributions (1 = Poisson)
    "conceded_shape": 1000,     # uncertainty in a team's expected goals against (lower = more)
    "newcomer_prior_minutes": 90,# how much the position/team average counts for players with no PL history
    "finishing_shrink": 40,     # xG of evidence needed before a player's own finishing counts half
    "assist_shrink": 7,         # same for FPL assists vs xA (assists stick to players much more)
}

RATE_STATS = ["xg", "xa", "dc", "saves", "bonus", "yellow", "red", "own_goal"]
# Stats whose per-match count depends on the opponent's defence (attacking) or attack (saves)
ATTACKING = {"xg", "xa", "bonus"}


# ---------- probability helpers ----------

def poisson_tail(lam, n):
    """P(X >= n) for X ~ Poisson(lam)."""
    if n <= 0:
        return 1.0
    term = math.exp(-lam)
    cdf = term
    for k in range(1, n):
        term *= lam / k
        cdf += term
    return max(0.0, 1.0 - cdf)


def negbin_tail(mean, dispersion, n):
    """P(X >= n) for a count with the given mean and variance = dispersion * mean."""
    if mean <= 0:
        return 0.0 if n > 0 else 1.0
    if dispersion <= 1.01:
        return poisson_tail(mean, n)
    r = mean / (dispersion - 1)
    p = r / (r + mean)
    term = p ** r
    cdf = term
    for k in range(1, n):
        term *= (k - 1 + r) / k * (1 - p)
        cdf += term
    return max(0.0, 1.0 - cdf)


def mixed_poisson_zero(lam, shape):
    """P(X = 0) when X ~ Poisson(L) and L is uncertain (gamma with mean lam and this shape)."""
    return (1 + lam / shape) ** -shape


def expected_floor_div(lam, k, shape=None):
    """E[floor(X / k)] for a Poisson count, e.g. points lost per 2 goals conceded.

    With `shape`, the rate itself is uncertain (gamma-distributed), which widens the spread.
    """
    if shape is None:
        return sum(poisson_tail(lam, j * k) for j in range(1, 30))
    dispersion = 1 + lam / shape
    return sum(negbin_tail(lam, dispersion, j * k) for j in range(1, 30))


# ---------- helpers for the FPL data ----------

def previous_season_name(bootstrap):
    first_deadline = bootstrap["events"][0]["deadline_time"]
    year = int(first_deadline[:4])
    return f"{year - 1}/{str(year)[2:]}"


def season_start_year(season_name):
    return int(season_name[:4])


def match_rows(summary, before_gw):
    return [r for r in summary["history"] if r["round"] < before_gw]


def team_side(row, fixtures_by_id):
    fixture = fixtures_by_id[row["fixture"]]
    return fixture["team_h"] if row["was_home"] else fixture["team_a"]


# ---------- team ratings ----------

class TeamRatings:
    """Multiplicative attack/defence ratings: expected xG = mu * home * attack[team] * defence[opponent]."""

    def __init__(self, mu, home, attack, defence):
        self.mu = mu
        self.home = home
        self.attack = attack
        self.defence = defence

    def home_factor(self, is_home):
        return self.home if is_home else 1 / self.home

    def expected_goals(self, team, opponent, is_home):
        return self.mu * self.home_factor(is_home) * self.attack[team] * self.defence[opponent]


def fit_team_ratings(bootstrap, fixtures, summaries, before_gw, params=PARAMS):
    fixtures_by_id = {f["id"]: f for f in fixtures}
    team_ids = [t["id"] for t in bootstrap["teams"]]

    # Team xG in each fixture = sum of its players' xG
    xg = defaultdict(float)
    for summary in summaries.values():
        for row in match_rows(summary, before_gw):
            if row["fixture"] in fixtures_by_id:
                xg[(row["fixture"], team_side(row, fixtures_by_id))] += float(row["expected_goals"])

    matches = []  # (team, opponent, is_home, weight, attacking output)
    for f in fixtures:
        if not f["finished"] or f["event"] is None or f["event"] >= before_gw:
            continue
        weight = params["decay"] ** (before_gw - 1 - f["event"])
        gw_ = params["goal_weight"]
        home_out = (1 - gw_) * xg[(f["id"], f["team_h"])] + gw_ * f["team_h_score"]
        away_out = (1 - gw_) * xg[(f["id"], f["team_a"])] + gw_ * f["team_a_score"]
        matches.append((f["team_h"], f["team_a"], True, weight, home_out))
        matches.append((f["team_a"], f["team_h"], False, weight, away_out))

    # Prior from FPL's own pre-season strength (2-5 scale)
    prior_att, prior_def = {}, {}
    for t in bootstrap["teams"]:
        s = ((t.get("strength_overall_home") or 3) + (t.get("strength_overall_away") or 3)) / 2
        prior_att[t["id"]] = math.exp(params["team_prior_scale"] * (s - 3))
        prior_def[t["id"]] = math.exp(-params["team_prior_scale"] * (s - 3))

    attack, defence = dict(prior_att), dict(prior_def)
    if not matches:
        return TeamRatings(1.35, 1.1, attack, defence)

    total_w = sum(m[3] for m in matches)
    mu = sum(m[3] * m[4] for m in matches) / total_w
    home_out = sum(m[3] * m[4] for m in matches if m[2])
    away_out = sum(m[3] * m[4] for m in matches if not m[2])
    # Shrink the observed home advantage toward the usual ~10%
    k_home = 60
    home = math.sqrt((home_out + k_home * 1.1) / (away_out + k_home / 1.1))

    k = params["team_prior_matches"]
    for _ in range(20):
        num_a, den_a = defaultdict(float), defaultdict(float)
        num_d, den_d = defaultdict(float), defaultdict(float)
        for team, opp, is_home, w, out in matches:
            hf = home if is_home else 1 / home
            num_a[team] += w * out
            den_a[team] += w * mu * hf * defence[opp]
            num_d[opp] += w * out
            den_d[opp] += w * mu * hf * attack[team]
        attack = {t: (num_a[t] + k * prior_att[t]) / (den_a[t] + k) for t in team_ids}
        defence = {t: (num_d[t] + k * prior_def[t]) / (den_d[t] + k) for t in team_ids}
        # Keep the average team at 1.0 so mu stays the league average
        mean_a = sum(attack.values()) / len(attack)
        mean_d = sum(defence.values()) / len(defence)
        attack = {t: v / mean_a for t, v in attack.items()}
        defence = {t: v / mean_d for t, v in defence.items()}

    return TeamRatings(mu, home, attack, defence)


# ---------- player rates ----------

def match_penalties(row, is_taker=True):
    """Expected number of penalties a player took in a match.

    FPL records missed penalties but not scored ones. A scored penalty adds 0.76 xG, so a
    listed taker who scored in a game with 0.76+ xG *may* have scored one, but a striker can
    easily get that much xG from open play. So weigh the two explanations: how often his team
    gets a penalty, against how often he'd reach that xG from open play alone.
    """
    if PENALTY_MODE[0] == "none":
        return 0.0
    missed = row["penalties_missed"]
    if not is_taker:
        return missed
    xg = float(row["expected_goals"]) - PENALTY_XG * missed
    possible = min(int((xg + 0.02) // PENALTY_XG), row["goals_scored"])
    if possible <= 0:
        return missed
    if PENALTY_MODE[0] == "hard":
        return missed + possible
    prior = PENALTY_RATE * PENALTY_TAKE
    typical = max(TYPICAL_XG.get(row["element"], 0.1), 0.02)
    open_play_chance = math.exp(-PENALTY_XG / typical)  # chance of 0.76+ open-play xG in a match
    return missed + possible * prior / (prior + (1 - prior) * open_play_chance)


def season_penalties(season, is_taker=False):
    """Rough penalty attempts in a past season (only season totals are available).

    From misses for anyone; for a player who is his club's taker now, assume he also took
    his team's usual share then, so those penalties aren't counted twice (once inside his
    past xG and again in the separate penalty part).
    """
    if PENALTY_MODE[0] == "none":
        return 0
    missed = season["penalties_missed"]
    attempts = missed / (1 - PENALTY_CONVERSION)
    if is_taker:
        attempts = max(attempts, season["minutes"] / 90 * PENALTY_RATE * PENALTY_TAKE)
    cap = min(season["goals_scored"] + missed, float(season["expected_goals"]) / PENALTY_XG)
    return min(attempts, cap)


def row_stat(row, stat):
    return {
        "xg": lambda r: max(float(r["expected_goals"]) - PENALTY_XG * match_penalties(r, r["element"] in TAKERS), 0.0),
        "xa": lambda r: float(r["expected_assists"]),
        "dc": lambda r: r["defensive_contribution"],
        "saves": lambda r: r["saves"],
        "bonus": lambda r: r["bonus"],
        "yellow": lambda r: r["yellow_cards"],
        "red": lambda r: r["red_cards"],
        "own_goal": lambda r: r["own_goals"],
    }[stat](row)


def past_stat(season, stat, is_taker=False):
    return {
        "xg": lambda s: max(float(s["expected_goals"]) - PENALTY_XG * season_penalties(s, is_taker), 0.0),
        "xa": lambda s: float(s["expected_assists"]),
        "dc": lambda s: s.get("defensive_contribution") or 0,
        "saves": lambda s: s["saves"],
        "bonus": lambda s: s["bonus"],
        "yellow": lambda s: s["yellow_cards"],
        "red": lambda s: s["red_cards"],
        "own_goal": lambda s: s["own_goals"],
    }[stat](season)


def season_has(season, stat):
    """Whether a past season's totals include this stat. FPL added xG/xA in 2022/23 and
    defensive contributions in 2025/26; earlier seasons show 0, which would look like real data."""
    if stat in ("xg", "xa", "goals", "assists"):
        return season_start_year(season["season_name"]) >= 2022
    if stat == "dc":
        return "defensive_contribution" in season and season_start_year(season["season_name"]) >= 2025
    return True


def opponent_adjustment(row, stat, ratings, fixtures_by_id):
    """How much easier than average this match was for this stat (1.0 = average opponent)."""
    fixture = fixtures_by_id.get(row["fixture"])
    if fixture is None:
        return 1.0
    opp = row["opponent_team"]
    hf = ratings.home_factor(row["was_home"])
    if stat in ATTACKING:
        return ratings.defence[opp] * hf
    if stat == "saves":
        return ratings.attack[opp] / hf
    return 1.0


class PopulationStats:
    """League-wide facts the player model needs: position averages, conversion ratios and noise levels."""

    def __init__(self, bootstrap, summaries, before_gw, prev_season):
        self.position_mean = {}
        self.dispersion = {}
        elements = {p["id"]: p for p in bootstrap["elements"]}

        # Average per-90 rate by position among last season's regulars
        # (this season's matches if last season didn't record the stat)
        for pos in POSITIONS:
            for stat in RATE_STATS:
                total = minutes = 0.0
                for pid, s in summaries.items():
                    if elements[pid]["element_type"] != pos:
                        continue
                    past = [x for x in s["history_past"] if x["season_name"] == prev_season]
                    if not past or past[0]["minutes"] < 900 or not season_has(past[0], stat):
                        continue
                    total += past_stat(past[0], stat)
                    minutes += past[0]["minutes"]
                if not minutes:
                    for pid, s in summaries.items():
                        if elements[pid]["element_type"] == pos:
                            for r in match_rows(s, before_gw):
                                total += row_stat(r, stat)
                                minutes += r["minutes"]
                self.position_mean[(pos, stat)] = total / minutes * 90 if minutes else 0.0

        # Dispersion: variance of per-match counts relative to a Poisson-like baseline
        for stat in RATE_STATS:
            num = den = 0.0
            for s in summaries.values():
                rows = [r for r in match_rows(s, before_gw) if r["minutes"] >= 60]
                mins = sum(r["minutes"] for r in rows)
                if len(rows) < 3 or mins == 0:
                    continue
                total = sum(row_stat(r, stat) for r in rows)
                rate = total / mins
                for r in rows:
                    expected = rate * r["minutes"]
                    num += (row_stat(r, stat) - expected) ** 2
                    den += expected
            self.dispersion[stat] = max(num / den, 0.05) if den else 1.0

        # How many real goals / FPL assists the league gets per unit of xG / xA
        # (FPL awards assists xA doesn't count, e.g. rebounds and deflections)
        self.conversion = {}
        for stat, actual, expected in [("goals", "goals_scored", "expected_goals"),
                                       ("assists", "assists", "expected_assists")]:
            got = exp = 0.0
            for s in summaries.values():
                for x in s["history_past"][-3:]:
                    if season_has(x, stat):
                        got += x[actual]
                        exp += float(x[expected])
                for r in match_rows(s, before_gw):
                    got += r[actual]
                    exp += float(r[expected])
            self.conversion[stat] = got / exp if exp else 1.0

        self.penalty_rate = PENALTY_RATE

        # Start rate of players with no recent Premier League history
        starts = matches = 0
        for s in summaries.values():
            if any(x["season_name"] == prev_season and x["minutes"] > 0 for x in s["history_past"]):
                continue
            for r in match_rows(s, before_gw):
                starts += r["starts"]
                matches += 1
        self.newcomer_start = (starts + 0.3) / (matches + 1)

        # Typical minutes, measured from this season's matches (2025/26 values as a fallback)
        start_rows = [r for s in summaries.values() for r in match_rows(s, before_gw) if r["starts"]]
        sub_rows = [r for s in summaries.values() for r in match_rows(s, before_gw)
                    if not r["starts"] and r["minutes"] > 0]
        k = 200
        self.start_minutes = (sum(r["minutes"] for r in start_rows) + 83 * k) / (len(start_rows) + k)
        self.p60 = (sum(r["minutes"] >= 60 for r in start_rows) + 0.93 * k) / (len(start_rows) + k)
        self.sub_minutes = (sum(r["minutes"] for r in sub_rows) + 18 * k) / (len(sub_rows) + k)

    def team_prior_rate(self, pos, stat, team, ratings):
        """An average player in this position, adjusted for how good their team is."""
        rate = self.position_mean[(pos, stat)]
        if stat in ATTACKING:
            return rate * ratings.attack[team]
        if stat == "saves":
            return rate * ratings.defence[team]
        return rate


class PlayerModel:
    def __init__(self, player, summary, ratings, population, fixtures_by_id, before_gw,
                 prev_season, params=PARAMS, use_news=True):
        self.player = player
        self.id = player["id"]
        self.team = player["team"]
        self.pos = POSITIONS[player["element_type"]]
        self.params = params
        rows = match_rows(summary, before_gw)
        self.weights = [params["decay"] ** (before_gw - 1 - r["round"]) for r in rows]

        # Previous seasons: last season in full, older seasons count half per year
        prev_year = season_start_year(prev_season)
        past = []
        for s in summary["history_past"]:
            age = prev_year - season_start_year(s["season_name"])
            if 0 <= age <= 2 and s["minutes"] > 0:
                past.append((s, 0.5 ** age))

        # --- per-90 rates ---
        self.rates = {}
        this_rates, this_minutes = {}, sum(w * r["minutes"] for w, r in zip(self.weights, rows))
        for stat in RATE_STATS:
            adj_total = sum(w * row_stat(r, stat) / opponent_adjustment(r, stat, ratings, fixtures_by_id)
                            for w, r in zip(self.weights, rows) if r["minutes"] > 0)
            this_rates[stat] = adj_total / this_minutes * 90 if this_minutes else 0.0

        # Prior: past seasons if the player has them; otherwise an average player in the same
        # position at the same club, given only a little weight so this season's games dominate
        prior_rates = {}
        past_minutes = sum(s["minutes"] * w for s, w in past)
        for stat in RATE_STATS:
            team_rate = population.team_prior_rate(player["element_type"], stat, self.team, ratings)
            # Defensive contributions were only recorded from 2025/26 onwards
            seasons = [(s, w) for s, w in past if season_has(s, stat)]
            stat_minutes = sum(s["minutes"] * w for s, w in seasons)
            if stat_minutes > 0:
                past_rate = sum(past_stat(s, stat, self.id in TAKERS) * w for s, w in seasons) / stat_minutes * 90
                trust = stat_minutes / (stat_minutes + 900)
                prior_rates[stat] = trust * past_rate + (1 - trust) * team_rate
            else:
                prior_rates[stat] = team_rate
        trust_past = past_minutes / (past_minutes + 900)
        prior_weight = (trust_past * params["prior_minutes"]
                        + (1 - trust_past) * params["newcomer_prior_minutes"])

        # Steady blend: the more he plays this season, the more this season counts
        for stat in RATE_STATS:
            k = prior_weight
            total = this_minutes + k
            self.rates[stat] = ((this_rates[stat] * this_minutes + prior_rates[stat] * k) / total
                                if total else prior_rates[stat])
        self.dc_dispersion = params["dc_dispersion"]

        # Finishing: players who consistently beat xG (or collect more FPL assists than xA)
        # keep some of that edge. Pulled toward the league-wide ratio by `*_shrink` xG of evidence.
        self.finishing = {}
        for stat, actual, expected, shrink in [("goals", "goals_scored", "expected_goals", "finishing_shrink"),
                                               ("assists", "assists", "expected_assists", "assist_shrink")]:
            with_xg = [s for s, _ in past if season_has(s, stat)]
            got = sum(s[actual] for s in with_xg) + sum(r[actual] for r in rows)
            exp = sum(float(s[expected]) for s in with_xg) + sum(float(r[expected]) for r in rows)
            k = params[shrink]
            league = population.conversion[stat]
            self.finishing[stat] = (got + k * league) / (exp + k)

        # --- minutes ---
        minute_weights = [params["minutes_decay"] ** (before_gw - 1 - r["round"]) for r in rows]
        minute_rows = list(zip(minute_weights, rows))
        if use_news:
            # Trailing blank matches during an injury aren't the player being dropped: ignore them
            injured_until = self._injury_cleared_at()
            while minute_rows and minute_rows[-1][1]["minutes"] == 0:
                kickoff = datetime.fromisoformat(minute_rows[-1][1]["kickoff_time"].replace("Z", "+00:00"))
                if injured_until is None or kickoff > injured_until:
                    break
                minute_rows.pop()

        last = [s for s, w in past if w == 1.0]
        if last:
            prior_start = min(last[0]["starts"] / 34, 0.95)
        else:
            prior_start = population.newcomer_start
        # Share of recent matches started (injury absences excluded): ranks the club's depth chart
        n_eff = sum(w for w, _ in minute_rows)
        this_start = sum(w * r["starts"] for w, r in minute_rows) / n_eff if n_eff else prior_start
        k_start = params["start_prior_matches"]
        self.prior_start = prior_start
        self.p_start = (this_start * n_eff + prior_start * k_start) / (n_eff + k_start)
        self.match_starts = [(r["fixture"], w, r["starts"]) for w, r in minute_rows]
        self.picked = params["picked_prior"]  # set by the Predictor's depth chart

        self.penalty_order = player.get("penalties_order")
        self.penalty_rate = population.penalty_rate

        starts = [(w, r) for w, r in minute_rows if r["starts"]]
        subs = [(w, r) for w, r in minute_rows if not r["starts"]]
        self.p60 = shrunk_mean([(w, r["minutes"] >= 60) for w, r in starts], population.p60, 2)
        self.start_minutes = shrunk_mean([(w, r["minutes"]) for w, r in starts], population.start_minutes, 2)
        self.p_sub = shrunk_mean([(w, r["minutes"] > 0) for w, r in subs], 0.3, params["sub_prior_matches"])
        self.sub_minutes = shrunk_mean([(w, r["minutes"]) for w, r in subs if r["minutes"] > 0],
                                       population.sub_minutes, 2)

        self.use_news = use_news

    def _injury_cleared_at(self):
        """Until when blank matches can be put down to injury or suspension (None if they can't)."""
        if self.player["status"] in ("i", "d", "s"):
            return datetime.now(timezone.utc)  # still out
        added = self.player.get("news_added")
        if not added or self.player.get("news"):
            return None
        # Fit again: FPL cleared the news at this time, so only earlier blanks were injury
        return datetime.fromisoformat(added.replace("Z", "+00:00"))

    def availability(self, deadline):
        """Chance the player is available for a gameweek with this deadline."""
        if not self.use_news:
            return self.params["backtest_availability"]
        p = self.player
        status = p["status"]
        chance = p["chance_of_playing_next_round"]
        if status == "a" and chance is None:
            return 1.0
        if status == "u":
            return 0.0

        back = news_return_date(p.get("news") or "", deadline.year)
        if back is not None:
            return 1.0 if deadline >= back else 0.0

        now = datetime.now(timezone.utc)
        weeks_ahead = max((deadline - now).days / 7, 0)
        base = (chance if chance is not None else (0 if status in ("i", "s", "n") else 100)) / 100
        # Without a return date, assume a gradual recovery over the following weeks
        return min(1.0, base + (1 - base) * min(weeks_ahead / 4, 1.0) * (0.5 if status == "i" else 1.0))

    def fixture_points(self, fixture, ratings, scoring, deadline, p_start=None, penalty_share=0.0,
                       availability=None):
        """Expected points and a breakdown for one fixture.

        `p_start` is the chance of starting including availability; by default it's this
        player's own record, but the Predictor passes one adjusted for teammates' availability.
        """
        is_home = fixture["team_h"] == self.team
        opp = fixture["team_a"] if is_home else fixture["team_h"]
        a = self.availability(deadline) if availability is None else availability
        ps = a * self.p_start if p_start is None else min(p_start, a)
        psub = (a - ps) * self.p_sub
        exp_minutes = ps * self.start_minutes + psub * self.sub_minutes

        att_mult = ratings.defence[opp] * ratings.home_factor(is_home)
        conceded_90 = ratings.expected_goals(opp, self.team, not is_home)
        pos = self.pos
        r = self.rates

        parts = {}
        parts["appearance"] = ps * (self.p60 * 2 + (1 - self.p60)) + psub
        parts["goals"] = (r["xg"] * self.finishing["goals"] * exp_minutes / 90 * att_mult
                          * scoring["goals_scored"][pos])
        parts["assists"] = (r["xa"] * self.finishing["assists"] * exp_minutes / 90 * att_mult
                            * scoring["assists"])

        cs_minutes = max(self.start_minutes, 60)
        shape = self.params["conceded_shape"]
        parts["clean_sheet"] = (ps * self.p60 * mixed_poisson_zero(conceded_90 * cs_minutes / 90, shape)
                                * scoring["clean_sheets"][pos])
        gc_pts = scoring["goals_conceded"][pos]
        parts["goals_conceded"] = gc_pts * (
            ps * expected_floor_div(conceded_90 * self.start_minutes / 90, 2, shape)
            + psub * expected_floor_div(conceded_90 * self.sub_minutes / 90, 2, shape)) if gc_pts else 0.0

        threshold = DC_THRESHOLD[pos]
        if threshold and scoring["defensive_contribution"][pos]:
            dc = r["dc"]
            parts["defensive"] = scoring["defensive_contribution"][pos] * (
                ps * negbin_tail(dc * self.start_minutes / 90, self.dc_dispersion, threshold)
                + psub * negbin_tail(dc * self.sub_minutes / 90, self.dc_dispersion, threshold))
        else:
            parts["defensive"] = 0.0

        if pos == "GKP":
            save_mult = ratings.attack[opp] / ratings.home_factor(is_home)
            parts["saves"] = scoring["saves"] * (
                ps * expected_floor_div(r["saves"] * save_mult * self.start_minutes / 90, 3))
            # Penalty saves: the opponent's expected penalties x the share keepers save
            opp_pens = self.penalty_rate * ratings.attack[opp] * ratings.defence[self.team] / ratings.home_factor(is_home)
            parts["saves"] += (opp_pens * PENALTY_SAVED * ps * self.start_minutes / 90
                               * scoring["penalties_saved"])
        else:
            parts["saves"] = 0.0

        parts["bonus"] = r["bonus"] * exp_minutes / 90 * math.sqrt(att_mult)
        parts["cards"] = (r["yellow"] * scoring["yellow_cards"] + r["red"] * scoring["red_cards"]
                          + r["own_goal"] * scoring["own_goals"]) * exp_minutes / 90

        # Penalties: team's expected penalties x chance he's the one taking them while on the pitch
        team_pens = self.penalty_rate * ratings.attack[self.team] * att_mult
        attempts = team_pens * penalty_share * exp_minutes / 90
        parts["penalties"] = attempts * (PENALTY_CONVERSION * scoring["goals_scored"][pos]
                                         + (1 - PENALTY_CONVERSION) * scoring["penalties_missed"])

        return sum(parts.values()), parts, a


def shrunk_mean(pairs, prior, k):
    """Weighted mean of (weight, value) pairs, pulled toward `prior` as if it had weight k."""
    num = sum(w * float(v) for w, v in pairs) + prior * k
    den = sum(w for w, _ in pairs) + k
    return num / den


MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


def news_return_date(news, year):
    """Parse 'Expected back 18 Oct' / 'Suspended until 04 Oct' style news."""
    m = re.search(r"(?:back|until)\s+(\d{1,2})\s+([A-Za-z]{3})", news)
    if not m:
        return None
    month = MONTHS.get(m.group(2).lower())
    if not month:
        return None
    return datetime(year, month, int(m.group(1)), tzinfo=timezone.utc)


# ---------- putting it together ----------

_BUILD_LOCK = threading.Lock()  # building uses module-level tables (TAKERS, TYPICAL_XG, ...)


class Predictor:
    """Builds every player's model once, then answers xP questions for upcoming gameweeks."""

    def __init__(self, bootstrap, fixtures, summaries, before_gw, params=PARAMS, use_news=True):
        with _BUILD_LOCK:
            self._build(bootstrap, fixtures, summaries, before_gw, params, use_news)

    def _build(self, bootstrap, fixtures, summaries, before_gw, params, use_news):
        self.bootstrap = bootstrap
        self.fixtures = fixtures
        PENALTY_MODE[0] = params["penalty_mode"]
        TAKERS.clear()
        TAKERS.update(p["id"] for p in bootstrap["elements"] if p.get("penalties_order"))
        prev_season_name = previous_season_name(bootstrap)
        TYPICAL_XG.clear()
        for p in bootstrap["elements"]:
            summary = summaries.get(p["id"])
            if not summary:
                continue
            past = [x for x in summary["history_past"] if x["season_name"] == prev_season_name]
            rows = [r for r in match_rows(summary, before_gw) if r["minutes"] > 0]
            if past and past[0]["minutes"] >= 900:
                TYPICAL_XG[p["id"]] = float(past[0]["expected_goals"]) / past[0]["minutes"] * 90
            elif sum(r["minutes"] for r in rows) >= 270:
                TYPICAL_XG[p["id"]] = sum(float(r["expected_goals"]) for r in rows) / sum(r["minutes"] for r in rows) * 90
        self.scoring = bootstrap["game_config"]["scoring"]
        self.before_gw = before_gw
        fixtures_by_id = {f["id"]: f for f in fixtures}
        prev_season = previous_season_name(bootstrap)

        self.ratings = fit_team_ratings(bootstrap, fixtures, summaries, before_gw, params)
        population = PopulationStats(bootstrap, summaries, before_gw, prev_season)
        self.players = {}
        for p in bootstrap["elements"]:
            if p["id"] in summaries:
                self.players[p["id"]] = PlayerModel(p, summaries[p["id"]], self.ratings, population,
                                                    fixtures_by_id, before_gw, prev_season, params, use_news)

        self.deadlines = {}
        for e in bootstrap["events"]:
            self.deadlines[e["id"]] = datetime.fromisoformat(e["deadline_time"].replace("Z", "+00:00"))
        self.fixtures_by_gw_team = defaultdict(list)
        for f in fixtures:
            if f["event"] is not None:
                self.fixtures_by_gw_team[(f["event"], f["team_h"])].append(f)
                self.fixtures_by_gw_team[(f["event"], f["team_a"])].append(f)

        self.params = params
        self.slots = starting_slots(summaries, self.players, fixtures_by_id, before_gw, params)
        self.squads = defaultdict(list)
        self.squads_by_team = defaultdict(list)
        for model in self.players.values():
            self.squads[(model.team, model.pos)].append(model)
            self.squads_by_team[model.team].append(model)
        self._build_depth_charts(summaries, fixtures_by_id, before_gw)
        self._start_cache = {}
        self._cache = {}

    def _build_depth_charts(self, summaries, fixtures_by_id, before_gw):
        """Rank each club's players per position and learn how often each is picked when a place is open.

        A place is open to a player in a match if fewer players ranked above him started than the
        club started in that position. A backup who starts when the first choice is out is doing
        exactly what a backup does, so it doesn't make him a rival for the first-choice spot.
        """
        started = defaultdict(set)  # (team, pos, fixture) -> players who started
        for pid, model in self.players.items():
            for r in match_rows(summaries[pid], before_gw):
                if r["starts"] and r["fixture"] in fixtures_by_id:
                    started[(team_side(r, fixtures_by_id), model.pos, r["fixture"])].add(pid)

        self.depth = {}
        for (team, pos), group in self.squads.items():
            ranked = sorted(group, key=lambda m: (-m.p_start, -m.prior_start))
            rank = {m.id: i for i, m in enumerate(ranked)}
            for model in group:
                picked = chances = 0.0
                for fixture, w, did_start in model.match_starts:
                    starters = started.get((team, pos, fixture), set())
                    above = sum(1 for pid in starters if rank.get(pid, len(ranked)) < rank[model.id])
                    if did_start or above < len(starters):
                        picked += w * did_start
                        chances += w
                k = self.params["picked_prior_matches"]
                model.picked = (picked + k * self.params["picked_prior"]) / (chances + k)
            self.depth[(team, pos)] = ranked

    def fitness(self, model, gw):
        """Chance he's available: injury news, plus the risk of a yellow card ban."""
        return model.availability(self.deadlines[gw]) * (1 - self.ban_risk(model, gw))

    def ban_risk(self, model, gw):
        """Chance he's serving a one-match ban for reaching the yellow card limit.

        Premier League rule: 5 yellows by GW19 or 10 by GW32 means a ban. One booking away from
        the limit, a ban in a later gameweek means being booked in the match before it.
        """
        if not model.use_news:
            return 0.0
        yellows = model.player["yellow_cards"]
        for limit, last_gw in ((5, 19), (10, 32)):
            if yellows == limit - 1 and gw - 1 <= last_gw:
                ahead = gw - self.before_gw
                if ahead < 1:
                    return 0.0
                per_match = min(model.rates["yellow"] * model.start_minutes / 90, 0.9)
                return (1 - per_match) ** (ahead - 1) * per_match
        return 0.0

    def selection_chance(self, model):
        """Chance he's picked if a place is open to him (before fitness)."""
        a, b = self.params["start_calibration"]
        chance = 1 / (1 + math.exp(-(a + b * logit(model.picked))))
        return chance

    def start_chance(self, player_id, gw):
        """Chance of starting in `gw`: places are filled down the depth chart by players who are fit and picked."""
        model = self.players[player_id]
        key = (model.team, model.pos, gw)
        if key not in self._start_cache:
            slots = self.slots.get((model.team, model.pos), 0.0)
            low, frac = int(slots), slots - int(slots)
            ahead = [1.0]  # probability distribution of how many players ranked above are in contention
            chances = {}
            for m in self.depth[(model.team, model.pos)]:
                contender = self.fitness(m, gw) * self.selection_chance(m)
                room = (1 - frac) * sum(ahead[:low]) + frac * sum(ahead[:low + 1])
                chances[m.id] = contender * room
                nxt = [0.0] * (len(ahead) + 1)
                for k, p in enumerate(ahead):
                    nxt[k] += p * (1 - contender)
                    nxt[k + 1] += p * contender
                ahead = nxt
            self._start_cache[key] = chances
        return self._start_cache[key][player_id]

    def penalty_share(self, player_id, gw):
        """Chance he's the taker for a penalty awarded while he's on the pitch."""
        model = self.players[player_id]
        order = model.penalty_order
        if self.params["penalty_mode"] == "none":
            return 0.0  # penalties stay inside each player's xG
        if not order:
            return 0.02
        share = PENALTY_TAKE
        # Takers ahead of him in the order take it if they're on the pitch
        for rival in self.squads_by_team[model.team]:
            if rival.penalty_order and rival.penalty_order < order:
                on_pitch = self.start_chance(rival.id, gw) * rival.start_minutes / 90
                share *= 1 - on_pitch
        return share

    def xp(self, player_id, gw):
        """Expected points for a player in a gameweek (0 if their team has no fixture)."""
        key = (player_id, gw)
        if key not in self._cache:
            model = self.players.get(player_id)
            total = 0.0
            if model:
                ps = self.start_chance(player_id, gw)
                pen = self.penalty_share(player_id, gw)
                fit = self.fitness(model, gw)
                for f in self.fixtures_by_gw_team[(gw, model.team)]:
                    total += model.fixture_points(f, self.ratings, self.scoring, self.deadlines[gw],
                                                  ps, pen, fit)[0]
            self._cache[key] = total
        return self._cache[key]

    def breakdown(self, player_id, gw):
        model = self.players[player_id]
        totals = defaultdict(float)
        ps = self.start_chance(player_id, gw)
        pen = self.penalty_share(player_id, gw)
        fit = self.fitness(model, gw)
        for f in self.fixtures_by_gw_team[(gw, model.team)]:
            _, parts, _ = model.fixture_points(f, self.ratings, self.scoring, self.deadlines[gw], ps, pen, fit)
            for k, v in parts.items():
                totals[k] += v
        return {k: round(v, 2) for k, v in totals.items()}

    def fixture_count(self, team_id, gw):
        return len(self.fixtures_by_gw_team[(gw, team_id)])


def starting_slots(summaries, players, fixtures_by_id, before_gw, params):
    """How many players each club usually starts in each position (recent matches count more)."""
    counts = defaultdict(float)   # (team, pos) -> weighted starters
    matches = defaultdict(float)  # team -> weighted matches
    seen = set()
    for pid, model in players.items():
        for r in match_rows(summaries[pid], before_gw):
            if r["fixture"] not in fixtures_by_id:
                continue
            team = team_side(r, fixtures_by_id)
            w = params["minutes_decay"] ** (before_gw - 1 - r["round"])
            counts[(team, model.pos)] += w * r["starts"]
            if (team, r["fixture"]) not in seen:
                seen.add((team, r["fixture"]))
                matches[team] += w
    return {(team, pos): c / matches[team] for (team, pos), c in counts.items() if matches[team]}


def logit(p):
    p = min(max(p, 0.001), 0.999)
    return math.log(p / (1 - p))
