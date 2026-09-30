# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

FPL (Fantasy Premier League) assistant: a Flask backend that pulls live data from the public FPL API, predicts expected points (xP) with its own model, and plans lineups, transfers and chips; plus a single static HTML page that displays the results. There is no database, no auth, no build step, and no test suite; `backtest.py` is the way to check model changes.

## Running

There is no `requirements.txt`; the only third-party dependencies are `flask` and `requests`.

Backend modules use bare imports (`from fpl import ...`), so run everything from inside `backend/`:

```bash
cd backend
python3 app.py                                              # dev server (debug) on http://127.0.0.1:5050
python3 -m flask --app app run --host=0.0.0.0 --port=5050   # reachable from a phone on the same Wi-Fi
python3 backtest.py                          # this season so far vs form / points-per-game baselines
python3 backtest.py --season 2025-26         # a full past season (~15s; downloads data on first run)
python3 backtest.py --season 2025-26 --tune  # tune PARAMS on odd GWs, report on even GWs (~10 min)
python3 backtest.py --season 2025-26 --tune-starts   # same for the start-chance calibration
```

Port 5000 is taken by macOS AirPlay Receiver (returns an empty 403), so use 5050. Everything hits `https://fantasy.premierleague.com/api` live; responses are cached as JSON in `backend/.cache/` (gitignored) with per-endpoint TTLs in `fpl.py`.

## Architecture

`fpl.py` → `model.py` → `planner.py` → `recommender.py` → `app.py` → `frontend/index.html`

- **`fpl.py`**: API wrappers with disk caching (`cached_get`). `get_all_element_summaries` fetches every player's `element-summary` (per-match history + `history_past` season totals) in parallel, which is about 4s cold. `get_active_squad` handles Free Hit reverts.
- **`model.py`**: the xP model. `Predictor(bootstrap, fixtures, summaries, before_gw)` only uses matches with `round < before_gw`, which is what makes backtesting honest. xP is a probability-weighted expectation summed over FPL's actual scoring rules (read from `game_config.scoring`), not a weighted score. Key ideas:
  - Team attack/defence ratings are fitted from this season's xG (summed from player rows), with FPL's `strength_overall_*` as a weak prior.
  - Per-90 rates are a steady blend: this season (opponent-adjusted, recency-decayed) plus past seasons worth `prior_minutes` of evidence, so this season takes over as minutes accumulate. There is deliberately no "role change" detection. It was tried, it over-reacted to short streaks, and the backtest preferred the steady blend. Players with no PL history get only a small prior (the average for their position scaled by their team's attack rating, `newcomer_prior_minutes`). Price is deliberately not used as a quality signal.
  - Penalties count in full, but are predicted from who takes them now rather than from past penalty xG. `match_penalties` estimates penalties a player took (FPL records misses but not scored penalties). A listed taker's scored, 0.76+ xG game counts as a penalty only in proportion to how unlikely that much open-play xG is for him (`TYPICAL_XG`), so big-chance strikers aren't stripped of open-play xG. They're removed from the `xg` rate and come back as the `penalties` part: long-run penalty rate (`PENALTY_RATE`) × team attack × opponent factor × `penalty_share` (club taker order, discounted by the chance a higher-order taker is on the pitch) × time on pitch. Keepers get penalty saves the same way.
  - Goals use xG × a per-player finishing multiplier and assists use xA × a per-player multiplier, both shrunk toward the league ratio (`finishing_shrink`, `assist_shrink`; fitted on season-to-season persistence). FPL awards about 1.4 assists per xA league-wide, so raw xA badly underestimates assists.
  - Starting uses a depth chart per club and position (`Predictor._build_depth_charts`, `start_chance`). Players are ranked by recent start share (`minutes_decay`, injury absences excluded via `_injury_cleared_at`). Each player's `picked` rate is learned only from matches where a place was open to him. The club's usual number of places (`starting_slots`) is then filled down the ranking by players who are fit (`fitness` = injury news × yellow-card ban risk) and picked (`selection_chance`, calibrated by `start_calibration`). A fit first choice starts ~95–98%, and a backup starts when someone ahead of him is out.
  - The backtest has no historical injury news, so it gives everyone `backtest_availability` (the share of regulars fit at a typical moment, measured from FPL's flags).
  - Clean sheets/goals conceded (`conceded_shape`) and defensive contributions (`dc_dispersion`) have spread settings. The full-season tuning found plain Poisson best for both.
  - Past seasons only feed a stat if that season recorded it (`season_has`): FPL shows xG/xA as 0 before 2022/23 and defensive contributions as absent before 2025/26, which otherwise look like real zeros.
  - `PARAMS` holds the tunable constants. Change them only with backtest evidence: tune on one season's odd GWs, then confirm on its even GWs **and** on another season (2024/25), because three gameweeks of the live season are far too noisy to tune on.
- **`history.py`**: loads past seasons from the public vaastav/Fantasy-Premier-League dataset (cached in `backend/.cache/history/`) into the same shapes as the live API, linking players across seasons by `code`. Limits: penalty order and positions are end-of-season snapshots, and the dataset's `xP` column leaks results, so it's not used.
- **`planner.py`**: `best_lineup` (formation-constrained greedy, which is optimal for these bounds), free-transfer replay from entry history, selling prices (FPL keeps half of any rise; Free Hit transfers ignored), `TransferPlanner` (singles plus pairs over a 6-GW horizon with future weeks discounted by 0.9, −4 hits, a small value for rolling), and a rough budget-aware Free Hit XI.
- **`recommender.py`**: `analyse_team` assembles the `/analysis` payload (squad xP and breakdowns, lineup and captain, transfer options, top players per position, chip advice, alerts). The `Predictor` is cached in memory for 5 minutes.
- **`app.py`**: `/` serves the frontend, `/analysis?team_id=` and `/price-changes` (FPL's own `price_change_*` fields: `price_change_projections[0]` is tonight with a −5..+5 `likelihood`; `price_change_percent` is progress where ±100 triggers a change; locked or calibrating players are skipped).
- **`frontend/index.html`**: a single self-contained file (inline CSS and JS, Google Fonts only) using same-origin `fetch`, so it must be served by Flask.

## FPL API conventions to keep in mind

- `now_cost`, `entry_history.bank`/`value` and transfer costs are in tenths of £m.
- `element_type` is the position: 1=GK, 2=DEF, 3=MID, 4=FWD.
- The current gameweek is the event with `is_next` (falling back to `is_current`). The squad comes from `gameweek - 1` picks, or `gameweek - 2` if that week's `active_chip == "freehit"`. The upcoming "Pick Team" squad (`/my-team/`) needs login, so pending transfers are invisible until the deadline.
- FPL's `ep_next` equals `form` for almost every player, so don't treat it as a prediction.
- `strength_attack_*`/`strength_defence_*` were all `0` in 2026/27; only `strength_overall_*` (2–5) is populated.
- Defensive contribution points need 10 (DEF) or 12 (MID/FWD) in the per-match `defensive_contribution` stat. The model's scoring reproduced every 2026/27 match row exactly.
- Chips come twice (GW1–19 and GW20–38); the windows are in `bootstrap["chips"]`. Up to 5 free transfers can be banked (`max_extra_free_transfers` + 1).
- FPL team IDs are per-season; an old ID 404s on the picks endpoint.
- Many numeric fields (`form`, `ep_next`, `selected_by_percent`, `expected_goals`, `threat`) arrive as strings.
