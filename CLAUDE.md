# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

FPL (Fantasy Premier League) assistant: a Flask backend that pulls live data from the public FPL API, predicts expected points (xP) with its own model, and plans lineups, transfers and chips; plus a single static HTML page that displays the results. There is no database, no auth, no build step, and no test suite; `backtest.py` is the way to check model changes.

## Goal and deployment context
- This app will be linked from Oscar's portfolio website as a live project.
  Visitors will mostly be recruiters and classmates, often without an FPL team.
- Near-term goal (do this first): deploy a working public demo.
  - One service at one URL: Flask serves both the API and the frontend.
  - A "Try with an example team" button so visitors without a team ID can use it.
  - A clear loading state for the slow transfer/chip planning (5–15s).
  - Disable or hide the team-news feature in the public version (API cost,
    and a personal Claude subscription isn't for powering a public service).
  - Any secrets or keys go in environment variables, never in the code.
  - The repo will be made public, so keep it clean and include a good README.
- Long-term goal (don't build yet): a public product with many users, installable
  on phones, and a shared team-news feature using the paid API.
- The name shouldn't make it look like an official Premier League product,
  and the historical dataset must be credited.

## Running

Install dependencies with `python3 -m pip install -r requirements.txt` (flask, requests, pulp, highspy). The optimiser uses the HiGHS solver: PuLP's bundled CBC binary is Intel-only and fails on Apple Silicon with "Bad CPU type".

Backend modules use bare imports (`from fpl import ...`), so run everything from inside `backend/`:

```bash
cd backend
python3 app.py                                              # dev server (debug) on http://127.0.0.1:5050
python3 -m flask --app app run --host=0.0.0.0 --port=5050 --reload   # reachable from a phone; reloads on code changes
python3 backtest.py                          # this season so far vs form / points-per-game baselines
python3 backtest.py --season 2025-26         # a full past season (~15s; downloads data on first run)
python3 backtest.py --season 2025-26 --tune  # tune PARAMS on odd GWs, report on even GWs (~10 min)
python3 backtest.py --season 2025-26 --tune-starts   # same for the start-chance calibration
```

Public deployment is split in two. **GitHub Pages** hosts the page: `.github/workflows/publish-site.yml` runs `scripts/build_static.py` twice a day and on each push to `main`. That script runs the backend in-process and writes `site/` (the page, `config.js` with `XPOINTS_CONFIG = {static, api, exampleTeamId, generated}`, and `data/*.json` for the example team's analysis/chips/accuracy plus price changes). In static mode the page's `resolveURL` serves those files and sends other teams to `CONFIG.api` (Render), which allows the Pages origin via `ALLOWED_ORIGINS` (CORS). The workflow caches `backend/predictions` between runs so accuracy snapshots persist. The Render free instance is very slow for full team planning (minutes), which is why the example team is precomputed.

Production server (Render, `render.yaml`): `cd backend && gunicorn app:app --workers 1 --threads 4 --timeout 300`. Keep **one worker** so the model and caches are shared. Settings are environment variables: `EXAMPLE_TEAM_ID`, `WARM_EXAMPLE=1` (precompute the example team on startup and after each model rebuild), `MODEL_TTL_SECONDS` (1800 in production, 300 locally) and `MALLOC_ARENA_MAX=2`. The free plan has 512 MB RAM and very little CPU: optimiser solves are serialised app-wide (`optimizer._SOLVE_LOCK`), because parallel solves used over 1 GB; results are cached per team until the model rebuilds (`recommender.cached_result`); and the chip planner checks the Wildcard every week for 4 weeks, then every other week. `/health` is the health check and `/config` gives the page the example team.

Without `--reload`, the server refuses API requests with a 503 "restart the app" error once any backend `.py` file changes after startup (`refuse_stale_code`), because the frontend is served fresh and would otherwise mismatch. Port 5000 is taken by macOS AirPlay Receiver (returns an empty 403), so use 5050. Everything hits `https://fantasy.premierleague.com/api` live; responses are cached as JSON in `backend/.cache/` (gitignored) with per-endpoint TTLs in `fpl.py`.

## Architecture

`fpl.py` → `model.py` → `planner.py` → `recommender.py` → `app.py` → `frontend/index.html`

- **`fpl.py`**: API wrappers with disk caching (`cached_get`). `get_all_element_summaries` fetches every player's `element-summary` (per-match history + `history_past` season totals) in parallel, which is about 4s cold. `get_active_squad` handles Free Hit reverts.
- **`model.py`**: the xP model. Building a `Predictor` takes `_BUILD_LOCK` because it fills module-level tables (`TAKERS`, `TYPICAL_XG`, `PENALTY_MODE`). `Predictor(bootstrap, fixtures, summaries, before_gw)` only uses matches with `round < before_gw`, which is what makes backtesting honest. xP is a probability-weighted expectation summed over FPL's actual scoring rules (read from `game_config.scoring`), not a weighted score. Key ideas:
  - Team attack/defence ratings are fitted from this season's xG (summed from player rows), with FPL's `strength_overall_*` as a weak prior.
  - Per-90 rates are a steady blend: this season (opponent-adjusted, recency-decayed) plus past seasons worth `prior_minutes` of evidence, so this season takes over as minutes accumulate. There is deliberately no "role change" detection. It was tried, it over-reacted to short streaks, and the backtest preferred the steady blend. Players with no PL history get only a small prior (the average for their position scaled by their team's attack rating, `newcomer_prior_minutes`). Price is deliberately not used as a quality signal.
  - Penalties count in full, but are predicted from who takes them now rather than from past penalty xG. `match_penalties` estimates penalties a player took (FPL records misses but not scored penalties). A listed taker's scored, 0.76+ xG game counts as a penalty only in proportion to how unlikely that much open-play xG is for him (`TYPICAL_XG`), so big-chance strikers aren't stripped of open-play xG. They're removed from the `xg` rate and come back as the `penalties` part: long-run penalty rate (`PENALTY_RATE`) × team attack × opponent factor × `penalty_share` (club taker order, discounted by the chance a higher-order taker is on the pitch) × time on pitch. Keepers get penalty saves the same way.
  - One freak match can't dominate a rate: each match's opponent-adjusted xG/xA is capped per 90 (`match_cap`, validated on both past seasons; it doesn't clip consistently elite players).
  - Goals use xG × a per-player finishing multiplier and assists use xA × a per-player multiplier, both shrunk toward the league ratio (`finishing_shrink`, `assist_shrink`; fitted on season-to-season persistence). FPL awards about 1.4 assists per xA league-wide, so raw xA badly underestimates assists.
  - Starting uses a depth chart per club and position (`Predictor._build_depth_charts`, `start_chance`). Players are ranked by recent start share (`minutes_decay`, injury absences excluded via `_injury_cleared_at`). Each player's `picked` rate is learned only from matches where a place was open to him. The club's usual number of places (`starting_slots`) is then filled down the ranking by players who are fit (`fitness` = injury news × yellow-card ban risk) and picked (`selection_chance`, calibrated by `start_calibration`). A fit first choice starts ~95–98%, and a backup starts when someone ahead of him is out.
  - The backtest has no historical injury news, so it gives everyone `backtest_availability` (the share of regulars fit at a typical moment, measured from FPL's flags).
  - Clean sheets/goals conceded (`conceded_shape`) and defensive contributions (`dc_dispersion`) have spread settings. The full-season tuning found plain Poisson best for both.
  - Past seasons only feed a stat if that season recorded it (`season_has`): FPL shows xG/xA as 0 before 2022/23 and defensive contributions as absent before 2025/26, which otherwise look like real zeros.
  - `Predictor.simulate`/`upside` draw whole-gameweek outcomes from the same rates (used for captain haul/blank chances). Bonus is allocated to returns and scaled to the expected bonus. Validated on 2024/25 and 2025/26: predicted haul and blank chances match actual frequencies within a few points, though the biggest haul chances are slightly underestimated.
  - `PARAMS` holds the tunable constants. Change them only with backtest evidence: tune on one season's odd GWs, then confirm on its even GWs **and** on another season (2024/25), because three gameweeks of the live season are far too noisy to tune on.
- **`history.py`**: loads past seasons from the public vaastav/Fantasy-Premier-League dataset (cached in `backend/.cache/history/`) into the same shapes as the live API, linking players across seasons by `code`. Limits: penalty order and positions are end-of-season snapshots, and the dataset's `xP` column leaks results, so it's not used.
- **`planner.py`**: `best_lineup` (formation-constrained greedy, which is optimal for these bounds), free-transfer replay from entry history, selling prices (FPL keeps half of any rise; Free Hit transfers ignored), chip windows, and planning constants (`FUTURE_WEIGHTS` = measured prediction accuracy by weeks ahead, `HIT_COST`, `ROLL_VALUE`, `MIN_GAIN_PER_MOVE`, `WILDCARD_THRESHOLD`).
- **`optimizer.py`**: exact squad optimisation (PuLP + HiGHS). `optimise` finds the best squad for a fixed number of moves or unlimited moves (Wildcard/Free Hit), maximising weighted XI + captain points plus a little bench value under budget/selling prices, 2-5-5-3, max 3 per club and formation rules. `plan_transfers` is the multi-week planner. It tracks squad, bank, free transfers (+1/week, max 5) and hits week by week, so "move now with a hit" competes with "move next week for free", and free transfers banked after the horizon are worth `FT_VALUE` each. `recommender.py` solves it once per possible number of moves this week (in parallel threads) and recommends by `MIN_GAIN_PER_MOVE` (free moves) and `MIN_GAIN_PER_HIT` (moves that cost a hit, on top of the hit). `squad_value` mirrors the objective for a fixed squad, and `best_single_transfers` uses it for the alternatives list.
- **`chips.py`**: the chip planner behind `/chips` (slow, about 10s, so the page loads it after `/analysis`, reusing the transfer plan via `recommender._chip_context`). It values every available chip in every week up to its deadline, using the squad the recommended plan expects that week. The Wildcard is valued from the squad *before* that week's transfers, against the normal plan from the same week. Chips are assigned to distinct weeks by brute force, and `advice` only says "play this week" when the value clearly stands out (`PLAY_NOW`, `STANDOUT`); otherwise it says "hold" as insurance. International breaks are detected from deadline gaps of 12+ days.
- **`recommender.py`**: `analyse_team` assembles the `/analysis` payload. The page's headline number per player is `points_if_starts` with `start_chance` shown beside it, while lineup, captain and transfer decisions use the chance-adjusted `xp`. It also returns (squad xP and breakdowns, lineup and captain, transfer options, top players per position, chip advice, alerts). The `Predictor` is cached in memory for 5 minutes.
- **`accuracy.py`**: the Accuracy page. `save_snapshot` records the upcoming gameweek's predictions every time `get_predictor` builds the model before a deadline (`backend/predictions/<season>/gw<N>.json`, gitignored, source "live"). Finished gameweeks without a snapshot are rebuilt like the backtest (source "rebuilt", no injury news from the time) and cached. `accuracy_report` compares predictions with actual points and FPL form, and optionally the manager's own picks for the latest finished gameweek.
- **`app.py`**: `/` serves the frontend, `/analysis?team_id=`, `/chips?team_id=`, `/accuracy?team_id=` (team optional) and `/price-changes` (FPL's own `price_change_*` fields: `price_change_projections[0]` is tonight with a −5..+5 `likelihood`; `price_change_percent` is progress where ±100 triggers a change; locked or calibrating players are skipped).
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

