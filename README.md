# xPoints

**A fantasy football planner.** xPoints predicts how many points every player will score, then plans your best lineup and captain, which transfers to make (and when), and when to play each chip.

Enter a Fantasy Premier League team ID, or try it with the example team.

> xPoints is an independent project and isn't affiliated with or endorsed by the Premier League or Fantasy Premier League.

## What it does

- **Predicted points for every player**, shown as "points if he starts" alongside his chance of starting.
- **Best lineup and captain** for the coming gameweek, with each captain option's chance of a big haul or a blank.
- **Transfer plan across the next 6 gameweeks.** It knows you gain a free transfer each week, so it only suggests a points hit when that clearly beats waiting a week, and it values keeping transfers in the bank.
- **Chip plan for the half-season.** Wildcard, Free Hit, Bench Boost and Triple Captain are valued in every remaining gameweek. It says to play one only when a week clearly stands out; otherwise it advises holding it as insurance.
- **Accuracy tracking.** Each gameweek's predictions are saved before the deadline and compared with what players actually scored.
- **Price change alerts** from FPL's own price projections.

## How the predictions work

FPL's own "expected points" is essentially recent form. xPoints instead builds each prediction from FPL's scoring rules, as an average over everything that could happen in a match, weighted by how likely it is:

- **Minutes:** the chance a player is fit and starts, from each club's depth chart (who starts when everyone is available, and who steps in when someone is injured) and FPL's injury news.
- **Goals and assists:** expected goals (xG) and assists (xA) per 90 minutes, adjusted for the opponent, with a per-player finishing record. Penalties are predicted from each club's current taker order. A single freak match (a hat-trick, say) is capped so it can't dominate a rating.
- **Clean sheets, goals conceded, defensive contributions, saves, bonus and cards,** using team attack and defence ratings fitted from this season's xG.
- **This season blended with past seasons,** with this season counting more as a player racks up minutes.

The squad planning uses exact optimisation (PuLP with the HiGHS solver) under FPL's rules: budget and selling prices, positions, at most three players per club, valid formations, and hits.

### Tested on past seasons

For every gameweek from GW3 to GW38, the model predicted each player's points using only information available before that gameweek. The table compares it with FPL's form.

| Season | Model: average error | Form: average error | Model: top 50 picks' actual points | Form: top 50 picks' actual points |
|---|---|---|---|---|
| 2024/25 | **1.01** | 1.05 | **4.28** | 3.69 |
| 2025/26 | **1.00** | 1.05 | **4.22** | 3.64 |

The model's settings were tuned on the odd gameweeks of 2025/26, so **2024/25 is a fully independent test.** The players it rates highest also score what it predicts: across both seasons, the top 50 each week were predicted about 4.25 points and actually averaged about 4.2.

## Running it locally

Requires Python 3.12+.

```bash
python3 -m pip install -r requirements.txt
cd backend
python3 app.py            # http://127.0.0.1:5050
```

Useful commands (run from `backend/`):

```bash
python3 backtest.py --season 2025-26          # test the model on a full past season
python3 backtest.py --season 2025-26 --tune   # tune the model's settings (about 10 minutes)
```

Everything is fetched live from the public FPL API and cached in `backend/.cache/`. No API keys are needed.

## Deployment

The site has two parts, so visitors never wait for a server to wake up:

- **The page, on GitHub Pages.** A scheduled GitHub Actions workflow (`.github/workflows/publish-site.yml`) runs twice a day. It calculates the example team's predictions, transfer plan and chip plan, plus price changes and accuracy, and publishes them as static files alongside the page (`scripts/build_static.py`). The page and the example team load instantly.
- **The calculation server, on Render** (`render.yaml`, free plan). It's only used when someone enters their own team ID. Flask with gunicorn serves the same API, and allows the GitHub Pages site to call it (`ALLOWED_ORIGINS`).

Server settings (environment variables):

| Variable | Purpose |
|---|---|
| `EXAMPLE_TEAM_ID` | The team the "Try with an example team" button loads |
| `WARM_EXAMPLE` | `1` to precompute the example team in the background |
| `MODEL_TTL_SECONDS` | How long to reuse the built model before refreshing FPL data |
| `ALLOWED_ORIGINS` | Sites allowed to call the API (the GitHub Pages address, and any custom domain) |

Build settings (GitHub repository variables, all optional): `XPOINTS_API_URL` (the calculation server), `EXAMPLE_TEAM_ID`, `SITE_DOMAIN` (a custom domain).

You can also run everything as one service: `app.py` serves the page itself, as when running locally.

## Project structure

```
backend/
  app.py          Flask routes (also serves the page)
  fpl.py          FPL API client with disk caching
  model.py        the points prediction model
  optimizer.py    squad and multi-week transfer optimisation
  chips.py        chip planning
  planner.py      lineups, free transfers, selling prices
  recommender.py  assembles everything for a team, with caching
  accuracy.py     saves predictions and compares them with results
  backtest.py     tests the model on past gameweeks
  history.py      loads past seasons for testing
frontend/
  index.html      the whole page (HTML, CSS and JavaScript)
scripts/
  build_static.py builds the static site with pre-calculated example data
```

## Credits

- Live data: the public [Fantasy Premier League](https://fantasy.premierleague.com) API.
- Historical data for testing: [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League) by Vaastav Anand.
