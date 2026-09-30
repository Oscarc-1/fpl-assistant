import os

from flask import Flask, jsonify, request, send_from_directory

from accuracy import accuracy_report
from fpl import get_bootstrap_data, get_fixtures, get_all_element_summaries, get_user_squad
from recommender import (EXAMPLE_TEAM_ID, WARM_EXAMPLE, analyse_team, cached_result, get_predictor,
                         plan_chips, warm_example)


def predictor_summaries(bootstrap):
    return get_all_element_summaries([p["id"] for p in bootstrap["elements"]])

app = Flask(__name__)

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))


def code_changed_at():
    return max(os.path.getmtime(os.path.join(BACKEND_DIR, f)) for f in os.listdir(BACKEND_DIR) if f.endswith(".py"))


STARTED_WITH = code_changed_at()


@app.before_request
def refuse_stale_code():
    """If the code changed since the app started (and it isn't auto-reloading), say so plainly
    instead of answering with an old version the page no longer matches."""
    if request.path != "/" and code_changed_at() > STARTED_WITH:
        return jsonify({"error": "The app's code has been updated since it was started. Restart it in Terminal "
                                 "(Ctrl + C, then run the start command again) and try again."}), 503

@app.route("/")
def index():
    frontend_path = os.path.join(os.path.dirname(__file__), "../frontend")
    return send_from_directory(frontend_path, "index.html")

@app.route("/health")
def health():
    """For the host's health checks: cheap, doesn't touch FPL or the model."""
    return jsonify({"ok": True})


@app.route("/config")
def config():
    """Settings the page needs (the example team to offer visitors)."""
    return jsonify({"example_team_id": EXAMPLE_TEAM_ID})


@app.route("/analysis")
def analysis():
    team_id = request.args.get("team_id", "").strip()
    if not team_id.isdigit():
        return jsonify({"error": "Enter your numeric FPL team ID"}), 400

    try:
        return jsonify(cached_result("analysis", int(team_id), analyse_team))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/chips")
def chips():
    team_id = request.args.get("team_id", "").strip()
    if not team_id.isdigit():
        return jsonify({"error": "Enter your numeric FPL team ID"}), 400
    try:
        return jsonify(cached_result("chips", int(team_id), plan_chips))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/accuracy")
def accuracy():
    team_id = request.args.get("team_id", "").strip()
    try:
        predictor = get_predictor()  # also records this gameweek's predictions
        data = get_bootstrap_data()
        return jsonify(accuracy_report(data, get_fixtures(), predictor_summaries(data),
                                       int(team_id) if team_id.isdigit() else None, get_user_squad))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/price-changes")
def price_changes():
    try:
        data = get_bootstrap_data()
        teams_by_id = {t["id"]: t["name"] for t in data["teams"]}
        settings = data["game_config"]["settings"]
        status = data["game_config"]["status"]

        risers = []
        fallers = []

        for p in data["elements"]:
            # Players who just changed price are locked, and new players are still calibrating
            if p["price_change_locked_until"] or p["price_change_calibrating"]:
                continue
            projections = p["price_change_projections"]
            if not projections:
                continue

            # FPL's own projection for the next price update (offset 0 = tonight)
            tonight = projections[0]
            likelihood = tonight["likelihood"]
            if likelihood == 0:
                continue

            entry = {
                "player": p["web_name"],
                "team": teams_by_id.get(p["team"], "?"),
                "cost": p["now_cost"] / 10,
                "selected_by": float(p["selected_by_percent"]),
                "progress": float(p["price_change_percent"]),
                "projected_progress": float(tonight["projected_percent"]),
                "likelihood": likelihood,
                "hourly_rate": p["price_change_hourly_rate"]
            }

            if likelihood > 0:
                risers.append(entry)
            else:
                fallers.append(entry)

        risers.sort(key=lambda x: (x["likelihood"], x["projected_progress"]), reverse=True)
        fallers.sort(key=lambda x: (x["likelihood"], x["projected_progress"]))

        deadlines = settings.get("price_change_deadlines") or []

        return jsonify({
            "risers": risers[:8],
            "fallers": fallers[:8],
            "next_change": deadlines[0] if deadlines else None,
            "last_updated": status.get("price_change_last_updated")
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if WARM_EXAMPLE:
    warm_example()


if __name__ == "__main__":
    app.run(debug=True, port=5050)  # reloads automatically when the code changes