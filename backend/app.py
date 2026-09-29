import os

from flask import Flask, jsonify, request, send_from_directory

from fpl import get_bootstrap_data
from recommender import analyse_team

app = Flask(__name__)

@app.route("/")
def index():
    frontend_path = os.path.join(os.path.dirname(__file__), "../frontend")
    return send_from_directory(frontend_path, "index.html")

@app.route("/analysis")
def analysis():
    team_id = request.args.get("team_id", "").strip()
    if not team_id.isdigit():
        return jsonify({"error": "Enter your numeric FPL team ID"}), 400

    try:
        return jsonify(analyse_team(int(team_id)))
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


if __name__ == "__main__":
    app.run(debug=True, port=5050)