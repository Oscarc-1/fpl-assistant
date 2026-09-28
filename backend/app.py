from flask import Flask, jsonify, request, send_from_directory
from fpl import get_bootstrap_data, get_current_gameweek, get_user_squad, get_fixtures, get_team_strengths
from recommender import analyse_team
import os

app = Flask(__name__)

@app.route("/")
def index():
    frontend_path = os.path.join(os.path.dirname(__file__), "../frontend")
    return send_from_directory(frontend_path, "index.html")

@app.route("/recommend")
def recommend():
    team_id = request.args.get("team_id")
    if not team_id:
        return jsonify({"error": "No team ID provided"}), 400

    try:
        results = analyse_team(int(team_id))
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    


@app.route("/price-changes")
def price_changes():
    try:
        data = get_bootstrap_data()
        players = data["elements"]
        total_managers = data["total_players"]
        teams_by_id = {t["id"]: t["name"] for t in data["teams"]}

        risers = []
        fallers = []

        for p in players:
            selected_by = float(p["selected_by_percent"])
            if selected_by == 0:
                continue

            net = p["transfers_in_event"] - p["transfers_out_event"]
            # How many managers own this player
            owners = total_managers * (selected_by / 100)
            # Net transfer ratio relative to ownership
            ownership_ratio = net / owners

            entry = {
                "player": p["web_name"],
                "team": teams_by_id.get(p["team"], "?"),
                "cost": p["now_cost"] / 10,
                "net": net,
                "ownership_ratio": round(ownership_ratio * 100, 2),
                "selected_by": selected_by
            }

            if ownership_ratio > 0.005:  # 0.5% threshold for rising
                risers.append(entry)
            elif ownership_ratio < -0.005:  # 0.5% threshold for falling
                fallers.append(entry)

        risers.sort(key=lambda x: x["ownership_ratio"], reverse=True)
        fallers.sort(key=lambda x: x["ownership_ratio"])

        return jsonify({
            "risers": risers[:5],
            "fallers": fallers[:5],
            "total_managers": total_managers
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/fixtures")
def fixtures():
    team_id = request.args.get("team_id")
    if not team_id:
        return jsonify({"error": "No team ID provided"}), 400

    try:
        data = get_bootstrap_data()
        all_fixtures = get_fixtures()
        gameweek = get_current_gameweek(data["events"])
        team_strengths = get_team_strengths(data["teams"])

        from fpl import get_user_squad
        squad_data = get_user_squad(int(team_id), gameweek - 1)
        picks = squad_data["picks"]
        players_by_id = {p["id"]: p for p in data["elements"]}
        teams_by_id = {t["id"]: t["name"] for t in data["teams"]}

        squad_fixtures = []
        for pick in picks:
            player = players_by_id[pick["element"]]
            team_id_player = player["team"]

            # Get next 5 fixtures for this player's team
            upcoming = []
            for fixture in all_fixtures:
                if fixture["event"] and fixture["event"] >= gameweek:
                    if fixture["team_h"] == team_id_player:
                        upcoming.append({
                            "opponent": teams_by_id.get(fixture["team_a"], "?"),
                            "difficulty": fixture["team_h_difficulty"],
                            "home": True
                        })
                    elif fixture["team_a"] == team_id_player:
                        upcoming.append({
                            "opponent": teams_by_id.get(fixture["team_h"], "?"),
                            "difficulty": fixture["team_a_difficulty"],
                            "home": False
                        })
                if len(upcoming) == 5:
                    break

            squad_fixtures.append({
                "player": player["web_name"],
                "team": teams_by_id.get(team_id_player, "?"),
                "position": player["element_type"],
                "is_starter": pick["multiplier"] > 0,
                "fixtures": upcoming
            })

        return jsonify({"players": squad_fixtures, "gameweek": gameweek})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    
if __name__ == "__main__":
    app.run(debug=True)