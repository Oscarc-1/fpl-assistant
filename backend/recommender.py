from fpl import get_bootstrap_data, get_fixtures, get_user_squad, get_current_gameweek, get_team_strengths, get_fixture_difficulty
from scorer import score_player

def analyse_team(team_id):
    print("Fetching data...")
    data = get_bootstrap_data()
    fixtures = get_fixtures()

    gameweek = get_current_gameweek(data["events"])
    all_players = data["elements"]
    team_strengths = get_team_strengths(data["teams"])
    players_by_id = {p["id"]: p for p in all_players}

    # Get user squad
    squad_data = get_user_squad(team_id, gameweek - 1)
    bank = squad_data["entry_history"]["bank"] / 10
    squad_value = squad_data["entry_history"]["value"] / 10

    all_picks = squad_data["picks"]
    squad_ids = [pick["element"] for pick in all_picks]
    starting_picks = [pick for pick in all_picks if pick["multiplier"] > 0]
    bench_picks = [pick for pick in all_picks if pick["multiplier"] == 0]

    # Score every player in the squad
    def score_pick(pick):
        player = players_by_id[pick["element"]]
        fix_diff = get_fixture_difficulty(fixtures, player["team"], gameweek, next_n=3)
        team_strength = team_strengths.get(player["team"], 2.5)
        score = score_player(player, fix_diff, team_strength)
        return {
            "id": player["id"],
            "player": f"{player['first_name']} {player['second_name']}",
            "cost": player["now_cost"] / 10,
            "form": float(player["form"]),
            "ep_next": float(player.get("ep_next") or 0),
            "position": player["element_type"],
            "score": score or 0,
            "is_starter": pick["multiplier"] > 0
        }

    starting_scored = [score_pick(p) for p in starting_picks]
    bench_scored = [score_pick(p) for p in bench_picks]
    all_squad_scored = starting_scored + bench_scored

    # --- CAPTAIN / VICE CAPTAIN ---
    sorted_starters = sorted(starting_scored, key=lambda x: x["score"], reverse=True)
    captain = sorted_starters[0] if len(sorted_starters) > 0 else None
    vice_captain = sorted_starters[1] if len(sorted_starters) > 1 else None

    # --- BENCH SWAPS ---
    bench_swaps = []
    for bench_player in bench_scored:
        for starter in starting_scored:
            if bench_player["position"] != starter["position"]:
                continue
            improvement = round(bench_player["score"] - starter["score"], 3)
            if improvement > 0.3:  # only suggest meaningful improvements
                bench_swaps.append({
                    "player_out": starter["player"],
                    "player_in": bench_player["player"],
                    "improvement": improvement,
                    "position": starter["position"]
                })

    bench_swaps.sort(key=lambda x: x["improvement"], reverse=True)

    # --- TRANSFER RECOMMENDATIONS ---
    recommendations = []
    for starter in starting_scored:
        position = starter["position"]
        starter_cost = starter["cost"]

        for player_in in all_players:
            if player_in["element_type"] != position:
                continue
            if player_in["id"] in squad_ids:
                continue
            player_in_cost = player_in["now_cost"] / 10
            if player_in_cost > starter_cost + bank:
                continue

            fix_diff = get_fixture_difficulty(fixtures, player_in["team"], gameweek, next_n=3)
            team_strength = team_strengths.get(player_in["team"], 2.5)
            score_in = score_player(player_in, fix_diff, team_strength)
            if score_in is None:
                continue

            improvement = round(score_in - starter["score"], 3)
            if improvement <= 0:
                continue

            # verdict based on improvement size
            if improvement > 1.0:
                verdict = "Strong"
            elif improvement > 0.5:
                verdict = "Moderate"
            else:
                verdict = "Marginal"

            recommendations.append({
                "player_out": starter["player"],
                "player_out_cost": starter_cost,
                "player_out_ep": starter["ep_next"],
                "player_in": f"{player_in['first_name']} {player_in['second_name']}",
                "player_in_cost": player_in_cost,
                "player_in_ep": float(player_in.get("ep_next") or 0),
                "score_improvement": improvement,
                "position": position,
                "verdict": verdict
            })

    recommendations.sort(key=lambda x: x["score_improvement"], reverse=True)

    # deduplicate
    seen_out = set()
    deduplicated = []
    for rec in recommendations:
        if rec["player_out"] not in seen_out:
            seen_out.add(rec["player_out"])
            deduplicated.append(rec)

    # if no strong transfers, suggest rolling
    roll_transfer = len([r for r in deduplicated if r["verdict"] == "Strong"]) == 0

    # --- DIFFERENTIALS ---
    differentials = []
    for player in all_players:
        selected_by = float(player.get("selected_by_percent", 0))
        if selected_by > 10.0:
            continue
        if player["id"] in squad_ids:
            continue

        fix_diff = get_fixture_difficulty(fixtures, player["team"], gameweek, next_n=3)
        team_strength = team_strengths.get(player["team"], 2.5)
        score = score_player(player, fix_diff, team_strength)
        if score is None:
            continue

        differentials.append({
            "player": f"{player['first_name']} {player['second_name']}",
            "web_name": player["web_name"],
            "team": player["team"],
            "cost": player["now_cost"] / 10,
            "selected_by": selected_by,
            "score": score,
            "position": player["element_type"],
            "ep_next": float(player.get("ep_next") or 0)
        })

    differentials.sort(key=lambda x: x["score"], reverse=True)

    # resolve team names
    teams_by_id = {t["id"]: t["name"] for t in data["teams"]}
    for d in differentials:
        d["team"] = teams_by_id.get(d["team"], "?")

    return {
        "squad_value": squad_value,
        "bank": bank,
        "gameweek": gameweek,
        "captain": captain,
        "vice_captain": vice_captain,
        "bench_swaps": bench_swaps[:3],
        "recommendations": deduplicated[:10],
        "roll_transfer": roll_transfer,
        "differentials": differentials[:5]
    }


if __name__ == "__main__":
    team_id = input("Enter your FPL team ID: ")
    results = analyse_team(int(team_id))

    print(f"\nCaptain: {results['captain']['player']}")
    print(f"Vice Captain: {results['vice_captain']['player']}")

    print(f"\nBench swaps:")
    if results["bench_swaps"]:
        for swap in results["bench_swaps"]:
            print(f"  Start {swap['player_in']} over {swap['player_out']}")
    else:
        print("  No bench swaps needed")

    print(f"\nTransfer recommendations:")
    for rec in results["recommendations"]:
        print(f"  [{rec['verdict']}] {rec['player_out']} → {rec['player_in']} (+{rec['score_improvement']})")

    if results["roll_transfer"]:
        print("\n  No strong transfers available — consider rolling your transfer this week")