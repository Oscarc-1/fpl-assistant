import requests

FPL_BASE_URL = "https://fantasy.premierleague.com/api"

def get_bootstrap_data():
    response = requests.get(f"{FPL_BASE_URL}/bootstrap-static/")
    return response.json()

def get_fixtures():
    response = requests.get(f"{FPL_BASE_URL}/fixtures/")
    return response.json()

def get_user_squad(team_id, gameweek):
    response = requests.get(f"{FPL_BASE_URL}/entry/{team_id}/event/{gameweek}/picks/")
    return response.json()

def get_current_gameweek(events):
    for event in events:
        if event["is_next"]:
            return event["id"]
    return None

def get_team_strengths(teams):
    raw_strengths = {}
    for team in teams:
        avg_strength = (team["strength_attack_home"] + 
                       team["strength_attack_away"] + 
                       team["strength_defence_home"] + 
                       team["strength_defence_away"]) / 4
        raw_strengths[team["id"]] = avg_strength

    max_strength = max(raw_strengths.values())
    min_strength = min(raw_strengths.values())

    normalised = {}
    for team_id, strength in raw_strengths.items():
        normalised[team_id] = round((strength - min_strength) / (max_strength - min_strength) * 5, 2)
    
    return normalised

def get_fixture_difficulty(fixtures, team_id, gameweek, next_n=3):
    upcoming = []
    for fixture in fixtures:
        if fixture["event"] and fixture["event"] >= gameweek:
            if fixture["team_h"] == team_id:
                upcoming.append(fixture["team_h_difficulty"])
            elif fixture["team_a"] == team_id:
                upcoming.append(fixture["team_a_difficulty"])
        if len(upcoming) == next_n:
            break
    if not upcoming:
        return 3  # default medium difficulty
    return round(sum(upcoming) / len(upcoming), 2)

# Test it
if __name__ == "__main__":
    print("Fetching FPL data...")
    data = get_bootstrap_data()
    fixtures = get_fixtures()
    
    gameweek = get_current_gameweek(data["events"])
    print(f"Current gameweek: {gameweek}")
    
    teams = {team["id"]: team["name"] for team in data["teams"]}
    strengths = get_team_strengths(data["teams"])
    
    print(f"Team strengths sample:")
    for team_id, strength in list(strengths.items())[:5]:
        print(f"  {teams[team_id]}: {strength}")
