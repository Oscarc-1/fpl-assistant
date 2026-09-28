def score_player(player, fixture_difficulty, team_strength):
    
    # --- FILTERS ---
    # If chance of playing is known and below 75%, skip this player
    chance = player.get("chance_of_playing_next_round")
    if chance is not None and chance < 75:
        return None

    # --- RAW DATA ---
    form = float(player.get("form", 0))
    points_per_game = float(player.get("points_per_game", 0))
    ep_next = float(player.get("ep_next", 0))
    starts_per_90 = float(player.get("starts_per_90", 0))
    bonus = float(player.get("bonus", 0))
    transfers_in_event = float(player.get("transfers_in_event", 0))

    # --- NORMALISE ---
    # We scale values so they are comparable to each other
    starts_score = min(starts_per_90, 1.0)  # already a 0-1 value
    transfers_score = min(transfers_in_event / 100000, 1.0)
    bonus_score = min(bonus / 50, 1.0)
    fixture_score = (6 - fixture_difficulty) / 5  # easier fixture = higher score
    team_score = team_strength / 5

    # --- WEIGHTED SCORE ---
    score = (
        form * 0.15 +
        fixture_score * 0.25 +
        points_per_game * 0.3 +
        team_score * 0.10 +
        starts_score * 0.10 +
        transfers_score * 0.05 +
        bonus_score * 0.05
    )

    return round(score, 3)