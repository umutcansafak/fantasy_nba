"""Fake two-team league built from a public-pool snapshot, for trying the app without credentials."""
import pandas as pd

DEMO_TEAMS = pd.DataFrame({"team_id": [1, 2], "abbrev": ["ME", "OPP"], "name": ["Demo: Me", "Demo: Opponent"],
                           "owner": ["", ""], "wins": [0, 0], "losses": [0, 0], "ties": [0, 0]})


def demo_league(snap: dict) -> dict:
    top = snap["players"].sort_values("pct_owned", ascending=False).head(26)["player_id"].tolist()
    teams = [1, 2, 2, 1] * 6 + [1, 2]  # snake draft order
    snap = dict(snap)
    snap["rosters"] = pd.DataFrame({"team_id": teams, "player_id": top, "lineup_slot": 0})
    snap["players"] = snap["players"].assign(
        on_team_id=snap["players"]["player_id"].map(dict(zip(top, teams))).fillna(0).astype(int))
    snap["teams"] = DEMO_TEAMS
    snap["meta"] = {**snap["meta"], "team_id": 1}
    return snap
