"""Free-agent pickups ranked by how much they change the projected matchup."""
import pandas as pd

from .nba_schedule import games_by_team_period
from .projection import Simulator, compare


def weakest_player(sim: Simulator, roster: pd.DataFrame) -> int:
    """Lowest-value player not parked on IR: the default drop candidate."""
    healthy = roster[roster["lineup_slot"] != 13]
    vals = healthy["player_id"].map(lambda p: sim.values.get(p, -99.0))
    return int(healthy["player_id"].iloc[vals.to_numpy().argmin()])


def free_agent_candidates(sim: Simulator, periods: list[int], n: int = 25) -> pd.DataFrame:
    """Shortlist available players by value x games this week (cheap pre-filter)."""
    fa = sim.players[(sim.players["on_team_id"] == 0) &
                     (~sim.players["injury_status"].isin(["OUT", "SUSPENSION", "INJURY_RESERVE"]))].copy()
    games = games_by_team_period(sim.pro, periods).sum(axis=1)
    fa["games"] = fa["pro_team_id"].map(games).fillna(0).astype(int)
    fa["value"] = fa["player_id"].map(sim.values).fillna(-99)
    fa["score"] = fa["value"].clip(lower=-3) + 3
    fa["score"] *= fa["games"]
    return fa[fa["games"] > 0].sort_values("score", ascending=False).head(n)


def recommend(sim: Simulator, team_id: int, opp_id: int, periods: list[int],
              drop_id: int | None = None, n: int = 25) -> tuple[pd.DataFrame, int]:
    roster = sim.roster_of(team_id)
    drop_id = drop_id or weakest_player(sim, roster)
    me = sim.project_team(team_id, roster, periods, 0)
    opp = sim.project_team(opp_id, sim.roster_of(opp_id), periods, 1)
    base, base_overall = compare(me, opp, sim.cat_ids)
    rows = []
    for _, fa in free_agent_candidates(sim, periods, n).iterrows():
        new_roster = pd.concat([roster[roster["player_id"] != drop_id],
                                pd.DataFrame({"player_id": [fa["player_id"]], "lineup_slot": [12]})])
        # same seed as the baseline so the delta reflects the swap, not sampling noise
        table, overall = compare(sim.project_team(team_id, new_roster, periods, 0), opp, sim.cat_ids)
        delta = (table.set_index("category")["p_win"] - base.set_index("category")["p_win"])
        rows.append({
            "player_id": fa["player_id"], "name": fa["name"], "team": fa["pro_team"],
            "status": fa["status"], "games": fa["games"], "pct_owned": fa["pct_owned"],
            "d_expected_cats": overall["expected_cats_won"] - base_overall["expected_cats_won"],
            "d_p_win_matchup": overall["p_win_matchup"] - base_overall["p_win_matchup"],
            **{f"d_{k}": v for k, v in delta.items()},
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("d_expected_cats", ascending=False, ignore_index=True)
    return out, drop_id
