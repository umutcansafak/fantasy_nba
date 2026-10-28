"""How each player on a roster helps or hurts every category."""
import pandas as pd

from .config import CATEGORIES
from .projection import Simulator, category_zscores


def roster_contributions(sim: Simulator, team_id: int, periods: list[int] | None = None) -> pd.DataFrame:
    """One row per rostered player: per-game rates, per-category z-scores and their total.

    With `periods`, also `exp_games` (expected games he starts in that window) so callers can
    weight z-scores by how much each player will actually play.
    """
    cats = [CATEGORIES[c]["name"] for c in sim.cat_ids]
    z = category_zscores(sim.rates, sim.cat_ids)
    roster = sim.roster_of(team_id)
    roster = roster[roster["player_id"].isin(z.index)]
    p = sim.players.loc[roster["player_id"], ["name", "pro_team", "injury_status"]].reset_index(drop=True)
    p["player_id"] = roster["player_id"].to_numpy()
    p["lineup_slot"] = roster["lineup_slot"].to_numpy()
    rates = sim.rates.loc[p["player_id"]].reset_index(drop=True)
    for cid in sim.cat_ids:
        cat = CATEGORIES[cid]
        if "ratio" in cat:
            made, att = cat["ratio"]
            p[f"raw_{cat['name']}"] = (rates[made] / rates[att].where(rates[att] > 0)).to_numpy()
            p[f"att_{cat['name']}"] = rates[att].to_numpy()
        else:
            p[f"raw_{cat['name']}"] = rates[cat["name"]].to_numpy()
    zz = z.loc[p["player_id"], cats].reset_index(drop=True)
    p[cats] = zz.to_numpy()
    p["total"] = zz.sum(axis=1).to_numpy()
    if periods is not None:
        starts = sim.plan_starts(roster, periods).groupby("player_id")["play_prob"].sum()
        p["exp_games"] = p["player_id"].map(starts).fillna(0.0)
    return p.sort_values("total", ascending=False).reset_index(drop=True)


DISPLAY_ORDER = ["PTS", "REB", "AST", "STL", "BLK", "3PM", "FG%", "FT%", "TO"]


def ordered_categories(cat_ids: list[str]) -> list[str]:
    """League category names in the familiar box-score order."""
    names = [CATEGORIES[c]["name"] for c in cat_ids]
    return sorted(names, key=lambda n: DISPLAY_ORDER.index(n) if n in DISPLAY_ORDER else len(DISPLAY_ORDER))
