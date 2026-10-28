"""How each player on a roster helps or hurts every category."""
import pandas as pd

from .config import CATEGORIES, RAW_STATS
from .projection import Simulator, category_zscores


def season_rates(players: pd.DataFrame) -> pd.DataFrame:
    """This season's actual per-game averages (no projection mixed in), for players who have played."""
    played = players[players["season_gp"].fillna(0) > 0]
    rates = pd.DataFrame({s: played[f"season_{s}"].fillna(0).to_numpy() for s in RAW_STATS},
                         index=played["player_id"].to_numpy())
    rates["GP"] = played["season_gp"].to_numpy()
    return rates


def roster_contributions(sim: Simulator, team_id: int, periods: list[int] | None = None) -> pd.DataFrame:
    """One row per rostered player: this season's per-game rates, per-category z-scores and their total.

    z-scores compare against the top 200 scorers this season. Players without a game yet get NaN.
    With `periods`, also `exp_games` (expected games he starts in that window) so callers can
    weight z-scores by how much each player will actually play.
    """
    cats = [CATEGORIES[c]["name"] for c in sim.cat_ids]
    rates = season_rates(sim.snap["players"])
    z = category_zscores(rates, sim.cat_ids)
    roster = sim.roster_of(team_id)
    roster = roster[roster["player_id"].isin(sim.players.index)]
    p = sim.players.loc[roster["player_id"], ["name", "pro_team", "injury_status"]].reset_index(drop=True)
    p["player_id"] = roster["player_id"].to_numpy()
    p["lineup_slot"] = roster["lineup_slot"].to_numpy()
    r = rates.reindex(p["player_id"]).reset_index(drop=True)
    p["gp"] = r["GP"].fillna(0).astype(int).to_numpy()
    for cid in sim.cat_ids:
        cat = CATEGORIES[cid]
        if "ratio" in cat:
            made, att = cat["ratio"]
            p[f"raw_{cat['name']}"] = (r[made] / r[att].where(r[att] > 0)).to_numpy()
            p[f"att_{cat['name']}"] = r[att].to_numpy()
        else:
            p[f"raw_{cat['name']}"] = r[cat["name"]].to_numpy()
    zz = z.reindex(p["player_id"])[cats].reset_index(drop=True)
    p[cats] = zz.to_numpy()
    p["total"] = zz.sum(axis=1, min_count=1).to_numpy()
    if periods is not None:
        starts = sim.plan_starts(roster, periods).groupby("player_id")["play_prob"].sum()
        p["exp_games"] = p["player_id"].map(starts).fillna(0.0)
    return p.sort_values("total", ascending=False, na_position="last").reset_index(drop=True)


DISPLAY_ORDER = ["PTS", "REB", "AST", "STL", "BLK", "3PM", "FG%", "FT%", "TO"]


def ordered_categories(cat_ids: list[str]) -> list[str]:
    """League category names in the familiar box-score order."""
    names = [CATEGORIES[c]["name"] for c in cat_ids]
    return sorted(names, key=lambda n: DISPLAY_ORDER.index(n) if n in DISPLAY_ORDER else len(DISPLAY_ORDER))
