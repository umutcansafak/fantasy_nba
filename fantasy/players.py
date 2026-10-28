"""How each player on a roster helps or hurts every category."""
import pandas as pd

from .config import CATEGORIES, RAW_STATS
from .nba_schedule import games_by_team_period
from .projection import Simulator, apply_zscores, zscore_params


def season_rates(players: pd.DataFrame) -> pd.DataFrame:
    """This season's actual per-game averages (no projection mixed in), for players who have played."""
    played = players[players["season_gp"].fillna(0) > 0]
    rates = pd.DataFrame({s: played[f"season_{s}"].fillna(0).to_numpy() for s in RAW_STATS},
                         index=played["player_id"].to_numpy())
    rates["GP"] = played["season_gp"].to_numpy()
    return rates


def roster_contributions(sim: Simulator, team_id: int, matchup_period: int | None = None) -> pd.DataFrame:
    """Contributions of one fantasy team's roster; the week view respects its daily lineup limits."""
    return contributions(sim, sim.roster_of(team_id), matchup_period)


def free_agent_contributions(sim: Simulator, matchup_period: int | None = None) -> pd.DataFrame:
    """Contributions of every unrostered player who has played this season (week view: every game he plays)."""
    taken = set(sim.snap["rosters"]["player_id"])
    pool = sim.players[~sim.players["player_id"].isin(taken) & (sim.players["season_gp"].fillna(0) > 0)]
    roster = pd.DataFrame({"player_id": pool["player_id"].to_numpy(), "lineup_slot": 0})
    return contributions(sim, roster, matchup_period, lineup=False)


def contributions(sim: Simulator, roster: pd.DataFrame, matchup_period: int | None = None,
                  lineup: bool = True) -> pd.DataFrame:
    """One row per player in `roster` (player_id, lineup_slot): this season's per-game rates,
    per-category z-scores and their total.

    z-scores compare against this season's top 200 scorers. Players without a game yet get NaN.
    With `matchup_period`, also that week's impact in `wk_<cat>` / `wk_total`: the z-scores of the games
    he already played that week (`played_games`) plus his season z-scores times the games he is
    expected to play in the rest of it (`exp_games`): starts within the lineup limits when `lineup`,
    otherwise every NBA game he is available for.
    """
    cats = [CATEGORIES[c]["name"] for c in sim.cat_ids]
    rates = season_rates(sim.snap["players"])
    params = zscore_params(rates, sim.cat_ids)
    z = apply_zscores(rates, params)
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

    if matchup_period is not None:
        week = [int(x) for x in sim.meta["matchup_periods"][matchup_period]]
        left = sim.periods_for(matchup_period, remaining_only=True)
        logs = sim.snap["game_logs"]
        logs = logs[(logs["season"] == sim.meta["season"]) & logs["player_id"].isin(p["player_id"])
                    & logs["scoring_period"].isin([d for d in week if d not in left])]
        done = apply_zscores(logs, params).groupby(logs["player_id"]).sum()
        p["played_games"] = p["player_id"].map(logs.groupby("player_id").size()).fillna(0).astype(int)
        if not left:
            starts = pd.Series(dtype=float)
        elif lineup:
            starts = sim.plan_starts(roster, left).groupby("player_id")["play_prob"].sum()
        else:
            grid = games_by_team_period(sim.pro, left)
            starts = pd.Series({pid: sum(sim._availability(pid, slot, d) for d in left
                                         if team in grid.index and grid.at[team, d] > 0)
                                for pid, slot, team in zip(roster["player_id"], roster["lineup_slot"],
                                                           sim.players.loc[roster["player_id"], "pro_team_id"])},
                               dtype=float)
        p["exp_games"] = p["player_id"].map(starts).fillna(0.0)
        for c in cats:
            played = p["player_id"].map(done[c]).fillna(0.0) if c in done else 0.0
            p[f"wk_{c}"] = played + p[c].fillna(0.0) * p["exp_games"]
        p["wk_total"] = p[[f"wk_{c}" for c in cats]].sum(axis=1)
    return p.sort_values("total", ascending=False, na_position="last").reset_index(drop=True)


DISPLAY_ORDER = ["PTS", "REB", "AST", "STL", "BLK", "3PM", "FG%", "FT%", "TO"]


def ordered_categories(cat_ids: list[str]) -> list[str]:
    """League category names in the familiar box-score order."""
    names = [CATEGORIES[c]["name"] for c in cat_ids]
    return sorted(names, key=lambda n: DISPLAY_ORDER.index(n) if n in DISPLAY_ORDER else len(DISPLAY_ORDER))


def team_overview(sim: Simulator) -> pd.DataFrame:
    """Every fantasy team's summed season z-scores per category (all rostered players), plus a total."""
    cats = [CATEGORIES[c]["name"] for c in sim.cat_ids]
    rows = []
    for tid, name in sim.snap["teams"].set_index("team_id")["name"].items():
        d = roster_contributions(sim, tid)
        rows.append({"team_id": tid, "team": name, **d[cats].sum().to_dict()})
    out = pd.DataFrame(rows)
    out["total"] = out[cats].sum(axis=1)
    return out.sort_values("total", ascending=False).reset_index(drop=True)
