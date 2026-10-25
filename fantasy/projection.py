"""Weekly matchup projection via Monte Carlo over players' real game logs.

For each fantasy team and matchup week:
  1. find each rostered player's NBA games in the week,
  2. drop unavailable players (IR slot, OUT without a return date in range),
  3. each day, start the most valuable players that fit the active lineup slots,
  4. for every simulated week, draw each start from the player's recent box
     scores (bootstrap; this season weighted above last season), and sum.
Ratio categories (FG%, FT%) are total makes / total attempts, as ESPN scores them.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import CATEGORIES, RAW_STATS, STAT_IDS
from .lineup import active_slots, fill_lineup
from .nba_schedule import games_by_team_period, period_dates

CURRENT_SEASON_WEIGHT = 3.0
PREV_SEASON_GAMES = 20
PRIOR_GAMES = 8          # shrinkage strength toward projection / last season
MIN_LOG_GAMES = 5        # below this, synthesize games around the blended mean
OUT_STATUSES = {"OUT", "SUSPENSION", "INJURY_RESERVE"}
PLAY_PROB = {"DAY_TO_DAY": 0.7, "QUESTIONABLE": 0.6, "DOUBTFUL": 0.25, "PROBABLE": 0.9}


# ----------------------------------------------------------------- player rates

def blended_rates(players: pd.DataFrame) -> pd.DataFrame:
    """Per-game means: this season shrunk toward ESPN's projection (or last season)."""
    out = pd.DataFrame(index=players["player_id"])
    n = players.get("season_gp", pd.Series(0.0, index=players.index)).fillna(0).to_numpy()
    for s in RAW_STATS:
        cur = players.get(f"season_{s}", pd.Series(np.nan, index=players.index)).fillna(0).to_numpy()
        prior = players.get(f"proj_{s}", pd.Series(np.nan, index=players.index))
        prior = prior.fillna(players.get(f"prev_{s}", pd.Series(np.nan, index=players.index))).to_numpy()
        prior = np.where(np.isnan(prior), cur, prior)
        out[s] = (n * cur + PRIOR_GAMES * prior) / (n + PRIOR_GAMES)
    return out


def player_values(rates: pd.DataFrame, cat_ids: list[str], pool_size: int = 200) -> pd.Series:
    """Sum of per-category z-scores (ratio cats weighted by attempt volume)."""
    ref = rates.sort_values("PTS", ascending=False).head(pool_size)
    total = pd.Series(0.0, index=rates.index)
    for cid in cat_ids:
        cat = CATEGORIES[cid]
        if "ratio" in cat:
            made, att = cat["ratio"]
            league_pct = ref[made].sum() / max(ref[att].sum(), 1e-9)
            col = rates[made] - league_pct * rates[att]
            ref_col = ref[made] - league_pct * ref[att]
        else:
            col, ref_col = rates[cat["name"]], ref[cat["name"]]
        z = (col - ref_col.mean()) / (ref_col.std() or 1.0)
        total += -z if cat.get("lower_is_better") else z
    return total


# ------------------------------------------------------------------ simulation

@dataclass
class TeamProjection:
    team_id: int
    samples: pd.DataFrame                 # n_sims x categories (+ raw totals)
    starts: pd.DataFrame                  # player-level games / starts in the week
    daily: pd.DataFrame = field(default=None)  # period -> starters count


class Simulator:
    def __init__(self, snap: dict, n_sims: int = 5000, seed: int = 7):
        self.snap = snap
        self.meta = snap["meta"]
        self.n_sims = n_sims
        self.seed = seed
        self.cat_ids = self.meta["category_ids"]
        self.players = snap["players"].set_index("player_id", drop=False)
        self.rates = blended_rates(snap["players"])
        self.values = player_values(self.rates, self.cat_ids)
        self.pro = snap["pro_schedule"]
        self.dates = period_dates(self.pro)
        self.slots = active_slots(self.meta["lineup_slot_counts"])
        self._pools = self._build_pools(snap["game_logs"])

    def _build_pools(self, logs: pd.DataFrame) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        season = self.meta["season"]
        pools = {}
        logs = logs.sort_values("scoring_period")
        for pid, g in logs.groupby("player_id"):
            cur = g[g["season"] == season]
            prev = g[g["season"] == season - 1].tail(PREV_SEASON_GAMES)
            both = pd.concat([cur, prev])
            w = np.r_[np.full(len(cur), CURRENT_SEASON_WEIGHT), np.ones(len(prev))]
            pools[pid] = (both[RAW_STATS].to_numpy(float), w / w.sum())
        return pools

    def _synthetic_pool(self, pid, rng, n=40):
        """Players with few box scores: Poisson games around the blended mean."""
        mean = self.rates.loc[pid, RAW_STATS].to_numpy(float) if pid in self.rates.index else np.zeros(len(RAW_STATS))
        games = rng.poisson(np.clip(mean, 0, None), size=(n, len(RAW_STATS))).astype(float)
        i = {s: k for k, s in enumerate(RAW_STATS)}
        for made, att in (("FGM", "FGA"), ("FTM", "FTA"), ("3PM", "FGM")):
            games[:, i[made]] = np.minimum(games[:, i[made]], games[:, i[att]])
        return games, np.full(n, 1.0 / n)

    def _availability(self, pid, lineup_slot, period) -> float:
        """Probability the player suits up on `period` (0 = not available)."""
        if lineup_slot == 13:  # on IR slot
            return 0.0
        p = self.players.loc[pid] if pid in self.players.index else None
        if p is None:
            return 1.0
        status = p["injury_status"]
        if status in OUT_STATUSES:
            ret = p.get("expected_return")
            return 1.0 if ret is not None and pd.notna(ret) and self.dates.get(period) and \
                self.dates[period] >= pd.Timestamp(ret).date() else 0.0
        return PLAY_PROB.get(status, 1.0)

    def plan_starts(self, roster: pd.DataFrame, periods: list[int]) -> pd.DataFrame:
        """roster: player_id, lineup_slot. Returns one row per (player, period) start."""
        grid = games_by_team_period(self.pro, periods)
        rows = []
        for period in periods:
            cands, probs = [], {}
            for pid, slot in zip(roster["player_id"], roster["lineup_slot"]):
                if pid not in self.players.index:
                    continue
                team = self.players.at[pid, "pro_team_id"]
                if team not in grid.index or grid.at[team, period] == 0:
                    continue
                prob = self._availability(pid, slot, period)
                if prob <= 0:
                    continue
                probs[pid] = prob
                cands.append((pid, float(self.values.get(pid, 0.0)) * prob, list(self.players.at[pid, "eligible_slots"])))
            for pid in fill_lineup(cands, self.slots):
                rows.append({"player_id": pid, "period": period, "date": self.dates.get(period), "play_prob": probs[pid]})
        return pd.DataFrame(rows, columns=["player_id", "period", "date", "play_prob"])

    def project_team(self, team_id: int, roster: pd.DataFrame, periods: list[int],
                     seed_offset: int = 0) -> TeamProjection:
        starts = self.plan_starts(roster, periods)
        totals = np.zeros((self.n_sims, len(RAW_STATS)))
        for pid, g in starts.groupby("player_id"):
            # per-player stream: swapping one player leaves everyone else's draws unchanged
            rng = np.random.default_rng([self.seed, seed_offset, int(pid)])
            games, w = self._pools.get(pid, (None, None))
            if games is None or len(games) < MIN_LOG_GAMES:
                games, w = self._synthetic_pool(pid, rng)
            idx = rng.choice(len(games), size=(self.n_sims, len(g)), p=w)
            played = rng.random((self.n_sims, len(g))) < g["play_prob"].to_numpy()
            totals += (games[idx] * played[..., None]).sum(axis=1)
        samples = add_ratio_categories(pd.DataFrame(totals, columns=RAW_STATS), self.cat_ids)
        summary = (starts.groupby("player_id").agg(starts=("period", "size"), exp_games=("play_prob", "sum"))
                   .join(self.players[["name", "pro_team", "injury_status"]]).reset_index())
        all_games = games_by_team_period(self.pro, periods).sum(axis=1)
        summary["nba_games"] = summary["player_id"].map(lambda p: int(all_games.get(self.players.at[p, "pro_team_id"], 0)))
        return TeamProjection(team_id, samples, summary, starts.groupby("period").size())

    def roster_of(self, team_id: int) -> pd.DataFrame:
        r = self.snap["rosters"]
        return r[r["team_id"] == team_id][["player_id", "lineup_slot"]]

    def first_open_period(self) -> int:
        """Scoring period of the snapshot's calendar day (ET); earlier days are already in ESPN's totals."""
        collected = self.meta.get("collected_at")
        if collected:
            day = pd.Timestamp(collected).date()
            for p, d in self.dates.items():
                if d == day:
                    return p
        return self.meta.get("current_scoring_period") or 0

    def periods_for(self, matchup_period: int, remaining_only: bool = True) -> list[int]:
        periods = [int(p) for p in self.meta["matchup_periods"][matchup_period]]
        cur = self.first_open_period()
        if remaining_only:
            # today's games may not have started; keep today in the window
            periods = [p for p in periods if p >= cur]
        return periods


def add_ratio_categories(samples: pd.DataFrame, cat_ids: list[str]) -> pd.DataFrame:
    for cid in cat_ids:
        cat = CATEGORIES[cid]
        if "ratio" in cat:
            made, att = cat["ratio"]
            samples[cat["name"]] = samples[made] / samples[att].replace(0, np.nan)
    return samples


def actual_totals(snap: dict, team_id: int, matchup_period: int) -> dict[str, float]:
    """Raw stat totals ESPN has already recorded for a team in a matchup (by stat name)."""
    m = snap.get("matchups")
    if m is None or m.empty:
        return {}
    rows = m[(m["matchup_period"] == matchup_period) & (m["team_id"] == team_id)].dropna(subset=["stat_id"])
    by_id = dict(zip(rows["stat_id"].astype(str), rows["score"].astype(float)))
    return {name: by_id[STAT_IDS[name]] for name in RAW_STATS if STAT_IDS[name] in by_id}


def with_actuals(proj: TeamProjection, actual: dict[str, float], cat_ids: list[str]) -> TeamProjection:
    """Add already-played totals to every simulated rest-of-week, then recompute percentages."""
    if not actual:
        return proj
    samples = proj.samples[RAW_STATS].copy()
    for name, value in actual.items():
        samples[name] += value
    return TeamProjection(proj.team_id, add_ratio_categories(samples, cat_ids), proj.starts, proj.daily)


# ---------------------------------------------------------------- comparison

def compare(me: TeamProjection, opp: TeamProjection, cat_ids: list[str],
            lo: float = 0.10, hi: float = 0.90) -> tuple[pd.DataFrame, dict]:
    """Per-category ranges and win probabilities; ties count as half a win."""
    rows, wins = [], np.zeros(len(me.samples))
    losses = np.zeros(len(me.samples))
    for cid in cat_ids:
        cat = CATEGORIES[cid]
        name = cat["name"]
        a, b = me.samples[name].to_numpy(), opp.samples[name].to_numpy()
        a, b = np.nan_to_num(a, nan=0.0), np.nan_to_num(b, nan=0.0)
        better = (a < b) if cat.get("lower_is_better") else (a > b)
        worse = (a > b) if cat.get("lower_is_better") else (a < b)
        p_win = better.mean() + 0.5 * (1 - better.mean() - worse.mean())
        wins += better
        losses += worse
        rows.append({
            "category": name, "stat_id": cid, "lower_is_better": bool(cat.get("lower_is_better")),
            "me_mean": a.mean(), "me_lo": np.quantile(a, lo), "me_hi": np.quantile(a, hi),
            "opp_mean": b.mean(), "opp_lo": np.quantile(b, lo), "opp_hi": np.quantile(b, hi),
            "p_win": p_win,
        })
    table = pd.DataFrame(rows)
    table["verdict"] = pd.cut(table["p_win"], [-0.01, 0.35, 0.65, 1.01], labels=["likely loss", "swing", "likely win"])
    overall = {
        "expected_cats_won": float(table["p_win"].sum()),
        "p_win_matchup": float((wins > losses).mean() + 0.5 * (wins == losses).mean()),
        "me_starts": int(me.starts["starts"].sum()) if len(me.starts) else 0,
        "opp_starts": int(opp.starts["starts"].sum()) if len(opp.starts) else 0,
    }
    return table, overall


def next_matchup_period(meta: dict) -> int:
    """On the last day of a week (Sunday), the week to plan for is the next one."""
    cur_period = meta.get("current_scoring_period") or 0
    for mp, periods in sorted(meta["matchup_periods"].items()):
        if max(periods) > cur_period:
            return mp
    return max(meta["matchup_periods"])


def opponent_of(snap: dict, team_id: int, matchup_period: int) -> int | None:
    m = snap.get("matchups")
    if m is None or m.empty:
        return None
    row = m[(m["matchup_period"] == matchup_period) & (m["team_id"] == team_id)]
    return None if row.empty or pd.isna(row["opponent_id"].iloc[0]) else int(row["opponent_id"].iloc[0])
