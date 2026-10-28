"""Season so far from ESPN's recorded weekly matchup totals: league table, category record, weekly trend.

Counting categories are summed over the weeks; FG%/FT% are total makes / total attempts, as ESPN
scores them. The live week is included in totals (every team has played the same days) but not in
the win-loss record until it is over.
"""
import pandas as pd

from .config import CATEGORIES, STAT_NAMES
from .nba_schedule import period_dates
from .players import ordered_categories

RAW = ["PTS", "REB", "AST", "STL", "BLK", "3PM", "TO", "FGM", "FGA", "FTM", "FTA"]


def weekly_scores(snap: dict) -> pd.DataFrame:
    """One row per (matchup_period, team): raw weekly totals, opponent, and the category values."""
    m = snap.get("matchups")
    if m is None or m.empty:
        return pd.DataFrame()
    cur = snap["meta"].get("current_matchup_period") or m["matchup_period"].max()
    m = m.dropna(subset=["stat_id", "score"])
    m = m[m["matchup_period"] <= cur]
    m = m.assign(stat=m["stat_id"].astype(str).map(STAT_NAMES)).dropna(subset=["stat"])
    w = m.pivot_table(index=["matchup_period", "team_id", "opponent_id"], columns="stat", values="score",
                      aggfunc="first").reset_index()
    for s in RAW:
        if s not in w:
            w[s] = 0.0
    return add_categories(w, snap["meta"]["category_ids"])


def add_categories(df: pd.DataFrame, cat_ids: list[str]) -> pd.DataFrame:
    for cid in cat_ids:
        cat = CATEGORIES[cid]
        if "ratio" in cat:
            made, att = cat["ratio"]
            df[cat["name"]] = df[made] / df[att].where(df[att] > 0)
    return df


def season_table(snap: dict) -> pd.DataFrame:
    """Per team: season totals (percentages from makes/attempts) and the league rank in each category."""
    w = weekly_scores(snap)
    if w.empty:
        return w
    cat_ids = snap["meta"]["category_ids"]
    t = add_categories(w.groupby("team_id")[RAW].sum().reset_index(), cat_ids)
    t["team"] = t["team_id"].map(snap["teams"].set_index("team_id")["name"])
    for c in ordered_categories(cat_ids):
        lower = any(CATEGORIES[i].get("lower_is_better") for i in cat_ids if CATEGORIES[i]["name"] == c)
        t[f"rank_{c}"] = t[c].rank(ascending=lower, method="min").astype(int)
    t["avg_rank"] = t[[f"rank_{c}" for c in ordered_categories(cat_ids)]].mean(axis=1)
    return t.sort_values("avg_rank").reset_index(drop=True)


def completed_weeks(snap: dict) -> list[int]:
    """Matchup periods whose last day is before the snapshot's day (results are final)."""
    meta = snap["meta"]
    dates = period_dates(snap["pro_schedule"])
    today = pd.Timestamp(meta["collected_at"]).date() if meta.get("collected_at") else None
    out = []
    for mp, days in sorted(meta["matchup_periods"].items(), key=lambda kv: int(kv[0])):
        last = dates.get(max(days))
        if last is not None and today is not None and last < today:
            out.append(int(mp))
    return out


def week_results(snap: dict, team_id: int) -> pd.DataFrame:
    """Per week (completed or live): my value, opponent value and W/L/T in each category."""
    w = weekly_scores(snap)
    if w.empty:
        return w
    cats = ordered_categories(snap["meta"]["category_ids"])
    lower = {CATEGORIES[i]["name"]: bool(CATEGORIES[i].get("lower_is_better")) for i in snap["meta"]["category_ids"]}
    done = set(completed_weeks(snap))
    teams = snap["teams"].set_index("team_id")["name"].to_dict()
    me = w[w["team_id"] == team_id]
    rows = []
    for _, r in me.iterrows():
        o = w[(w["matchup_period"] == r["matchup_period"]) & (w["team_id"] == r["opponent_id"])]
        if o.empty:
            continue
        o = o.iloc[0]
        row = {"matchup_period": int(r["matchup_period"]), "opponent": teams.get(int(r["opponent_id"]), "?"),
               "final": int(r["matchup_period"]) in done}
        for c in cats:
            a, b = r[c], o[c]
            res = "T" if pd.isna(a) or pd.isna(b) or a == b else ("W" if (a < b) == lower[c] else "L")
            row.update({f"me_{c}": a, f"opp_{c}": b, f"res_{c}": res})
        row["W"] = sum(row[f"res_{c}"] == "W" for c in cats)
        row["L"] = sum(row[f"res_{c}"] == "L" for c in cats)
        row["T"] = len(cats) - row["W"] - row["L"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("matchup_period").reset_index(drop=True)


def category_record(results: pd.DataFrame, cats: list[str]) -> pd.DataFrame:
    """W-L-T per category over completed weeks only."""
    done = results[results["final"]] if not results.empty else results
    return pd.DataFrame([{"category": c, "W": int((done[f"res_{c}"] == "W").sum()),
                          "L": int((done[f"res_{c}"] == "L").sum()),
                          "T": int((done[f"res_{c}"] == "T").sum())} for c in cats])


def weekly_ranks(snap: dict, team_id: int) -> pd.DataFrame:
    """My league rank in each category, week by week."""
    w = weekly_scores(snap)
    if w.empty:
        return w
    cat_ids = snap["meta"]["category_ids"]
    rows = []
    for mp, g in w.groupby("matchup_period"):
        row = {"matchup_period": int(mp)}
        for cid in cat_ids:
            c = CATEGORIES[cid]["name"]
            ranks = g[c].rank(ascending=bool(CATEGORIES[cid].get("lower_is_better")), method="min")
            mine = ranks[g["team_id"] == team_id]
            row[c] = int(mine.iloc[0]) if len(mine) and pd.notna(mine.iloc[0]) else None
        rows.append(row)
    return pd.DataFrame(rows)
