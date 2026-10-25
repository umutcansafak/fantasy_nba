"""NBA schedule helpers. One fantasy scoring period = one calendar day (ET)."""
from datetime import date, timedelta

import pandas as pd

DEFAULT_SLOT_COUNTS = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 11: 3, 12: 3, 13: 1}


def period_dates(pro: pd.DataFrame) -> dict[int, date]:
    """Scoring period -> calendar date, including off days without games."""
    known = pro.groupby("scoring_period")["date"].min()
    start = min(d - timedelta(days=int(p) - 1) for p, d in known.items())
    last = int(known.index.max())
    return {p: start + timedelta(days=p - 1) for p in range(1, last + 1)}


def period_for(day: date, pro: pd.DataFrame) -> int:
    dates = period_dates(pro)
    start = dates[1]
    return max(1, (day - start).days + 1)


def calendar_weeks(pro: pd.DataFrame) -> dict[int, list[int]]:
    """ESPN fantasy weeks: week 1 runs from opening night to the first Sunday, then Mon-Sun."""
    dates = period_dates(pro)
    weeks, current = {}, []
    for p in sorted(dates):
        current.append(p)
        if dates[p].weekday() == 6:  # Sunday closes a week
            weeks[len(weeks) + 1], current = current, []
    if current:
        weeks[len(weeks) + 1] = current
    return weeks


def resolve_matchup_periods(pro: pd.DataFrame, matchup_weeks: dict[int, list[int]] | None) -> dict[int, list[int]]:
    """Matchup period -> scoring periods (days).

    ESPN's scheduleSettings.matchupPeriods maps each matchup to calendar weeks,
    e.g. {18: [18], 19: [19, 20]} for 2-week playoff rounds. Without league
    settings every week is its own matchup.
    """
    weeks = calendar_weeks(pro)
    if not matchup_weeks:
        return weeks
    return {mp: [d for w in ws for d in weeks.get(int(w), [])] for mp, ws in matchup_weeks.items()}


def games_by_team_period(pro: pd.DataFrame, periods) -> pd.DataFrame:
    """Pivot: rows = NBA team id, columns = scoring period, values = games (0/1)."""
    sub = pro[pro["scoring_period"].isin(list(periods))]
    grid = sub.pivot_table(index="pro_team_id", columns="scoring_period", values="game_id",
                           aggfunc="count", fill_value=0)
    return grid.reindex(columns=list(periods), fill_value=0)
