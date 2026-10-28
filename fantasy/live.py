"""Live view of the current matchup: score so far, projected final, and which categories are still in play."""
import numpy as np
import pandas as pd

from .config import CATEGORIES
from .projection import Simulator, actual_totals, compare, opponent_of, with_actuals

# P(win) bins for the rest of the week; "swing" is where a move can still change the result.
STATUS_BINS = [-0.01, 0.10, 0.35, 0.65, 0.90, 1.01]
STATUS_LABELS = ["Lost", "Leaning loss", "Swing", "Leaning win", "Locked"]


def category_value(totals: dict[str, float], cid: str) -> float | None:
    """A category's value from raw stat totals; None when a percentage has no attempts yet."""
    cat = CATEGORIES[cid]
    if "ratio" in cat:
        made, att = cat["ratio"]
        return totals[made] / totals[att] if totals.get(att) else None
    return totals.get(cat["name"], 0.0)


def leader(me: float | None, opp: float | None, lower_is_better: bool) -> str:
    if me is None or opp is None or me == opp:
        return "tie"
    return "me" if (me < opp) == lower_is_better else "opp"


def live_matchup(sim: Simulator, snap: dict, me_id: int) -> dict | None:
    """Current week vs the scheduled opponent, or None outside the season."""
    meta = snap["meta"]
    mp = meta.get("current_matchup_period")
    opp_id = opponent_of(snap, me_id, mp) if mp else None
    if not mp or opp_id is None or mp not in meta["matchup_periods"]:
        return None
    week = [int(p) for p in meta["matchup_periods"][mp]]
    left = sim.periods_for(mp, remaining_only=True)

    now_me, now_opp = actual_totals(snap, me_id, mp), actual_totals(snap, opp_id, mp)
    if left:
        me = with_actuals(sim.project_team(me_id, sim.roster_of(me_id), left, 0), now_me, sim.cat_ids)
        opp = with_actuals(sim.project_team(opp_id, sim.roster_of(opp_id), left, 1), now_opp, sim.cat_ids)
        table, overall = compare(me, opp, sim.cat_ids)
        today = left[0]
        today_me, today_opp = int(me.daily.get(today, 0)), int(opp.daily.get(today, 0))
    else:  # week over: the recorded totals are final
        table, overall, today_me, today_opp = None, None, 0, 0

    rows = []
    for cid in sim.cat_ids:
        cat = CATEGORIES[cid]
        a, b = category_value(now_me, cid), category_value(now_opp, cid)
        row = {"category": cat["name"], "lower_is_better": bool(cat.get("lower_is_better")),
               "pct": "ratio" in cat, "me_now": a, "opp_now": b,
               "leader": leader(a, b, bool(cat.get("lower_is_better")))}
        if table is not None:
            t = table.set_index("category").loc[cat["name"]]
            row.update(me_final=t["me_mean"], opp_final=t["opp_mean"], p_win=float(t["p_win"]))
        else:
            row["p_win"] = {"me": 1.0, "opp": 0.0, "tie": 0.5}[row["leader"]]
            row.update(me_final=a, opp_final=b)
        rows.append(row)
    cats = pd.DataFrame(rows)
    cats["status"] = pd.cut(cats["p_win"], STATUS_BINS, labels=STATUS_LABELS).astype(str)

    lead = cats["leader"]
    return {
        "mp": mp, "opp_id": opp_id, "cats": cats,
        "first_day": sim.dates[week[0]], "last_day": sim.dates[week[-1]],
        "days_total": len(week), "days_left": len(left), "started": bool(now_me or now_opp),
        "score_now": (int((lead == "me").sum()), int((lead == "opp").sum()), int((lead == "tie").sum())),
        "expected_cats": float(cats["p_win"].sum()),
        "p_win": overall["p_win_matchup"] if overall else float(np.sign((lead == "me").sum() - (lead == "opp").sum()) / 2 + 0.5),
        "starts_left": (overall["me_starts"], overall["opp_starts"]) if overall else (0, 0),
        "starts_today": (today_me, today_opp),
    }
