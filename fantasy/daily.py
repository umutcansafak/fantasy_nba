"""Machine-readable morning brief (`daily.json` on the Pages site) for a scheduled Claude task.

Everything a short daily summary needs, already computed: yesterday's box scores for my roster and
my opponent's day, where the live matchup stands, today's lineup picture, the best pickups for this
week's matchup (TO punted on my side), and injury / roster changes since the previous snapshot.
"""
import json
from datetime import timedelta
from pathlib import Path

import pandas as pd

from . import store
from .config import BENCH_SLOTS, CATEGORIES, DEFAULT_CATEGORY_WEIGHTS, PRO_TEAMS, SLOT_NAMES
from .fit import FitModel
from .live import live_matchup
from .nba_schedule import games_by_team_period
from .players import ordered_categories, roster_contributions
from .projection import Simulator
from .recommend import recommend

BOX = ["MIN", "PTS", "REB", "AST", "STL", "BLK", "3PM", "TO", "FGM", "FGA", "FTM", "FTA"]
N_PICKUPS = 8


def _r(v, nd=1):
    return None if v is None or pd.isna(v) else round(float(v), nd)


def _status(s):
    return "active" if s == "ACTIVE" else str(s).replace("_", " ").lower()


def _yesterday(sim, snap, team_id, period):
    """Box score per rostered player for one day, plus the team's totals (every rostered player)."""
    logs = snap["game_logs"]
    roster = sim.roster_of(team_id)
    day = logs[(logs["season"] == sim.meta["season"]) & (logs["scoring_period"] == period)
               & logs["player_id"].isin(roster["player_id"])]
    slot = dict(zip(roster["player_id"], roster["lineup_slot"]))
    players = [{"name": sim.players.at[r["player_id"], "name"], "current_slot": SLOT_NAMES.get(int(slot[r["player_id"]]), "?"),
                **{s: _r(r[s], 0) for s in BOX}}
               for _, r in day.sort_values("PTS", ascending=False).iterrows()]
    tot = day[BOX].sum()
    totals = {s: _r(tot[s], 0) for s in BOX}
    totals["FG%"] = _r(tot["FGM"] / tot["FGA"], 3) if tot["FGA"] else None
    totals["FT%"] = _r(tot["FTM"] / tot["FTA"], 3) if tot["FTA"] else None
    return {"players": players, "totals": totals}


def _today(sim, team_id, period):
    """Who has a game today, who the lineup filler would start, and lineup problems."""
    roster = sim.roster_of(team_id)
    grid = games_by_team_period(sim.pro, [period])
    starts = set(sim.plan_starts(roster, [period])["player_id"])
    pro = sim.pro[sim.pro["scoring_period"] == period].set_index("pro_team_id")["opponent"].to_dict()
    rows, issues = [], []
    for pid, slot in zip(roster["player_id"], roster["lineup_slot"]):
        if pid not in sim.players.index:
            continue
        p = sim.players.loc[pid]
        team = p["pro_team_id"]
        plays = team in grid.index and grid.at[team, period] > 0
        status = _status(p["injury_status"])
        active_slot = int(slot) not in BENCH_SLOTS
        rows.append({"name": p["name"], "team": p["pro_team"], "slot": SLOT_NAMES.get(int(slot), "?"),
                     "status": status, "game_today": bool(plays),
                     "opponent": PRO_TEAMS.get(pro.get(team), pro.get(team)) if plays else None,
                     "suggested_start": pid in starts})
        if active_slot and p["injury_status"] in ("OUT", "INJURY_RESERVE", "SUSPENSION"):
            issues.append(f"{p['name']} is {status} but in an active slot ({SLOT_NAMES.get(int(slot))})")
        if not active_slot and int(slot) != 13 and pid in starts:
            issues.append(f"{p['name']} is on the bench but has a game and should start")
    return rows, issues


def _changes(snap, prev, team_ids):
    """Injury-status changes and adds/drops on the given teams since the previous snapshot."""
    if prev is None:
        return {"injuries": [], "roster_moves": []}
    teams = snap["teams"].set_index("team_id")["name"].to_dict()
    now_p = snap["players"].set_index("player_id")
    was_p = prev["players"].set_index("player_id")
    injuries, moves = [], []
    for tid in team_ids:
        now_r = set(snap["rosters"].query("team_id == @tid")["player_id"])
        was_r = set(prev["rosters"].query("team_id == @tid")["player_id"])
        name = lambda pid: now_p.at[pid, "name"] if pid in now_p.index else was_p.at[pid, "name"] if pid in was_p.index else str(pid)
        moves += [{"team": teams[tid], "added": name(p)} for p in now_r - was_r]
        moves += [{"team": teams[tid], "dropped": name(p)} for p in was_r - now_r]
        for pid in now_r & was_r:
            if pid in now_p.index and pid in was_p.index and now_p.at[pid, "injury_status"] != was_p.at[pid, "injury_status"]:
                injuries.append({"team": teams[tid], "player": now_p.at[pid, "name"],
                                 "from": _status(was_p.at[pid, "injury_status"]),
                                 "to": _status(now_p.at[pid, "injury_status"])})
    return {"injuries": injuries, "roster_moves": moves}


def _pickups(sim, me_id, opp_id, periods, weights):
    """Free agents simulated into this week's matchup in place of my lowest-value player, ranked by the
    weighted change in categories won (my punts ignored), with each one's rest-of-season fit."""
    cats = ordered_categories(sim.cat_ids)
    mine = roster_contributions(sim, me_id).assign(w=lambda d: sum(d[c] * weights[c] for c in cats))
    played = mine[mine["gp"] > 0].sort_values("w")
    drop = int(played["player_id"].iloc[0]) if len(played) else None
    recs, drop = recommend(sim, me_id, opp_id, periods, drop_id=drop, n=25)
    if recs.empty:
        return {"drop": None, "players": []}
    recs["week_gain"] = sum(recs[f"d_{c}"] * weights[c] for c in cats)
    recs = recs.sort_values("week_gain", ascending=False).head(N_PICKUPS)
    fit = FitModel(sim).fit(me_id, recs["player_id"].tolist(), drop).set_index("player_id")
    out = []
    for _, r in recs.iterrows():
        f = fit.loc[r["player_id"]] if r["player_id"] in fit.index else None
        out.append({
            "name": r["name"], "team": r["team"], "status": _status(sim.players.at[r["player_id"], "injury_status"]), "games_left_this_week": int(r["games"]),
            "pct_owned": _r(r["pct_owned"], 0),
            "this_week_extra_cats": _r(r["week_gain"], 2),
            "this_week_win_rate_change_pts": {c: _r(r[f"d_{c}"] * 100, 0) for c in cats if weights[c] > 0},
            "rest_of_season_fit_cats_per_week": None if f is None else _r(sum(f[c] * weights[c] for c in cats), 2),
        })
    return {"drop": sim.players.at[drop, "name"] if drop in sim.players.index else None, "players": out}


def build(sim: Simulator, snap: dict, me_id: int, root: Path) -> dict:
    meta = snap["meta"]
    cats = ordered_categories(sim.cat_ids)
    weights = {c: DEFAULT_CATEGORY_WEIGHTS.get(c, 1.0) for c in cats}
    teams = snap["teams"].set_index("team_id")["name"].to_dict()
    today = sim.first_open_period()
    days = store.list_snapshots(root)
    day = pd.Timestamp(meta["collected_at"]).date().isoformat()
    prev_days = [d for d in days if d < day]
    prev = store.load_snapshot(prev_days[-1], root) if prev_days else None

    lv = live_matchup(sim, snap, me_id)
    out = {
        "updated": meta["collected_at"], "today_et": sim.dates[today].isoformat() if today in sim.dates else day,
        "league": meta.get("league_name"), "my_team": teams[me_id],
        "my_punts": [c for c in cats if weights[c] == 0],
        "notes": "Win-rate changes are percentage points. Yesterday's totals include every rostered player, "
                 "even ones on the bench, so they can differ slightly from ESPN's scored totals; current_slot is today's "
                 "slot, not necessarily the one the player was in yesterday. Pickup gains are vs dropping "
                 "pickups.drop; small or negative gains mean no free agent clearly helps.",
    }
    opp_id = lv["opp_id"] if lv else None
    if lv:
        c = lv["cats"]
        out["matchup"] = {
            "week": lv["mp"], "opponent": teams[opp_id],
            "dates": [lv["first_day"].isoformat(), lv["last_day"].isoformat()],
            "days_left_including_today": lv["days_left"],
            "score_now_w_l_t": list(lv["score_now"]), "projected_cats_won": _r(lv["expected_cats"], 1),
            "chance_to_win_week": _r(lv["p_win"], 2),
            "starts_left_me_vs_opp": list(lv["starts_left"]), "starts_today_me_vs_opp": list(lv["starts_today"]),
            "categories": [{"category": r.category, "me_now": _r(r.me_now, 3), "opp_now": _r(r.opp_now, 3),
                            "me_projected": _r(r.me_final, 3), "opp_projected": _r(r.opp_final, 3),
                            "chance_to_win": _r(r.p_win, 2), "status": r.status,
                            "punted": weights[r.category] == 0} for r in c.itertuples()],
        }
    yday = today - 1
    if yday in sim.dates:
        out["yesterday"] = {"date": sim.dates[yday].isoformat(), "me": _yesterday(sim, snap, me_id, yday)}
        if opp_id is not None:
            out["yesterday"]["opponent"] = _yesterday(sim, snap, opp_id, yday)["totals"]
    lineup, issues = _today(sim, me_id, today)
    out["today"] = {"my_roster": lineup, "lineup_issues": issues}
    out["changes_since_previous_snapshot"] = _changes(snap, prev, [t for t in (me_id, opp_id) if t is not None])
    if lv and lv["days_left"]:
        out["pickups"] = _pickups(sim, me_id, opp_id, sim.periods_for(lv["mp"]), weights)
    return out


def write(sim, snap, me_id, root: Path, out_dir: Path) -> Path:
    path = out_dir / "daily.json"
    path.write_text(json.dumps(build(sim, snap, me_id, root), indent=1, default=str))
    return path
