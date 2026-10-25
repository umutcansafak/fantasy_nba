"""Matchup dashboard: `streamlit run app.py`."""
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fantasy import store
from fantasy.config import CATEGORIES
from fantasy.demo import demo_league
from fantasy.projection import (Simulator, actual_totals, compare, next_matchup_period, opponent_of,
                                with_actuals)
from fantasy.recommend import recommend, weakest_player

# Categorical slots 1-2 of the validated reference palette (me = blue, opponent = orange).
ME, OPP = "#2a78d6", "#eb6834"
DATA_ROOT = Path(os.getenv("FANTASY_DATA", store.DATA_DIR))

st.set_page_config(page_title="Fantasy NBA matchup", layout="wide")


@st.cache_data(show_spinner=False, max_entries=5)
def load(day, version=None):
    """`version` (meta.json mtime) busts the cache when a day's snapshot is re-collected."""
    return store.load_snapshot(day, root=DATA_ROOT)


@st.cache_resource(show_spinner="Building player pools…", max_entries=3)
def simulator(day, version, demo, n_sims):
    snap = load(day, version)
    return Simulator(demo_league(snap) if demo else snap, n_sims=n_sims)


def fmt(cat_name, v):
    return f"{v:.1%}" if cat_name.endswith("%") else f"{v:,.0f}"


def category_chart(row, me_name, opp_name):
    """10th-90th percentile band + mean marker for each side, on one shared axis."""
    name = row["category"]
    fig = go.Figure()
    for label, color, prefix in ((opp_name, OPP, "opp"), (me_name, ME, "me")):
        lo, hi, mean = row[f"{prefix}_lo"], row[f"{prefix}_hi"], row[f"{prefix}_mean"]
        hover = f"<b>{label}</b><br>expected {fmt(name, mean)}<br>80% range {fmt(name, lo)}–{fmt(name, hi)}<extra></extra>"
        fig.add_trace(go.Scatter(x=[lo, hi], y=[label, label], mode="lines", line=dict(color=color, width=8),
                                 opacity=0.35, hovertemplate=hover, showlegend=False))
        fig.add_trace(go.Scatter(x=[mean], y=[label], mode="markers+text", text=[fmt(name, mean)],
                                 textposition="top center", marker=dict(color=color, size=12),
                                 hovertemplate=hover, showlegend=False))
    span = max(row["me_hi"], row["opp_hi"]) - min(row["me_lo"], row["opp_lo"])
    pad = span * 0.15 or 1
    fig.update_xaxes(range=[min(row["me_lo"], row["opp_lo"]) - pad, max(row["me_hi"], row["opp_hi"]) + pad],
                     tickformat=".1%" if name.endswith("%") else ",.0f", showgrid=True, nticks=5)
    fig.update_yaxes(showgrid=False, range=[-0.5, 1.8])  # headroom for the value labels
    better = "lower is better" if row["lower_is_better"] else ""
    fig.update_layout(
        title=dict(text=f"<b>{name}</b>  ·  P(win) {row['p_win']:.0%}  ·  {row['verdict']}"
                        + (f"  <span style='font-size:11px'>({better})</span>" if better else ""), font=dict(size=14)),
        height=190, margin=dict(l=10, r=10, t=40, b=20), hovermode="closest",
    )
    return fig


def games_chart(me_daily, opp_daily, dates, me_name, opp_name):
    fig = go.Figure()
    for label, color, daily in ((me_name, ME, me_daily), (opp_name, OPP, opp_daily)):
        y = [int(daily.get(p, 0)) for p in dates]
        fig.add_trace(go.Bar(x=[dates[p].strftime("%a %b %d") for p in dates], y=y, name=label,
                             marker=dict(color=color, line=dict(width=0)), text=y, textposition="outside",
                             hovertemplate=f"<b>{label}</b><br>%{{x}}: %{{y}} starters<extra></extra>"))
    fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.08, height=300,
                      margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.12),
                      yaxis=dict(title="players starting", rangemode="tozero"))
    return fig


# ------------------------------------------------------------------- sidebar

days = store.list_snapshots(DATA_ROOT)
if not days:
    st.error("No data yet. Run `python -m fantasy.collect` (or `--public` to try it without a league).")
    st.stop()

with st.sidebar:
    st.header("Setup")
    day = st.selectbox("Snapshot", days[::-1], help="Data collected on this day (ET)")
    version = (store.snapshot_dir(day, DATA_ROOT) / "meta.json").stat().st_mtime
    snap = load(day, version)
    demo = snap["rosters"].empty
    if demo:
        st.info("No league in this snapshot (public pool). Showing a demo league of two drafted teams.")
    n_sims = st.select_slider("Simulations", [1000, 2000, 5000, 10000], value=5000)
    sim = simulator(day, version, demo, n_sims)
    s = sim.snap
    teams = s["teams"].set_index("team_id")["name"].to_dict()
    my_default = s["meta"].get("team_id") or next(iter(teams))
    me_id = st.selectbox("My team", list(teams), index=list(teams).index(my_default), format_func=teams.get)

    periods_map = s["meta"]["matchup_periods"]
    nxt = next_matchup_period(s["meta"])
    mps = sorted(periods_map)
    mp = st.selectbox("Matchup week", mps, index=mps.index(nxt),
                      format_func=lambda m: f"Week {m}: {sim.dates[min(periods_map[m])]:%b %d} – {sim.dates[max(periods_map[m])]:%b %d}")
    others = [t for t in teams if t != me_id]
    scheduled = opponent_of(s, me_id, mp)
    opp_id = st.selectbox("Opponent", others, index=others.index(scheduled) if scheduled in others else 0,
                          format_func=lambda t: teams[t] + ("  (scheduled)" if t == scheduled else ""))
    remaining = st.toggle("Only days not yet played", value=True,
                          help="For the current week: totals ESPN already recorded + simulated games from today on.")

periods = sim.periods_for(mp, remaining_only=remaining)
if not periods:
    st.warning("This week is already over. Pick a later week or turn off 'Only days not yet played'.")
    st.stop()

me = sim.project_team(me_id, sim.roster_of(me_id), periods, 0)
opp = sim.project_team(opp_id, sim.roster_of(opp_id), periods, 1)
live = remaining and mp == s["meta"].get("current_matchup_period")
if live:
    me = with_actuals(me, actual_totals(s, me_id, mp), sim.cat_ids)
    opp = with_actuals(opp, actual_totals(s, opp_id, mp), sim.cat_ids)
table, overall = compare(me, opp, sim.cat_ids)
me_name, opp_name = teams[me_id], teams[opp_id]

st.title(f"{me_name} vs {opp_name}")
st.caption(f"Week {mp} · {sim.dates[periods[0]]:%a %b %d} – {sim.dates[periods[-1]]:%a %b %d} · "
           f"{n_sims:,} simulated weeks · data from {day}"
           + (" · includes totals already played this week" if live else ""))

k = st.columns(4)
k[0].metric("Expected categories won", f"{overall['expected_cats_won']:.1f} / {len(table)}")
k[1].metric("P(win matchup)", f"{overall['p_win_matchup']:.0%}")
k[2].metric("My starts", overall["me_starts"], delta=overall["me_starts"] - overall["opp_starts"])
k[3].metric("Opponent starts", overall["opp_starts"])

tab_cats, tab_games, tab_fa, tab_trends = st.tabs(["Categories", "Games & lineups", "Free agents", "Trends"])

# --------------------------------------------------------------- categories

with tab_cats:
    swing = table[table["verdict"] == "swing"]["category"].tolist()
    if swing:
        st.markdown(f"**Focus on:** {', '.join(swing)}: these are close enough that a pickup or a streaming plan "
                    "can change the result.")
    st.caption("Bars show the middle 80% of simulated weeks; the dot is the expected total. "
               "Charts are ordered closest race first.")
    ordered = table.assign(closeness=(table["p_win"] - 0.5).abs()).sort_values("closeness")
    cols = st.columns(3)
    for i, (_, row) in enumerate(ordered.iterrows()):
        cols[i % 3].plotly_chart(category_chart(row, me_name, opp_name), width="stretch",
                                 config={"displayModeBar": False}, key=f"cat-{row['category']}")

    with st.expander("Table view"):
        view = table[["category", "me_lo", "me_mean", "me_hi", "opp_lo", "opp_mean", "opp_hi", "p_win", "verdict"]].copy()
        for c in ["me_lo", "me_mean", "me_hi", "opp_lo", "opp_mean", "opp_hi"]:
            view[c] = [fmt(n, v) for n, v in zip(view["category"], view[c])]
        view["p_win"] = view["p_win"].map("{:.0%}".format)
        st.dataframe(view.rename(columns={"me_lo": "me p10", "me_mean": "me exp", "me_hi": "me p90",
                                          "opp_lo": "opp p10", "opp_mean": "opp exp", "opp_hi": "opp p90"}),
                     hide_index=True, width="stretch")

# ------------------------------------------------------------ games/lineups

with tab_games:
    dates = {p: sim.dates[p] for p in periods}
    st.plotly_chart(games_chart(me.daily, opp.daily, dates, me_name, opp_name), width="stretch",
                    config={"displayModeBar": False})
    st.caption("Starters per day after filling active lineup slots; players beyond the slot limit sit on the bench.")
    c1, c2 = st.columns(2)
    for col, proj, label in ((c1, me, me_name), (c2, opp, opp_name)):
        roster = sim.roster_of(proj.team_id).merge(sim.players[["name", "pro_team", "injury_status"]].reset_index(),
                                                    on="player_id", how="left")
        st_ = roster.merge(proj.starts[["player_id", "nba_games", "starts", "exp_games"]], on="player_id", how="left")
        st_ = st_.fillna({"nba_games": 0, "starts": 0, "exp_games": 0})
        col.subheader(label)
        col.dataframe(st_[["name", "pro_team", "injury_status", "nba_games", "starts", "exp_games"]]
                      .sort_values(["starts", "nba_games"], ascending=False),
                      hide_index=True, width="stretch",
                      column_config={"exp_games": st.column_config.NumberColumn("expected games", format="%.1f")})

# ------------------------------------------------------------- free agents

with tab_fa:
    roster = sim.roster_of(me_id)
    names = sim.players["name"].to_dict()
    default_drop = weakest_player(sim, roster)
    choices = roster["player_id"].tolist()
    drop = st.selectbox("Player to drop", choices, index=choices.index(default_drop),
                        format_func=lambda p: names.get(p, str(p)) + ("  (lowest value)" if p == default_drop else ""))
    n_fa = st.slider("Free agents to test", 10, 50, 25, step=5)
    if st.button("Simulate pickups", type="primary"):
        with st.spinner("Re-running the matchup for each candidate…"):
            recs, _ = recommend(sim, me_id, opp_id, periods, drop_id=drop, n=n_fa)
        if recs.empty:
            st.info("No free agents with games in this window.")
        else:
            cat_cols = [f"d_{CATEGORIES[c]['name']}" for c in sim.cat_ids]
            shown = recs[["name", "team", "status", "games", "pct_owned", "d_expected_cats", "d_p_win_matchup", *cat_cols]]
            st.dataframe(
                shown, hide_index=True, width="stretch",
                column_config={
                    "pct_owned": st.column_config.NumberColumn("% owned", format="%.0f"),
                    "d_expected_cats": st.column_config.NumberColumn("Δ exp. cats", format="%+.2f"),
                    "d_p_win_matchup": st.column_config.NumberColumn("Δ P(win)", format="%+.1%"),
                    **{c: st.column_config.NumberColumn(c[2:], format="%+.0%") for c in cat_cols},
                })
            st.caption("Deltas are versus your current roster for this week, using the same simulated games for "
                       "everyone else, so differences come from the swap itself.")

# ------------------------------------------------------------------ trends

with tab_trends:
    m = s.get("matchups")
    if m is None or m.empty or m["stat_id"].isna().all():
        st.info("Weekly results appear here once your league has completed matchups.")
    else:
        m = m.dropna(subset=["stat_id", "score"])
        cur = s["meta"].get("current_matchup_period") or m["matchup_period"].max()
        m = m[m["matchup_period"] <= cur]
        cols = st.columns(3)
        for i, cid in enumerate(sim.cat_ids):
            name = CATEGORIES[cid]["name"]
            d = m[m["stat_id"] == cid]
            if d.empty:
                continue
            league_med = d.groupby("matchup_period")["score"].median()
            mine = d[d["team_id"] == me_id].set_index("matchup_period")["score"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=league_med.index, y=league_med.values, name="League median", mode="lines",
                                     line=dict(color="#8a8984", width=2, dash="dot")))
            fig.add_trace(go.Scatter(x=mine.index, y=mine.values, name=me_name, mode="lines+markers",
                                     line=dict(color=ME, width=2), marker=dict(size=8)))
            fig.update_layout(title=dict(text=f"<b>{name}</b>", font=dict(size=14)), height=220,
                              margin=dict(l=10, r=10, t=40, b=20), hovermode="x unified",
                              legend=dict(orientation="h", y=-0.25), xaxis=dict(title=None, dtick=1),
                              yaxis=dict(tickformat=".1%" if name.endswith("%") else ",.0f"))
            cols[i % 3].plotly_chart(fig, width="stretch", config={"displayModeBar": False},
                                     key=f"trend-{name}")
        st.caption("Your weekly category totals (blue) against the league median (dotted). The current week is partial.")
