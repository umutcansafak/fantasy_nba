"""Matchup dashboard: `streamlit run app.py`."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fantasy import store
from fantasy.config import CATEGORIES
from fantasy.app_data import DATA_ROOT, load, simulator, snapshot_version
from fantasy.projection import (actual_totals, compare, next_matchup_period, opponent_of,
                                with_actuals)
from fantasy.league_page import week_data
from fantasy.live import live_matchup
from fantasy.contrib_view import show_contributions
from fantasy.players import roster_contributions
from fantasy.recommend import recommend, weakest_player

# Categorical slots 1-2 of the validated reference palette (me = blue, opponent = orange).
ME, OPP, OTHER = "#2a78d6", "#eb6834", "#8a8984"
# Diverging pair for win probability: blue = winning side, red = losing side, gray = swing midpoint.
WIN, LOSS = "#2a78d6", "#e34948"
STATUS_ICON = {"Locked": "🔒", "Leaning win": "↗", "Swing": "⚖", "Leaning loss": "↘", "Lost": "✕"}
st.set_page_config(page_title="Fantasy NBA matchup", layout="wide")


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


@st.cache_data(show_spinner="Simulating every team…", max_entries=6)
def league_week(day, version, demo, n_sims, mp):
    sim = simulator(day, version, demo, n_sims)
    return week_data(sim, sim.snap, mp, f"Week {mp}")


def league_chart(cat, team_names, visible, me_id, opp_id):
    """All teams ranked by expected total; hidden teams keep their true rank."""
    ranked = sorted(cat["rows"], key=lambda r: r["mean"], reverse=not cat["lower"])
    rank = {r["id"]: i + 1 for i, r in enumerate(ranked)}
    rows = [r for r in ranked if r["id"] in visible]
    fmt_v = (lambda v: f"{v:.1%}") if cat["pct"] else (lambda v: f"{v:,.0f}")
    fig = go.Figure()
    labels = []
    for r in rows:
        color = ME if r["id"] == me_id else OPP if r["id"] == opp_id else OTHER
        label = f"#{rank[r['id']]}  {team_names[r['id']]}"
        labels.append(label)
        hover = (f"<b>{label}</b><br>expected {fmt_v(r['mean'])}<br>"
                 f"80% range {fmt_v(r['lo'])}–{fmt_v(r['hi'])}<extra></extra>")
        fig.add_trace(go.Scatter(x=[r["lo"], r["hi"]], y=[label, label], mode="lines", opacity=0.35,
                                 line=dict(color=color, width=9), hovertemplate=hover, showlegend=False))
        fig.add_trace(go.Scatter(x=[r["mean"]], y=[label], mode="markers+text", text=[fmt_v(r["mean"])],
                                 textposition="middle right", marker=dict(color=color, size=11),
                                 hovertemplate=hover, showlegend=False,
                                 textfont=dict(size=12, weight=700 if r["id"] == me_id else 400)))
    fig.update_yaxes(categoryorder="array", categoryarray=labels[::-1], showgrid=False)
    fig.update_xaxes(tickformat=".1%" if cat["pct"] else ",.0f", showgrid=True, nticks=6)
    fig.update_layout(height=70 + 34 * max(len(rows), 1), margin=dict(l=10, r=40, t=10, b=30), hovermode="closest")
    return fig


@st.cache_data(show_spinner="Simulating the rest of this week…", max_entries=6)
def live_week(day, version, demo, n_sims, me_id):
    sim = simulator(day, version, demo, n_sims)
    return live_matchup(sim, sim.snap, me_id)


def pwin_chart(cats):
    """One dot per category on a 0-100% win-probability axis; the swing band is shaded."""
    d = cats.sort_values("p_win")
    colors = [WIN if p >= 0.65 else LOSS if p <= 0.35 else OTHER for p in d["p_win"]]
    fig = go.Figure()
    fig.add_vrect(x0=35, x1=65, fillcolor=OTHER, opacity=0.12, line_width=0)
    fig.add_vline(x=50, line=dict(color=OTHER, width=1, dash="dot"))
    fig.add_trace(go.Scatter(
        x=d["p_win"] * 100, y=d["category"], mode="markers+text", marker=dict(color=colors, size=13),
        text=[f"{p:.0%}  {STATUS_ICON[s]} {s}" for p, s in zip(d["p_win"], d["status"])],
        textposition="middle right", cliponaxis=False, showlegend=False,
        hovertemplate="%{y}: %{x:.0f}% chance to win<extra></extra>"))
    fig.add_annotation(x=50, y=1.0, yref="paper", yanchor="bottom", text="swing zone", showarrow=False,
                       font=dict(size=11, color=OTHER))
    fig.update_layout(height=60 + 34 * len(d), margin=dict(l=10, r=150, t=24, b=30), hovermode="closest",
                      xaxis=dict(range=[0, 100], ticksuffix="%", dtick=25, showgrid=True,
                                 title="chance to win the category by Sunday"),
                      yaxis=dict(showgrid=False))
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
    version = snapshot_version(day)
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

tab_live, tab_cats, tab_players, tab_league, tab_games, tab_fa = st.tabs(
    ["This week", "Categories", "My players", "League", "Games & lineups", "Free agents"])

# ---------------------------------------------------------------- this week

with tab_live:
    lv = live_week(day, version, demo, n_sims, me_id)
    if lv is None:
        st.info("No matchup is in progress for this team.")
    else:
        lv_opp = teams[lv["opp_id"]]
        played = lv["days_total"] - lv["days_left"]
        st.subheader(f"Week {lv['mp']}: {me_name} vs {lv_opp}")
        st.caption(f"{lv['first_day']:%a %b %d} – {lv['last_day']:%a %b %d} · day {min(played + 1, lv['days_total'])} "
                   f"of {lv['days_total']} · {lv['days_left']} days left incl. today · data from {day}. "
                   "Always the live week vs your scheduled opponent, whatever the sidebar says.")
        w, l, t = lv["score_now"]
        sm, so = lv["starts_left"]
        k = st.columns(4)
        k[0].metric("Score now", f"{w}–{l}–{t}" if lv["started"] else "not started",
                    help="Categories you lead, trail and are tied in with the totals ESPN has recorded so far")
        k[1].metric("Projected final", f"{lv['expected_cats']:.1f} cats", help="Expected categories won by Sunday")
        k[2].metric("Chance to win the week", f"{lv['p_win']:.0%}")
        k[3].metric("Starts left", f"{sm} vs {so}", delta=sm - so,
                    help="Players you can start for the rest of the week after lineup limits")
        st.caption(f"Starting today: {lv['starts_today'][0]} for you, {lv['starts_today'][1]} for {lv_opp}.")

        cats = lv["cats"]
        groups = {s: cats[cats["status"] == s]["category"].tolist() for s in STATUS_ICON}
        st.markdown("  \n".join(
            f"{STATUS_ICON[s]} **{s}:** {', '.join(groups[s])}" for s in STATUS_ICON if groups[s]))
        if groups["Swing"]:
            st.caption("Swing categories are where a streamer, a lineup change or benching a risky player "
                       "can still change the result.")

        c1, c2 = st.columns([1, 1])
        c1.plotly_chart(pwin_chart(cats), width="stretch", config={"displayModeBar": False}, key="live-pwin")

        def show(row, col):
            v = row[col]
            if v is None or pd.isna(v):
                return "–"
            return f"{v:.1%}" if row["pct"] else f"{v:,.0f}"
        view = pd.DataFrame({
            "Category": cats["category"],
            "Now (you – opp)": [f"{show(r, 'me_now')} – {show(r, 'opp_now')}" for _, r in cats.iterrows()],
            "Projected final": [f"{show(r, 'me_final')} – {show(r, 'opp_final')}" for _, r in cats.iterrows()],
            "Status": [f"{STATUS_ICON[s]} {s}" for s in cats["status"]],
        }).loc[cats["p_win"].sort_values().index]
        c2.dataframe(view, hide_index=True, width="stretch", height=36 * len(view) + 40)
        st.caption("Projected final = totals already recorded + the average of the simulated remaining days. "
                   "TO: lower is better.")

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

# -------------------------------------------------------------- my players

with tab_players:
    show_contributions(sim, roster_contributions(sim, me_id, mp), weekly_note="your roster", key="contrib")

# ------------------------------------------------------------------ league

with tab_league:
    lw = league_week(day, version, demo, n_sims, mp)
    if lw is None:
        st.info("No days left to simulate in this week.")
    else:
        st.caption(f"Every team's projected totals for week {mp} ({lw['note']}), independent of opponent. "
                   "Bands are the middle 80% of simulated weeks; ranks compare expected totals.")
        ids = list(teams)
        visible = st.pills("Teams shown (click to toggle)", ids, selection_mode="multi", default=ids,
                           format_func=lambda t: teams[t] + (" (you)" if t == me_id else " (opponent)" if t == opp_id else ""),
                           key="league_teams")
        visible = set(visible or [])
        cats = lw["cats"]
        def my_rank(c):
            ranked = sorted(c["rows"], key=lambda r: r["mean"], reverse=not c["lower"])
            return next(i + 1 for i, r in enumerate(ranked) if r["id"] == me_id)
        sub = st.tabs([f"{c['name']} · #{my_rank(c)}" for c in cats])
        for t, c in zip(sub, cats):
            with t:
                st.markdown(f"**{c['name']}**: you rank **#{my_rank(c)} of {len(c['rows'])}**"
                            + (" · lower is better" if c["lower"] else ""))
                if not visible:
                    st.info("All teams are hidden.")
                else:
                    st.plotly_chart(league_chart(c, teams, visible, me_id, opp_id), width="stretch",
                                    config={"displayModeBar": False}, key=f"league-{c['name']}")

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
