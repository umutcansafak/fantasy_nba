"""Season so far: league table per category, my category record, and weekly trends."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fantasy import store
from fantasy.app_data import DATA_ROOT, load, snapshot_version
from fantasy.players import ordered_categories
from fantasy.season import category_record, season_table, week_results, weekly_ranks, weekly_scores

# Diverging: blue = good rank / win, red = bad rank / loss, gray midpoint; my series blue, league gray.
GOOD, BAD, MID, ME, OTHER = "#2a78d6", "#e34948", "#e6e5e0", "#2a78d6", "#8a8984"

st.set_page_config(page_title="Season", layout="wide")


def fmt(c, v):
    return "–" if v is None or pd.isna(v) else f"{v:.1%}" if c.endswith("%") else f"{v:,.0f}"


def rank_heatmap(t: pd.DataFrame, cats: list[str], me_id: int, n: int):
    """Teams x categories coloured by league rank (1 = best); cell text = season value and rank."""
    rows = t.iloc[::-1]
    z = rows[[f"rank_{c}" for c in cats]].to_numpy(float)
    text = [[f"{fmt(c, r[c])}<br>#{int(r[f'rank_{c}'])}" for c in cats] for _, r in rows.iterrows()]
    labels = [f"{'▶ ' if tid == me_id else ''}{name}  (avg #{a:.1f})"
              for tid, name, a in zip(rows["team_id"], rows["team"], rows["avg_rank"])]
    fig = go.Figure(go.Heatmap(
        z=z, x=cats, y=labels, zmin=1, zmax=n, xgap=2, ygap=2, colorscale=[[0, GOOD], [0.5, MID], [1, BAD]],
        showscale=False, text=text, texttemplate="%{text}", textfont=dict(size=11),
        hovertemplate="<b>%{y}</b><br>%{x}: %{text}<extra></extra>"))
    fig.update_layout(height=70 + 42 * len(rows), margin=dict(l=10, r=10, t=30, b=10),
                      xaxis=dict(side="top", showgrid=False), yaxis=dict(showgrid=False))
    return fig


def results_heatmap(r: pd.DataFrame, cats: list[str]):
    """Weeks x categories: W blue, L red, T gray; cell text = my value vs theirs."""
    rows = r.iloc[::-1]
    code = {"W": 1, "T": 0, "L": -1}
    z = [[code[x[f"res_{c}"]] for c in cats] for _, x in rows.iterrows()]
    text = [[f"<b>{x[f'res_{c}']}</b><br>{fmt(c, x[f'me_{c}'])}<br>{fmt(c, x[f'opp_{c}'])}" for c in cats]
            for _, x in rows.iterrows()]
    labels = [f"Week {x['matchup_period']} vs {x['opponent']}  {x['W']}–{x['L']}–{x['T']}"
              + ("" if x["final"] else " (in progress)") for _, x in rows.iterrows()]
    fig = go.Figure(go.Heatmap(z=z, x=cats, y=labels, zmin=-1, zmax=1, xgap=2, ygap=2, showscale=False,
                               colorscale=[[0, BAD], [0.5, MID], [1, GOOD]], text=text, texttemplate="%{text}",
                               textfont=dict(size=11), hovertemplate="<b>%{y}</b><br>%{x}: %{text}<extra></extra>"))
    fig.update_layout(height=70 + 62 * len(rows), margin=dict(l=10, r=10, t=30, b=10),
                      xaxis=dict(side="top", showgrid=False), yaxis=dict(showgrid=False))
    return fig


def ranks_heatmap(wr: pd.DataFrame, cats: list[str], n: int, live: int | None):
    rows = wr.iloc[::-1]
    z = rows[cats].to_numpy(float)
    labels = [f"Week {mp}" + (" (in progress)" if mp == live else "") for mp in rows["matchup_period"]]
    fig = go.Figure(go.Heatmap(z=z, x=cats, y=labels, zmin=1, zmax=n, xgap=2, ygap=2, showscale=False,
                               colorscale=[[0, GOOD], [0.5, MID], [1, BAD]],
                               text=[["–" if np.isnan(v) else f"#{int(v)}" for v in r] for r in z],
                               texttemplate="%{text}", hovertemplate="%{y} · %{x}: %{text}<extra></extra>"))
    fig.update_layout(height=70 + 36 * len(rows), margin=dict(l=10, r=10, t=30, b=10),
                      xaxis=dict(side="top", showgrid=False), yaxis=dict(showgrid=False))
    return fig


days = store.list_snapshots(DATA_ROOT)
if not days:
    st.error("No data yet. Run `python -m fantasy.collect`.")
    st.stop()
day = days[-1]
snap = load(day, snapshot_version(day))
if snap["rosters"].empty or snap["matchups"].empty:
    st.info("The season view needs a real league snapshot with matchups.")
    st.stop()
teams = snap["teams"].set_index("team_id")["name"].to_dict()
with st.sidebar:
    default_me = snap["meta"].get("team_id") or next(iter(teams))
    me_id = st.selectbox("My team", list(teams), index=list(teams).index(default_me), format_func=teams.get)

cats = ordered_categories(snap["meta"]["category_ids"])
table = season_table(snap)
results = week_results(snap, me_id)
record = category_record(results, cats)
live = next((int(r["matchup_period"]) for _, r in results.iterrows() if not r["final"]), None)
n = len(table)
mine = table[table["team_id"] == me_id].iloc[0]

st.title("Season so far")
st.caption(f"ESPN's recorded totals through {day} · FG%/FT% = total makes ÷ attempts · "
           + (f"week {live} is in progress: counted in totals and ranks, not yet in the win–loss record"
              if live else "all weeks final"))

done = results[results["final"]]
k = st.columns(4)
k[0].metric("Matchup record", f"{int((done['W'] > done['L']).sum())}–{int((done['W'] < done['L']).sum())}–"
            f"{int((done['W'] == done['L']).sum())}", help="Completed weeks: won–lost–tied")
k[1].metric("Category record", f"{record['W'].sum()}–{record['L'].sum()}–{record['T'].sum()}",
            help="Every category of every completed week")
k[2].metric("Average category rank", f"#{mine['avg_rank']:.1f} of {n}",
            help=f"League position by average rank: #{int(table.index[table['team_id'] == me_id][0]) + 1}")
ranks = {c: int(mine[f"rank_{c}"]) for c in cats}
best = sorted(cats, key=ranks.get)
k[3].metric("Best / worst", f"{best[0]} #{ranks[best[0]]} · {best[-1]} #{ranks[best[-1]]}")

st.subheader("League table")
st.caption("Season totals for every team, coloured by league rank in each category (blue = top, red = bottom). "
           "Teams are ordered by their average rank across categories. TO: fewest is #1.")
st.plotly_chart(rank_heatmap(table, cats, me_id, n), width="stretch", config={"displayModeBar": False}, key="table")

st.subheader("My category record")
wlt = record.set_index("category")
st.dataframe(pd.DataFrame([[f"{wlt.at[c, 'W']}–{wlt.at[c, 'L']}–{wlt.at[c, 'T']}" for c in cats],
                           [f"#{ranks[c]}" for c in cats]], columns=cats, index=["W–L–T", "Season rank"]),
             width="stretch")
st.plotly_chart(results_heatmap(results, cats), width="stretch", config={"displayModeBar": False}, key="results")
st.caption("Each week's result per category: your total on top, your opponent's below.")

st.subheader("Weekly trend")
tab_rank, tab_totals = st.tabs(["Rank per week", "Totals per week"])
with tab_rank:
    st.plotly_chart(ranks_heatmap(weekly_ranks(snap, me_id), cats, n, live), width="stretch",
                    config={"displayModeBar": False}, key="ranks")
    st.caption("Your league rank in each category, week by week.")
with tab_totals:
    w = weekly_scores(snap)
    cols = st.columns(3)
    for i, c in enumerate(cats):
        league_med = w.groupby("matchup_period")[c].median()
        mine_w = w[w["team_id"] == me_id].set_index("matchup_period")[c]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=league_med.index, y=league_med.values, name="League median", mode="lines+markers",
                                 line=dict(color=OTHER, width=2, dash="dot"), marker=dict(size=8)))
        fig.add_trace(go.Scatter(x=mine_w.index, y=mine_w.values, name=teams[me_id], mode="lines+markers",
                                 line=dict(color=ME, width=2), marker=dict(size=8)))
        fig.update_layout(title=dict(text=f"<b>{c}</b>", font=dict(size=14)), height=220,
                          margin=dict(l=10, r=10, t=40, b=20), hovermode="x unified",
                          legend=dict(orientation="h", y=-0.25), xaxis=dict(title=None, dtick=1),
                          yaxis=dict(tickformat=".1%" if c.endswith("%") else ",.0f"))
        cols[i % 3].plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=f"trend-{c}")
    st.caption("Your weekly totals (blue) against the league median (dotted)."
               + (f" Week {live} is partial." if live else ""))
