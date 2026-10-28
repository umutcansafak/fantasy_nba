"""Streamlit view of per-category player contributions, shared by the dashboard and the Players page."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .players import ordered_categories

# Diverging pair: blue = helps, red = hurts, light gray midpoint.
WIN, LOSS, MID = "#2a78d6", "#e34948", "#e6e5e0"


def contribution_heatmap(d, cats, weekly):
    """Players x categories, diverging around 0: blue helps, red hurts, gray is about average."""
    rows = d.iloc[::-1]  # best player on top
    z = rows[cats].to_numpy(dtype=float)
    z = np.where(abs(z) < 0.05, 0.0, z)  # no "-0.0"
    lim = max(1.0, float(np.nanmax(abs(z)))) if weekly and not np.isnan(z).all() else 3.0
    games = ([f"{pg} played + {eg:.1f} left" for pg, eg in zip(rows["played_games"], rows["exp_games"])] if weekly
             else [f"{g} GP" for g in rows["gp"]])
    labels = [f"{n}  ({'no games' if pd.isna(t) else f'{t:+.1f}'}, {g})"
              + (f" · {s.replace('_', ' ').lower()}" if s != "ACTIVE" else "")
              for n, t, g, s in zip(rows["name"], rows["total"], games, rows["injury_status"])]
    def raw(r, c):
        v = r[f"raw_{c}"]
        if pd.isna(v):
            return "–"
        if c.endswith("%"):
            return f"{v:.1%} on {r[f'att_{c}']:.1f} att"
        return f"{v:.1f}"
    hover = [[f"<b>{r['name']}</b><br>{c}: {raw(r, c)} per game<br>"
              + (f"{r[c]:+.1f} this week ({r['played_games']} played + {r['exp_games']:.1f} expected)" if weekly
                 else f"{r[c]:+.1f} vs average")
              for c in cats] for _, r in rows.iterrows()]
    fig = go.Figure(go.Heatmap(
        z=z, x=cats, y=labels, zmid=0, zmin=-lim, zmax=lim, xgap=2, ygap=2,
        colorscale=[[0, LOSS], [0.5, MID], [1, WIN]], text=[["–" if np.isnan(v) else f"{v:+.1f}" for v in r] for r in z],
        texttemplate="%{text}", textfont=dict(size=12), hovertext=hover, hovertemplate="%{hovertext}<extra></extra>",
        colorbar=dict(title=dict(text="helps →" if not weekly else "impact", side="right"), thickness=10)))
    fig.update_layout(height=60 + 30 * len(rows), margin=dict(l=10, r=10, t=30, b=10),
                      xaxis=dict(side="top", showgrid=False), yaxis=dict(showgrid=False))
    return fig


def team_heatmap(teams: pd.DataFrame, cats: list[str], highlight: int | None = None):
    """Teams x categories: summed season z-scores of each roster."""
    rows = teams.iloc[::-1]
    z = rows[cats].to_numpy(dtype=float)
    lim = max(1.0, float(np.nanmax(abs(z))))
    labels = [f"{'▶ ' if t == highlight else ''}{n}  ({v:+.1f})" for t, n, v in zip(rows["team_id"], rows["team"], rows["total"])]
    fig = go.Figure(go.Heatmap(
        z=z, x=cats, y=labels, zmid=0, zmin=-lim, zmax=lim, xgap=2, ygap=2,
        colorscale=[[0, LOSS], [0.5, MID], [1, WIN]], text=[[f"{v:+.1f}" for v in r] for r in z],
        texttemplate="%{text}", textfont=dict(size=12),
        hovertemplate="<b>%{y}</b><br>%{x}: %{z:+.1f}<extra></extra>",
        colorbar=dict(title=dict(text="helps →", side="right"), thickness=10)))
    fig.update_layout(height=60 + 30 * len(rows), margin=dict(l=10, r=10, t=30, b=10),
                      xaxis=dict(side="top", showgrid=False), yaxis=dict(showgrid=False))
    return fig


def show_contributions(sim, contrib: pd.DataFrame, weekly_note: str, key: str, summary: bool = True):
    """Season / week toggle, heatmap, and the per-player helps/hurts list."""
    mode = st.segmented_control("Show", ["Season so far", "This week"], default="Season so far", key=f"{key}_mode",
                                help="Season so far: his actual per-game averages this season vs the top 200 scorers. "
                                     "This week: the games he already played in the selected week, plus his season "
                                     "averages times the games he is expected to play in the rest of it.")
    weekly = mode == "This week"
    cat_names = ordered_categories(sim.cat_ids)
    shown = contrib.copy()
    if weekly:
        shown[cat_names] = shown[[f"wk_{c}" for c in cat_names]].to_numpy()
        shown["total"] = shown["wk_total"]
        shown = shown.sort_values("total", ascending=False, na_position="last")
        st.caption(f"This week = z-scores of the games already played this week (every game by a player now on "
                   f"{weekly_note}, even on the bench) + season averages × expected games for the days left.")
    st.caption("Only this season's games, no projections. Each cell is a z-score against this season's top 200 "
               "scorers: +1 means one standard deviation better than average in that category, negative means he "
               "pulls you down. TO is flipped (more turnovers = red). FG%/FT% account for volume: a 45% shooter on "
               "20 attempts hurts more than one on 5. After each name: his total and games played (GP); "
               "small GP = noisy.")
    st.plotly_chart(contribution_heatmap(shown, cat_names, weekly), width="stretch",
                    config={"displayModeBar": False}, key=key)

    if summary:
        team = shown[cat_names].sum()
        st.markdown("**Roster in a nutshell** " + ("(this week)" if weekly else "(season so far, per game, all players)")
                    + ": strongest " + ", ".join(f"{c} {team[c]:+.1f}" for c in team.sort_values(ascending=False).index[:3])
                    + " · weakest " + ", ".join(f"{c} {team[c]:+.1f}" for c in team.sort_values().index[:3]))
    lines = []
    for _, r in shown.iterrows():
        vals = r[cat_names].astype(float)
        if vals.isna().all():
            lines.append(f"- **{r['name']}**: no games this season yet")
            continue
        ups = [f"{c} {vals[c]:+.1f}" for c in vals.sort_values(ascending=False).index[:3] if vals[c] >= 0.5]
        downs = [f"{c} {vals[c]:+.1f}" for c in vals.sort_values().index[:2] if vals[c] <= -0.5]
        status = "" if r["injury_status"] == "ACTIVE" else f" _({r['injury_status'].replace('_', ' ').lower()})_"
        lines.append(f"- **{r['name']}**{status}: helps {', '.join(ups) or 'nothing much'}"
                     + (f" · hurts {', '.join(downs)}" if downs else ""))
    with st.expander("Player by player"):
        st.markdown("\n".join(lines))
