"""Streamlit view of per-category player contributions, shared by the dashboard and the Players page."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .config import DEFAULT_CATEGORY_WEIGHTS
from .players import ordered_categories

# Diverging pair: blue = helps, red = hurts, light gray midpoint.
WIN, LOSS, MID = "#2a78d6", "#e34948", "#e6e5e0"


def category_weights(cats: list[str]) -> dict[str, float]:
    """Sidebar weights per category, kept across pages; 0 = punted (ignored in totals)."""
    with st.sidebar:
        with st.expander("Category weights", expanded=True):
            st.caption("Used for player totals and fit scores on the contribution pages. 0 = punt.")
            weights = {}
            for c in cats:
                k = f"w_{c}"
                if k not in st.session_state:  # widget state is dropped on page switch; restore it
                    st.session_state[k] = st.session_state.get(f"_keep_{k}", DEFAULT_CATEGORY_WEIGHTS.get(c, 1.0))
                weights[c] = st.number_input(c, 0.0, 3.0, step=0.25, key=k)
                st.session_state[f"_keep_{k}"] = weights[c]
    return weights


def col_label(c: str, w: float) -> str:
    return f"{c} (punted)" if w == 0 else c if w == 1 else f"{c} ×{w:g}"


def weighted_total(df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    return sum(df[c] * w for c, w in weights.items()).where(df[list(weights)].notna().any(axis=1))


def contribution_heatmap(d, cats, weekly, weights=None):
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
        z=z, x=[col_label(c, (weights or {}).get(c, 1.0)) for c in cats], y=labels, zmid=0, zmin=-lim, zmax=lim,
        xgap=2, ygap=2,
        colorscale=[[0, LOSS], [0.5, MID], [1, WIN]], text=[["–" if np.isnan(v) else f"{v:+.1f}" for v in r] for r in z],
        texttemplate="%{text}", textfont=dict(size=12), hovertext=hover, hovertemplate="%{hovertext}<extra></extra>",
        colorbar=dict(title=dict(text="helps →" if not weekly else "impact", side="right"), thickness=10)))
    fig.update_layout(height=60 + 30 * len(rows), margin=dict(l=10, r=10, t=30, b=10),
                      xaxis=dict(side="top", showgrid=False), yaxis=dict(showgrid=False))
    return fig


def team_heatmap(teams: pd.DataFrame, cats: list[str], highlight: int | None = None, weights=None):
    """Teams x categories: summed season z-scores of each roster; total uses the weights."""
    weights = weights or {c: 1.0 for c in cats}
    teams = teams.assign(total=weighted_total(teams, weights)).sort_values("total", ascending=False)
    rows = teams.iloc[::-1]
    z = rows[cats].to_numpy(dtype=float)
    lim = max(1.0, float(np.nanmax(abs(z))))
    labels = [f"{'▶ ' if t == highlight else ''}{n}  ({v:+.1f})" for t, n, v in zip(rows["team_id"], rows["team"], rows["total"])]
    fig = go.Figure(go.Heatmap(
        z=z, x=[col_label(c, weights[c]) for c in cats], y=labels, zmid=0, zmin=-lim, zmax=lim, xgap=2, ygap=2,
        colorscale=[[0, LOSS], [0.5, MID], [1, WIN]], text=[[f"{v:+.1f}" for v in r] for r in z],
        texttemplate="%{text}", textfont=dict(size=12),
        hovertemplate="<b>%{y}</b><br>%{x}: %{z:+.1f}<extra></extra>",
        colorbar=dict(title=dict(text="helps →", side="right"), thickness=10)))
    fig.update_layout(height=60 + 30 * len(rows), margin=dict(l=10, r=10, t=30, b=10),
                      xaxis=dict(side="top", showgrid=False), yaxis=dict(showgrid=False))
    return fig


def show_contributions(sim, contrib: pd.DataFrame, weekly_note: str, key: str, summary: bool = True,
                       weights: dict[str, float] | None = None):
    """Season / week toggle, heatmap, and the per-player helps/hurts list."""
    mode = st.segmented_control("Show", ["Season so far", "This week"], default="Season so far", key=f"{key}_mode",
                                help="Season so far: his actual per-game averages this season vs the top 200 scorers. "
                                     "This week: the games he already played in the selected week, plus his season "
                                     "averages times the games he is expected to play in the rest of it.")
    weekly = mode == "This week"
    cat_names = ordered_categories(sim.cat_ids)
    weights = weights or category_weights(cat_names)
    shown = contrib.copy()
    if weekly:
        shown[cat_names] = shown[[f"wk_{c}" for c in cat_names]].to_numpy()
    shown["total"] = weighted_total(shown, weights)
    shown = shown.sort_values("total", ascending=False, na_position="last")
    if weekly:
        st.caption(f"This week = z-scores of the games already played this week (every game by a player now on "
                   f"{weekly_note}, even on the bench) + season averages × expected games for the days left.")
    st.caption("Only this season's games, no projections. Each cell is a z-score against this season's top 200 "
               "scorers: +1 means one standard deviation better than average in that category, negative means he "
               "pulls you down. TO is flipped (more turnovers = red). FG%/FT% account for volume: a 45% shooter on "
               "20 attempts hurts more than one on 5. After each name: his total and games played (GP); "
               "small GP = noisy.")
    punted = [c for c in cat_names if weights[c] == 0]
    if punted:
        st.caption(f"Punting {', '.join(punted)}: still shown, but left out of each player's total.")
    st.plotly_chart(contribution_heatmap(shown, cat_names, weekly, weights), width="stretch",
                    config={"displayModeBar": False}, key=key)

    if summary:
        team = shown[[c for c in cat_names if weights[c] > 0]].sum()
        st.markdown("**Roster in a nutshell** " + ("(this week)" if weekly else "(season so far, per game, all players)")
                    + ": strongest " + ", ".join(f"{c} {team[c]:+.1f}" for c in team.sort_values(ascending=False).index[:3])
                    + " · weakest " + ", ".join(f"{c} {team[c]:+.1f}" for c in team.sort_values().index[:3]))
    lines = []
    for _, r in shown.iterrows():
        vals = r[[c for c in cat_names if weights[c] > 0]].astype(float)
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
    return weights


def fit_heatmap(fit: pd.DataFrame, cats: list[str], weights: dict[str, float]):
    """Candidates x categories: change in my weekly win rate (percentage points), blue up, red down."""
    rows = fit.iloc[::-1]
    z = rows[cats].to_numpy(dtype=float) * 100
    z = np.where(abs(z) < 0.5, 0.0, z)
    lim = max(5.0, float(np.nanmax(abs(z)))) if len(z) else 5.0
    labels = [f"{n}  ({v:+.2f})" + (f" · {s.replace('_', ' ').lower()}" if s != "ACTIVE" else "")
              for n, v, s in zip(rows["name"], rows["fit"], rows["injury_status"])]
    fig = go.Figure(go.Heatmap(
        z=z, x=[col_label(c, weights[c]) for c in cats], y=labels, zmid=0, zmin=-lim, zmax=lim, xgap=2, ygap=2,
        colorscale=[[0, LOSS], [0.5, MID], [1, WIN]], text=[[f"{v:+.0f}" for v in r] for r in z],
        texttemplate="%{text}", textfont=dict(size=12),
        hovertemplate="<b>%{y}</b><br>%{x}: %{z:+.1f} pts of weekly win rate<extra></extra>",
        colorbar=dict(title=dict(text="pts win rate", side="right"), thickness=10)))
    fig.update_layout(height=60 + 30 * len(rows), margin=dict(l=10, r=10, t=30, b=10),
                      xaxis=dict(side="top", showgrid=False), yaxis=dict(showgrid=False))
    return fig


def show_fit(model, sim, me_id: int, candidates: pd.DataFrame, mine: pd.DataFrame, weights: dict[str, float], key: str):
    """'If he were on my team': swap each candidate in for one of my players and show my category changes."""
    cats = ordered_categories(sim.cat_ids)
    mine = mine.assign(total=weighted_total(mine, weights))
    order = mine.sort_values("total", na_position="first")["player_id"].tolist()
    played = mine[mine["gp"] > 0].sort_values("total")
    default = int(played["player_id"].iloc[0]) if len(played) else order[0]
    names = mine.set_index("player_id")["name"].to_dict()
    drop = st.selectbox("…in place of my player", order, index=order.index(default), key=f"{key}_drop",
                        format_func=lambda p: names[p] + ("  (lowest weighted total)" if p == default else ""))
    fit = model.fit(me_id, candidates["player_id"].tolist(), drop)
    if fit.empty:
        st.info("No players to compare.")
        return
    fit = fit.merge(candidates[["player_id", "name", "injury_status"]], on="player_id")
    fit["fit"] = sum(fit[c] * weights[c] for c in cats)
    fit = fit.sort_values("fit", ascending=False)
    base = model.baseline(me_id)
    st.caption("Change in **your** chance of winning each category in an average week against the rest of the league "
               "(percentage points), rest of season, this season's stats only. The number after each name is the "
               "weighted sum: extra categories won per week. Gains shrink where you already win "
               f"({', '.join(f'{c} {base[c]:.0%}' for c in cats)} now), so the same player can be worth more or less "
               "to you than his own team.")
    st.plotly_chart(fit_heatmap(fit, cats, weights), width="stretch", config={"displayModeBar": False},
                    key=f"{key}_fit")
