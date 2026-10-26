"""Trade ideas: rest-of-season, win-win swaps with every other team."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fantasy import store
from fantasy.app_data import DATA_ROOT, load, simulator, snapshot_version
from fantasy.trades import suggest

# Diverging pair of the reference palette: gains blue, losses red; text stays neutral.
GAIN, LOSS, ME, OTHER = "#2a78d6", "#e34948", "#2a78d6", "#8a8984"

st.set_page_config(page_title="Trade ideas", layout="wide")


@st.cache_data(show_spinner="Searching every 1-for-1 and 2-for-2 trade in the league (about 20 seconds)…",
               max_entries=4)
def trade_ideas(day, version, me_id, include_injured):
    sim = simulator(day, version, False, 1000)
    return suggest(sim, me_id, include_injured=include_injured)


def delta_chart(cats: dict, title: str):
    """Per-category change in win rate vs the average opponent (percentage points)."""
    names = list(cats)[::-1]
    vals = [cats[n] * 100 for n in names]
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h", marker=dict(color=[GAIN if v >= 0 else LOSS for v in vals], line=dict(width=0)),
        text=[f"{v:+.0f}" if abs(v) >= 0.5 else "0" for v in vals], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:+.1f} pts win rate<extra></extra>"))
    lim = max(5, max(abs(v) for v in vals) * 1.3)
    fig.update_layout(title=dict(text=title, font=dict(size=14)), height=330, bargap=0.35,
                      margin=dict(l=10, r=30, t=40, b=30),
                      xaxis=dict(range=[-lim, lim], zeroline=True, zerolinewidth=1, ticksuffix=" pts", showgrid=True),
                      yaxis=dict(showgrid=False))
    return fig


def strength_chart(strengths: pd.DataFrame, me_id: int):
    d = strengths.sort_values("strength")
    colors = [ME if t == me_id else OTHER for t in d["team_id"]]
    fig = go.Figure(go.Bar(x=d["strength"], y=d["team"], orientation="h", marker=dict(color=colors, line=dict(width=0)),
                           text=d["strength"].map("{:.2f}".format), textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}: %{x:.2f} categories per week<extra></extra>"))
    fig.update_layout(height=34 * len(d) + 40, margin=dict(l=10, r=40, t=10, b=30), bargap=0.3,
                      xaxis=dict(title="expected categories won per week vs average opponent", range=[0, 9]),
                      yaxis=dict(showgrid=False))
    return fig


days = store.list_snapshots(DATA_ROOT)
if not days:
    st.error("No data yet. Run `python -m fantasy.collect`.")
    st.stop()
day = days[-1]
version = snapshot_version(day)
snap = load(day, version)
if snap["rosters"].empty:
    st.info("Trade ideas need a real league snapshot (this one is the public player pool).")
    st.stop()
teams = snap["teams"].set_index("team_id")["name"].to_dict()

with st.sidebar:
    st.header("Trade settings")
    default_me = snap["meta"].get("team_id") or next(iter(teams))
    me_id = st.selectbox("My team", list(teams), index=list(teams).index(default_me), format_func=teams.get)
    include_injured = st.toggle("Include injured players", value=False,
                                help="OUT/IR players are valued at half their games for the rest of the season.")

ideas, strengths = trade_ideas(day, version, me_id, include_injured)

st.title("Trade ideas")
st.caption(f"Rest of the season · data from {day} · playoff weeks count double · recalculated when new data "
           "arrives, so a good trade stays listed as long as it remains one.")

with st.expander("How trades are scored", expanded=False):
    st.markdown(
        "- **Gain** = change in expected categories won **per week** against the average opponent, "
        "for the rest of the season. +0.25 means about one extra category every four weeks.\n"
        "- Every 1-for-1 and 2-for-2 swap with every team is checked, then the best are re-scored with real "
        "daily lineups (positions and slot limits).\n"
        "- **Fairness (0-100)** averages two things: how evenly the gain is split, and whether the players "
        "*look* equally valuable by standard 9-cat rankings, which is how most managers judge an offer.\n"
        "- Win-win works because categories matter differently to each team: your extra scoring is worth "
        "little to you (you already win PTS) but a lot to a team that loses it.")

with st.expander("League strength today", expanded=False):
    st.plotly_chart(strength_chart(strengths, me_id), width="stretch", config={"displayModeBar": False})

# filters + table, full width so every column fits
if ideas.empty:
    st.info("No trades found that improve your team.")
    st.stop()
c1, c2, c3 = st.columns(3)
sizes = c1.multiselect("Trade size", ["1-for-1", "2-for-2"], default=["1-for-1", "2-for-2"])
min_fair = c2.slider("Minimum fairness", 0, 100, 50, step=5)
sort = c3.selectbox("Sort by", ["Best for me", "Most mutual", "Fairest"])
partners = st.multiselect("Trade partners", sorted(ideas["partner"].unique()), placeholder="All teams")
win_win = st.toggle("Only win-win (both teams improve)", value=True)

view = ideas[ideas["size"].isin(sizes) & (ideas["fairness"] >= min_fair)]
if win_win:
    view = view[view["win_win"]]
if partners:
    view = view[view["partner"].isin(partners)]
key = {"Best for me": view["my_gain"], "Most mutual": view[["my_gain", "their_gain"]].min(axis=1),
       "Fairest": view["fairness"]}[sort]
view = view.loc[key.sort_values(ascending=False).index].reset_index(drop=True)

st.caption(f"{len(view)} of {len(ideas)} checked trades match. Select a row for the category breakdown.")
table = view[["partner", "you_give", "you_get", "my_gain", "their_gain", "fairness"]]
picked = st.dataframe(
    table, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row", key="trades",
    column_config={
        "partner": "Partner", "you_give": "You give", "you_get": "You get",
        "my_gain": st.column_config.NumberColumn("Your gain", format="%+.2f", help="categories per week"),
        "their_gain": st.column_config.NumberColumn("Their gain", format="%+.2f", help="categories per week"),
        "fairness": st.column_config.ProgressColumn("Fairness", min_value=0, max_value=100, format="%.0f"),
    })

rows = picked.selection.rows if picked and picked.selection else []
if not view.empty:
    t = view.iloc[rows[0] if rows else 0]
    st.divider()
    st.subheader(f"{t['you_give']}  ⇄  {t['you_get']}  ·  with {t['partner']}")
    def summary(cats):
        ups = [f"{k} {v * 100:+.0f}" for k, v in sorted(cats.items(), key=lambda kv: -kv[1]) if v * 100 >= 0.5]
        downs = [f"{k} {v * 100:+.0f}" for k, v in sorted(cats.items(), key=lambda kv: kv[1]) if v * 100 <= -0.5]
        return ("**Wins:** " + (", ".join(ups) or "none") + "  \n**Loses:** " + (", ".join(downs) or "none"))
    k = st.columns(4)
    k[0].metric("Your gain", f"{t['my_gain']:+.2f} cats/week")
    k[1].metric(f"{t['partner']} gain", f"{t['their_gain']:+.2f} cats/week")
    k[2].metric("Fairness", f"{t['fairness']:.0f} / 100")
    k[3].metric("Looks even (by rankings)", f"{t['value_balance']:.0%}")
    a, b = st.columns(2)
    with a:
        st.plotly_chart(delta_chart(t["my_cats"], "You: change in weekly win rate per category"),
                        width="stretch", config={"displayModeBar": False}, key="d-me")
        st.markdown(summary(t["my_cats"]))
    with b:
        st.plotly_chart(delta_chart(t["their_cats"], f"{t['partner']}: change in weekly win rate per category"),
                        width="stretch", config={"displayModeBar": False}, key="d-them")
        st.markdown(summary(t["their_cats"]))
    st.caption("Points = percentage-point change in how often that category is won in an average week. "
               "Projections lean on last season until this season has more games.")
