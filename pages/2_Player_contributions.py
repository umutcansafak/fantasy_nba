"""Every team's players (and free agents): how each one helps or hurts every category this season."""
import streamlit as st

from fantasy import store
from fantasy.app_data import DATA_ROOT, load, simulator, snapshot_version
from fantasy.contrib_view import show_contributions, team_heatmap
from fantasy.players import free_agent_contributions, ordered_categories, roster_contributions, team_overview
from fantasy.projection import next_matchup_period

st.set_page_config(page_title="Player contributions", layout="wide")

FREE_AGENTS = -1


@st.cache_data(show_spinner="Scoring every roster…", max_entries=4)
def overview(day, version):
    return team_overview(simulator(day, version, False, 1000))


@st.cache_data(show_spinner="Scoring players…", max_entries=24)
def team_contrib(day, version, team_id, mp):
    sim = simulator(day, version, False, 1000)
    if team_id == FREE_AGENTS:
        return free_agent_contributions(sim, mp)
    return roster_contributions(sim, team_id, mp)


days = store.list_snapshots(DATA_ROOT)
if not days:
    st.error("No data yet. Run `python -m fantasy.collect`.")
    st.stop()
day = days[-1]
version = snapshot_version(day)
snap = load(day, version)
if snap["rosters"].empty:
    st.info("Player contributions need a real league snapshot (this one is the public player pool).")
    st.stop()
sim = simulator(day, version, False, 1000)
teams = snap["teams"].set_index("team_id")["name"].to_dict()
me_id = snap["meta"].get("team_id")
mp = snap["meta"].get("current_matchup_period") or next_matchup_period(snap["meta"])
if not sim.periods_for(mp):
    mp = next_matchup_period(snap["meta"])

st.title("Player contributions")
st.caption(f"This season's games only · data from {day} · week view = week {mp}")

cats = ordered_categories(sim.cat_ids)
with st.expander("Teams at a glance", expanded=True):
    st.caption("Each team's roster summed (season so far, per game, every rostered player). "
               "Blue = the roster is strong in that category, red = weak.")
    st.plotly_chart(team_heatmap(overview(day, version), cats, me_id), width="stretch",
                    config={"displayModeBar": False}, key="overview")

options = [t for t in teams if t != me_id] + [FREE_AGENTS] + ([me_id] if me_id in teams else [])
pick = st.selectbox("Team", options, format_func=lambda t: "Free agents" if t == FREE_AGENTS
                    else teams[t] + (" (you)" if t == me_id else ""))
contrib = team_contrib(day, version, pick, mp)

if pick == FREE_AGENTS:
    c1, c2 = st.columns([1, 2])
    n = c1.slider("Players shown", 10, 60, 25, step=5)
    min_gp = c2.slider("Minimum games played", 1, int(max(contrib["gp"].max(), 1)), 1)
    contrib = contrib[contrib["gp"] >= min_gp].head(n)
    show_contributions(sim, contrib, weekly_note="the free-agent list", key="fa", summary=False)
else:
    show_contributions(sim, contrib, weekly_note="this roster", key=f"team-{pick}")
