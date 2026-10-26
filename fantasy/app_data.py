"""Cached data loaders shared by the Streamlit pages (one cache across pages)."""
import os
from pathlib import Path

import streamlit as st

from . import store
from .demo import demo_league
from .projection import Simulator

DATA_ROOT = Path(os.getenv("FANTASY_DATA", store.DATA_DIR))


def snapshot_version(day: str) -> float:
    """meta.json mtime: changes when a day's snapshot is re-collected, busting the caches."""
    return (store.snapshot_dir(day, DATA_ROOT) / "meta.json").stat().st_mtime


@st.cache_data(show_spinner=False, max_entries=5)
def load(day, version=None):
    return store.load_snapshot(day, root=DATA_ROOT)


@st.cache_resource(show_spinner="Building player pools…", max_entries=3)
def simulator(day, version, demo, n_sims):
    snap = load(day, version)
    return Simulator(demo_league(snap) if demo else snap, n_sims=n_sims)
