"""End-to-end check without a league: `python -m tests.smoke_test [data_root]`.

Snake-drafts two fake 13-man teams from a public-pool snapshot (collect --public)
and projects a matchup week between them.
"""
import sys
import time
from pathlib import Path

import pandas as pd

from fantasy import store
from fantasy.demo import demo_league
from fantasy.lineup import fill_lineup
from fantasy.projection import Simulator, compare, next_matchup_period


def test_lineup():
    # PG-only player and a G-eligible player competing for PG + G slots: both must start.
    slots = [0, 5]
    picked = fill_lineup([(1, 9.0, [0, 5]), (2, 5.0, [0]), (3, 1.0, [1])], slots)
    assert picked == {1, 2}, picked
    # More candidates than slots: lowest value sits.
    picked = fill_lineup([(1, 3.0, [11]), (2, 2.0, [11]), (3, 1.0, [11])], [11, 11])
    assert picked == {1, 2}, picked


def main(root):
    test_lineup()
    snap = demo_league(store.load_snapshot(root=root))
    t0 = time.time()
    sim = Simulator(snap, n_sims=5000)
    mp = next_matchup_period(snap["meta"]) + 1  # a full future week
    periods = sim.periods_for(mp)
    a = sim.project_team(1, sim.roster_of(1), periods, 0)
    b = sim.project_team(2, sim.roster_of(2), periods, 1)
    table, overall = compare(a, b, snap["meta"]["category_ids"])
    print(f"matchup period {mp}: scoring periods {periods[0]}-{periods[-1]}  ({time.time() - t0:.1f}s)")
    pd.set_option("display.width", 160)
    print(table.round(3).to_string(index=False))
    print(overall)
    print(a.starts.sort_values("starts", ascending=False).to_string(index=False))
    assert len(table) == 9
    assert table["p_win"].between(0, 1).all()
    assert (table["me_lo"] <= table["me_hi"]).all()
    assert a.starts["starts"].sum() <= len(sim.slots) * len(periods)
    print("OK")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else store.DATA_DIR)
