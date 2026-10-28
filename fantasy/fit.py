"""How a player would fit *my* roster: change in my weekly category win rates if he replaced one of my players.

Built on the trade model's weekly normal approximation (rest-of-season games per week, my lineup start
factor, every other team as the opponent pool), but with per-game means from this season's actual
averages only. Because win probability saturates, a player adds little in a category I already win
and a lot in one I'm close in, which a plain z-score can't show.
"""
import numpy as np
import pandas as pd

from .config import CATEGORIES
from .players import season_rates
from .projection import Simulator
from .trades import STATS, TradeModel


class FitModel(TradeModel):
    def _per_game_moments(self):
        """Trade-model moments, with means replaced by this season's per-game averages where available."""
        mean, var, cov = super()._per_game_moments()
        season = season_rates(self.snap["players"])
        for pid in season.index:
            if int(pid) in mean:
                mean[int(pid)] = season.loc[pid, STATS].to_numpy(float)
        return mean, var, cov

    def fit(self, me: int, candidates: list[int], drop: int) -> pd.DataFrame:
        """Per candidate: change in my mean win probability per category (vs every other team) after
        swapping `drop` for him. Columns: player_id, one per category name."""
        names = [c["name"] for c in self.cats]
        others = [t for t in self.teams if t != me]
        base = self._base()
        o_mu = np.stack([base[o][0] for o in others])
        o_sd = np.stack([base[o][1] for o in others])
        f = self.start_factor[me]
        roster = self.roster[me]
        m0 = self.team_moments(roster, f)
        p0 = self.win_probs(*self.cat_dists(*m0), o_mu, o_sd).mean(0)
        d_out = self.contrib(drop, f) if drop in self.mean else (0, 0, 0)
        rows = []
        for pid in candidates:
            if pid not in self.mean or pid in roster:
                continue
            c_in = self.contrib(pid, f)
            m1 = tuple(m0[i] - d_out[i] + c_in[i] for i in range(3))
            p1 = self.win_probs(*self.cat_dists(*m1), o_mu, o_sd).mean(0)
            rows.append({"player_id": pid, **dict(zip(names, p1 - p0))})
        return pd.DataFrame(rows, columns=["player_id", *names])

    def baseline(self, me: int) -> dict[str, float]:
        """My current mean win probability per category vs every other team."""
        others = [t for t in self.teams if t != me]
        base = self._base()
        p = self.win_probs(*base[me], np.stack([base[o][0] for o in others]),
                           np.stack([base[o][1] for o in others])).mean(0)
        return {c["name"]: float(v) for c, v in zip(self.cats, p)}
