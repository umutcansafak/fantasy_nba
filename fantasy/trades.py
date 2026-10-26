"""Rest-of-season trade finder for H2H categories.

Team strength = expected categories won per week against the average league
opponent. Each team's weekly category totals are modelled as normal
distributions built from players' per-game means/variances x expected starts
per week over the remaining schedule (playoff weeks weighted more). A trade is
win-win when both teams' strength goes up; because category needs differ, the
same swap can help both sides.

Search is exhaustive over 1-for-1 and 2-for-2 swaps with an analytic model; the
best candidates are then re-scored with real daily lineup filling (positions,
slot limits) to catch trades that look good on paper but can't be started.
"""
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from .config import CATEGORIES, RAW_STATS
from .projection import Simulator

STATS = ["PTS", "REB", "AST", "STL", "BLK", "3PM", "TO", "FGM", "FGA", "FTM", "FTA"]
IDX = {s: i for i, s in enumerate(STATS)}
PLAYOFF_WEIGHT = 2.0
VERIFY_DAYS = 28
OUT_AVAILABILITY = 0.5   # rest-of-season share for OUT players without a return date
DTD_AVAILABILITY = 0.9


def _phi(x):
    """Standard normal CDF (Abramowitz-Stegun 7.1.26, |error| < 1.5e-7), vectorized."""
    z = np.abs(x) / np.sqrt(2.0)
    t = 1.0 / (1.0 + 0.3275911 * z)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    erf = 1.0 - poly * np.exp(-z * z)
    return 0.5 * (1.0 + np.sign(x) * erf)


@dataclass
class Trade:
    partner: int
    give: tuple
    get: tuple
    my_gain: float
    their_gain: float
    my_cats: dict
    their_cats: dict


class TradeModel:
    def __init__(self, sim: Simulator):
        self.sim = sim
        self.snap = sim.snap
        meta = self.snap["meta"]
        self.cat_ids = sim.cat_ids
        self.cats = [CATEGORIES[c] for c in self.cat_ids]
        rosters = self.snap["rosters"]
        self.roster = {int(t): g["player_id"].astype(int).tolist() for t, g in rosters.groupby("team_id")}
        self.slot = dict(zip(rosters["player_id"].astype(int), rosters["lineup_slot"]))
        self.teams = sorted(self.roster)
        self.weekly_games = self._weekly_games(meta)
        self.mean, self.var, self.cov = self._per_game_moments()
        self.start_factor = {t: self._start_factor(self.roster[t]) for t in self.teams}

    # ------------------------------------------------------------ inputs

    def _weekly_games(self, meta) -> dict[int, float]:
        """Weighted average NBA games per fantasy week over the rest of the season, per player."""
        mps = {int(k): v for k, v in meta["matchup_periods"].items()}
        cur = meta.get("current_matchup_period") or min(mps)
        regular = meta.get("regular_matchups") or sum(1 for d in mps.values() if len(d) <= 7)
        open_day = self.sim.first_open_period()
        pro = self.sim.pro
        team_games, weights = {}, 0.0
        for mp, days in mps.items():
            days = [d for d in days if d >= open_day] if mp == cur else days
            if mp < cur or not days:
                continue
            w = PLAYOFF_WEIGHT if mp > regular else 1.0
            # a two-week playoff round counts as two weeks of games
            n_weeks = max(1.0, len(days) / 7.0)
            games = pro[pro["scoring_period"].isin(days)].groupby("pro_team_id").size() / n_weeks
            for team, g in games.items():
                team_games[team] = team_games.get(team, 0.0) + w * n_weeks * g
            weights += w * n_weeks
        players = self.sim.players
        out = {}
        for pid in players.index:
            g = team_games.get(players.at[pid, "pro_team_id"], 0.0) / max(weights, 1e-9)
            out[int(pid)] = g * self._availability(pid)
        return out

    def _availability(self, pid) -> float:
        p = self.sim.players.loc[pid]
        if self.slot.get(int(pid)) == 13 or p["injury_status"] in ("OUT", "SUSPENSION", "INJURY_RESERVE"):
            return OUT_AVAILABILITY
        return DTD_AVAILABILITY if p["injury_status"] == "DAY_TO_DAY" else 1.0

    def _per_game_moments(self):
        """Per-game mean (blended rates), variance and makes/attempts covariance per player."""
        rates = self.sim.rates
        mean, var, cov = {}, {}, {}
        for pid in self.sim.players.index:
            pid = int(pid)
            m = rates.loc[pid, STATS].to_numpy(float) if pid in rates.index else np.zeros(len(STATS))
            games, w = self.sim._pools.get(pid, (None, None))
            if games is not None and len(games) >= 5:
                g = games[:, [RAW_STATS.index(x) for x in STATS]]
                mu = (g * w[:, None]).sum(0)
                v = (((g - mu) ** 2) * w[:, None]).sum(0)
                c_fg = (((g[:, IDX["FGM"]] - mu[IDX["FGM"]]) * (g[:, IDX["FGA"]] - mu[IDX["FGA"]])) * w).sum()
                c_ft = (((g[:, IDX["FTM"]] - mu[IDX["FTM"]]) * (g[:, IDX["FTA"]] - mu[IDX["FTA"]])) * w).sum()
            else:  # Poisson-like fallback
                v, c_fg, c_ft = np.maximum(m, 0.0), m[IDX["FGM"]], m[IDX["FTM"]]
            mean[pid], var[pid], cov[pid] = m, v, np.array([c_fg, c_ft])
        return mean, var, cov

    def _start_factor(self, roster: list[int]) -> float:
        """Share of player-games that fit in the lineup over the next weeks (positions + slots)."""
        open_day = self.sim.first_open_period()
        days = [d for d in sorted(self.sim.dates) if d >= open_day][:VERIFY_DAYS]
        frame = pd.DataFrame({"player_id": roster, "lineup_slot": [self.slot.get(p, 12) for p in roster]})
        starts = self.sim.plan_starts(frame, days)
        grid = self.sim.pro[self.sim.pro["scoring_period"].isin(days)].groupby("pro_team_id").size()
        games = sum(grid.get(self.sim.players.at[p, "pro_team_id"], 0) for p in roster if p in self.sim.players.index)
        return min(1.0, len(starts) / games) if games else 1.0

    # ----------------------------------------------------------- scoring

    def contrib(self, pid: int, factor: float):
        """Weekly (mean, var, cov) a player adds to a team with the given start factor."""
        g = self.weekly_games.get(pid, 0.0) * factor
        return self.mean[pid] * g, self.var[pid] * g, self.cov[pid] * g

    def team_moments(self, roster, factor):
        parts = [self.contrib(p, factor) for p in roster if p in self.mean]
        return tuple(np.sum([x[i] for x in parts], axis=0) for i in range(3))

    def cat_dists(self, mean, var, cov):
        """Category means/stds from team moments; arrays broadcast over leading dims."""
        mus, sds = [], []
        for cat in self.cats:
            if "ratio" in cat:
                made, att = cat["ratio"]
                k = 0 if made == "FGM" else 1
                M, A = mean[..., IDX[made]], np.maximum(mean[..., IDX[att]], 1e-9)
                r = M / A
                v = (var[..., IDX[made]] - 2 * r * cov[..., k] + r * r * var[..., IDX[att]]) / (A * A)
                mus.append(r)
                sds.append(np.sqrt(np.maximum(v, 1e-12)))
            else:
                mus.append(mean[..., IDX[cat["name"]]])
                sds.append(np.sqrt(np.maximum(var[..., IDX[cat["name"]]], 1e-9)))
        return np.stack(mus, -1), np.stack(sds, -1)

    def win_probs(self, mu, sd, opp_mu, opp_sd):
        """P(win) per category vs each opponent: (..., n_opp, n_cat)."""
        diff = mu[..., None, :] - opp_mu
        z = diff / np.sqrt(sd[..., None, :] ** 2 + opp_sd ** 2)
        lower = np.array([bool(c.get("lower_is_better")) for c in self.cats])
        return np.where(lower, 1 - _phi(z), _phi(z))

    def _base(self):
        dists = {}
        for t in self.teams:
            dists[t] = self.cat_dists(*self.team_moments(self.roster[t], self.start_factor[t]))
        return dists

    def strengths(self) -> pd.DataFrame:
        """Expected categories won per week vs the average opponent, for every team."""
        d = self._base()
        rows = []
        for t in self.teams:
            others = [o for o in self.teams if o != t]
            p = self.win_probs(d[t][0], d[t][1], np.stack([d[o][0] for o in others]), np.stack([d[o][1] for o in others]))
            rows.append({"team_id": t, "strength": float(p.mean(0).sum()),
                         **{c["name"]: float(p[:, i].mean()) for i, c in enumerate(self.cats)}})
        return pd.DataFrame(rows)

    # ------------------------------------------------------------ search

    def _side(self, roster, factor, out_sets, in_sets):
        """Team moments after removing each out-set and adding each in-set: (n_out, n_in, ...)."""
        base = self.team_moments(roster, factor)
        def sums(sets):
            arr = [[self.contrib(p, factor) for p in s] for s in sets]
            return [np.array([[c[i] for c in a] for a in arr]).sum(1) for i in range(3)]
        o, n = sums(out_sets), sums(in_sets)
        return tuple(base[i] - o[i][:, None] + n[i][None, :] for i in range(3))

    def search(self, me: int, sizes=(1, 2), partners=None, tradeable=None) -> list[Trade]:
        tradeable = tradeable or (lambda pid: True)
        base = self._base()
        trades = []
        for b in partners or [t for t in self.teams if t != me]:
            others = [o for o in self.teams if o not in (me, b)]
            o_mu = np.stack([base[o][0] for o in others])
            o_sd = np.stack([base[o][1] for o in others])
            p_me0 = self.win_probs(*base[me], np.vstack([o_mu, base[b][0][None]]), np.vstack([o_sd, base[b][1][None]]))
            p_b0 = self.win_probs(*base[b], np.vstack([o_mu, base[me][0][None]]), np.vstack([o_sd, base[me][1][None]]))
            mine = [p for p in self.roster[me] if tradeable(p)]
            theirs = [p for p in self.roster[b] if tradeable(p)]
            for k in sizes:
                G, R = list(combinations(mine, k)), list(combinations(theirs, k))
                if not G or not R:
                    continue
                a_mu, a_sd = self.cat_dists(*self._side(self.roster[me], self.start_factor[me], G, R))
                b_mu, b_sd = self.cat_dists(*self._side(self.roster[b], self.start_factor[b], R, G))
                b_mu, b_sd = b_mu.transpose(1, 0, 2), b_sd.transpose(1, 0, 2)  # index as (give, get)
                # each side vs the untouched teams, plus vs the other side after the trade
                pa = np.concatenate([self.win_probs(a_mu, a_sd, o_mu, o_sd),
                                     self._pairwise(a_mu, a_sd, b_mu, b_sd)[..., None, :]], axis=-2)
                pb = np.concatenate([self.win_probs(b_mu, b_sd, o_mu, o_sd),
                                     self._pairwise(b_mu, b_sd, a_mu, a_sd)[..., None, :]], axis=-2)
                d_me = pa.mean(-2) - p_me0.mean(0)          # (nG, nR, cats)
                d_b = pb.mean(-2) - p_b0.mean(0)
                gain_me, gain_b = d_me.sum(-1), d_b.sum(-1)
                for i, j in zip(*np.nonzero((gain_me > 0) & (gain_b > -0.05))):
                    trades.append(Trade(b, G[i], R[j], float(gain_me[i, j]), float(gain_b[i, j]),
                                        dict(zip([c["name"] for c in self.cats], d_me[i, j])),
                                        dict(zip([c["name"] for c in self.cats], d_b[i, j]))))
        return trades

    def _pairwise(self, mu, sd, o_mu, o_sd):
        """P(win) per category between matching trade outcomes (same give/get index)."""
        z = (mu - o_mu) / np.sqrt(sd ** 2 + o_sd ** 2)
        lower = np.array([bool(c.get("lower_is_better")) for c in self.cats])
        return np.where(lower, 1 - _phi(z), _phi(z))

    def verify(self, me: int, t: Trade) -> Trade:
        """Re-score with real lineup filling for both new rosters (positions, slot limits)."""
        a = [p for p in self.roster[me] if p not in t.give] + list(t.get)
        b = [p for p in self.roster[t.partner] if p not in t.get] + list(t.give)
        fa, fb = self._start_factor(a), self._start_factor(b)
        base = self._base()
        others = [o for o in self.teams if o not in (me, t.partner)]
        a_d = self.cat_dists(*self.team_moments(a, fa))
        b_d = self.cat_dists(*self.team_moments(b, fb))
        def strength(d, opp_d, base_self, base_opp):
            opp_mu = np.stack([base[o][0] for o in others] + [opp_d[0]])
            opp_sd = np.stack([base[o][1] for o in others] + [opp_d[1]])
            p = self.win_probs(d[0], d[1], opp_mu, opp_sd).mean(0)
            p0 = self.win_probs(*base_self, np.stack([base[o][0] for o in others] + [base_opp[0]]),
                                np.stack([base[o][1] for o in others] + [base_opp[1]])).mean(0)
            return p - p0
        d_me = strength(a_d, b_d, base[me], base[t.partner])
        d_b = strength(b_d, a_d, base[t.partner], base[me])
        names = [c["name"] for c in self.cats]
        return Trade(t.partner, t.give, t.get, float(d_me.sum()), float(d_b.sum()),
                     dict(zip(names, d_me)), dict(zip(names, d_b)))


# ------------------------------------------------------------- fairness

def perceived_values(sim: Simulator, player_ids) -> dict[int, float]:
    """Positive 'how good does he look' scale from per-game 9-cat value (what managers see)."""
    v = sim.values.reindex(list(player_ids)).fillna(sim.values.min())
    floor = v.min() - 1.0
    return {int(p): float(x - floor) for p, x in v.items()}


def fairness(t: Trade, worth: dict[int, float]) -> tuple[float, float, float]:
    """(score 0-100, gain balance 0-1, value balance 0-1)."""
    if t.my_gain > 0 and t.their_gain > 0:
        balance = min(t.my_gain, t.their_gain) / max(t.my_gain, t.their_gain)
    else:
        balance = 0.0
    give, get = sum(worth[p] for p in t.give), sum(worth[p] for p in t.get)
    looks = min(give, get) / max(give, get) if max(give, get) > 0 else 1.0
    return 100 * (0.5 * balance + 0.5 * looks), balance, looks


def suggest(sim: Simulator, me: int, sizes=(1, 2), include_injured=False, top=40) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (suggestions, team strengths). Suggestions are verified with real lineups."""
    model = TradeModel(sim)
    ok = (lambda p: True) if include_injured else \
        (lambda p: model._availability(p) >= DTD_AVAILABILITY and model.slot.get(p) != 13)
    worth = perceived_values(sim, model.slot)
    raw = model.search(me, sizes=sizes, tradeable=ok)
    # verify a few per partner so one team can't crowd out the rest: the best for me, and the
    # most mutually beneficial (the ones the other manager has a real reason to accept)
    per_partner = max(3, top // max(len(model.teams) - 1, 1))
    picked = {}
    for t in raw:  # one bucket per (partner, trade size) so 2-for-2s don't hide 1-for-1s
        picked.setdefault((t.partner, len(t.give)), [])
    for key in picked:
        ts = [t for t in raw if (t.partner, len(t.give)) == key and t.their_gain > 0]
        best_me = sorted(ts, key=lambda t: -t.my_gain)[:per_partner]
        mutual = sorted(ts, key=lambda t: -min(t.my_gain, t.their_gain))[:per_partner]
        picked[key] = list({(t.give, t.get): t for t in best_me + mutual}.values())
    checked = [model.verify(me, t) for ts in picked.values() for t in ts]
    names = sim.players["name"].to_dict()
    team_names = sim.snap["teams"].set_index("team_id")["name"].to_dict()
    rows = []
    for t in checked:
        score, balance, looks = fairness(t, worth)
        rows.append({
            "partner_id": t.partner, "partner": team_names.get(t.partner, t.partner),
            "give_ids": t.give, "get_ids": t.get,
            "you_give": " + ".join(names.get(p, str(p)) for p in t.give),
            "you_get": " + ".join(names.get(p, str(p)) for p in t.get),
            "size": f"{len(t.give)}-for-{len(t.get)}",
            "my_gain": t.my_gain, "their_gain": t.their_gain,
            "fairness": score, "gain_balance": balance, "value_balance": looks,
            "win_win": t.my_gain > 0 and t.their_gain > 0,
            "my_cats": t.my_cats, "their_cats": t.their_cats,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["win_win", "my_gain"], ascending=False, ignore_index=True)
    strengths = model.strengths()
    strengths["team"] = strengths["team_id"].map(team_names)
    return out, strengths.sort_values("strength", ascending=False, ignore_index=True)
