"""Static HTML matchup report for GitHub Pages: `python -m fantasy.report [--out site]`.

One page, my matchup only: the live week (ESPN totals so far + simulated rest)
and next week's preview, each with the 9 category range charts, starters per
day, and free-agent pickups.
"""
import argparse
import html
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import league_page, players_page, store
from .live import live_matchup
from .players import ordered_categories, roster_contributions
from .config import CATEGORIES, ROOT
from .projection import (Simulator, actual_totals, compare, next_matchup_period, opponent_of,
                         with_actuals)
from .recommend import recommend

esc = html.escape


def fmt(cat, v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    return f"{v:.1%}" if cat.endswith("%") else f"{v:,.0f}"


def nice_ticks(lo, hi, n=4):
    span = hi - lo or 1.0
    raw = span / n
    mag = 10 ** np.floor(np.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=raw)
    start = np.ceil(lo / step) * step
    return [t for t in np.arange(start, hi + step * 0.01, step) if lo <= t <= hi]


def category_svg(row, me_name, opp_name):
    """Two rows (me, opponent): 80% range band + expected-value dot, shared x scale."""
    name = row["category"]
    W, H, L, R = 340, 118, 92, 14
    lo = min(row["me_lo"], row["opp_lo"])
    hi = max(row["me_hi"], row["opp_hi"])
    pad = (hi - lo) * 0.12 or abs(hi) * 0.05 or 1
    lo, hi = lo - pad, hi + pad
    x = lambda v: L + (v - lo) / (hi - lo) * (W - L - R)
    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(name)} projection">']
    for t in nice_ticks(lo, hi):
        parts.append(f'<line class="grid" x1="{x(t):.1f}" x2="{x(t):.1f}" y1="14" y2="{H - 22}"/>'
                     f'<text class="tick" x="{x(t):.1f}" y="{H - 6}" text-anchor="middle">{fmt(name, t)}</text>')
    for i, (label, cls, pre) in enumerate(((me_name, "me", "me"), (opp_name, "opp", "opp"))):
        y = 36 + i * 40
        a, b, m = row[f"{pre}_lo"], row[f"{pre}_hi"], row[f"{pre}_mean"]
        tip = f"{label}: expected {fmt(name, m)}, 80% range {fmt(name, a)}–{fmt(name, b)}"
        parts.append(
            f'<g class="{cls}"><title>{esc(tip)}</title>'
            f'<text class="lab" x="{L - 8}" y="{y + 4}" text-anchor="end">{esc(label[:14])}</text>'
            f'<rect class="band" x="{x(a):.1f}" y="{y - 5}" width="{max(x(b) - x(a), 2):.1f}" height="10" rx="5"/>'
            f'<rect class="hit" x="{x(a) - 6:.1f}" y="{y - 14}" width="{max(x(b) - x(a), 2) + 12:.1f}" height="28"/>'
            f'<circle class="dot" cx="{x(m):.1f}" cy="{y}" r="6"/>'
            f'<text class="val" x="{x(m):.1f}" y="{y - 11}" text-anchor="middle">{fmt(name, m)}</text></g>')
    parts.append("</svg>")
    return "".join(parts)


def daily_svg(me_daily, opp_daily, dates, me_name, opp_name):
    W, H, L, B = 1000, 220, 30, 28
    n = len(dates)
    top = max([int(me_daily.get(p, 0)) for p in dates] + [int(opp_daily.get(p, 0)) for p in dates] + [1])
    step = (W - L) / n
    bw = min(30, step / 3)
    y = lambda v: H - B - v / top * (H - B - 20)
    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Starters per day">']
    for t in nice_ticks(0, top, 3):
        parts.append(f'<line class="grid" x1="{L}" x2="{W}" y1="{y(t):.1f}" y2="{y(t):.1f}"/>'
                     f'<text class="tick" x="{L - 6}" y="{y(t) + 4:.1f}" text-anchor="end">{t:.0f}</text>')
    for i, (p, d) in enumerate(dates.items()):
        cx = L + step * (i + 0.5)
        for j, (cls, daily, label) in enumerate((("me", me_daily, me_name), ("opp", opp_daily, opp_name))):
            v = int(daily.get(p, 0))
            bx = cx - bw - 1 + j * (bw + 2)
            parts.append(f'<g class="{cls}"><title>{esc(label)}: {v} starters on {d:%a %b %d}</title>'
                         f'<path class="bar" d="M{bx:.1f},{H - B} V{y(v) + 4:.1f} q0,-4 4,-4 h{bw - 8:.1f} q4,0 4,4 V{H - B} Z"/>'
                         f'<text class="val" x="{bx + bw / 2:.1f}" y="{y(v) - 5:.1f}" text-anchor="middle">{v}</text></g>')
        parts.append(f'<text class="tick" x="{cx:.1f}" y="{H - 8}" text-anchor="middle">{d:%a %d}</text>')
    parts.append("</svg>")
    return "".join(parts)


STATUS_ICON = {"Locked": "🔒", "Leaning win": "↗", "Swing": "⚖", "Leaning loss": "↘", "Lost": "✕"}


def pwin_svg(cats):
    """One dot per category on a 0-100% win-probability axis, swing zone (35-65%) shaded."""
    d = cats.sort_values("p_win", ascending=False)
    W, L, R, top, rh = 440, 40, 125, 20, 26
    H = top + rh * len(d) + 28
    x = lambda p: L + p * (W - L - R)
    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Chance to win each category">',
             f'<rect class="zone" x="{x(.35):.1f}" y="{top - 6}" width="{x(.65) - x(.35):.1f}" height="{rh * len(d) + 6}"/>',
             f'<text class="tick" x="{x(.5):.1f}" y="{top - 9}" text-anchor="middle">swing zone</text>']
    for t in (0, .25, .5, .75, 1):
        parts.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{top - 6}" y2="{top + rh * len(d)}"/>'
                     f'<text class="tick" x="{x(t):.1f}" y="{H - 8}" text-anchor="middle">{t:.0%}</text>')
    for i, r in enumerate(d.itertuples()):
        cy = top + rh * i + rh / 2
        side = "pw" if r.p_win >= .65 else "pl" if r.p_win <= .35 else "ps"
        parts.append(f'<g><title>{esc(r.category)}: {r.p_win:.0%} chance to win</title>'
                     f'<text class="lab" x="{L - 8}" y="{cy + 4:.1f}" text-anchor="end">{esc(r.category)}</text>'
                     f'<circle class="dot {side}" cx="{x(r.p_win):.1f}" cy="{cy:.1f}" r="6.5"/>'
                     f'<text class="lab" x="{x(r.p_win) + 12:.1f}" y="{cy + 4:.1f}">{r.p_win:.0%} '
                     f'{STATUS_ICON[r.status]} {esc(r.status)}</text></g>')
    parts.append("</svg>")
    return "".join(parts)


def live_html(sim, snap, me_id):
    """Where the live week stands: score now, projected final, and each category's status."""
    lv = live_matchup(sim, snap, me_id)
    if lv is None or not lv["days_left"]:
        return ""
    teams = snap["teams"].set_index("team_id")["name"].to_dict()
    opp = teams[lv["opp_id"]]
    cats = lv["cats"]
    w, l, t = lv["score_now"]
    sm, so = lv["starts_left"]
    played = lv["days_total"] - lv["days_left"]
    groups = "".join(
        f"<li>{STATUS_ICON[s]} <b>{s}:</b> {esc(', '.join(g))}</li>"
        for s in STATUS_ICON if (g := cats[cats["status"] == s]["category"].tolist()))
    def v(r, col):
        return fmt(r["category"], r[col])
    rows = "".join(
        f"<tr><td>{esc(r['category'])}</td><td>{v(r, 'me_now')} – {v(r, 'opp_now')}</td>"
        f"<td>{v(r, 'me_final')} – {v(r, 'opp_final')}</td><td>{STATUS_ICON[r['status']]} {esc(r['status'])}</td></tr>"
        for _, r in cats.sort_values("p_win").iterrows())
    return f"""
<section>
  <h2>Right now: {esc(teams[me_id])} <span class="vs">vs</span> {esc(opp)}</h2>
  <p class="muted">Week {lv['mp']} · {lv['first_day']:%a %b %d} – {lv['last_day']:%a %b %d} ·
     day {played + 1} of {lv['days_total']} · {lv['days_left']} days left incl. today</p>
  <div class="kpis">
    <div><span>Score now (W–L–T)</span><b>{f"{w}–{l}–{t}" if lv['started'] else "not started"}</b></div>
    <div><span>Projected final</span><b>{lv['expected_cats']:.1f}<small> cats</small></b></div>
    <div><span>Chance to win the week</span><b>{lv['p_win']:.0%}</b></div>
    <div><span>Starts left (me / them)</span><b>{sm}<small> / {so}</small></b>
      <small class="muted">today {lv['starts_today'][0]} / {lv['starts_today'][1]}</small></div>
  </div>
  <ul class="status">{groups}</ul>
  <div class="split">
    <div class="chart">{pwin_svg(cats)}</div>
    <div class="scroll"><table><thead><tr><th>Cat</th><th>Now (me – opp)</th><th>Projected final</th><th>Status</th></tr></thead>
      <tbody>{rows}</tbody></table></div>
  </div>
  <p class="muted">Locked ≥ 90% · Swing 35–65% · Lost ≤ 10% chance to win the category by Sunday.
     Projected final = totals recorded so far + the average simulated rest of the week. TO: lower is better.</p>
</section>"""


def heat_cell(v, lim):
    """Diverging cell: blue helps, red hurts; intensity mixes the pole into the card surface."""
    if pd.isna(v):
        return '<td class="z muted">–</td>'
    v = v if abs(v) >= 0.05 else 0.0  # no "-0.0"
    share = min(abs(v) / lim, 1) * 70
    pole = "--pw" if v >= 0 else "--pl"
    return (f'<td class="z" style="background:color-mix(in oklab,var({pole}) {share:.0f}%,var(--card))">'
            f"{v:+.1f}</td>")


def players_html(sim, snap, me_id):
    """Each rostered player's z-score per category, per game and weighted by games left this week."""
    mp = snap["meta"].get("current_matchup_period") or next_matchup_period(snap["meta"])
    if not sim.periods_for(mp):
        mp = next_matchup_period(snap["meta"])
    d = roster_contributions(sim, me_id, mp)
    if d.empty:
        return ""
    cats = ordered_categories(sim.cat_ids)

    def table(view, weekly):
        lim = max(1.0, float(np.nanmax(view[cats].abs().to_numpy(dtype=float)))) if weekly and view[cats].notna().any().any() else 3.0
        body = "".join(
            f"<tr><td>{esc(r['name'])}"
            + ("" if r["injury_status"] == "ACTIVE" else f" <span class=muted>{esc(r['injury_status'].replace('_', ' ').lower())}</span>")
            + f"</td><td class=z>{r['gp']}</td><td class=z>{'–' if pd.isna(r['total']) else format(r['total'], '+.1f')}</td>"
            + (f"<td class=z>{r['played_games']} + {r['exp_games']:.1f}</td>" if weekly else "")
            + "".join(heat_cell(r[c], lim) for c in cats) + "</tr>"
            for _, r in view.iterrows())
        head = ("<th>Player</th><th>GP</th><th>Total</th>" + ("<th>Played + left</th>" if weekly else "")
                + "".join(f"<th>{esc(c)}</th>" for c in cats))
        return f"<div class='scroll'><table class='heat'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"

    weekly = d.copy()
    weekly[cats] = weekly[[f"wk_{c}" for c in cats]].to_numpy()
    weekly["total"] = weekly["wk_total"]
    weekly = weekly.sort_values("total", ascending=False, na_position="last")
    team = d[cats].sum()
    return f"""
<section>
  <h2>My players</h2>
  <p class="muted">This season's games only, no projections. How each player helps (blue) or hurts (red) every
     category: z-score vs this season's top 200 scorers, so +1 = one standard deviation better than average.
     TO is flipped; FG%/FT% weigh shooting volume. GP = games played; small GP = noisy.
     The week view adds up the games already played this week and season averages × expected starts left.</p>
  <div class="seg" role="tablist">
    <button class="on" data-show="pg">Season so far</button><button data-show="wk">Week {mp}: played + rest</button></div>
  <div data-view="pg">{table(d, False)}</div>
  <div data-view="wk" hidden>{table(weekly, True)}</div>
  <p><b>Roster in a nutshell (season so far, per game):</b> strongest
     {esc(", ".join(f"{c} {team[c]:+.1f}" for c in team.sort_values(ascending=False).index[:3]))} ·
     weakest {esc(", ".join(f"{c} {team[c]:+.1f}" for c in team.sort_values().index[:3]))}</p>
  <script>document.querySelectorAll('.seg button').forEach(b=>b.onclick=()=>{{
    document.querySelectorAll('.seg button').forEach(x=>x.classList.toggle('on',x===b));
    document.querySelectorAll('[data-view]').forEach(v=>v.hidden=v.dataset.view!==b.dataset.show);}});</script>
</section>"""


def section(sim, snap, me_id, mp, title, n_fa):
    teams = snap["teams"].set_index("team_id")["name"].to_dict()
    opp_id = opponent_of(snap, me_id, mp)
    if opp_id is None:
        return f"<section><h2>{esc(title)}</h2><p class='muted'>No opponent scheduled.</p></section>"
    periods = sim.periods_for(mp)
    if not periods:
        return ""
    live = mp == snap["meta"].get("current_matchup_period")
    me = sim.project_team(me_id, sim.roster_of(me_id), periods, 0)
    opp = sim.project_team(opp_id, sim.roster_of(opp_id), periods, 1)
    if live:
        me = with_actuals(me, actual_totals(snap, me_id, mp), sim.cat_ids)
        opp = with_actuals(opp, actual_totals(snap, opp_id, mp), sim.cat_ids)
    table, overall = compare(me, opp, sim.cat_ids)
    me_name, opp_name = teams[me_id], teams[opp_id]
    all_days = sim.meta["matchup_periods"][mp]
    span = f"{sim.dates[all_days[0]]:%a %b %d} – {sim.dates[all_days[-1]]:%a %b %d}"
    note = (f"Totals already played + {len(periods)} simulated day{'s' * (len(periods) != 1)} remaining"
            if live else f"{len(periods)} days simulated")

    swing = table[table["verdict"] == "swing"]["category"].tolist()
    ordered = table.assign(c=(table["p_win"] - 0.5).abs()).sort_values("c")
    cards = "".join(
        f'<figure class="card"><figcaption><b>{esc(r["category"])}</b>'
        f'<span class="pill {str(r["verdict"]).replace(" ", "-")}">{r["p_win"]:.0%} · {esc(str(r["verdict"]))}</span>'
        f'{"<span class=muted> lower wins</span>" if r["lower_is_better"] else ""}</figcaption>'
        f'{category_svg(r, me_name, opp_name)}</figure>'
        for _, r in ordered.iterrows())

    rows = "".join(
        f"<tr><td>{esc(r['category'])}</td><td>{fmt(r['category'], r['me_lo'])}–{fmt(r['category'], r['me_hi'])}</td>"
        f"<td>{fmt(r['category'], r['me_mean'])}</td><td>{fmt(r['category'], r['opp_mean'])}</td>"
        f"<td>{fmt(r['category'], r['opp_lo'])}–{fmt(r['category'], r['opp_hi'])}</td><td>{r['p_win']:.0%}</td></tr>"
        for _, r in table.iterrows())

    starts = (sim.roster_of(me_id).merge(sim.players[["name", "pro_team", "injury_status"]].reset_index(), on="player_id")
              .merge(me.starts[["player_id", "nba_games", "starts"]], on="player_id", how="left")
              .fillna({"nba_games": 0, "starts": 0}).sort_values(["starts", "nba_games"], ascending=False))
    start_rows = "".join(
        f"<tr><td>{esc(r['name'])}</td><td>{esc(r['pro_team'])}</td>"
        f"<td>{'' if r['injury_status'] == 'ACTIVE' else esc(r['injury_status'].replace('_', ' ').title())}</td>"
        f"<td>{int(r['nba_games'])}</td><td>{int(r['starts'])}</td></tr>" for _, r in starts.iterrows())

    fa_html = ""
    if n_fa:
        recs, drop_id = recommend(sim, me_id, opp_id, periods, n=n_fa)
        if not recs.empty:
            cat_names = [CATEGORIES[c]["name"] for c in sim.cat_ids]
            def best(r):
                gains = sorted(((r[f"d_{c}"], c) for c in cat_names), reverse=True)[:2]
                return ", ".join(f"{c} {g:+.0%}" for g, c in gains if g > 0.005) or "–"
            fa_rows = "".join(
                f"<tr><td>{esc(r['name'])}</td><td>{esc(r['team'])}</td><td>{int(r['games'])}</td>"
                f"<td>{r['d_expected_cats']:+.2f}</td><td>{r['d_p_win_matchup']:+.0%}</td><td>{esc(best(r))}</td></tr>"
                for _, r in recs.head(8).iterrows())
            fa_html = (f"<h3>Free-agent pickups</h3><p class='muted'>Each one simulated in place of "
                       f"<b>{esc(sim.players.at[drop_id, 'name'])}</b> (lowest-value player on the roster).</p>"
                       f"<div class='scroll'><table><thead><tr><th>Player</th><th>Team</th><th>Games</th>"
                       f"<th>Δ exp. cats</th><th>Δ P(win)</th><th>Biggest gains</th></tr></thead>"
                       f"<tbody>{fa_rows}</tbody></table></div>")

    if live:
        title += ", category ranges"
    return f"""
<section>
  <h2>{esc(title)}: {esc(me_name)} <span class="vs">vs</span> {esc(opp_name)}</h2>
  <p class="muted">Week {mp} · {span} · {note}</p>
  {"" if live else f"""<div class="kpis">
    <div><span>Expected categories</span><b>{overall['expected_cats_won']:.1f}<small> / {len(table)}</small></b></div>
    <div><span>P(win matchup)</span><b>{overall['p_win_matchup']:.0%}</b></div>
    <div><span>Starts (me / them)</span><b>{overall['me_starts']}<small> / {overall['opp_starts']}</small></b></div>
  </div>"""}
  {f"<p class='focus'><b>Focus on:</b> {esc(', '.join(swing))}</p>" if swing else ""}
  <p class="legend"><i class="sw me"></i>{esc(me_name)} <i class="sw opp"></i>{esc(opp_name)}
     <span class="muted">· band = middle 80% of simulated weeks, dot = expected · closest races first</span></p>
  <div class="grid">{cards}</div>
  <details><summary>Table view</summary><div class="scroll"><table>
    <thead><tr><th>Cat</th><th>Me 80%</th><th>Me exp.</th><th>Opp exp.</th><th>Opp 80%</th><th>P(win)</th></tr></thead>
    <tbody>{rows}</tbody></table></div></details>
  <h3>Starters per day</h3>
  <div class="chart">{daily_svg(me.daily, opp.daily, {p: sim.dates[p] for p in periods}, me_name, opp_name)}</div>
  <details><summary>My roster: games and starts</summary><div class="scroll"><table>
    <thead><tr><th>Player</th><th>Team</th><th>Status</th><th>NBA games</th><th>Starts</th></tr></thead>
    <tbody>{start_rows}</tbody></table></div></details>
  {fa_html}
</section>"""


CSS = """
:root{--bg:#fcfcfb;--card:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--muted:#6f6e69;--line:#e6e5e0;
--me:#2a78d6;--opp:#eb6834;--pw:#2a78d6;--pl:#e34948;--ps:#8a8984;--win:#1f7a3d;--loss:#b3261e;--swing:#8a5a00;color-scheme:light}
@media (prefers-color-scheme:dark){:root{--bg:#121211;--card:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;
--muted:#9b9a92;--line:#2f2f2c;--me:#3987e5;--opp:#d95926;--pw:#3987e5;--pl:#e66767;--win:#6cc287;--loss:#f08b84;--swing:#e3b04b;color-scheme:dark}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1120px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:28px;margin:0 0 4px}h2{font-size:21px;margin:40px 0 2px}h3{font-size:16px;margin:28px 0 8px}
.vs{color:var(--muted);font-weight:400}.muted{color:var(--muted)}
.kpis{display:flex;flex-wrap:wrap;gap:12px;margin:16px 0}
.kpis div{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 16px;min-width:150px;flex:1}
.kpis span{display:block;color:var(--ink2);font-size:13px}.kpis b{font-size:28px}.kpis small{font-size:15px;color:var(--muted)}
.focus{margin:8px 0}.legend{font-size:13px;color:var(--ink2)}
.sw{display:inline-block;width:10px;height:10px;border-radius:50%;margin:0 5px 0 10px;vertical-align:-1px}
.sw.me{background:var(--me)}.sw.opp{background:var(--opp)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}
.card{margin:0;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px 4px}
.card figcaption{display:flex;gap:8px;align-items:center;font-size:14px}
.pill{font-size:12px;padding:1px 8px;border-radius:99px;border:1px solid currentColor}
.likely-win{color:var(--win)}.likely-loss{color:var(--loss)}.swing{color:var(--swing)}
svg{width:100%;height:auto;display:block;overflow:visible}
.grid line,.chart line{stroke:var(--line);stroke-width:1}
.tick{fill:var(--muted);font-size:11px}.lab{fill:var(--ink2);font-size:12px}.val{fill:var(--ink);font-size:11px;font-weight:600}
.me .band{fill:var(--me);opacity:.32}.opp .band{fill:var(--opp);opacity:.32}
.me .dot,.me .bar{fill:var(--me)}.opp .dot,.opp .bar{fill:var(--opp)}
.dot{stroke:var(--card);stroke-width:2}.hit{fill:transparent}
g:hover .band{opacity:.5}
.chart{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
details{margin:12px 0}summary{cursor:pointer;color:var(--ink2)}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px;margin-top:8px}
th,td{text-align:left;padding:6px 10px;border-bottom:1px solid var(--line);white-space:nowrap}
th{color:var(--ink2);font-weight:600}td:nth-child(n+3){font-variant-numeric:tabular-nums}
footer{margin-top:48px;color:var(--muted);font-size:13px}
.status{list-style:none;padding:0;margin:8px 0 16px}.status li{margin:2px 0}
.split{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px;align-items:start}
.zone{fill:var(--ps);opacity:.14}.dot.pw{fill:var(--pw)}.dot.pl{fill:var(--pl)}.dot.ps{fill:var(--ps)}
.kpis div small.muted{display:block;font-size:12px}
[hidden]{display:none!important}
.heat td.z{text-align:center;font-variant-numeric:tabular-nums;min-width:52px}.heat th{text-align:center}.heat th:first-child{text-align:left}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:8px;overflow:hidden;margin:4px 0 8px}
.seg button{background:var(--card);color:var(--ink2);border:0;padding:6px 12px;font:inherit;font-size:13px;cursor:pointer}
.seg button.on{background:var(--ink);color:var(--bg)}
nav.top{display:flex;gap:16px;margin-bottom:16px;font-size:14px}nav.top a{color:var(--ink2)}nav.top a.on{color:var(--ink);font-weight:600}
"""


def report_weeks(meta) -> list[tuple[int, str]]:
    """The live week (+ next week's preview), or just the coming week once the live one is over."""
    cur, nxt = meta.get("current_matchup_period"), next_matchup_period(meta)
    if cur and nxt == cur:
        return [(cur, "This week")] + ([(cur + 1, "Next week")] if cur + 1 in meta["matchup_periods"] else [])
    return [(nxt, "Next week")]


def nav(active: str) -> str:
    links = (("index", "index.html", "My matchup"), ("league", "league.html", "League view"),
             ("players", "players.html", "Player contributions"))
    return '<nav class="top">' + "".join(
        f'<a href="{href}"{" class=on" if key == active else ""}>{label}</a>' for key, href, label in links) + "</nav>"


def build(out: Path, n_sims: int = 5000, n_fa: int = 20) -> Path:
    snap = store.load_snapshot()
    sim = Simulator(snap, n_sims=n_sims)
    meta = snap["meta"]
    me_id = meta.get("team_id")
    if me_id is None:
        raise SystemExit("TEAM_ID is not set.")
    weeks = report_weeks(meta)
    sections = [section(sim, snap, me_id, mp, title, n_fa) for mp, title in weeks]
    updated = datetime.fromisoformat(meta["collected_at"]).strftime("%a %b %d, %Y %H:%M ET")
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(meta.get('league_name') or 'Fantasy')} · Matchup report</title><style>{CSS}</style></head>
<body><main>
{nav('index')}
<h1>{esc(meta.get('league_name') or 'Fantasy NBA')} matchup report</h1>
<p class="muted">Updated {updated} · {n_sims:,} simulated weeks from players' real box scores</p>
{live_html(sim, snap, me_id)}
{players_html(sim, snap, me_id)}
{''.join(sections)}
<footer>Projections bootstrap each starter's recent games (this season weighted above last; last season's games
rescaled to full-season averages for players with 50+ games), fill daily lineups
by value within ESPN slot limits, and skip players listed OUT. Ties count as half a win.</footer>
</main></body></html>"""
    out.mkdir(parents=True, exist_ok=True)
    path = out / "index.html"
    path.write_text(page)
    league_weeks = [w for w in (league_page.week_data(sim, snap, mp, title) for mp, title in weeks) if w]
    (out / "league.html").write_text(league_page.render(
        esc(meta.get("league_name") or "Fantasy NBA"), updated, me_id, snap["teams"], league_weeks, CSS, nav("league")))
    if not snap["rosters"].empty:
        (out / "players.html").write_text(players_page.render(
            esc(meta.get("league_name") or "Fantasy NBA"), updated, players_page.build_data(sim, snap, me_id),
            CSS, nav("players")))
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "site")
    ap.add_argument("--sims", type=int, default=5000)
    ap.add_argument("--fa", type=int, default=20, help="free agents to simulate per week (0 = skip)")
    args = ap.parse_args()
    print(build(args.out, args.sims, args.fa))


if __name__ == "__main__":
    main()
