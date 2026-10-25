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

from . import store
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

    return f"""
<section>
  <h2>{esc(title)}: {esc(me_name)} <span class="vs">vs</span> {esc(opp_name)}</h2>
  <p class="muted">Week {mp} · {span} · {note}</p>
  <div class="kpis">
    <div><span>Expected categories</span><b>{overall['expected_cats_won']:.1f}<small> / {len(table)}</small></b></div>
    <div><span>P(win matchup)</span><b>{overall['p_win_matchup']:.0%}</b></div>
    <div><span>{'Starts left' if live else 'Starts'} (me / them)</span><b>{overall['me_starts']}<small> / {overall['opp_starts']}</small></b></div>
  </div>
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
--me:#2a78d6;--opp:#eb6834;--win:#1f7a3d;--loss:#b3261e;--swing:#8a5a00;color-scheme:light}
@media (prefers-color-scheme:dark){:root{--bg:#121211;--card:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;
--muted:#9b9a92;--line:#2f2f2c;--me:#3987e5;--opp:#d95926;--win:#6cc287;--loss:#f08b84;--swing:#e3b04b;color-scheme:dark}}
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
"""


def build(out: Path, n_sims: int = 5000, n_fa: int = 20) -> Path:
    snap = store.load_snapshot()
    sim = Simulator(snap, n_sims=n_sims)
    meta = snap["meta"]
    me_id = meta.get("team_id")
    if me_id is None:
        raise SystemExit("TEAM_ID is not set.")
    cur = meta.get("current_matchup_period")
    nxt = next_matchup_period(meta)
    sections = []
    if cur and nxt == cur:
        sections.append(section(sim, snap, me_id, cur, "This week", n_fa))
        if cur + 1 in meta["matchup_periods"]:
            sections.append(section(sim, snap, me_id, cur + 1, "Next week", n_fa))
    else:  # Sunday after the last games, or before the season: plan the coming week
        sections.append(section(sim, snap, me_id, nxt, "Next week", n_fa))
    updated = datetime.fromisoformat(meta["collected_at"]).strftime("%a %b %d, %Y %H:%M ET")
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(meta.get('league_name') or 'Fantasy')} · Matchup report</title><style>{CSS}</style></head>
<body><main>
<h1>{esc(meta.get('league_name') or 'Fantasy NBA')} matchup report</h1>
<p class="muted">Updated {updated} · {n_sims:,} simulated weeks from players' real box scores</p>
{''.join(sections)}
<footer>Projections bootstrap each starter's recent games (this season weighted above last), fill daily lineups
by value within ESPN slot limits, and skip players listed OUT. Ties count as half a win.</footer>
</main></body></html>"""
    out.mkdir(parents=True, exist_ok=True)
    path = out / "index.html"
    path.write_text(page)
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
