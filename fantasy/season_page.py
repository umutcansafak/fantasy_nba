"""Static "Season so far" page for GitHub Pages: league table by category rank, my category record,
each week's results and my weekly ranks. Same numbers as the Streamlit Season page."""
import html

import pandas as pd

from .players import ordered_categories
from .season import category_record, season_table, week_results, weekly_ranks

esc = html.escape

PAGE_CSS = """
.heat td.c{text-align:center;font-variant-numeric:tabular-nums;min-width:64px;line-height:1.25}
.heat th{text-align:center}.heat th:first-child,.heat td:first-child{text-align:left}
.heat tr.me td{font-weight:600}.heat td.c small{display:block;color:var(--ink2);font-size:11px}
"""


def fmt(c, v):
    return "–" if v is None or pd.isna(v) else f"{v:.1%}" if c.endswith("%") else f"{v:,.0f}"


def tone(score: float) -> str:
    """score in [-1, 1]: blue for good, red for bad, mixed into the card surface."""
    share = min(abs(score), 1) * 70
    pole = "--pw" if score >= 0 else "--pl"
    return f"background:color-mix(in oklab,var({pole}) {share:.0f}%,var(--card))"


def rank_tone(rank, n):
    mid = (n + 1) / 2
    return tone((mid - rank) / (mid - 1)) if n > 1 else ""


def render(league_name: str, updated: str, snap: dict, me_id: int, css: str, nav: str) -> str:
    cats = ordered_categories(snap["meta"]["category_ids"])
    table = season_table(snap)
    results = week_results(snap, me_id)
    record = category_record(results, cats)
    wr = weekly_ranks(snap, me_id)
    n = len(table)
    mine = table[table["team_id"] == me_id].iloc[0]
    ranks = {c: int(mine[f"rank_{c}"]) for c in cats}
    live = next((int(r["matchup_period"]) for _, r in results.iterrows() if not r["final"]), None)
    done = results[results["final"]]
    w_, l_, t_ = int((done["W"] > done["L"]).sum()), int((done["W"] < done["L"]).sum()), int((done["W"] == done["L"]).sum())
    best = sorted(cats, key=ranks.get)
    head = "".join(f"<th>{esc(c)}</th>" for c in cats)

    league_rows = "".join(
        f"<tr class=\"{'me' if r['team_id'] == me_id else ''}\"><td>{'▶ ' if r['team_id'] == me_id else ''}{esc(r['team'])}</td>"
        f"<td class=c>#{r['avg_rank']:.1f}</td>"
        + "".join(f"<td class=c style=\"{rank_tone(r[f'rank_{c}'], n)}\">{fmt(c, r[c])}<small>#{int(r[f'rank_{c}'])}</small></td>"
                  for c in cats) + "</tr>"
        for _, r in table.iterrows())
    rec = record.set_index("category")
    record_rows = ("<tr><td>W–L–T</td>" + "".join(
        f"<td class=c>{rec.at[c, 'W']}–{rec.at[c, 'L']}–{rec.at[c, 'T']}</td>" for c in cats) + "</tr>"
        + "<tr><td>Season rank</td>" + "".join(f"<td class=c style=\"{rank_tone(ranks[c], n)}\">#{ranks[c]}</td>" for c in cats)
        + "</tr>")
    res_tone = {"W": tone(0.8), "L": tone(-0.8), "T": ""}
    week_rows = "".join(
        f"<tr><td>Week {r['matchup_period']} vs {esc(r['opponent'])} <span class=muted>{r['W']}–{r['L']}–{r['T']}"
        f"{'' if r['final'] else ' · in progress'}</span></td>"
        + "".join(f"<td class=c style=\"{res_tone[r[f'res_{c}']]}\"><b>{r[f'res_{c}']}</b><small>"
                  f"{fmt(c, r[f'me_{c}'])} – {fmt(c, r[f'opp_{c}'])}</small></td>" for c in cats) + "</tr>"
        for _, r in results.iterrows())
    rank_rows = "".join(
        f"<tr><td>Week {int(r['matchup_period'])}{' <span class=muted>in progress</span>' if r['matchup_period'] == live else ''}</td>"
        + "".join("<td class=c>–</td>" if pd.isna(r[c]) else
                  f"<td class=c style=\"{rank_tone(r[c], n)}\">#{int(r[c])}</td>" for c in cats) + "</tr>"
        for _, r in wr.iterrows())

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Season so far · {league_name}</title><style>{css}{PAGE_CSS}</style></head>
<body><main>
{nav}
<h1>Season so far</h1>
<p class="muted">Updated {updated} · ESPN's recorded totals · FG%/FT% = total makes ÷ attempts{
    f" · week {live} is in progress: counted in totals and ranks, not yet in the record" if live else ""}</p>
<div class="kpis">
  <div><span>Matchup record</span><b>{w_}–{l_}–{t_}</b></div>
  <div><span>Category record</span><b>{record['W'].sum()}–{record['L'].sum()}–{record['T'].sum()}</b></div>
  <div><span>Average category rank</span><b>#{mine['avg_rank']:.1f}<small> of {n}</small></b></div>
  <div><span>Best / worst</span><b>{esc(best[0])} #{ranks[best[0]]}<small> · {esc(best[-1])} #{ranks[best[-1]]}</small></b></div>
</div>
<section>
  <h2>League table</h2>
  <p class="muted">Season totals for every team, coloured by league rank in each category (blue = top, red = bottom),
     ordered by average rank. TO: fewest is #1.</p>
  <div class="scroll"><table class="heat"><thead><tr><th>Team</th><th>Avg rank</th>{head}</tr></thead>
    <tbody>{league_rows}</tbody></table></div>
</section>
<section>
  <h2>My category record</h2>
  <p class="muted">Completed weeks only.</p>
  <div class="scroll"><table class="heat"><thead><tr><th></th>{head}</tr></thead><tbody>{record_rows}</tbody></table></div>
  <h3>Week by week</h3>
  <p class="muted">Result per category, with your total – your opponent's.</p>
  <div class="scroll"><table class="heat"><thead><tr><th>Week</th>{head}</tr></thead><tbody>{week_rows}</tbody></table></div>
  <h3>My rank per week</h3>
  <div class="scroll"><table class="heat"><thead><tr><th>Week</th>{head}</tr></thead><tbody>{rank_rows}</tbody></table></div>
</section>
<footer>Ranks compare every team's totals for the same days, so a live week is fair to compare but not final.</footer>
</main></body></html>"""
