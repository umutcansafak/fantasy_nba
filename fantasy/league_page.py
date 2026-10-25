"""League page: every team's projected week, one tab per category, teams toggleable.

Data is computed in Python and embedded as JSON; a small vanilla-JS renderer
draws the ranked range chart so toggling teams re-ranks and rescales in place.
"""
import json

import numpy as np

from .config import CATEGORIES
from .projection import actual_totals, opponent_of, with_actuals


def week_data(sim, snap, mp: int, title: str) -> dict | None:
    periods = sim.periods_for(mp)
    if not periods:
        return None
    meta = snap["meta"]
    live = mp == meta.get("current_matchup_period")
    teams = snap["teams"]
    projections = {}
    for i, tid in enumerate(teams["team_id"]):
        proj = sim.project_team(int(tid), sim.roster_of(int(tid)), periods, i)
        if live:
            proj = with_actuals(proj, actual_totals(snap, int(tid), mp), sim.cat_ids)
        projections[int(tid)] = proj
    cats = []
    for cid in sim.cat_ids:
        cat = CATEGORIES[cid]
        rows = []
        for tid, proj in projections.items():
            v = np.nan_to_num(proj.samples[cat["name"]].to_numpy(), nan=0.0)
            rows.append({"id": tid, "mean": float(v.mean()), "lo": float(np.quantile(v, 0.1)),
                         "hi": float(np.quantile(v, 0.9))})
        cats.append({"name": cat["name"], "lower": bool(cat.get("lower_is_better")), "pct": "ratio" in cat,
                     "rows": rows})
    days = meta["matchup_periods"][mp]
    return {
        "mp": mp, "title": title, "live": live,
        "span": f"{sim.dates[days[0]]:%a %b %d} – {sim.dates[days[-1]]:%a %b %d}",
        "note": (f"totals already played + {len(periods)} simulated days" if live else f"{len(periods)} days simulated"),
        "opp": opponent_of(snap, meta["team_id"], mp),
        "starts": {tid: int(p.starts["starts"].sum()) if len(p.starts) else 0 for tid, p in projections.items()},
        "cats": cats,
    }


def render(league_name: str, updated: str, me_id: int, teams, weeks: list[dict], css: str, nav: str) -> str:
    data = {
        "me": me_id,
        "teams": [{"id": int(r.team_id), "name": r.name, "abbrev": r.abbrev} for r in teams.itertuples()],
        "weeks": weeks,
    }
    payload = json.dumps(data).replace("</", "<\\/")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>League view · {league_name}</title><style>{css}{LEAGUE_CSS}</style></head>
<body><main>
{nav}
<h1>{league_name}: where every team stands</h1>
<p class="muted">Updated {updated} · each team's projected totals for the week, independent of opponent</p>
<div id="weeks" class="seg" role="tablist" aria-label="Week"></div>
<p id="weekinfo" class="muted"></p>
<div id="cats" class="cattabs" role="tablist" aria-label="Category"></div>
<section class="card league">
  <div class="charthead"><h2 id="cattitle"></h2><span id="catnote" class="muted"></span></div>
  <div id="chart" class="scroll"></div>
  <p class="legend"><i class="sw me"></i>You <i class="sw opp"></i>This week's opponent <i class="sw other"></i>Other teams
    <span class="muted">· band = middle 80% of simulated weeks, dot = expected · hover a row for numbers</span></p>
</section>
<div class="chiphead"><h3>Teams</h3>
  <button class="link" data-act="all">Show all</button><button class="link" data-act="mine">Only me + opponent</button></div>
<div id="chips" class="chips"></div>
<footer>Ranks compare expected totals; percentages are total makes ÷ attempts. TO: lower is better, so the lowest is ranked #1.</footer>
</main>
<script id="data" type="application/json">{payload}</script>
<script>{LEAGUE_JS}</script>
</body></html>"""


LEAGUE_CSS = """
[hidden]{display:none!important}
:root{--other:#8a8984}@media (prefers-color-scheme:dark){:root{--other:#8f8e88}}
nav.top{display:flex;gap:16px;margin-bottom:16px;font-size:14px}nav.top a{color:var(--ink2)}nav.top a.on{color:var(--ink);font-weight:600}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:8px;overflow:hidden;margin-top:12px}
.seg button{background:var(--card);color:var(--ink2);border:0;padding:6px 14px;font:inherit;cursor:pointer}
.seg button[aria-selected=true]{background:var(--ink);color:var(--bg)}
.cattabs{display:grid;grid-template-columns:repeat(auto-fill,minmax(96px,1fr));gap:8px;margin:12px 0}
.cattabs button{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:8px 10px;text-align:left;
  font:inherit;color:var(--ink);cursor:pointer}
.cattabs button b{display:block;font-size:14px}.cattabs button span{font-size:12px;color:var(--ink2)}
.cattabs button[aria-selected=true]{border-color:var(--me);box-shadow:inset 0 0 0 1px var(--me)}
.card.league{padding:14px 16px}.charthead{display:flex;gap:12px;align-items:baseline;flex-wrap:wrap}
.charthead h2{margin:0;font-size:19px}
#chart svg{min-width:560px}
.row .band{opacity:.3}.row:hover .band{opacity:.5}.row .lab{font-size:13px}.row .rk{fill:var(--muted);font-size:12px}
.row.me .lab{font-weight:700;fill:var(--ink)}.row .num{fill:var(--ink2);font-size:12px;font-variant-numeric:tabular-nums}
.row.me .band,.row.me .dot{fill:var(--me)}.row.opp .band,.row.opp .dot{fill:var(--opp)}
.row.other .band,.row.other .dot{fill:var(--other)}.row .hit{fill:transparent}.gridl{stroke:var(--line);stroke-width:1}
.sw.other{background:var(--other)}
.chiphead{display:flex;gap:12px;align-items:baseline;margin-top:20px}.chiphead h3{margin:0}
button.link{background:none;border:0;color:var(--me);font:inherit;font-size:14px;cursor:pointer;padding:0}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px}
.chips button{display:inline-flex;align-items:center;gap:6px;background:var(--card);color:var(--ink);border:1px solid var(--line);
  border-radius:99px;padding:5px 12px;font:inherit;font-size:14px;cursor:pointer}
.chips button i{width:9px;height:9px;border-radius:50%;background:var(--other)}
.chips button.me i{background:var(--me)}.chips button.opp i{background:var(--opp)}
.chips button[aria-pressed=false]{color:var(--muted);border-style:dashed}.chips button[aria-pressed=false] i{opacity:.25}
"""

LEAGUE_JS = r"""
const D = JSON.parse(document.getElementById('data').textContent);
const teamById = Object.fromEntries(D.teams.map(t => [t.id, t]));
const state = {week: 0, cat: 0, hidden: new Set()};
try { const h = JSON.parse(localStorage.getItem('hiddenTeams') || '[]'); h.forEach(id => state.hidden.add(id)); } catch (e) {}
const save = () => { try { localStorage.setItem('hiddenTeams', JSON.stringify([...state.hidden])); } catch (e) {} };
const fmt = (c, v) => c.pct ? (v * 100).toFixed(1) + '%' : Math.round(v).toLocaleString();
const esc = s => String(s).replace(/[&<>"]/g, ch => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[ch]));
const week = () => D.weeks[state.week];
const roleOf = id => id === D.me ? 'me' : id === week().opp ? 'opp' : 'other';

function ranked(c, rows) {
  return [...rows].sort((a, b) => c.lower ? a.mean - b.mean : b.mean - a.mean);
}

function ticks(lo, hi, n) {
  const raw = (hi - lo) / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= raw) || raw;
  const out = []; for (let t = Math.ceil(lo / step) * step; t <= hi + 1e-9; t += step) out.push(t);
  return out;
}

function renderWeeks() {
  const el = document.getElementById('weeks');
  el.hidden = D.weeks.length < 2;
  el.innerHTML = D.weeks.map((w, i) =>
    `<button role="tab" aria-selected="${i === state.week}" data-week="${i}">${esc(w.title)} · Week ${w.mp}</button>`).join('');
  const w = week(), opp = teamById[w.opp];
  document.getElementById('weekinfo').textContent =
    `Week ${w.mp} · ${w.span} · ${w.note}` + (opp ? ` · your opponent: ${opp.name}` : '');
}

function renderCats() {
  const w = week();
  document.getElementById('cats').innerHTML = w.cats.map((c, i) => {
    const full = ranked(c, c.rows), r = full.findIndex(x => x.id === D.me) + 1;
    return `<button role="tab" aria-selected="${i === state.cat}" data-cat="${i}"><b>${esc(c.name)}</b>` +
           `<span>you #${r} of ${full.length}</span></button>`;
  }).join('');
}

function renderChart() {
  const w = week(), c = w.cats[state.cat];
  const full = ranked(c, c.rows), rank = Object.fromEntries(full.map((r, i) => [r.id, i + 1]));
  const rows = full.filter(r => !state.hidden.has(r.id));
  document.getElementById('cattitle').textContent = c.name;
  document.getElementById('catnote').textContent = (c.lower ? 'lower is better · ' : '') +
    `you rank #${rank[D.me]} of ${full.length} by expected total`;
  const el = document.getElementById('chart');
  if (!rows.length) { el.innerHTML = '<p class="muted">All teams are hidden. Pick some below.</p>'; return; }
  const W = 900, L = 200, R = 90, RH = 34, T = 8, B = 30, H = T + rows.length * RH + B;
  let lo = Math.min(...rows.map(r => r.lo)), hi = Math.max(...rows.map(r => r.hi));
  const pad = (hi - lo) * 0.06 || Math.abs(hi) * 0.05 || 1; lo -= pad; hi += pad;
  const x = v => L + (v - lo) / (hi - lo) * (W - L - R);
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(c.name)} by team">`;
  for (const t of ticks(lo, hi, 5)) {
    s += `<line class="gridl" x1="${x(t)}" x2="${x(t)}" y1="${T}" y2="${H - B + 4}"/>` +
         `<text class="tick" x="${x(t)}" y="${H - 8}" text-anchor="middle">${fmt(c, t)}</text>`;
  }
  rows.forEach((r, i) => {
    const y = T + i * RH + RH / 2, t = teamById[r.id], role = roleOf(r.id);
    const tip = `#${rank[r.id]} ${t.name}: expected ${fmt(c, r.mean)}, 80% range ${fmt(c, r.lo)}–${fmt(c, r.hi)}`;
    s += `<g class="row ${role}"><title>${esc(tip)}</title>` +
         `<rect class="hit" x="0" y="${y - RH / 2}" width="${W}" height="${RH}"/>` +
         `<text class="rk" x="8" y="${y + 4}">#${rank[r.id]}</text>` +
         `<text class="lab" x="40" y="${y + 4}">${esc(t.name.length > 20 ? t.name.slice(0, 19) + '…' : t.name)}</text>` +
         `<rect class="band" x="${x(r.lo)}" y="${y - 5}" width="${Math.max(x(r.hi) - x(r.lo), 2)}" height="10" rx="5"/>` +
         `<circle class="dot" cx="${x(r.mean)}" cy="${y}" r="6"/>` +
         `<text class="num" x="${W - 8}" y="${y + 4}" text-anchor="end">${fmt(c, r.mean)}</text></g>`;
  });
  el.innerHTML = s + '</svg>';
}

function renderChips() {
  const w = week();
  document.getElementById('chips').innerHTML = D.teams.map(t =>
    `<button class="${roleOf(t.id)}" aria-pressed="${!state.hidden.has(t.id)}" data-team="${t.id}">` +
    `<i></i>${esc(t.name)}${t.id === D.me ? ' (you)' : t.id === w.opp ? ' (opponent)' : ''}</button>`).join('');
}

function render() { renderWeeks(); renderCats(); renderChart(); renderChips(); }

document.addEventListener('click', e => {
  const b = e.target.closest('button'); if (!b) return;
  if (b.dataset.week) { state.week = +b.dataset.week; }
  else if (b.dataset.cat) { state.cat = +b.dataset.cat; }
  else if (b.dataset.team) {
    const id = +b.dataset.team; state.hidden.has(id) ? state.hidden.delete(id) : state.hidden.add(id); save();
  } else if (b.dataset.act === 'all') { state.hidden.clear(); save(); }
  else if (b.dataset.act === 'mine') {
    state.hidden = new Set(D.teams.map(t => t.id).filter(id => id !== D.me && id !== week().opp)); save();
  } else return;
  render();
});
render();
"""
