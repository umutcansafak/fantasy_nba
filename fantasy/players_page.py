"""Static "Player contributions" page for GitHub Pages: every roster's (and the free agents') per-category
z-scores this season, plus how each player would fit my team. Same numbers as the Streamlit page; the
interactivity (team picker, who to replace) runs on embedded JSON.
"""
import json

import numpy as np
import pandas as pd

from .config import DEFAULT_CATEGORY_WEIGHTS
from .fit import FitModel
from .players import free_agent_contributions, ordered_categories, roster_contributions

N_FREE_AGENTS = 30
FA_MIN_GP = 2


def _num(v, nd=2):
    return None if v is None or pd.isna(v) else round(float(v), nd)


def build_data(sim, snap, me_id: int) -> dict:
    cats = ordered_categories(sim.cat_ids)
    weights = {c: DEFAULT_CATEGORY_WEIGHTS.get(c, 1.0) for c in cats}
    teams = snap["teams"].set_index("team_id")["name"].to_dict()

    def players(d: pd.DataFrame) -> list[dict]:
        return [{"id": int(r["player_id"]), "name": r["name"], "team": r["pro_team"],
                 "status": "" if r["injury_status"] == "ACTIVE" else r["injury_status"].replace("_", " ").lower(),
                 "gp": int(r["gp"]), "z": [_num(r[c]) for c in cats]} for _, r in d.iterrows()]

    groups, rosters = [], {}
    for tid, name in teams.items():
        d = roster_contributions(sim, tid)
        rosters[tid] = d
        groups.append({"id": int(tid), "name": name, "me": tid == me_id, "players": players(d)})
    fa = free_agent_contributions(sim)
    fa = fa[fa["gp"] >= FA_MIN_GP]
    fa = fa.assign(w=sum(fa[c] * w for c, w in weights.items())).sort_values("w", ascending=False).head(N_FREE_AGENTS)
    groups.append({"id": -1, "name": "Free agents", "me": False, "players": players(fa)})

    model = FitModel(sim)
    mine = rosters[me_id].assign(w=sum(rosters[me_id][c] * w for c, w in weights.items()))
    candidates = [p["id"] for g in groups if not g["me"] for p in g["players"]]
    fit = {}
    for drop in mine["player_id"]:
        f = model.fit(me_id, candidates, int(drop))
        fit[int(drop)] = {int(r["player_id"]): [_num(r[c], 4) for c in cats] for _, r in f.iterrows()}
    played = mine[mine["gp"] > 0].sort_values("w")
    default_drop = int(played["player_id"].iloc[0] if len(played) else mine["player_id"].iloc[0])
    base = model.baseline(me_id)
    return {
        "cats": cats, "weights": [weights[c] for c in cats], "me": int(me_id), "groups": groups,
        "drops": [{"id": int(r["player_id"]), "name": r["name"]} for _, r in mine.sort_values("w").iterrows()],
        "defaultDrop": default_drop, "fit": fit, "baseline": [_num(base[c], 3) for c in cats],
    }


PAGE_CSS = """
.heat td.z{text-align:center;font-variant-numeric:tabular-nums;min-width:52px}.heat th{text-align:center}
.heat th:first-child,.heat td:first-child{text-align:left}.heat tr.me td{font-weight:600}
.controls{display:flex;flex-wrap:wrap;gap:12px;align-items:end;margin:12px 0}
.controls label{display:flex;flex-direction:column;font-size:13px;color:var(--ink2);gap:4px}
select{font:inherit;padding:6px 8px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--ink)}
"""

PAGE_JS = r"""
const D = JSON.parse(document.getElementById('data').textContent);
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const fmt = (v, d=1) => v == null ? '–' : (Math.abs(v) < 0.5 * 10 ** -d ? 0 : v).toFixed(d).replace(/^(?!-)/, '+');
const cell = (v, lim, d=1) => {
  if (v == null) return '<td class="z muted">–</td>';
  const share = Math.min(Math.abs(v) / lim, 1) * 70, pole = v >= 0 ? '--pw' : '--pl';
  return `<td class="z" style="background:color-mix(in oklab,var(${pole}) ${share.toFixed(0)}%,var(--card))">${fmt(v, d)}</td>`;
};
const colName = (c, w) => w === 0 ? `${c} (punted)` : w === 1 ? c : `${c} ×${w}`;
const total = (z, w) => z.every(v => v == null) ? null : z.reduce((s, v, i) => s + (v ?? 0) * w[i], 0);
const ones = D.cats.map(() => 1);

function overview() {
  const rows = D.groups.filter(g => g.id !== -1).map(g => {
    const sums = D.cats.map((_, i) => g.players.reduce((s, p) => s + (p.z[i] ?? 0), 0));
    return {g, sums, total: sums.reduce((a, b) => a + b, 0), mine: total(sums, D.weights)};
  }).sort((a, b) => b.total - a.total);
  const lim = Math.max(1, ...rows.flatMap(r => r.sums.map(Math.abs)));
  document.getElementById('overview').innerHTML =
    `<table class="heat"><thead><tr><th>Team</th><th>Total</th>${D.cats.map(c => `<th>${c}</th>`).join('')}</tr></thead><tbody>` +
    rows.map(r => `<tr class="${r.g.me ? 'me' : ''}"><td>${r.g.me ? '▶ ' : ''}${esc(r.g.name)}</td>` +
      `<td class="z">${fmt(r.total)}${r.g.me && D.weights.some(w => w !== 1) ? ` <span class="muted">(${fmt(r.mine)} yours)</span>` : ''}</td>` +
      r.sums.map(v => cell(v, lim)).join('') + '</tr>').join('') + '</tbody></table>';
}

function roster(g) {
  // my roster and free agents are judged with my weights; other teams' players with standard 9-cat
  const w = g.me || g.id === -1 ? D.weights : ones;
  const ps = g.players.map(p => ({...p, t: total(p.z, w)})).sort((a, b) => (b.t ?? -1e9) - (a.t ?? -1e9));
  document.getElementById('rosternote').textContent = g.me || g.id === -1
    ? (D.weights.some(x => x !== 1) ? 'Totals use your weights (punted categories left out).' : '')
    : `Totals count all 9 categories equally: your punt is yours, not ${g.name}'s.`;
  document.getElementById('roster').innerHTML =
    `<table class="heat"><thead><tr><th>Player</th><th>GP</th><th>Total</th>${D.cats.map((c, i) => `<th>${esc(colName(c, w[i]))}</th>`).join('')}</tr></thead><tbody>` +
    ps.map(p => `<tr><td>${esc(p.name)} <span class="muted">${esc(p.team)}${p.status ? ' · ' + esc(p.status) : ''}</span></td>` +
      `<td class="z">${p.gp}</td><td class="z">${fmt(p.t)}</td>` + p.z.map(v => cell(v, 3)).join('') + '</tr>').join('') +
    '</tbody></table>';
}

function fitTable(g) {
  const sec = document.getElementById('fitsec');
  sec.hidden = g.me;
  if (g.me) return;
  const drop = +document.getElementById('drop').value, f = D.fit[drop] || {};
  const rows = g.players.filter(p => f[p.id]).map(p => ({p, d: f[p.id].map(v => v * 100), t: total(f[p.id], D.weights)}))
    .sort((a, b) => b.t - a.t);
  const lim = Math.max(5, ...rows.flatMap(r => r.d.map(Math.abs)));
  document.getElementById('fit').innerHTML = rows.length ? (
    `<table class="heat"><thead><tr><th>Player</th><th>Fit</th>${D.cats.map((c, i) => `<th>${esc(colName(c, D.weights[i]))}</th>`).join('')}</tr></thead><tbody>` +
    rows.map(r => `<tr><td>${esc(r.p.name)}${r.p.status ? ` <span class="muted">${esc(r.p.status)}</span>` : ''}</td>` +
      `<td class="z">${fmt(r.t, 2)}</td>` + r.d.map(v => cell(v, lim, 0)).join('') + '</tr>').join('') +
    '</tbody></table>') : '<p class="muted">No players to compare.</p>';
}

const pick = document.getElementById('team'), drop = document.getElementById('drop');
const order = [...D.groups.filter(g => !g.me && g.id !== -1), ...D.groups.filter(g => g.id === -1), ...D.groups.filter(g => g.me)];
pick.innerHTML = order.map(g => `<option value="${g.id}">${esc(g.name)}${g.me ? ' (you)' : ''}</option>`).join('');
drop.innerHTML = D.drops.map(p => `<option value="${p.id}"${p.id === D.defaultDrop ? ' selected' : ''}>${esc(p.name)}${p.id === D.defaultDrop ? ' (lowest weighted total)' : ''}</option>`).join('');
document.getElementById('baseline').textContent = D.cats.map((c, i) => `${c} ${Math.round(D.baseline[i] * 100)}%`).join(', ');
const punted = D.cats.filter((_, i) => D.weights[i] === 0);
document.getElementById('weights').textContent = punted.length ? `Your punt: ${punted.join(', ')}.` : '';
let saved = null;
try { saved = localStorage.getItem('players-team'); } catch (e) {}
if (saved && order.some(g => String(g.id) === saved)) pick.value = saved;
const draw = () => {
  const g = D.groups.find(x => String(x.id) === pick.value);
  try { localStorage.setItem('players-team', pick.value); } catch (e) {}
  roster(g); fitTable(g);
};
pick.onchange = draw; drop.onchange = draw;
overview(); draw();
"""


def render(league_name: str, updated: str, data: dict, css: str, nav: str) -> str:
    payload = json.dumps(data).replace("</", "<\\/")
    me = next(g["name"] for g in data["groups"] if g["me"])
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Player contributions · {league_name}</title><style>{css}{PAGE_CSS}</style></head>
<body><main>
{nav}
<h1>Player contributions</h1>
<p class="muted">Updated {updated} · this season's games only, no projections · <span id="weights"></span></p>

<section>
  <h2>Teams at a glance</h2>
  <p class="muted">Each roster summed (per game, every rostered player). Blue = strong in that category, red = weak.
     Totals count all 9 categories for every team.</p>
  <div class="scroll" id="overview"></div>
</section>

<section>
  <h2>Players</h2>
  <div class="controls"><label>Team<select id="team"></select></label></div>
  <p class="muted">Each cell is a z-score against this season's top 200 scorers: +1 = one standard deviation better
     than average. TO is flipped (more turnovers = red); FG%/FT% weigh shooting volume. GP = games played; small GP = noisy.
     <span id="rosternote"></span></p>
  <div class="scroll" id="roster"></div>
</section>

<section id="fitsec">
  <h2>If they played for {me}</h2>
  <div class="controls"><label>…in place of my player<select id="drop"></select></label></div>
  <p class="muted">Change in <b>your</b> chance of winning each category in an average week against the rest of the
     league (percentage points), rest of season, this season's stats only. Fit = the weighted sum: extra categories
     won per week. Gains shrink where you already win (<span id="baseline"></span> now).</p>
  <div class="scroll" id="fit"></div>
</section>
<footer>The other team keeps the player in its own strength, so fit ignores how losing him would weaken them.</footer>
</main>
<script id="data" type="application/json">{payload}</script>
<script>{PAGE_JS}</script>
</body></html>"""
