"""Static "Trade lab" page for GitHub Pages: build a hypothetical version of my roster (add anyone, remove my
players, equal counts) and see the resulting team's category win rates vs the league, category ranks and
heatmap.

Uses the fit model (`fantasy/fit.py`): a team's week is the sum of its players' weekly (mean, var, cov), so
the page re-sums exported per-player moments for any roster and scores it with the same normal
approximation in JS. Only my team changes; every other team stays as it is.
"""
import json

import numpy as np
import pandas as pd

from .config import CATEGORIES, DEFAULT_CATEGORY_WEIGHTS
from .fit import FitModel
from .players import contributions, ordered_categories
from .trades import IDX

LOW_GP = 8


def _num(v, nd=2):
    return None if v is None or pd.isna(v) else round(float(v), nd)


def _cat_defs(model) -> list[dict]:
    """Categories in model order, with how to compute each from the 11 summed stats."""
    out = []
    for c in model.cats:
        d = {"name": c["name"], "lower": bool(c.get("lower_is_better"))}
        if "ratio" in c:
            made, att = c["ratio"]
            d.update(made=IDX[made], att=IDX[att], k=0 if made == "FGM" else 1)
        else:
            d["stat"] = IDX[c["name"]]
        out.append(d)
    return out


def scenario(model: FitModel, me: int, add: list[int], drop: list[int]) -> dict:
    """Reference (Python) result of a roster change: per-category mean win prob vs every other team and my
    category means, the same numbers the page computes."""
    others = [t for t in model.teams if t != me]
    base = model._base()
    o_mu = np.stack([base[o][0] for o in others])
    o_sd = np.stack([base[o][1] for o in others])
    roster = [p for p in model.roster[me] if p not in drop] + list(add)
    mu, sd = model.cat_dists(*model.team_moments(roster, model.start_factor[me]))
    return {"p": model.win_probs(mu, sd, o_mu, o_sd).mean(0), "mu": mu}


def build_data(sim, snap, me_id: int) -> dict:
    model = FitModel(sim)
    f = model.start_factor[me_id]
    shown = ordered_categories(sim.cat_ids)
    defs = _cat_defs(model)
    names = [d["name"] for d in defs]
    weights = {c: DEFAULT_CATEGORY_WEIGHTS.get(c, 1.0) for c in names}
    teams = snap["teams"].set_index("team_id")["name"].to_dict()

    rosters = snap["rosters"][["player_id", "lineup_slot", "team_id"]]
    taken = set(rosters["player_id"])
    fa = sim.players[~sim.players["player_id"].isin(taken) & (sim.players["season_gp"].fillna(0) > 0)]
    pool = pd.concat([rosters, pd.DataFrame({"player_id": fa["player_id"].to_numpy(), "lineup_slot": 0, "team_id": -1})])
    pool = pool[pool["player_id"].isin(model.mean)]
    fantasy_team = dict(zip(pool["player_id"], pool["team_id"]))
    z = contributions(sim, pool[["player_id", "lineup_slot"]], lineup=False)

    players = []
    for _, r in z.iterrows():
        pid = int(r["player_id"])
        m, v, c = model.contrib(pid, f)
        tid = int(fantasy_team[pid])
        players.append({
            "id": pid, "name": r["name"], "nba": r["pro_team"], "team": teams.get(tid, "FA"), "mine": tid == me_id,
            "status": "" if r["injury_status"] == "ACTIVE" else r["injury_status"].replace("_", " ").lower(),
            "gp": int(r["gp"]), "z": [_num(r[c]) for c in shown],
            "m": [round(float(x), 4) for x in m], "v": [round(float(x), 4) for x in v],
            "c": [round(float(x), 4) for x in c],
        })

    base = model._base()
    others = [t for t in model.teams if t != me_id]
    return {
        "cats": defs, "shown": [names.index(c) for c in shown], "weights": [weights[c] for c in names],
        "players": players, "roster": [int(p) for p in model.roster[me_id] if p in model.mean],
        "opp": [{"name": teams[o], "mu": [float(x) for x in base[o][0]], "sd": [float(x) for x in base[o][1]]}
                for o in others],
        "me": teams[me_id], "lowGp": LOW_GP,
    }


# Pure maths (no DOM, so it can be checked under node against `scenario`): a roster's category distributions and
# mean win probability vs every other team (ported from TradeModel.cat_dists / win_probs / _phi).
LAB_MATH = r"""
function phi(x) {
  const z = Math.abs(x) / Math.SQRT2, t = 1 / (1 + 0.3275911 * z);
  const poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
  const erf = 1 - poly * Math.exp(-z * z);
  return 0.5 * (1 + Math.sign(x) * erf);
}
function teamResult(D, ids) {
  const byId = D.byId, m = Array(11).fill(0), v = Array(11).fill(0), c = [0, 0];
  for (const id of ids) {
    const p = byId[id];
    if (!p) continue;
    p.m.forEach((x, i) => m[i] += x); p.v.forEach((x, i) => v[i] += x); c[0] += p.c[0]; c[1] += p.c[1];
  }
  const mu = [], sd = [];
  for (const cat of D.cats) {
    if (cat.made !== undefined) {
      const M = m[cat.made], A = Math.max(m[cat.att], 1e-9), r = M / A;
      const vv = (v[cat.made] - 2 * r * c[cat.k] + r * r * v[cat.att]) / (A * A);
      mu.push(r); sd.push(Math.sqrt(Math.max(vv, 1e-12)));
    } else {
      mu.push(m[cat.stat]); sd.push(Math.sqrt(Math.max(v[cat.stat], 1e-9)));
    }
  }
  const p = D.cats.map((cat, i) => {
    let s = 0;
    for (const o of D.opp) {
      const zz = (mu[i] - o.mu[i]) / Math.sqrt(sd[i] ** 2 + o.sd[i] ** 2);
      s += cat.lower ? 1 - phi(zz) : phi(zz);
    }
    return s / D.opp.length;
  });
  const rank = D.cats.map((cat, i) => 1 + D.opp.filter(o => cat.lower ? o.mu[i] < mu[i] : o.mu[i] > mu[i]).length);
  return {mu, p, rank, exp: p.reduce((s, x, i) => s + x * D.weights[i], 0)};
}
"""

PAGE_CSS = """
.heat td.z{text-align:center;font-variant-numeric:tabular-nums;min-width:52px}.heat th{text-align:center}
.heat th:first-child,.heat td:first-child{text-align:left}.heat tr.add td:first-child{font-weight:600}
.heat tr.sum td{font-weight:600;border-top:2px solid var(--line)}
.controls{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:12px 0}
input,button.b{font:inherit;padding:7px 10px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--ink)}
input{flex:1;min-width:220px;max-width:460px}button.b{cursor:pointer}button.b.primary{background:var(--ink);color:var(--bg);border-color:var(--ink)}
.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px;align-items:start}
.list{list-style:none;padding:0;margin:0;background:var(--card);border:1px solid var(--line);border-radius:10px}
.list li{display:flex;gap:8px;align-items:center;padding:6px 10px;border-bottom:1px solid var(--line)}
.list li:last-child{border-bottom:0}.list li .n{flex:1;min-width:0}.list li.out .n{text-decoration:line-through;color:var(--muted)}
.list li button{font:inherit;font-size:13px;padding:3px 10px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--ink2);cursor:pointer}
.list li.empty{color:var(--muted)}.warn{color:var(--swing)}.balance{margin:12px 0;font-weight:600}
td.up{color:var(--win)}td.down{color:var(--loss)}
"""

PAGE_JS = r"""
const D = JSON.parse(document.getElementById('data').textContent);
D.byId = Object.fromEntries(D.players.map(p => [p.id, p]));
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const sign = (v, d=1) => (Math.abs(v) < 0.5 * 10 ** -d ? 0 : v).toFixed(d).replace(/^(?!-)/, '+');
const fmtZ = v => v == null ? '–' : sign(v);
const cell = (v, lim) => {
  if (v == null) return '<td class="z muted">–</td>';
  const share = Math.min(Math.abs(v) / lim, 1) * 70, pole = v >= 0 ? '--pw' : '--pl';
  return `<td class="z" style="background:color-mix(in oklab,var(${pole}) ${share.toFixed(0)}%,var(--card))">${fmtZ(v)}</td>`;
};
const label = p => `${p.name} · ${p.nba} · ${p.team}`;
const catVal = (i, x) => D.cats[i].made !== undefined ? (x * 100).toFixed(1) + '%' : x.toFixed(1);

let adds = [], drops = [];
const input = document.getElementById('who'), msg = document.getElementById('addmsg');
const byLabel = Object.fromEntries(D.players.filter(p => !p.mine).map(p => [label(p), p]));
document.getElementById('pl').innerHTML = D.players.filter(p => !p.mine)
  .sort((a, b) => a.name.localeCompare(b.name)).map(p => `<option value="${esc(label(p))}">`).join('');

function readHash() {
  const h = new URLSearchParams(location.hash.slice(1)), ids = k => (h.get(k) || '').split(',').filter(Boolean).map(Number);
  adds = ids('add').filter(id => D.byId[id] && !D.byId[id].mine);
  drops = ids('drop').filter(id => D.roster.includes(id));
}
function writeHash() {
  const h = adds.length || drops.length ? `#add=${adds.join(',')}&drop=${drops.join(',')}` : location.pathname + location.search;
  history.replaceState(null, '', h);
}
function add() {
  const v = input.value.trim(), p = byLabel[v] || D.players.find(x => !x.mine && x.name.toLowerCase() === v.toLowerCase());
  if (!p) { msg.textContent = v ? `No player matches “${v}”. Pick one from the list.` : ''; return; }
  if (!adds.includes(p.id)) adds.push(p.id);
  input.value = ''; msg.textContent = ''; draw();
}
document.getElementById('add').onclick = add;
input.onkeydown = e => { if (e.key === 'Enter') add(); };
document.getElementById('reset').onclick = () => { adds = []; drops = []; draw(); };

function lists() {
  const flag = p => (p.gp < D.lowGp ? ` <span class="warn" title="Few games: noisy">${p.gp} GP</span>` : ` <span class="muted">${p.gp} GP</span>`)
    + (p.status ? ` <span class="muted">· ${esc(p.status)}</span>` : '');
  document.getElementById('mine').innerHTML = D.roster.map(id => D.byId[id]).map(p => {
    const out = drops.includes(p.id);
    return `<li class="${out ? 'out' : ''}"><span class="n">${esc(p.name)} <span class="muted">${esc(p.nba)}</span>${flag(p)}</span>` +
      `<button data-drop="${p.id}">${out ? 'Undo' : 'Remove'}</button></li>`;
  }).join('');
  document.getElementById('added').innerHTML = adds.length ? adds.map(id => D.byId[id]).map(p =>
    `<li><span class="n">${esc(p.name)} <span class="muted">${esc(p.nba)} · from ${esc(p.team)}</span>${flag(p)}</span>` +
    `<button data-undo="${p.id}" aria-label="Take ${esc(p.name)} back out">✕</button></li>`).join('')
    : '<li class="empty">Nobody yet: type a name above.</li>';
  document.querySelectorAll('[data-drop]').forEach(b => b.onclick = () => {
    const id = +b.dataset.drop; drops = drops.includes(id) ? drops.filter(x => x !== id) : [...drops, id]; draw();
  });
  document.querySelectorAll('[data-undo]').forEach(b => b.onclick = () => { adds = adds.filter(x => x !== +b.dataset.undo); draw(); });
}

function results() {
  const bal = document.getElementById('balance'), res = document.getElementById('results'), diff = adds.length - drops.length;
  if (!adds.length && !drops.length) { bal.textContent = 'Add a player and remove one of yours to see the new team.'; res.hidden = true; return; }
  if (diff) {
    bal.innerHTML = `<span class="warn">${diff > 0 ? `Remove ${diff} more of your players` : `Add ${-diff} more player${diff < -1 ? 's' : ''}`}` +
      ` to balance the roster (${adds.length} in, ${drops.length} out).</span>`;
    res.hidden = true; return;
  }
  bal.textContent = `${adds.length} in, ${drops.length} out.`; res.hidden = false;
  const now = teamResult(D, D.roster), ids = [...D.roster.filter(id => !drops.includes(id)), ...adds], next = teamResult(D, ids);
  const punted = D.cats.filter((_, i) => D.weights[i] === 0).map(c => c.name);
  const d = next.exp - now.exp;
  document.getElementById('kpis').innerHTML =
    `<div><span>Expected categories / week now</span><b>${now.exp.toFixed(2)}</b></div>` +
    `<div><span>With these changes</span><b>${next.exp.toFixed(2)}</b></div>` +
    `<div><span>Change</span><b class="${d >= 0 ? 'up' : 'down'}">${sign(d, 2)}</b>` +
    `<small class="muted">${punted.length ? 'not counting ' + punted.join(', ') : ''}</small></div>`;
  document.getElementById('cats').innerHTML =
    '<thead><tr><th>Cat</th><th>Per week now</th><th>After</th><th>Win % now</th><th>After</th><th>Δ</th><th>Rank now</th><th>After</th></tr></thead><tbody>' +
    D.shown.map(i => {
      const dp = (next.p[i] - now.p[i]) * 100, dr = now.rank[i] - next.rank[i];
      const cls = Math.abs(dp) < 0.5 ? '' : dp > 0 ? 'up' : 'down';
      return `<tr><td>${esc(D.cats[i].name)}${D.weights[i] === 0 ? ' <span class="muted">(punted)</span>' : ''}${D.cats[i].lower ? ' <span class="muted">lower wins</span>' : ''}</td>` +
        `<td>${catVal(i, now.mu[i])}</td><td>${catVal(i, next.mu[i])}</td>` +
        `<td>${Math.round(now.p[i] * 100)}%</td><td>${Math.round(next.p[i] * 100)}%</td><td class="${cls}">${sign(dp, 0)}</td>` +
        `<td>${now.rank[i]}</td><td class="${dr > 0 ? 'up' : dr < 0 ? 'down' : ''}">${next.rank[i]}${dr ? ` (${dr > 0 ? '↑' : '↓'}${Math.abs(dr)})` : ''}</td></tr>`;
    }).join('') + '</tbody>';

  const ps = ids.map(id => D.byId[id]), tot = (list, w) => list.reduce((s, p) => s + p.z.reduce((a, v, j) => a + (v ?? 0) * w[j], 0), 0);
  const wShown = D.shown.map(i => D.weights[i]);
  const sums = list => D.shown.map((_, j) => list.reduce((s, p) => s + (p.z[j] ?? 0), 0));
  const before = sums(D.roster.map(id => D.byId[id])), after = sums(ps);
  const lim = Math.max(1, ...[...before, ...after].map(Math.abs));
  const rows = ps.map(p => ({p, t: p.z.every(v => v == null) ? null : tot([p], wShown)})).sort((a, b) => (b.t ?? -1e9) - (a.t ?? -1e9));
  const total = s => s.reduce((a, v, j) => a + v * wShown[j], 0);
  document.getElementById('heat').innerHTML =
    `<thead><tr><th>Player</th><th>GP</th><th>Total</th>${D.shown.map(i => `<th>${esc(D.cats[i].name)}${D.weights[i] === 0 ? ' (punted)' : ''}</th>`).join('')}</tr></thead><tbody>` +
    rows.map(({p, t}) => `<tr class="${adds.includes(p.id) ? 'add' : ''}"><td>${adds.includes(p.id) ? '＋ ' : ''}${esc(p.name)} <span class="muted">${esc(p.nba)}${p.status ? ' · ' + esc(p.status) : ''}</span></td>` +
      `<td class="z${p.gp < D.lowGp ? ' warn' : ''}">${p.gp}</td><td class="z">${fmtZ(t)}</td>${p.z.map(v => cell(v, 3)).join('')}</tr>`).join('') +
    `<tr class="sum"><td>Team now</td><td></td><td class="z">${fmtZ(total(before))}</td>${before.map(v => cell(v, lim)).join('')}</tr>` +
    `<tr class="sum"><td>Team after</td><td></td><td class="z">${fmtZ(total(after))}</td>${after.map(v => cell(v, lim)).join('')}</tr>` +
    `<tr class="sum"><td>Change</td><td></td><td class="z">${fmtZ(total(after) - total(before))}</td>${after.map((v, j) => cell(v - before[j], lim)).join('')}</tr></tbody>`;
}

function draw() { lists(); results(); writeHash(); }
readHash(); draw();
window.onhashchange = () => { readHash(); lists(); results(); };
"""


def render(league_name: str, updated: str, data: dict, css: str, nav: str) -> str:
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Trade lab · {league_name}</title><style>{css}{PAGE_CSS}</style></head>
<body><main>
{nav}
<h1>Trade lab</h1>
<p class="muted">Updated {updated} · build a hypothetical {data['me']} and see how it would do · this season's games only</p>

<section>
  <h2>Build the team</h2>
  <div class="controls">
    <input id="who" list="pl" placeholder="Type a player's name…" autocomplete="off" aria-label="Player to add">
    <datalist id="pl"></datalist>
    <button class="b primary" id="add">Add</button><button class="b" id="reset">Reset</button>
    <span class="muted" id="addmsg"></span>
  </div>
  <div class="cols">
    <div><h3>My roster</h3><ul class="list" id="mine"></ul></div>
    <div><h3>Adding</h3><ul class="list" id="added"></ul></div>
  </div>
  <p class="balance" id="balance"></p>
</section>

<section id="results" hidden>
  <h2>Resulting team</h2>
  <div class="kpis" id="kpis"></div>
  <h3>Categories vs the league</h3>
  <p class="muted">Per week = expected weekly total (FG%/FT% from makes and attempts). Win % = chance of winning the
     category in an average week against each other team. Rank = where that weekly total would place among all
     {len(data['opp']) + 1} teams.</p>
  <div class="scroll"><table id="cats"></table></div>
  <h3>Roster heatmap</h3>
  <p class="muted">Per-game z-scores vs this season's top 200 scorers (blue helps, red hurts; TO flipped). Added
     players marked ＋. GP in amber = fewer than {data['lowGp']} games, so the numbers are noisy.</p>
  <div class="scroll"><table class="heat" id="heat"></table></div>
</section>
<footer>Only {data['me']} changes; other teams keep their current rosters, so a trade partner's loss isn't modelled.
Players' per-game averages from this season carry forward unchanged, with rest-of-season games per week and your
current lineup start rate. The scenario is saved in the link, so you can bookmark or share it.</footer>
</main>
<script id="data" type="application/json">{payload}</script>
<script>{LAB_MATH}{PAGE_JS}</script>
</body></html>"""
