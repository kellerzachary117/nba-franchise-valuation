# Usage: python src/make_dashboard.py data/panel.csv   (writes docs/index.html)
# Refits the same models as nba_models.py, embeds the results and the panel in one HTML page that draws
# interactive charts with Plotly (loaded from a CDN, so the page needs an internet connection).
import sys, os, json, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings('ignore')

df = pd.read_csv(sys.argv[1])
root = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[1])))
os.makedirs(os.path.join(root, 'docs'), exist_ok=True)

df['log_val'] = np.log(df.forbes_value); df['log_pop'] = np.log(df.msa_pop)
df['log_mult'] = np.log(df.forbes_value / df.revenue)
df['owned'] = (df.arena_owned == 'Y').astype(int)
df['winpct10'] = df.win_pct * 10
def fit(f, d): return smf.ols(f, data=d).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(d.team)[0]})
A = fit('log_val ~ log_pop + winpct10 + playoff_score + arena_age + owned + C(season)', df)
B = fit('log_val ~ winpct10 + playoff_score + arena_age + C(team) + C(season)', df)
Cm = fit('log_mult ~ log_pop + winpct10 + playoff_score + arena_age + owned + C(season)', df[df.covid_finances == 0])

def eff(m, k, base=np.e):
    b, se = m.params[k], m.bse[k]
    return [100 * (base ** v - 1) for v in (b, b - 1.96 * se, b + 1.96 * se)]
effects = []
for key, label in [('winpct10', '+10 pts of win%'), ('playoff_score', '+1 playoff round'), ('arena_age', '+1 year of arena age')]:
    for name, m in [('A: between teams', A), ('B: within teams', B), ('C: revenue multiple', Cm)]:
        if key in m.params:
            e, lo, hi = eff(m, key)
            effects.append(dict(var=label, model=name, est=e, lo=lo, hi=hi, p=float(m.pvalues[key])))
pop = []
for name, m in [('A: between teams', A), ('C: revenue multiple', Cm)]:
    e, lo, hi = eff(m, 'log_pop', base=2)
    pop.append(dict(model=name, est=e, lo=lo, hi=hi, p=float(m.pvalues['log_pop'])))

r = df.copy()
for c in ['log_val', 'log_pop', 'winpct10', 'playoff_score', 'arena_age', 'owned']: r[c + '_d'] = r[c] - r.groupby('season')[c].transform('mean')
r2 = lambda f: smf.ols(f, data=r).fit().rsquared
variance = [dict(name='Market size', r2=r2('log_val_d ~ log_pop_d - 1')),
            dict(name='Winning', r2=r2('log_val_d ~ winpct10_d + playoff_score_d - 1')),
            dict(name='Arena', r2=r2('log_val_d ~ arena_age_d + owned_d - 1')),
            dict(name='All together', r2=r2('log_val_d ~ log_pop_d + winpct10_d + playoff_score_d + arena_age_d + owned_d - 1'))]
year_share = smf.ols('log_val ~ C(season)', df).fit().rsquared

fv = df.set_index(['team', 'season']).forbes_value
raw = [('Minnesota Timberwolves', 1500, 'May 2021', 2021, 'Control'), ('Phoenix Suns', 4000, 'Dec 2022', 2023, 'Control'),
       ('Milwaukee Bucks', 3500, 'Feb 2023', 2023, 'Minority'), ('Charlotte Hornets', 3000, 'Jun 2023', 2023, 'Control'),
       ('Dallas Mavericks', 3500, 'Dec 2023', 2024, 'Control'), ('Boston Celtics', 6100, 'Mar 2025', 2025, 'Control'),
       ('Los Angeles Lakers', 10000, 'Jun 2025', 2025, 'Control'), ('Portland Trail Blazers', 4250, 'Aug 2025', 2025, 'Control')]
sales = [dict(team=t, price=p, agreed=a, list=l, type=ty, forbes=int(fv[(t, l)]), prem=p / int(fv[(t, l)]) - 1) for t, p, a, l, ty in raw]
ctrl = pd.Series([s['prem'] for s in sales if s['type'] == 'Control'])

panel = [dict(team=x.team, season=int(x.season), value=float(x.forbes_value), pop=float(x.msa_pop), win=round(float(x.win_pct), 3),
              arena=int(x.arena_age), po=int(x.playoff_score)) for x in df.itertuples()]
DATA = dict(panel=panel, effects=effects, pop=pop, variance=variance, yearShare=year_share, sales=sales,
            medPrem=float(ctrl.median()), meanPrem=float(ctrl.mean()), nCtrl=int(len(ctrl)),
            elasticity=100 * (2 ** A.params['log_pop'] - 1), seasons=[int(s) for s in sorted(df.season.unique())])

HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NBA Franchise Values</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.27.0/plotly.min.js"></script>
<style>
:root{color-scheme:light;--surface:#fcfcfb;--card:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--grid:#e4e3df;--muted:#b9b8b2;--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--surface:#1a1a19;--card:#232322;--ink:#ffffff;--ink2:#c3c2b7;--grid:#3a3a38;--muted:#6b6a64;--s1:#3987e5;--s2:#d95926;--s3:#199e70}}
:root[data-theme="dark"]{color-scheme:dark;--surface:#1a1a19;--card:#232322;--ink:#ffffff;--ink2:#c3c2b7;--grid:#3a3a38;--muted:#6b6a64;--s1:#3987e5;--s2:#d95926;--s3:#199e70}
*{box-sizing:border-box}body{margin:0;background:var(--surface);color:var(--ink);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
main{max-width:1000px;margin:0 auto;padding:28px 16px 64px}
h1{font-size:26px;margin:0 0 4px}.lede{color:var(--ink2);margin:0 0 20px}
.filters{display:flex;flex-wrap:wrap;gap:20px;align-items:end;padding:14px 16px;background:var(--card);border:1px solid var(--grid);border-radius:10px;margin-bottom:20px}
.filters label{display:block;font-size:12px;color:var(--ink2);margin-bottom:4px}
.filters select,.filters input[type=range]{font:inherit;color:var(--ink);background:var(--surface);border:1px solid var(--grid);border-radius:6px;padding:6px 8px}
.filters input[type=range]{padding:0;width:200px;accent-color:var(--s1)}
.season-out{font-weight:700;font-size:18px;min-width:3.2ch;display:inline-block}
.card{background:var(--card);border:1px solid var(--grid);border-radius:10px;padding:18px 16px 10px;margin-bottom:20px}
.card h2{font-size:18px;margin:0 0 2px}.card p.sub{color:var(--ink2);margin:0 0 8px;font-size:13.5px}
.row{display:grid;grid-template-columns:3fr 1.4fr;gap:8px}@media (max-width:720px){.row{grid-template-columns:1fr}}
.chart{width:100%}
details{margin:6px 0 10px;color:var(--ink2);font-size:13px}summary{cursor:pointer}
table{border-collapse:collapse;width:100%;margin-top:6px}th,td{text-align:left;padding:4px 8px;border-bottom:1px solid var(--grid)}td.n,th.n{text-align:right}
.note{color:var(--ink2);font-size:13px}
</style></head><body><main>
<h1>What makes an NBA franchise valuable?</h1>
<p class="lede">30 teams, 2017 to <span id="lastSeason"></span>. Forbes values against market size, winning and arenas, plus how Forbes compares with real sale prices. Hover any point or bar for details.</p>
<div class="filters">
  <div><label for="season">Season (scatter)</label><input id="season" type="range" step="1"> <span class="season-out" id="seasonOut"></span></div>
  <div><label for="team">Highlight team</label><select id="team"></select></div>
</div>

<section class="card"><h2>Bigger markets are worth more</h2>
<p class="sub" id="scatterSub"></p><div id="scatter" class="chart" style="height:480px"></div>
<details><summary>Table view</summary><div id="scatterTable"></div></details></section>

<section class="card"><h2>The whole league rose together</h2>
<p class="sub">Forbes value by team over time. Year effects alone explain <span id="yearShare"></span> of the variance in log value.</p>
<div id="lines" class="chart" style="height:460px"></div></section>

<section class="card"><h2>Market size dominates; winning barely registers</h2>
<p class="sub">Effect on value (or revenue multiple) with 95% confidence intervals. An interval crossing zero means the effect is not clearly different from nothing.</p>
<div class="row"><div id="effects" class="chart" style="height:380px"></div><div id="popfx" class="chart" style="height:380px"></div></div>
<details><summary>Table view</summary><div id="effectsTable"></div></details></section>

<section class="card"><h2>Teams sell for more than Forbes says</h2>
<p class="sub" id="salesSub"></p><div id="sales" class="chart" style="height:420px"></div>
<details><summary>Table view</summary><div id="salesTable"></div></details>
<p class="note">Only 8 sales, hand-picked in the project's sales check. Treat the median as a rough signal, not a precise estimate.</p></section>

<section class="card"><h2>Location explains most of what teams are worth</h2>
<p class="sub">Share of cross-team value spread explained (R-squared, year effects removed).</p>
<div id="variance" class="chart" style="height:300px"></div></section>
</main>
<script>
const D = __DATA__;
const $ = id => document.getElementById(id);
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const short = t => ({'Portland Trail Blazers':'Blazers','LA Clippers':'Clippers'})[t] || t.split(' ').slice(-1)[0];
const money = v => v >= 1000 ? '$' + (v/1000).toFixed(v%1000?2:0).replace(/0$/,'') + 'B' : '$' + v + 'M';
const pct = (v, d=1) => (v>=0?'+':'') + v.toFixed(d) + '%';
const cfg = {displayModeBar:false, responsive:true};
const teams = [...new Set(D.panel.map(r => r.team))].sort();
const last = D.seasons[D.seasons.length-1], first = D.seasons[0];
const byTeam = {}; D.panel.forEach(r => (byTeam[r.team] = byTeam[r.team] || []).push(r));
Object.values(byTeam).forEach(a => a.sort((x,y) => x.season - y.season));
const lastVals = teams.map(t => [t, byTeam[t][byTeam[t].length-1].value]).sort((a,b) => b[1]-a[1]);
const DEFAULTS = {'Golden State Warriors':0,'Los Angeles Lakers':1,'New York Knicks':2,[lastVals[lastVals.length-1][0]]:3};
function layout(extra){
  const L = {paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{color:css('--ink2'),size:12},margin:{l:60,r:20,t:10,b:50},
    xaxis:{gridcolor:css('--grid'),zerolinecolor:css('--grid'),linecolor:css('--grid')},yaxis:{gridcolor:css('--grid'),zerolinecolor:css('--grid'),linecolor:css('--grid')},
    hoverlabel:{bgcolor:css('--card'),bordercolor:css('--grid'),font:{color:css('--ink')}},showlegend:false};
  for (const k in extra) L[k] = (typeof extra[k]==='object' && !Array.isArray(extra[k]) && L[k]) ? Object.assign({}, L[k], extra[k]) : extra[k];
  return L;
}
function table(el, head, rows){
  const t = document.createElement('table'), h = t.createTHead().insertRow();
  head.forEach((x,i) => { const c = document.createElement('th'); c.textContent = x; if(i) c.className='n'; h.appendChild(c); });
  const b = t.createTBody();
  rows.forEach(r => { const tr = b.insertRow(); r.forEach((x,i) => { const c = tr.insertCell(); c.textContent = x; if(i) c.className='n'; }); });
  el.replaceChildren(t);
}
function olsLine(xs, ys){
  const n = xs.length, mx = xs.reduce((a,b)=>a+b)/n, my = ys.reduce((a,b)=>a+b)/n;
  let sxy=0, sxx=0; for (let i=0;i<n;i++){ sxy += (xs[i]-mx)*(ys[i]-my); sxx += (xs[i]-mx)**2; }
  const b = sxy/sxx; return {b, a: my - b*mx};
}

function renderScatter(){
  const s = +$('season').value, hl = $('team').value, rows = D.panel.filter(r => r.season === s);
  const ln = olsLine(rows.map(r=>Math.log(r.pop)), rows.map(r=>Math.log(r.value)));
  const x0 = Math.min(...rows.map(r=>r.pop)), x1 = Math.max(...rows.map(r=>r.pop));
  const isHl = r => hl !== 'default' && r.team === hl;
  const pts = (list, color, size, labels) => ({type:'scatter', mode: labels?'markers+text':'markers', x:list.map(r=>r.pop), y:list.map(r=>r.value),
    text:list.map(r=>short(r.team)), textposition:'top right', textfont:{size:10,color:css('--ink2')},
    marker:{color, size, line:{color:css('--card'), width:2}},
    customdata:list.map(r=>[r.team, (r.win*100).toFixed(1), r.arena, r.po]),
    hovertemplate:'<b>%{customdata[0]}</b><br>Value: $%{y:,.0f}M<br>Metro pop: %{x:,.0f}<br>Win%: %{customdata[1]}<br>Arena age: %{customdata[2]} yrs<extra></extra>'});
  const traces = [
    {type:'scatter', mode:'lines', x:[x0,x1], y:[Math.exp(ln.a+ln.b*Math.log(x0)), Math.exp(ln.a+ln.b*Math.log(x1))], line:{color:css('--muted'),width:2}, hoverinfo:'skip'},
    pts(rows.filter(r=>!isHl(r)), css('--s1'), 11, false), pts(rows.filter(isHl), css('--s2'), 15, true)];
  if (hl === 'default') traces.push(pts(rows.filter(r => r.team in DEFAULTS), css('--s1'), 11, true));
  Plotly.react('scatter', traces, layout({margin:{l:70,r:20,t:10,b:55},
    xaxis:{type:'log',title:{text:'Metro population (log scale)'},tickvals:[1e6,2e6,5e6,1e7,2e7],ticktext:['1M','2M','5M','10M','20M']},
    yaxis:{type:'log',title:{text:'Forbes value (log scale)'},tickvals:[500,1000,2000,4000,8000,12000],ticktext:['$0.5B','$1B','$2B','$4B','$8B','$12B']}}), cfg);
  $('scatterSub').textContent = 'Season ' + s + '. A doubling of metro population goes with about ' + pct(D.elasticity,0) + ' value across all seasons (Model A). The grey line is this season alone.';
  table($('scatterTable'), ['Team','Value ($M)','Metro pop','Win %','Arena age'], rows.slice().sort((a,b)=>b.value-a.value).map(r=>[r.team, r.value.toLocaleString(), r.pop.toLocaleString(), (r.win*100).toFixed(1), r.arena]));
  $('seasonOut').textContent = s;
}
function renderLines(){
  const hl = $('team').value, s = +$('season').value, cols = [css('--s1'), css('--s2'), css('--s3'), css('--ink2')];
  const picked = hl === 'default' ? Object.keys(DEFAULTS) : [hl];
  const trace = (t, color, w) => ({type:'scatter', mode:'lines', x:byTeam[t].map(r=>r.season), y:byTeam[t].map(r=>r.value), name:short(t),
    line:{color, width:w}, hovertemplate:'<b>'+t+'</b><br>%{x}: $%{y:,.0f}M<extra></extra>'});
  const grey = teams.filter(t => !picked.includes(t)).map(t => trace(t, css('--muted'), 1));
  const hi = picked.map((t,i) => trace(t, hl==='default' ? cols[DEFAULTS[t]] : css('--s2'), 3));
  const ends = picked.map(t => ({t, y: byTeam[t][byTeam[t].length-1].value})).sort((a,b) => a.y - b.y);
  for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i-1].y < 550) ends[i].y = ends[i-1].y + 550;
  const labels = {type:'scatter', mode:'text', x:ends.map(()=>last), y:ends.map(e=>e.y), text:ends.map(e=>short(e.t)), textposition:'middle right', textfont:{color:css('--ink'),size:12}, hoverinfo:'skip'};
  Plotly.react('lines', [...grey, ...hi, labels], layout({margin:{l:60,r:80,t:10,b:40},
    xaxis:{dtick:1,range:[first-0.2,last+0.9]}, yaxis:{title:{text:'Forbes value'},tickvals:[2000,4000,6000,8000,10000],ticktext:['$2B','$4B','$6B','$8B','$10B']},
    shapes:[{type:'line',x0:s,x1:s,yref:'paper',y0:0,y1:1,line:{color:css('--ink2'),width:1,dash:'dot'}}], hovermode:'closest'}), cfg);
}
function renderEffects(){
  const vars = [...new Set(D.effects.map(e=>e.var))], models = [...new Set(D.effects.map(e=>e.model))], cols = [css('--s1'), css('--s2'), css('--s3')];
  const mk = (rows, ycat, off, color, name, showlegend) => ({type:'scatter', mode:'markers', name, showlegend,
    x:rows.map(e=>e.est), y:rows.map(e=>ycat(e)+off), error_x:{type:'data',symmetric:false,array:rows.map(e=>e.hi-e.est),arrayminus:rows.map(e=>e.est-e.lo),color,thickness:2.5,width:0},
    marker:{color,size:11,line:{color:css('--card'),width:2}},
    customdata:rows.map(e=>[e.model,e.lo.toFixed(2),e.hi.toFixed(2),e.p<0.001?'<0.001':e.p.toFixed(3)]),
    hovertemplate:'<b>%{customdata[0]}</b><br>Effect: %{x:+.2f}%<br>95% CI: %{customdata[1]}% to %{customdata[2]}%<br>p-value: %{customdata[3]}<extra></extra>'});
  const tr = models.map((m,i) => mk(D.effects.filter(e=>e.model===m), e=>vars.indexOf(e.var), (1-i)*0.22, cols[i], m, true));
  Plotly.react('effects', tr, layout({showlegend:true, legend:{orientation:'h',y:-0.22,font:{color:css('--ink2')}}, margin:{l:140,r:10,t:10,b:90},
    xaxis:{title:{text:'Change in value / multiple (%)'},zeroline:true,zerolinecolor:css('--ink2')},
    yaxis:{tickvals:vars.map((_,i)=>i),ticktext:vars,autorange:'reversed',zeroline:false,gridcolor:'rgba(0,0,0,0)'}}), cfg);
  const pm = D.pop.map((e,i)=>mk([e], ()=>i===0?0:1, 0, i===0?cols[0]:cols[2], e.model, false));
  Plotly.react('popfx', pm, layout({margin:{l:80,r:10,t:10,b:90},
    xaxis:{title:{text:'Doubling metro pop (%)'},rangemode:'tozero',zeroline:true,zerolinecolor:css('--ink2')},
    yaxis:{tickvals:[0,1],ticktext:['Model A','Model C'],range:[1.5,-0.5],zeroline:false,gridcolor:'rgba(0,0,0,0)'}}), cfg);
  table($('effectsTable'), ['Variable','Model','Effect %','95% CI low','95% CI high','p'],
    [...D.effects.map(e=>[e.var,e.model,e.est.toFixed(2),e.lo.toFixed(2),e.hi.toFixed(2),e.p.toFixed(3)]), ...D.pop.map(e=>['Doubling metro pop',e.model,e.est.toFixed(2),e.lo.toFixed(2),e.hi.toFixed(2),e.p.toFixed(3)])]);
}
function renderSales(){
  const S = D.sales, lbl = S.map(s => short(s.team) + (s.type==='Minority' ? ' (minority)' : ''));
  const bar = (name, ys, color, extra) => Object.assign({type:'bar', name, x:lbl, y:ys.map(v=>v/1000), marker:{color,line:{color:css('--card'),width:2}}, customdata:S.map(s=>[s.team,s.agreed,s.list,s.type,s.forbes,s.price,(s.prem*100).toFixed(0)])}, extra);
  const hover = '<b>%{customdata[0]}</b> (%{customdata[3]})<br>Agreed: %{customdata[1]}<br>Forbes %{customdata[2]} list: $%{customdata[4]:,}M<br>Sale price: $%{customdata[5]:,}M<br>Premium: %{customdata[6]}%<extra></extra>';
  Plotly.react('sales', [bar('Forbes list value', S.map(s=>s.forbes), css('--s1'), {hovertemplate:hover}),
    bar('Actual sale price', S.map(s=>s.price), css('--s2'), {hovertemplate:hover, text:S.map(s=>pct(s.prem*100,0)), textposition:'outside', textfont:{color:css('--ink')}, cliponaxis:false})],
    layout({barmode:'group',bargap:0.2,bargroupgap:0.04,showlegend:true,legend:{orientation:'h',y:1.1,font:{color:css('--ink2')}},margin:{l:60,r:10,t:30,b:60},
    yaxis:{title:{text:'$ billions'},ticksuffix:'B',tickprefix:'$'},xaxis:{gridcolor:'rgba(0,0,0,0)'}}), cfg);
  $('salesSub').textContent = 'Median control-sale premium ' + pct(D.medPrem*100,0) + ' (mean ' + pct(D.meanPrem*100,0) + ', n=' + D.nCtrl + '). Percent labels are sale price versus the Forbes list in force.';
  table($('salesTable'), ['Team','Type','Agreed','Forbes list ($M)','Sale price ($M)','Premium'], S.map(s=>[s.team,s.type,s.agreed,s.forbes.toLocaleString(),s.price.toLocaleString(),pct(s.prem*100,0)]));
}
function renderVariance(){
  const V = D.variance;
  Plotly.react('variance', [{type:'bar', orientation:'h', x:V.map(v=>v.r2), y:V.map(v=>v.name), text:V.map(v=>(v.r2*100).toFixed(0)+'%'), textposition:'outside', cliponaxis:false,
    marker:{color:V.map(v=>v.name==='All together'?css('--ink2'):css('--s1')),line:{color:css('--card'),width:2}}, textfont:{color:css('--ink')},
    hovertemplate:'<b>%{y}</b><br>Explains %{x:.1%} of the spread<extra></extra>'}],
    layout({margin:{l:110,r:40,t:10,b:50},xaxis:{range:[0,0.7],tickformat:'.0%',title:{text:'Share of cross-team value spread explained'}},yaxis:{autorange:'reversed',gridcolor:'rgba(0,0,0,0)'}}), cfg);
}
function renderAll(){ renderScatter(); renderLines(); renderEffects(); renderSales(); renderVariance(); }

$('lastSeason').textContent = last; $('yearShare').textContent = (D.yearShare*100).toFixed(0) + '%';
const sl = $('season'); sl.min = first; sl.max = last; sl.value = last;
const sel = $('team'); [['default','Default (Warriors, Lakers, Knicks, lowest)'], ...teams.map(t=>[t,t])].forEach(([v,l]) => { const o = document.createElement('option'); o.value = v; o.textContent = l; sel.appendChild(o); });
sl.addEventListener('input', () => { renderScatter(); renderLines(); }); sel.addEventListener('change', () => { renderScatter(); renderLines(); });
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', renderAll);
renderAll();
</script></body></html>'''

out = os.path.join(root, 'docs', 'index.html')
open(out, 'w').write(HTML.replace('__DATA__', json.dumps(DATA)))
print('wrote', out, f'({os.path.getsize(out)/1024:.0f} KB)')
