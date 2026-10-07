# Usage: python src/make_charts.py data/panel.csv   (writes PNGs to results/charts/)
# Refits the same models as nba_models.py and draws five charts from them.
import sys, os, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
warnings.filterwarnings('ignore')

SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
BLUE, ORANGE, AQUA, MUTED = '#2a78d6', '#eb6834', '#1baf7a', '#b9b8b2'
plt.rcParams.update({'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
    'text.color': INK, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.axisbelow': True,
    'font.size': 11, 'axes.titlesize': 14, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left'})

df = pd.read_csv(sys.argv[1])
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[1]))), 'results', 'charts')
os.makedirs(out, exist_ok=True)

df['log_val'] = np.log(df.forbes_value); df['log_pop'] = np.log(df.msa_pop)
df['log_mult'] = np.log(df.forbes_value / df.revenue)
df['owned'] = (df.arena_owned == 'Y').astype(int)
df['winpct10'] = df.win_pct * 10
def fit(f, d): return smf.ols(f, data=d).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(d.team)[0]})
A = fit('log_val ~ log_pop + winpct10 + playoff_score + arena_age + owned + C(season)', df)
B = fit('log_val ~ winpct10 + playoff_score + arena_age + C(team) + C(season)', df)
Cm = fit('log_mult ~ log_pop + winpct10 + playoff_score + arena_age + owned + C(season)', df[df.covid_finances == 0])
short = lambda t: t.split()[-1] if t not in ('Portland Trail Blazers', 'LA Clippers') else {'Portland Trail Blazers': 'Blazers', 'LA Clippers': 'Clippers'}[t]
def title(ax, t, sub):
    ax.set_title(t, pad=26); ax.text(0, 1.02, sub, transform=ax.transAxes, color=INK2, fontsize=10, va='bottom')
def save(fig, name): fig.tight_layout(); fig.savefig(os.path.join(out, name), dpi=160); plt.close(fig)

# 1. value vs metro population, team averages with season effects removed
r = df.copy()
for c in ['log_val', 'log_pop']: r[c + '_d'] = r[c] - r.groupby('season')[c].transform('mean')
t = r.groupby('team')[['log_val_d', 'log_pop_d']].mean()
fig, ax = plt.subplots(figsize=(10, 6.5))
x, y = t.log_pop_d / np.log(2), t.log_val_d
ax.scatter(x, y, s=70, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
b = np.polyfit(x, y, 1); xs = np.array([x.min(), x.max()]); ax.plot(xs, np.polyval(b, xs), color=MUTED, lw=2, zorder=2)
nudge = {'Sacramento Kings': (-30, 6), 'San Antonio Spurs': (6, 8), 'Portland Trail Blazers': (8, -12), 'Utah Jazz': (-26, 6),
         'Oklahoma City Thunder': (-14, -14), 'Milwaukee Bucks': (6, 8), 'Orlando Magic': (8, 4), 'Charlotte Hornets': (-44, -12)}
for team, xi, yi in zip(t.index, x, y): ax.annotate(short(team), (xi, yi), xytext=nudge.get(team, (6, 4)), textcoords='offset points', fontsize=8.5, color=INK2)
ax.set_xlabel('Metro population vs league-typical team (each step = doubling)')
ax.set_ylabel('Franchise value vs season average')
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{2**v:.1f}x'))
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{np.exp(v):.1f}x'))
title(ax, 'Bigger markets are worth more', f'Team averages, 2017-{df.season.max()}, season effects removed. Doubling metro population ~ +{100*(2**A.params["log_pop"]-1):.0f}% value (Model A)')
save(fig, '1_value_vs_market.png')

# 2. value over time, all teams, a few highlighted
piv = df.pivot(index='season', columns='team', values='forbes_value')
last = piv.iloc[-1]; hi = ['New York Knicks', 'Los Angeles Lakers', 'Golden State Warriors', last.idxmin()]
cols = dict(zip(hi, [BLUE, ORANGE, AQUA, INK2]))
fig, ax = plt.subplots(figsize=(10, 6.5))
for team in piv.columns:
    if team not in hi: ax.plot(piv.index, piv[team] / 1000, color=MUTED, lw=1, alpha=0.7, zorder=1)
for team in hi:
    ax.plot(piv.index, piv[team] / 1000, color=cols[team], lw=2.2, zorder=3)
    ax.annotate(short(team), (piv.index[-1], piv[team].iloc[-1] / 1000), xytext=(6, {'New York Knicks': -7, 'Los Angeles Lakers': 7}.get(team, 0)), textcoords='offset points', va='center', fontsize=10, color=INK)
ax.set_xlim(piv.index.min(), piv.index.max() + 1.2); ax.set_xticks(piv.index)
ax.set_ylabel('Forbes franchise value ($B)'); ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'${v:.0f}B'))
share = smf.ols('log_val ~ C(season)', df).fit().rsquared
title(ax, 'The whole league rose together', f'Forbes value by team. Year effects alone explain {100*share:.0f}% of the variance in log value')
save(fig, '2_value_over_time.png')

# 3. coefficient plot: % change with 95% CI
def eff(m, k): b, se = m.params[k], m.bse[k]; return [100*(np.exp(v)-1) for v in (b, b-1.96*se, b+1.96*se)]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5.5), gridspec_kw={'width_ratios': [3, 1.3]})
models = [('A: between teams', A, BLUE), ('B: within teams', B, ORANGE), ('C: revenue multiple', Cm, AQUA)]
vars1 = [('winpct10', '+10 pts of win%'), ('playoff_score', '+1 playoff round'), ('arena_age', '+1 year of arena age')]
for i, (lab, m, c) in enumerate(models):
    ys, es, lo, hi_ = [], [], [], []
    for j, (k, _) in enumerate(vars1):
        if k in m.params:
            e, l, h = eff(m, k); a1.errorbar(e, j + (1 - i) * 0.22, xerr=[[e - l], [h - e]], fmt='o', color=c, ms=8, mec=SURFACE, mew=2, lw=2, capsize=0, label=lab if j == 0 else None)
    if 'log_pop' in m.params and i != 1:
        b_, se = m.params['log_pop'], m.bse['log_pop']; f = lambda v: 100*(2**v-1)
        a2.errorbar(f(b_), 0 if i == 0 else -0.22, xerr=[[f(b_) - f(b_-1.96*se)], [f(b_+1.96*se) - f(b_)]], fmt='o', color=c, ms=8, mec=SURFACE, mew=2, lw=2)
a1.axvline(0, color=INK2, lw=1); a1.set_yticks(range(len(vars1))); a1.set_yticklabels([v[1] for v in vars1]); a1.invert_yaxis()
a1.set_xlabel('Change in value / multiple (%), 95% CI'); a1.grid(axis='y', visible=False)
a1.legend(frameon=False, loc='lower right', fontsize=9)
a2.axvline(0, color=INK2, lw=1); a2.set_yticks([0, -0.22]); a2.set_yticklabels(['Model A', 'Model C']); a2.grid(axis='y', visible=False)
a2.set_xlabel('Doubling metro pop (%), 95% CI'); a2.set_ylim(-0.6, 0.4)
fig.suptitle('Market size dominates; winning barely registers', x=0.01, y=0.98, ha='left', fontsize=14, fontweight='bold')
fig.text(0.01, 0.9, 'Winning is not significant within teams (B). Within a team, each year of arena age costs about 0.3% of value (B)', color=INK2, fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.87)); fig.savefig(os.path.join(out, '3_model_effects.png'), dpi=160); plt.close(fig)

# 4. Forbes list value vs actual sale price
fv = df.set_index(['team', 'season']).forbes_value
sales = [('Minnesota Timberwolves', 1500, 2021, 'Control'), ('Phoenix Suns', 4000, 2023, 'Control'), ('Milwaukee Bucks', 3500, 2023, 'Minority'),
         ('Charlotte Hornets', 3000, 2023, 'Control'), ('Dallas Mavericks', 3500, 2024, 'Control'), ('Boston Celtics', 6100, 2025, 'Control'),
         ('Los Angeles Lakers', 10000, 2025, 'Control'), ('Portland Trail Blazers', 4250, 2025, 'Control')]
s = pd.DataFrame(sales, columns=['team', 'price', 'list', 'type']); s['forbes'] = [fv[(a, b)] for a, b in zip(s.team, s.list)]
s['prem'] = s.price / s.forbes - 1
fig, ax = plt.subplots(figsize=(11, 6)); w = 0.38; idx = np.arange(len(s))
ax.bar(idx - w/2 - 0.01, s.forbes / 1000, w, color=BLUE, label='Forbes list value'); ax.bar(idx + w/2 + 0.01, s.price / 1000, w, color=ORANGE, label='Actual sale price')
for i, p in zip(idx, s.prem): ax.text(i, max(s.price[i], s.forbes[i]) / 1000 + 0.2, f'{p:+.0%}', ha='center', fontsize=10, color=INK)
ax.set_xticks(idx); ax.set_xticklabels([f'{short(t)}\n{"(minority)" if ty == "Minority" else ""}' for t, ty in zip(s.team, s.type)], fontsize=9)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'${v:.0f}B')); ax.set_ylabel('$ billions'); ax.grid(axis='x', visible=False); ax.legend(frameon=False, loc='upper left')
c = s[s.type == 'Control']
title(ax, 'Teams sell for more than Forbes says', f'Median control-sale premium {c.prem.median():+.0%} (mean {c.prem.mean():+.0%}, n={len(c)}). Percent = price vs Forbes list in force')
save(fig, '4_forbes_vs_sale_price.png')

# 5. share of cross-team spread explained
for col in ['log_val', 'log_pop', 'winpct10', 'playoff_score', 'arena_age', 'owned']: r[col + '_d'] = r[col] - r.groupby('season')[col].transform('mean')
r2 = lambda f: smf.ols(f, data=r).fit().rsquared
parts = [('Market size', r2('log_val_d ~ log_pop_d - 1')), ('Winning', r2('log_val_d ~ winpct10_d + playoff_score_d - 1')),
         ('Arena', r2('log_val_d ~ arena_age_d + owned_d - 1')),
         ('All together', r2('log_val_d ~ log_pop_d + winpct10_d + playoff_score_d + arena_age_d + owned_d - 1'))]
fig, ax = plt.subplots(figsize=(10, 4.6))
ax.barh([p[0] for p in parts], [p[1] for p in parts], color=[BLUE, BLUE, BLUE, INK2], height=0.6)
for i, (_, v) in enumerate(parts): ax.text(v + 0.01, i, f'{v:.0%}', va='center', fontsize=11, color=INK)
ax.invert_yaxis(); ax.set_xlim(0, 0.7); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.0%}')); ax.grid(axis='y', visible=False)
ax.set_xlabel('Share of cross-team value spread explained (R2, year effects removed)')
title(ax, 'Location explains most of what teams are worth', 'Market size alone gets nearly all of the explained spread')
save(fig, '5_variance_explained.png')
print('wrote 5 charts to', out)
