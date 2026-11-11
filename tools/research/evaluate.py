"""Every strategy over 3- and 6-month holds: in-sample, out-of-sample, by year.

    python tools/research/evaluate.py      (after build_panel.py)
"""
import warnings, logging, pickle, json
warnings.filterwarnings('ignore'); logging.disable(logging.CRITICAL)
import numpy as np, pandas as pd
import lib
from quant import strategies as ST, consensus as CONS, engine as EN
pd.set_option('display.width', 230)
df = lib.load()

def picks(strategy, n=20):
    out = {}
    for d, g in df.groupby('date'):
        sc = lib.score(g, strategy)
        out[d] = list(EN.diversified_head(sc, n).index)
    return out

def consensus_picks(n=20):
    base = ST.get('consensus')
    strat = ST.Strategy(key='cons', name='cons', horizon='long', description='', weights={},
                        components={k: v for k, v in base.components.items() if k != 'Insider buying'},
                        vote_pct=base.vote_pct)
    out = {}
    for d, g in df.groupby('date'):
        raw = g.set_index('ticker')
        sc, _d, _w = CONS.score(raw, strat, d.date())
        if sc.empty:
            out[d] = []; continue
        sc = sc.sort_values(['votes', 'composite'], ascending=False)
        sc['sector'] = raw['sector'].reindex(sc.index)
        out[d] = list(EN.diversified_head(sc, n).index)
    return out

ORDER = ['balanced', 'consensus', 'buffett', 'shareholder_yield', 'quality_momentum', 'style_factors',
         'momentum_trend', 'relative_strength', 'breakout', 'earnings_drift', 'pullback', 'mean_reversion']
allp = {k: (consensus_picks() if k == 'consensus' else picks(k)) for k in ORDER}
stats = {k: lib.portfolio_stats(df, p) for k, p in allp.items()}

rows = []
for k in ORDER:
    st = stats[k]
    r = {'strategy': ST.get(k).name if k != 'consensus' else 'Multi-Strategy Consensus', 'key': k}
    for tag, sl in (('is', st[st.index <= lib.IS_END]), ('oos', st[st.index >= lib.OOS_START]), ('all', st)):
        for h in (6, 3):
            s = lib.summarize(sl, h)
            r[f'{tag}{h}_ex'] = s.get('ex_spy_ann'); r[f'{tag}{h}_hit'] = s.get('hit_spy')
            if tag == 'all':
                r[f'all{h}_ret'] = s.get('ret_ann'); r[f'all{h}_spy'] = s.get('spy_ann'); r[f'all{h}_t'] = s.get('t_spy')
                r[f'all{h}_exew'] = s.get('ex_ew_ann'); r[f'all{h}_n'] = s.get('n'); r[f'all{h}_worst'] = s.get('worst')
    ex = (st['ret_6m'] - st['spy_6m']).dropna()
    for y in range(2020, 2027):
        a = ex[ex.index.year == y]
        r[f'y{y}'] = a.mean() * 2 if len(a) else None
    rows.append(r)
out = pd.DataFrame(rows)
show = out.set_index('key')[['is6_ex', 'oos6_ex', 'all6_ex', 'all6_hit', 'all6_t', 'is3_ex', 'oos3_ex', 'all3_ex', 'all3_hit']]
print((show * 100).round(1).assign(all6_t=out.set_index('key')['all6_t'].round(2)).to_string())
print('\nby formation year, 6m excess vs SPY (annualised %):')
print((out.set_index('key')[[f'y{y}' for y in range(2020, 2027)]] * 100).round(1).to_string())
b = df.groupby('date').agg(spy6=('spy_6m', 'first'), ew6=('fwd_6m', 'mean')).dropna()
print('\nequal-weight universe minus SPY, 6m annualised %, by year:', {y: round((b.ew6 - b.spy6)[b.index.year == y].mean() * 200, 1) for y in range(2020, 2027)})
print('all6: SPY %.1f%%/yr | balanced %.1f%%/yr' % (out.set_index('key').loc['balanced', 'all6_spy'] * 100, out.set_index('key').loc['balanced', 'all6_ret'] * 100))
# insider factor, where data exists
ins = [c for c in df.columns if c.startswith('INSIDER')]
for c in ins:
    part = df[df[c].notna()]
    print(c, 'coverage dates', part.date.nunique(), 'from', part.date.min().date() if len(part) else None, '| names/date', int(part.groupby('date').size().median()) if len(part) else 0)
    for h in (3, 6):
        s = lib.ic_summary(lib.factor_ic(part, c, f'exew_{h}m'), h)
        print(f'   {h}m IC {s["ic"]*100:.1f}, positive {s["pos"]*100:.0f}% of {s["n"]} dates, t={s["t"]:.2f}')
