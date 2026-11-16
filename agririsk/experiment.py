"""Expanding-window experiments, delayed calibration, paired block bootstrap."""
import json
import platform
import importlib.metadata
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .data import ROOT, build
from .models import MODELS, predict_at

def block_ci(values, block=3, draws=2000):
    a = np.asarray(values); n = len(a); rng = np.random.default_rng(42)
    samples = []
    for _ in range(draws):
        starts = rng.integers(0, n, size=int(np.ceil(n/block)))
        idx = np.concatenate([(s+np.arange(block)) % n for s in starts])[:n]
        samples.append(a[idx].mean())
    return np.quantile(samples, [.025, .975])

def run():
    df = build(); out = ROOT / 'reports'; out.mkdir(exist_ok=True)
    rows = []
    for i in range(48, len(df)):
        for model in MODELS:
            pred = predict_at(df, i, model)
            # Only forecast errors whose actuals have become available may calibrate t.
            previous = [abs(r['actual']-r['prediction']) for r in rows
                        if r['model'] == model and r['i'] <= i-2][-24:]
            n = len(previous)
            q = (np.sort(previous)[min(n-1, int(np.ceil((n+1)*.9))-1)] if n >= 10 else np.nan)
            rows.append({'i': i, 'month': df.index[i], 'information_through': df.index[i-2],
                         'model': model, 'actual': df.wheat_cad_t.iloc[i], 'prediction': pred,
                         'lower90': max(0, pred-q) if np.isfinite(q) else np.nan, 'upper90': pred+q,
                         'calibration_n': n,
                         'split': 'calibration' if df.index[i].year == 2021 else
                                  ('validation' if df.index[i].year < 2024 else 'test')})
    forecasts = pd.DataFrame(rows)
    forecasts.to_csv(out / 'predictions.csv', index=False)
    summaries = []
    for (split, model), g in forecasts[forecasts.split != 'calibration'].groupby(['split', 'model']):
        err = g.prediction-g.actual
        lo, hi = block_ci(abs(err))
        valid = g.lower90.notna()
        coverage = ((g.actual >= g.lower90) & (g.actual <= g.upper90))[valid].mean()
        interval_score = ((g.upper90-g.lower90) + 20*(g.lower90-g.actual).clip(lower=0)
                          + 20*(g.actual-g.upper90).clip(lower=0))[valid].mean()
        summaries.append({'split':split, 'model':model, 'n':len(g), 'MAE':abs(err).mean(),
                          'RMSE':np.sqrt((err**2).mean()), 'MAPE_pct':100*(abs(err)/g.actual).mean(),
                          'MAE_CI_low':lo, 'MAE_CI_high':hi, 'coverage90':coverage,
                          'interval_width':(g.upper90-g.lower90)[valid].mean(),
                          'interval_score90':interval_score})
    metrics = pd.DataFrame(summaries); metrics.to_csv(out/'metrics.csv', index=False)
    winner = metrics[metrics.split == 'validation'].sort_values('MAE').iloc[0].model
    ablations = []
    for split in ['validation', 'test']:
        pivot = forecasts[forecasts.split == split].pivot(index='month', columns='model', values='prediction')
        actual = df.loc[pivot.index, 'wheat_cad_t']
        for a, b in [('xgb_plus_fx','xgb_target_only'), ('xgb_full','xgb_plus_fx'), ('xgb_full','xgb_target_only'), ('xgb_full','persistence'), (winner,'persistence')]:
            delta = (pivot[a]-actual).abs()-(pivot[b]-actual).abs()
            for block in [3, 6]:
                lo, hi = block_ci(delta, block=block)
                ablations.append({'split': split, 'comparison':f'{a} minus {b}', 'block_months':block,
                                  'MAE_delta':delta.mean(), 'CI_low':lo, 'CI_high':hi})
    pd.DataFrame(ablations).to_csv(out/'ablations.csv',index=False)
    g = forecasts[(forecasts.model == winner) & (forecasts.split == 'test')].copy()
    g['material_budget_error_1000t_cad'] = 1000*(g.prediction-g.actual)
    g['margin_error_pp_600k_revenue'] = -100*g.material_budget_error_1000t_cad/600000
    g['large_move'] = abs(g.actual/df.wheat_cad_t.shift(2).loc[g.month].to_numpy()-1) >= .10
    g.to_csv(out/'business_backtest.csv',index=False)
    stress = g.groupby('large_move').apply(lambda a: pd.Series({'n':len(a), 'MAE':abs(a.prediction-a.actual).mean()}), include_groups=False)
    stress.to_csv(out/'shock_regimes.csv')
    (out/'selection.json').write_text(json.dumps({'selected_on_validation':winner,
       'selection_rule':'Lowest 2022-2023 MAE; no test-based retuning',
       'python':platform.python_version(),
       'versions':{p:importlib.metadata.version(p) for p in ['pandas','numpy','scikit-learn','xgboost']}},indent=2))
    plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'font.size':11})
    fig, ax = plt.subplots(figsize=(11,5),layout='constrained')
    ax.plot(g.month,g.actual,label='Observed wheat CAD/t',color='#173d35',lw=2.5)
    ax.plot(g.month,g.prediction,label=f'Validation-selected: {winner}',color='#d78529',lw=2)
    ax.fill_between(g.month,g.lower90,g.upper90,color='#d78529',alpha=.18,label='Nominal 90% interval')
    ax.set(title='AgriRisk AI | Untouched evaluation period: 2024–2025',ylabel='CAD per metric tonne')
    ax.legend(fontsize=9); fig.savefig(out/'forecast.png',dpi=160); plt.close(fig)
    test = metrics[metrics.split == 'test'].sort_values('MAE')
    fig, ax = plt.subplots(figsize=(10,5),layout='constrained')
    ax.barh(test.model,test.MAE,color='#367766'); ax.invert_yaxis()
    ax.set(title='Test-period forecasting error | Lower is better',xlabel='MAE, CAD per metric tonne')
    fig.savefig(out/'model_comparison.png',dpi=160); plt.close(fig)
    from .report import build_report
    build_report()
    print(metrics.round(3).to_string(index=False)); print('Validation-selected model:',winner)

if __name__ == '__main__': run()
