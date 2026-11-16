"""Build both report formats from executed experiment outputs."""
from html import escape
import json
import pandas as pd
from .data import ROOT

def table_md(df):
    cols=list(df.columns)
    return '| '+' | '.join(cols)+' |\n| '+' | '.join(['---']*len(cols))+' |\n'+'\n'.join('| '+' | '.join(str(v) for v in row)+' |' for row in df.itertuples(index=False,name=None))

def build_report():
    r=ROOT/'reports'; m=pd.read_csv(r/'metrics.csv'); a=pd.read_csv(r/'ablations.csv')
    selected=json.loads((r/'selection.json').read_text())['selected_on_validation']
    test=m[m.split=='test'].set_index('model'); val=m[m.split=='validation'].set_index('model')
    winner=test.loc[selected]; full=test.loc['xgb_full']
    pair=a[(a.split=='test') & (a.comparison=='xgb_full minus persistence') & (a.block_months==3)].iloc[0]
    manifest=json.loads((ROOT/'data/manifest.json').read_text())
    sections=[]
    def add(title,body): sections.append((title,body,None))
    def tab(title,df): sections.append((title,'',df))
    add('Abstract',f'''AgriRisk AI evaluates whether exchange-rate and related commodity information improves wheat procurement forecasts for a Canadian food manufacturer. The dataset contains {manifest['rows']} observed months from January 2017 through December 2025, combining World Bank wheat, maize and Brent prices with Bank of Canada USD/CAD observations. Seven forecasting variants are evaluated using expanding windows and a conservative publication-delay embargo.

The validation-selected model was {selected}, with test MAE of CAD {winner.MAE:.2f}/t. Full-feature XGBoost achieved CAD {full.MAE:.2f}/t on the same test months, but its paired difference from persistence was {pair.MAE_delta:.2f}/t (approximate 95% interval {pair.CI_low:.2f} to {pair.CI_high:.2f}). This does not establish superiority over persistence. A dashboard converts explicit price and currency shocks into purchasing costs and margins. Findings are retrospective, conditional on the source vintage, and not evidence of improved realised business profits.''')
    add('1. Research question and Food Science connection','''Primary question: Do lagged exchange rates and related commodity prices improve monthly CAD-denominated wheat-cost forecasts beyond simple historical baselines?

Secondary question: How do forecast errors and joint price/currency shocks translate into procurement-budget and operating-margin risk under fixed production assumptions?

Food Science connects through ingredient procurement, manufacturing yields, inventory losses, quality specifications and continuity of production. This first implementation models the purchasing-cost layer; it does not model food safety, spoilage, nutritional equivalence or optimal ingredient substitution. The selected case is a Canadian importer because the supplied FX source supports an internally consistent CAD/USD analysis. A Nigerian or other-country deployment needs local FX, tariffs, freight, supplier basis and firm-specific validation.

Hypotheses are that explicit FX features help a CAD cost model, and that adding wheat USD, maize and Brent context helps further. Graph features are not appropriate to this tabular time-series question. The ablation therefore removes groups of economic predictors, not transaction-network features.''')
    add('2. Data, rights and preprocessing',f'''Official source snapshots are retained with SHA-256 checksums and exact URLs in data/manifest.json. The World Bank workbook provides Wheat US HRW and Maize in USD/metric tonne and Brent in USD/barrel. Bank of Canada FXUSDCAD is CAD per USD. Daily FX is averaged within each month; the smallest month contains {manifest['minimum_daily_fx_observations']} observations. Calendar months are aligned without interpolation. The target is wheat_cad_t = wheat_usd_t × cad_per_usd. This product of monthly averages approximates benchmark purchasing exposure; it is not a daily transaction-weighted converted price.

The transformed study sample has {manifest['missing_values']} missing values. Non-numeric source markers are coerced to missing; missing or nonpositive/nonfinite values in the sample fail validation. No forward or backward imputation is applied. Lag construction creates expected warm-up missingness, and those training rows are dropped. No outlier deletion or winsorisation is performed because large movements are central to risk. Class imbalance is not applicable to continuous regression; rare shock regimes remain a sampling limitation.

The World Bank default dataset licence is CC BY 4.0 with additional terms and metadata/third-party exceptions. Bank of Canada data are used under its attribution-based permission to reproduce, subject to its exceptions and disclaimers; they are not labelled CC BY here. Full attribution and modifications are listed in docs/DATA_CARD.md, and original metadata are retained. Neither institution endorses this work. The code MIT licence does not replace data-source terms.''')
    add('3. Forecast timing and validation design','''Target t is the monthly average wheat cost for month t, with a planning decision at the beginning of t. Because the preceding month's commodity average may not yet have been published, every model is limited to observations through t−2. This is a two-step forecast from the latest included observation, not a one-step forecast with instantaneous monthly data. Both input features and training target labels for t−1 are embargoed.

Features begin after a 13-month warm-up. Initial model estimation uses available 2018–2020 history. Rolling forecasts during 2021 form calibration errors; 2022–2023 provides 24 validation months; 2024–2025 provides 24 final evaluation months. Every month uses an expanding training window. During test evaluation, newly available older test outcomes may enter the training/calibration window, as in a rolling operational process, but model choice and hyperparameters are not reselected on test results. The validation rule is lowest MAE.

StandardScaler for ridge is fitted inside each training window. XGBoost needs no scaling. Features for each series are lags 2, 3, 4, 7 and 13, a three-observation rolling mean ending t−2, and six-observation volatility of percentage returns ending t−2. Month sine/cosine terms are known in advance. No contemporaneous target-month market input is used.

The dataset was downloaded as a revised snapshot rather than from historical release vintages. Publication-delay controls prevent ordinary look-ahead in the code, but cannot eliminate revision-based hindsight. A true real-time evaluation requires archived releases and their actual publication timestamps.''')
    add('4. Models and controlled ablations','''Persistence predicts the most recent available CAD wheat price, y[t−2]. Seasonal naive predicts y[t−12]. The moving average uses y[t−4], y[t−3], y[t−2]. Ridge uses all features with alpha=10 and training-only standardisation.

XGBoost uses 80 trees, depth 2, learning rate 0.04, minimum child weight 5, L2 regularisation 10, full row/column sampling, square-error loss and seed 42. These are fixed compact research settings, not an optimised search. Target-only uses CAD wheat history and calendar terms. Plus-FX adds CAD/USD history. Full adds USD wheat, maize and Brent histories. Identical training windows and model settings isolate the predictive contribution of feature groups.

Past CAD wheat prices already contain historical FX effects. Consequently, the first ablation tests explicit FX information beyond the combined target, not whether exchange rates matter economically. The full block adds several related inputs together; it cannot establish which individual economic variable causes a change in accuracy. Further individual-variable and multi-seed ablations are future work.''')
    display=m[['split','model','MAE','RMSE','MAPE_pct','coverage90','interval_width']].copy().round(3)
    tab('5. Executed forecasting results',display)
    add('Result interpretation',f'''Errors are in CAD per metric tonne except MAPE (percent) and coverage (fraction). Persistence was chosen using validation MAE {val.loc['persistence','MAE']:.2f}; XGBoost full had validation MAE {val.loc['xgb_full','MAE']:.2f}. Although XGBoost full had the smallest observed test MAE, replacing the selected model after seeing the test would compromise the selection procedure.

For persistence, the approximate 95% block-bootstrap interval for test MAE is {winner.MAE_CI_low:.2f}–{winner.MAE_CI_high:.2f} CAD/t. At 1,000 tonnes, its average absolute benchmark procurement-budget error is about CAD {winner.MAE*1000:,.0f}. With fixed CAD 600,000 revenue, that corresponds to an average absolute margin-estimation error of {winner.MAE*1000/600000*100:.2f} percentage points. This is an accounting translation of benchmark price error, not an independent validation against actual company costs.''')
    tab('6. Ablation results and paired uncertainty',a[(a.block_months==3)&(a.comparison!='persistence minus persistence')].drop(columns='block_months').round(3))
    add('Ablation interpretation','''Negative MAE differences favour the first model. Explicit FX alone did not reliably help: the uncertainty interval crosses zero in both periods. Adding the full economic block reduced test error relative to both reduced XGBoost variants, with exploratory intervals below zero. However, full XGBoost versus persistence has an interval crossing zero. This is evidence for useful added context within this particular tree model, not broad evidence that AI beats a naive forecasting baseline.

Intervals use 2,000 paired circular moving-block bootstrap replicates with a fixed seed and three-month blocks. Six-month sensitivity results are also saved in ablations.csv. Resampling paired month-level loss differences preserves comparison alignment and some serial dependence. With only 24 months per evaluation period, estimates are unstable. Multiple comparisons are exploratory and not adjusted for multiplicity; no confirmatory significance claim is made.''')
    add('7. Predictive uncertainty and shocks','''A nominal 90% interval uses the finite-sample adjusted 90th percentile of the latest up to 24 available absolute rolling forecast errors, requiring at least 10. For target t, only errors for targets through t−2 enter calibration. Lower prices are clipped at zero. These are adaptive residual intervals inspired by conformal calibration; time dependence and nonstationarity mean there is no distribution-free coverage guarantee.

Report coverage, mean width and interval score together. The 90% interval score is width plus 20 times the distance of an actual outside the interval. Persistence covered all 24 test observations, but its average interval width was approximately CAD 152/t: high coverage came with broad uncertainty, partly inherited from more volatile earlier periods. Full XGBoost coverage was 87.5%; a 24-month sample is too small to assess nominal calibration precisely.

The selected-model test backtest labels a large move as an absolute change of at least 10% from the t−2 CAD benchmark. Only one test month qualifies, with absolute error about CAD 38.24/t versus CAD 14.48/t over the other 23 months. This small subgroup cannot establish shock robustness. The larger validation errors demonstrate substantial regime sensitivity. Imposed dashboard shocks have no assigned probability and cannot be scored as forecasts.''')
    add('8. Business scenario and sensitivity','''Material cost = Q × P × E × (1+sP) × (1+sE), where Q is tonnes, P is wheat USD/t, E is CAD/USD and sP/sE are user-specified fractional changes. Total operating cost adds Q × freight CAD/t plus other operating costs. Profit is revenue less total cost. Margin is 100 × profit/revenue. Financing, tax, tariffs, hedging, quality premiums and demand responses are excluded.

Illustration: 1,000 tonnes at USD 300/t and CAD 1.35/USD gives CAD 405,000 materials. With CAD 30/t freight, CAD 120,000 other costs and CAD 600,000 sales, profit is CAD 45,000 and margin 7.5%. A 20% wheat increase and 10% rise in CAD/USD jointly increase materials by 32% to CAD 534,600, leaving a CAD 84,600 loss and −14.1% margin. The interaction term explains why 20% plus 10% is not merely 30%.

Holding other inputs fixed, material cost has unit elasticity to wheat price and FX separately. Its derivatives are Q×E with respect to P and Q×P with respect to E. The dashboard sensitivity grid makes the joint interaction visible. Scenario arithmetic is checked automatically. A farmer receiving USD-linked revenue could have a different exposure direction; the importer model must not be reused unchanged for a producer.''')
    add('9. Operational prototype and reproducibility','''The Streamlit dashboard supports editable input prices, FX, volume, revenue, freight and other costs; interactive price/FX shocks; a margin sensitivity table; downloadable scenario grids; historical forecast plots; metric and ablation tables; and downloadable predictions. It is a local research demonstration, not a hosted production service or a live market feed.

Run python -m agririsk.experiment to regenerate processed data, rolling forecasts, tables, charts and this report from cached official files. Run python -m pytest -q to check accounting identities, invalid inputs, monthly integrity, feature timing, prediction invariance to future/unavailable data, delayed interval calibration and dashboard execution. The GitHub Actions workflow reproduces experiments and executes the tests. The included dependency versions and manifest support auditability; no external credentials are needed for the dataset sources.

The implementation was executed in the supplied environment. The source repository includes installation instructions in README.md and a short research presentation guide in docs/DEMO.md.''')
    add('10. Limitations, ethics and next research','''The sample covers one wheat benchmark, one currency pair and 108 months, with only 24 final test targets. Results do not generalise automatically across countries, crops, farmers or manufacturers. Current-vintage revisions, benchmark basis risk, unobserved contracts and monthly aggregation restrict operational interpretation. Oil is an economic context proxy, not a complete model of inflation, interest rates, monetary policy or geopolitical shocks. Feature relationships are predictive, not causal.

No private personal or firm data are used. Before operational deployment, protect supplier contracts and purchasing volumes, obtain appropriate permission for private records, and audit performance across small and large enterprises. Do not use price projections as evidence of food safety or as a reason to substitute ingredients without scientific and regulatory checks. Avoid presenting uncertain forecasts as guaranteed savings or institutional endorsements.

Priority extensions: (1) archived release vintages and longer harmonised FX histories; (2) additional crops, countries and genuine firm invoices; (3) isolated-variable ablations and nested time-series tuning; (4) macroeconomic inputs with release calendars; (5) multi-horizon forecasts and joint price/FX distributions; (6) prospective evaluation of procurement budgets; (7) inventory models with yield, shelf-life and spoilage constraints; and (8) hedging and cash-flow models only after adding contract and financing data.''')
    add('References','''World Bank Commodity Markets: https://www.worldbank.org/en/research/commodity-markets
World Bank dataset terms: https://www.worldbank.org/ext/en/legal/terms-conditions/datasets
World Bank data licensing: https://datacatalog.worldbank.org/public-licenses
Bank of Canada Valet: https://www.bankofcanada.ca/valet/docs/
Bank of Canada terms: https://www.bankofcanada.ca/terms/
Bank of Canada exchange-rate definitions: https://www.bankofcanada.ca/rates/exchange/daily-exchange-rates/
Exact source download URLs and hashes: data/manifest.json. Full data transformations and source acknowledgements: docs/DATA_CARD.md.''')
    md=['# AgriRisk AI — Technical Report\n\nCommodity price and agricultural financial risk analysis\n']
    html=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AgriRisk AI — Technical Report</title><style>body{font:17px/1.65 system-ui;color:#203b33;max-width:1100px;margin:40px auto;padding:0 24px}h1{font-size:42px;line-height:1.1}h2{margin-top:42px;border-top:1px solid #cddbd3;padding-top:22px}p{max-width:95ch}table{border-collapse:collapse;font-size:13px;width:100%;display:block;overflow:auto}th,td{padding:8px;border-bottom:1px solid #dbe5de;text-align:left}th{background:#eaf1ec}img{width:100%;height:auto}header{padding:36px;background:#173d35;color:#fff;border-radius:12px}footer{margin:40px 0;color:#60766c}@media print{body{font-size:11px}h2{break-after:avoid}table{font-size:9px}}</style><header><p>RESEARCH PROTOTYPE • FOOD SYSTEMS & FINANCE</p><h1>AgriRisk AI</h1><p>Commodity price and agricultural financial risk analysis</p></header>']
    for title,body,df in sections:
        md.append('## '+title+'\n\n'+(table_md(df) if df is not None else body))
        html.append('<h2>'+escape(title)+'</h2>')
        html.append(df.to_html(index=False,border=0) if df is not None else ''.join('<p>'+escape(p).replace('\n','<br>')+'</p>' for p in body.split('\n\n')))
        if title=='5. Executed forecasting results':
            md.append('![Test forecasts](forecast.png)\n\n![Model comparison](model_comparison.png)')
            html.append('<img src="forecast.png" alt="Observed and predicted wheat prices with intervals"><img src="model_comparison.png" alt="Test errors by model">')
    html.append('<footer>Reproducible research package • Results generated from executed experiments</footer></html>')
    (r/'TECHNICAL_REPORT.md').write_text('\n\n'.join(md))
    (r/'technical_report.html').write_text('\n'.join(html))

if __name__=='__main__': build_report()
