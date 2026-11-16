"""Run: streamlit run app.py"""
import json
import numpy as np
import pandas as pd
import streamlit as st
from agririsk.data import ROOT
from agririsk.risk import scenario

st.set_page_config(page_title='AgriRisk AI', page_icon='🌾', layout='wide')
st.title('AgriRisk AI')
st.caption('Wheat procurement • Currency exposure • Food manufacturing margins')
path = ROOT/'data/processed/monthly.csv'
if not path.exists():
    st.info('Run python -m agririsk.experiment first.'); st.stop()
df = pd.read_csv(path,parse_dates=['month']).set_index('month')
st.info('Historical research prototype: January 2017–December 2025. Scenario shocks are user assumptions, not forecasts or assigned probabilities.')
tab1, tab2, tab3 = st.tabs(['Cost & margin scenarios','Forecast evidence','Data & methods'])
with tab1:
    st.subheader('What happens to purchasing costs when wheat and FX move together?')
    cols = st.columns(3)
    with cols[0]:
        wheat = st.number_input('Wheat price (USD / metric tonne)',min_value=1.,value=float(df.wheat_usd_t.iloc[-1]))
        fx = st.number_input('CAD per USD',min_value=.001,value=float(df.cad_per_usd.iloc[-1]),format='%.4f')
        tonnes = st.number_input('Wheat purchased (metric tonnes)',min_value=0.,value=1000.)
    with cols[1]:
        revenue = st.number_input('Sales revenue (CAD)',min_value=1.,value=600000.)
        other = st.number_input('Other operating costs (CAD)',min_value=0.,value=120000.)
        freight = st.number_input('Freight (CAD / metric tonne)',min_value=0.,value=30.)
    with cols[2]:
        ws = st.slider('Global wheat price change (%)',-80,100,20)/100
        fs = st.slider('CAD per USD change (%)',-50,100,10)/100
        st.caption('A positive FX change means CAD depreciation: each USD costs more CAD. Freight is fixed in CAD in this demonstration.')
    base = scenario(wheat,fx,tonnes,revenue,other,freight)
    shocked = scenario(wheat,fx,tonnes,revenue,other,freight,ws,fs)
    a,b,c=st.columns(3)
    a.metric('Stressed material bill',f"CAD {shocked['material_cost_cad']:,.0f}",f"{shocked['material_cost_cad']-base['material_cost_cad']:,.0f} CAD",delta_color='inverse')
    b.metric('Stressed operating profit',f"CAD {shocked['profit_cad']:,.0f}",f"{shocked['profit_cad']-base['profit_cad']:,.0f} CAD")
    c.metric('Stressed margin',f"{shocked['margin_pct']:.1f}%",f"{shocked['margin_pct']-base['margin_pct']:.1f} pp")
    grid = pd.DataFrame({f'FX {f:+.0%}':{f'Wheat {w:+.0%}':scenario(wheat,fx,tonnes,revenue,other,freight,w,f)['margin_pct'] for w in [-.2,0,.1,.2,.4]} for f in [-.1,0,.1,.2]})
    st.subheader('Margin sensitivity (%)')
    st.dataframe(grid.style.format('{:.1f}').background_gradient(cmap='RdYlGn'),width='stretch')
    st.download_button('Download scenario grid',grid.to_csv(),'agririsk_scenarios.csv','text/csv')
    st.caption('Material cost = tonnes × wheat USD/t × CAD/USD × (1+wheat shock) × (1+FX shock). Revenue, volume, freight and other costs are held constant.')
with tab2:
    if not (ROOT/'reports/metrics.csv').exists(): st.info('Run the experiment to populate results.')
    else:
        selection=json.loads((ROOT/'reports/selection.json').read_text())
        st.write('Model chosen using 2022–2023 validation:',selection['selected_on_validation'])
        st.image(str(ROOT/'reports/forecast.png'))
        st.dataframe(pd.read_csv(ROOT/'reports/metrics.csv'),hide_index=True)
        st.subheader('Ablation: does adding FX and economic context help?')
        st.caption('Negative MAE differences favour the first model. Confidence intervals use paired circular block bootstrap.')
        st.dataframe(pd.read_csv(ROOT/'reports/ablations.csv'),hide_index=True)
        st.download_button('Download historical forecasts',(ROOT/'reports/predictions.csv').read_bytes(),'predictions.csv','text/csv')
with tab3:
    st.line_chart(df[['wheat_usd_t','wheat_cad_t']])
    st.caption('Wheat series have different currency units; the chart is for historical context.')
    st.markdown('Sources: [World Bank Pink Sheet](https://www.worldbank.org/en/research/commodity-markets) and [Bank of Canada Valet](https://www.bankofcanada.ca/valet/docs/). Modified by monthly aggregation, alignment and currency conversion. No endorsement is implied.')
    st.write('Forecasts use data from two months before the target or earlier. Revised historical data remain a limitation. This is regression: class balancing is not applicable. No causal or real-time accuracy claim is made.')
    st.json(json.loads((ROOT/'data/manifest.json').read_text()))
