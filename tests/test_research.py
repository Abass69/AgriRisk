import numpy as np
import pandas as pd
import pytest
from agririsk.data import ROOT, validate
from agririsk.models import features, predict_at, MODELS
from agririsk.risk import scenario

@pytest.fixture
def frame():
    return pd.read_csv(ROOT/'data/processed/monthly.csv',parse_dates=['month']).set_index('month')

def test_scenario_compounds_shocks():
    b=scenario(300,1.35)
    s=scenario(300,1.35,wheat_shock=.2,fx_shock=.1)
    assert s['material_cost_cad']==pytest.approx(b['material_cost_cad']*1.32)
    assert s['profit_cad']==pytest.approx(600000-s['total_cost_cad'])

@pytest.mark.parametrize('kwargs',[{'cad_per_usd':0},{'tonnes':-1},{'fx_shock':-1},{'wheat_usd_t':float('nan')}])
def test_invalid_risk_inputs(kwargs):
    params=dict(wheat_usd_t=300,cad_per_usd=1.35);params.update(kwargs)
    with pytest.raises(ValueError): scenario(**params)

def test_monthly_integrity(frame):
    validate(frame)
    with pytest.raises(ValueError): validate(frame.drop(frame.index[5]))
    frame.iloc[5,0]=np.nan
    with pytest.raises(ValueError): validate(frame)
    frame.iloc[5,0]=np.inf
    with pytest.raises(ValueError): validate(frame)

def test_features_exclude_unavailable_data(frame):
    changed=frame.copy();changed.iloc[59:]=changed.iloc[59:]*100
    pd.testing.assert_series_equal(features(frame).iloc[60],features(changed).iloc[60])

@pytest.mark.parametrize('model',MODELS)
def test_predictions_ignore_future_and_embargoed_labels(frame,model):
    changed=frame.copy();changed.iloc[59:]=changed.iloc[59:]*100
    assert predict_at(frame,60,model)==pytest.approx(predict_at(changed,60,model))

def test_evaluation_and_calibration_embargo():
    p=pd.read_csv(ROOT/'reports/predictions.csv',parse_dates=['month','information_through'])
    assert (p.information_through < p.month).all()
    assert set(p[p.split=='test'].month.dt.year)=={2024,2025}
    for _,g in p.groupby('model'):
        for _,r in g[g.split!='calibration'].iterrows():
            past=g[g.i<=r.i-2].tail(24)
            q=np.sort(abs(past.actual-past.prediction))[min(len(past)-1,int(np.ceil((len(past)+1)*.9))-1)]
            assert r.upper90==pytest.approx(r.prediction+q)

def test_dashboard_smoke():
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_file(str(ROOT/'app.py')).run(timeout=30)
    assert not app.exception
    assert len(app.metric)==3
    app.slider[0].set_value(40).run()
    assert not app.exception
