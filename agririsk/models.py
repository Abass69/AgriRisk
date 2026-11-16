"""All features for target month t use observations at t-2 or earlier."""
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

MODELS = ['persistence', 'seasonal_naive', 'moving_average', 'ridge',
          'xgb_target_only', 'xgb_plus_fx', 'xgb_full']

def features(df, mode='full'):
    cols = ['wheat_cad_t']
    if mode in ('fx', 'full'):
        cols += ['cad_per_usd']
    if mode == 'full':
        cols += ['wheat_usd_t', 'maize_usd_t', 'brent_usd_bbl']
    out = {}
    for col in cols:
        s = df[col]
        for lag in [2, 3, 4, 7, 13]:
            out[f'{col}_lag{lag}'] = s.shift(lag)
        out[f'{col}_mean3'] = s.shift(2).rolling(3).mean()
        out[f'{col}_vol6'] = s.pct_change(fill_method=None).shift(2).rolling(6).std()
    out['month_sin'] = np.sin(2*np.pi*df.index.month/12)
    out['month_cos'] = np.cos(2*np.pi*df.index.month/12)
    return pd.DataFrame(out, index=df.index)

def predict_at(df, i, model):
    y = df.wheat_cad_t
    # t-1 is intentionally embargoed, including training labels.
    if model == 'persistence': return float(y.iloc[i-2])
    if model == 'seasonal_naive': return float(y.iloc[i-12])
    if model == 'moving_average': return float(y.iloc[i-4:i-1].mean())
    mode = {'xgb_target_only': 'target', 'xgb_plus_fx': 'fx'}.get(model, 'full')
    x = features(df, mode)
    train = x.iloc[:i-1].dropna()
    if len(train) < 20: raise ValueError('Insufficient training history')
    if x.iloc[[i]].isna().any().any(): raise ValueError('Missing forecast features')
    estimator = (make_pipeline(StandardScaler(), Ridge(alpha=10)) if model == 'ridge' else
        XGBRegressor(n_estimators=80, max_depth=2, learning_rate=.04,
                     min_child_weight=5, reg_lambda=10, subsample=1,
                     colsample_bytree=1, objective='reg:squarederror',
                     n_jobs=1, random_state=42))
    estimator.fit(train, y.loc[train.index])
    return max(0., float(estimator.predict(x.iloc[[i]])[0]))
