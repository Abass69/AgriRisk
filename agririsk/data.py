"""Download, validate and transform official data; never silently substitute synthetic data."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
WB_URL = 'https://thedocs.worldbank.org/en/doc/74e8be41ceb20fa0da750cda2f6b9e4e-0050012026/related/CMO-Historical-Data-Monthly.xlsx'
FX_URL = 'https://www.bankofcanada.ca/valet/observations/FXUSDCAD/json?start_date=2017-01-01&end_date=2025-12-31'

def validate(frame):
    expected = pd.date_range('2017-01-01', '2025-12-01', freq='MS')
    if not frame.index.equals(expected):
        raise ValueError('Missing, duplicate or out-of-order months')
    if not np.isfinite(frame.to_numpy()).all() or not (frame > 0).all().all():
        raise ValueError('Missing or nonpositive observations; investigate upstream data')
    return frame

def build(refresh=False):
    raw = ROOT / 'data/raw'
    raw.mkdir(parents=True, exist_ok=True)
    entries = []
    for name, url in [('world_bank_monthly.xlsx', WB_URL), ('boc_fx.json', FX_URL)]:
        path = raw / name
        if refresh or not path.exists():
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            path.write_bytes(r.content)
        entries.append({'file': name, 'url': url, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    wb = pd.read_excel(raw / 'world_bank_monthly.xlsx', sheet_name='Monthly Prices', header=4)
    wb = wb[wb.iloc[:, 0].astype(str).str.fullmatch(r'\d{4}M\d{2}')].copy()
    wb.index = pd.to_datetime(wb.iloc[:, 0], format='%YM%m')
    mapping = {'Wheat, US HRW': 'wheat_usd_t', 'Maize': 'maize_usd_t', 'Crude oil, Brent': 'brent_usd_bbl'}
    wb = wb[list(mapping)].rename(columns=mapping).apply(pd.to_numeric, errors='coerce')
    obs = json.loads((raw / 'boc_fx.json').read_text())['observations']
    fx = pd.Series({pd.Timestamp(o['d']): float(o['FXUSDCAD']['v']) for o in obs if 'FXUSDCAD' in o})
    counts = fx.resample('MS').count()
    if (counts < 15).any():
        raise ValueError('Insufficient daily FX observations in a month')
    frame = wb.join(fx.resample('MS').mean().rename('cad_per_usd')).loc['2017':'2025']
    validate(frame)
    frame['wheat_cad_t'] = frame.wheat_usd_t * frame.cad_per_usd
    frame.index.name = 'month'
    (ROOT / 'data/processed').mkdir(exist_ok=True)
    frame.to_csv(ROOT / 'data/processed/monthly.csv')
    manifest = {'built_utc': datetime.now(timezone.utc).isoformat(), 'sources': entries,
                'start': str(frame.index.min().date()), 'end': str(frame.index.max().date()),
                'rows': len(frame), 'missing_values': int(frame.isna().sum().sum()),
                'minimum_daily_fx_observations': int(counts.min()),
                'vintage_warning': 'Current revised snapshot, not a real-time vintage archive.'}
    (ROOT / 'data/manifest.json').write_text(json.dumps(manifest, indent=2))
    return frame

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(); p.add_argument('--refresh', action='store_true')
    print(build(p.parse_args().refresh).tail())
