"""Build a validated, persistent rankings snapshot from the existing app's data logic.
Run via GitHub Actions; does not execute the Streamlit page.
"""
import ast
import json
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

ROOT = Path(__file__).resolve().parent
source = ROOT / 'app.py'
tree = ast.parse(source.read_text(encoding='utf-8'))
functions = {
    'safe_num', 'rating_label', 'analyst_rating_score',
    'inverse_percentile', 'direct_percentile', 'add_internal_score',
    'batch_prices', 'fetch_snapshot', 'build_dataset'
}
constants = {'HOME_UNIVERSE', 'WEIGHTS'}
selected = []
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in functions:
        selected.append(node)
    elif isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id in constants
        for target in node.targets
    ):
        selected.append(node)

found_functions = {n.name for n in selected if isinstance(n, ast.FunctionDef)}
found_constants = {
    t.id for n in selected if isinstance(n, ast.Assign)
    for t in n.targets if isinstance(t, ast.Name)
}
if functions - found_functions or constants - found_constants:
    raise RuntimeError('The app.py scoring functions have changed; refusing to publish an incompatible snapshot.')

module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
exec(compile(module, str(source), 'exec'), globals())

universe = HOME_UNIVERSE
expected = sum(len(v) for v in universe.values())
df = build_dataset(tuple((sector, tuple(tickers)) for sector, tickers in universe.items()))
if len(df) != expected:
    raise RuntimeError(f'Expected {expected} rows, got {len(df)}; previous snapshot preserved.')

# Require usable values across every sector, not just a high global count.
core = ['P/E', 'Forward P/E', 'Analyst Target', 'Price']
valid = df[core].notna().any(axis=1) & df['Internal Rating'].notna()
usable = int(valid.sum())
per_sector = df.assign(_valid=valid).groupby('Sector')['_valid'].sum()
if usable < 55 or (per_sector < 4).any():
    raise RuntimeError(
        f'Insufficient Yahoo data: {usable}/{expected}; sector counts: {per_sector.to_dict()}. '
        'Previous snapshot preserved.'
    )

# Write new data only after all checks succeed.
df = df.replace([np.inf, -np.inf], np.nan)
updated_at = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
rows = json.loads(df.to_json(orient='records'))
payload = {'updated_at': updated_at, 'usable': usable, 'rows': rows}
data_dir = ROOT / 'data'
data_dir.mkdir(exist_ok=True)
latest = data_dir / 'rankings_snapshot.json'
history_dir = data_dir / 'history'
history_dir.mkdir(exist_ok=True)
# Daily snapshots: one per UTC day, latest successful run of that day.
day = updated_at[:10]
historical = history_dir / f'{day}.json'
serialized = json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
historical.write_text(serialized, encoding='utf-8')
latest.write_text(serialized, encoding='utf-8')
print(f'Saved validated snapshot: {usable}/{expected} usable; {day}')
