"""Refresh the Home ranking snapshot without importing/running the Streamlit UI."""
import ast
import json
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

ROOT = Path(__file__).resolve().parent
tree = ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))
needed = {"safe_num", "rating_label", "analyst_rating_score", "inverse_percentile", "direct_percentile", "add_internal_score", "batch_prices",
          "fetch_snapshot", "build_dataset"}
nodes = []
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in needed:
        nodes.append(node)
    elif isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id in {"HOME_UNIVERSE"}
        for target in node.targets
    ):
        nodes.append(node)
module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
exec(compile(module, str(ROOT / "app.py"), "exec"), globals())
universe = globals()["HOME_UNIVERSE"]
df = build_dataset(tuple((k, tuple(v)) for k, v in universe.items()))
if df.empty:
    raise RuntimeError("No data returned; preserving previous snapshot.")
usable = int(df[["P/E", "Forward P/E", "EPS", "Analyst Target"]].notna().any(axis=1).sum())
if usable < 45:
    raise RuntimeError(f"Only {usable} usable records; preserving previous snapshot.")
df = df.replace([np.inf, -np.inf], np.nan)
payload = {
    "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
    "usable": usable,
    "rows": json.loads(df.to_json(orient="records"))
}
path = ROOT / "data" / "rankings_snapshot.json"
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
print(f"Updated {path.name}: {usable}/{len(df)} usable records")
