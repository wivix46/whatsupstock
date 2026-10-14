"""Build daily Highlights metrics; preserve previous good values on partial failures."""
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import yfinance as yf
import pandas as pd

ROOT = Path(__file__).resolve().parent
selection = json.loads((ROOT / "data/highlights.json").read_text(encoding="utf-8"))
output = ROOT / "data/highlights_metrics.json"
previous = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {}
old_stocks = previous.get("stocks", {})
stocks = {}

def finite(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None

def price_change(series, days):
    # Select the latest available close on/before the calendar lookback date.
    if len(series) < 2:
        return None
    target = series.index[-1] - pd.Timedelta(days=days)
    earlier = series.loc[series.index <= target]
    if earlier.empty:
        return None
    p0, p1 = finite(earlier.iloc[-1]), finite(series.iloc[-1])
    return round((p1 / p0 - 1) * 100, 2) if p0 and p1 else None

all_tickers = list(dict.fromkeys(row["Ticker"] for region in ("us", "europe") for row in selection[region]))
updated = 0
for ticker in all_tickers:
    old = old_stocks.get(ticker, {})
    try:
        t = yf.Ticker(ticker)
        # auto_adjust=True accounts for splits AND cash dividends; we use
        # auto_adjust=False + Close to exclude dividends while accounting for splits.
        history = t.history(period="15mo", interval="1d", auto_adjust=False, actions=False)
        if history.empty or "Close" not in history:
            raise ValueError("Missing price history")
        close = history["Close"].dropna()
        close = close[close > 0]
        if len(close) < 30:
            raise ValueError("Insufficient price history")
        # Yahoo Close is generally split-adjusted, not dividend-adjusted.
        month = price_change(close, 30)
        year = price_change(close, 365)
        # Sample at most 36 evenly spaced points for a lightweight sparkline.
        year_close = close.loc[close.index >= close.index[-1] - pd.Timedelta(days=365)]
        sampled = year_close.iloc[[round(i*(len(year_close)-1)/(min(36,len(year_close))-1)) for i in range(min(36,len(year_close)))]] if len(year_close)>1 else year_close
        trend = [round(float(v), 4) for v in sampled.tolist()]
        info = {}
        try:
            info = t.info or {}
        except Exception as ex:
            print(f"{ticker}: fundamentals unavailable: {ex}")
        pe = finite(info.get("trailingPE"))
        target = finite(info.get("targetMeanPrice"))
        last = finite(close.iloc[-1])
        upside = round((target / last - 1)*100, 2) if target and last else None
        stocks[ticker] = {
            "return_1m": month if month is not None else old.get("return_1m"),
            "return_1y": year if year is not None else old.get("return_1y"),
            "trend_1y": trend,
            "pe": pe if pe and pe > 0 else old.get("pe"),
            "analyst_upside": upside if upside is not None else old.get("analyst_upside"),
            "as_of": close.index[-1].strftime("%Y-%m-%d"),
        }
        updated += 1
        print(f"{ticker}: updated")
    except Exception as ex:
        print(f"{ticker}: retaining previous metrics: {ex}")
        if old:
            stocks[ticker] = old

if updated < 5 and not old_stocks:
    raise RuntimeError("Insufficient market data: refusing to create an empty snapshot")
if not stocks:
    raise RuntimeError("No valid highlights metrics")
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps({"updated_at":datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),"updated_count":updated,"stocks":stocks},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(f"Saved {len(stocks)} stocks; updated {updated}")
