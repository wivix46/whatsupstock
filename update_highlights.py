"""Build Highlights metrics, retaining previous valid fields on partial failures."""
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent
selection = json.loads((ROOT / 'data/highlights.json').read_text(encoding='utf-8'))
output = ROOT / 'data/highlights_metrics.json'
previous = json.loads(output.read_text(encoding='utf-8')) if output.exists() else {}
old_stocks = previous.get('stocks', {})
stocks = {}


def finite(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def price_change(series, days):
    if len(series) < 2:
        return None
    target = series.index[-1] - pd.Timedelta(days=days)
    earlier = series.loc[series.index <= target]
    if earlier.empty:
        return None
    start, end = finite(earlier.iloc[-1]), finite(series.iloc[-1])
    return round((end / start - 1) * 100, 2) if start and end else None


def clean_close(history):
    if history is None or history.empty or 'Close' not in history:
        return pd.Series(dtype=float)
    close = pd.to_numeric(history['Close'], errors='coerce').dropna()
    return close[close > 0].sort_index()


def sparkline(close):
    if len(close) < 2:
        return None
    year = close.loc[close.index >= close.index[-1] - pd.Timedelta(days=365)]
    if len(year) < 2:
        return None
    count = min(36, len(year))
    positions = [round(i * (len(year) - 1) / (count - 1)) for i in range(count)]
    return [round(float(value), 4) for value in year.iloc[positions]]


def keep_new_or_old(new, old, field):
    return new if new is not None else old.get(field)


all_tickers = list(dict.fromkeys(row['Ticker'] for region in ('us', 'europe') for row in selection[region]))
updated = 0
for ticker in all_tickers:
    old = old_stocks.get(ticker, {})
    record = dict(old)
    try:
        ticker_obj = yf.Ticker(ticker)
        close = pd.Series(dtype=float)
        try:
            close = clean_close(ticker_obj.history(period='15mo', interval='1d', auto_adjust=False, actions=False))
            # A short Yahoo response can be temporary; retry with a broader period.
            if len(close) < 30:
                fallback = clean_close(ticker_obj.history(period='5y', interval='1d', auto_adjust=False, actions=False))
                if len(fallback) > len(close):
                    close = fallback
        except Exception as exc:
            print(f'{ticker}: price history unavailable: {exc}')

        if len(close):
            month = price_change(close, 30)
            year = price_change(close, 365)
            trend = sparkline(close)
            record['return_1m'] = keep_new_or_old(month, old, 'return_1m')
            record['return_1y'] = keep_new_or_old(year, old, 'return_1y')
            record['trend_1y'] = keep_new_or_old(trend, old, 'trend_1y')
            record['as_of'] = close.index[-1].strftime('%Y-%m-%d')
            print(f'{ticker}: {len(close)} price observations; 1M={month}, 1Y={year}')
        else:
            print(f'{ticker}: no price history; retaining any previous price metrics')

        # Fundamentals can still be available when the price-history request fails.
        info = {}
        try:
            info = ticker_obj.info or {}
        except Exception as exc:
            print(f'{ticker}: fundamentals unavailable: {exc}')
        pe = finite(info.get('trailingPE'))
        record['pe'] = keep_new_or_old(pe if pe is not None and pe > 0 else None, old, 'pe')
        target = finite(info.get('targetMeanPrice'))
        last = finite(close.iloc[-1]) if len(close) else None
        upside = round((target / last - 1) * 100, 2) if target and last else None
        record['analyst_upside'] = keep_new_or_old(upside, old, 'analyst_upside')

        # Do not store empty records; they would hide a missing-data problem.
        if any(record.get(key) is not None for key in ('return_1m', 'return_1y', 'trend_1y', 'pe', 'analyst_upside')):
            stocks[ticker] = record
            updated += 1
            print(f'{ticker}: saved')
        else:
            print(f'{ticker}: no usable metrics')
    except Exception as exc:
        print(f'{ticker}: retaining previous metrics: {exc}')
        if old:
            stocks[ticker] = old

if updated < 5 and not old_stocks:
    raise RuntimeError('Insufficient market data: refusing to create an empty snapshot')
if not stocks:
    raise RuntimeError('No valid highlights metrics')
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps({
    'updated_at': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M'),
    'updated_count': updated,
    'stocks': stocks,
}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(f'Saved {len(stocks)} stocks; updated {updated}')
