
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import html
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(page_title="Whatsupstock", page_icon="📊", layout="wide")

SECTORS = {
    "Restaurants": ["MCD","SBUX","CMG","YUM","QSR","DRI","TXRH","DPZ","WING","CAVA","CAKE","WEN","JACK","SHAK"],
    "Semiconductors": ["NVDA","AVGO","AMD","QCOM","MU","TXN","INTC","ADI","NXPI","MCHP","ON","MPWR","MRVL","SWKS"],
    "Software": ["MSFT","ORCL","CRM","ADBE","NOW","INTU","PANW","SNOW","PLTR","DDOG","MDB","TEAM","ZS","HUBS"],
    "Technology Hardware": ["AAPL","DELL","HPE","HPQ","ANET","SMCI","NTAP","WDC","STX","PSTG","LOGI","ZBRA","FFIV","JNPR"],
    "Internet & Digital Platforms": ["GOOGL","META","NFLX","SPOT","UBER","ABNB","DASH","PINS","RDDT","SNAP","RBLX","TTD","MTCH","IAC"],
    "Payments & Financial Services": ["V","MA","AXP","PYPL","COF","DFS","FI","FIS","GPN","HOOD","SOFI","AFRM","XYZ","NU"],
    "Banks": ["JPM","BAC","WFC","C","GS","MS","USB","PNC","TFC","BK","STT","FITB","KEY","CFG"],
    "Insurance": ["BRK-B","PGR","ALL","CB","AIG","TRV","MET","PRU","AFL","HIG","CINF","ACGL","WRB","L"],
    "Pharma": ["LLY","JNJ","ABBV","MRK","PFE","BMY","AMGN","GILD","REGN","VRTX","BIIB","ZTS","ALNY","MRNA"],
    "Energy": ["XOM","CVX","COP","EOG","SLB","MPC","PSX","VLO","OXY","FANG","HAL","DVN","BKR","APA"],
    "Retail": ["WMT","COST","AMZN","HD","LOW","TGT","TJX","ROST","BBY","DG","DLTR","BURL","ULTA","KR"],
    "Auto": ["TSLA","GM","F","RIVN","LCID","TM","HMC","STLA","RACE","MBLY","APTV","BWA","LEA","GNTX"],
    "Aerospace & Defense": ["RTX","GE","BA","LMT","NOC","GD","LHX","HWM","TDG","TXT","HII","CW","BWXT","ACHR"],
    "Industrials": ["CAT","DE","HON","ETN","PH","EMR","ITW","MMM","ROK","CMI","PCAR","FAST","URI","GWW"],
    "Telecom": ["TMUS","VZ","T","CHTR","LUMN","ASTS","VOD","TEF","AMX","BCE","TU","RCI","USM","TDS"],
    "Consumer Brands": ["PG","KO","PEP","PM","MO","CL","KMB","MDLZ","KHC","GIS","HSY","KDP","STZ","MNST"]
}

MAX_STOCKS = 10

def safe_num(value):
    try:
        if value is None:
            return np.nan
        return float(value)
    except Exception:
        return np.nan

def normalize_yield(value):
    """
    Normalize Yahoo/yfinance dividend yield to a decimal fraction.
    Final form: 0.027 = 2.7%.
    """
    value = safe_num(value)
    if pd.isna(value) or value < 0:
        return np.nan
    if value > 1:
        value = value / 100.0
    # Discard clearly implausible values for ordinary listed equities.
    if value > 0.20:
        return np.nan
    return value

def rating_label(info):
    key = str(info.get("recommendationKey", "") or "").strip().lower()
    mapping = {
        "strong_buy": "Strong Buy",
        "strong buy": "Strong Buy",
        "buy": "Buy",
        "hold": "Hold",
        "underperform": "Sell",
        "sell": "Sell",
        "strong_sell": "Strong Sell",
        "strong sell": "Strong Sell",
    }
    return mapping.get(key, key.replace("_", " ").title() if key else "—")


@st.cache_data(ttl=1800, show_spinner=False)
def batch_market_data(symbols_tuple):
    """One batched Yahoo request for recent prices."""
    symbols = list(symbols_tuple)
    rows = []

    try:
        hist = yf.download(
            tickers=symbols,
            period="5d",
            interval="1d",
            auto_adjust=False,
            progress=False,
            group_by="ticker",
            threads=True,
        )
    except Exception:
        hist = pd.DataFrame()

    for symbol in symbols:
        price = np.nan
        try:
            sdf = hist.copy() if len(symbols) == 1 else hist[symbol].copy()
            if not sdf.empty and "Close" in sdf.columns:
                closes = pd.to_numeric(sdf["Close"], errors="coerce").dropna()
                if not closes.empty:
                    price = safe_num(closes.iloc[-1])
        except Exception:
            pass

        rows.append({"Ticker": symbol, "Batch Price": price})

    return pd.DataFrame(rows)


@st.cache_data(ttl=21600, show_spinner=False)
def fetch_fundamentals(symbol):
    """
    Robust per-ticker fundamental snapshot.
    Cached for 6 hours. Incomplete Yahoo responses are retried and are not
    considered valid merely because an object was returned.
    """
    best = {}

    for attempt in range(3):
        try:
            t = yf.Ticker(symbol)
            info = t.get_info() or {}

            # Keep the richest response seen so far.
            if len(info) > len(best):
                best = info

            # Require at least some genuinely useful fields before accepting.
            useful = sum(
                info.get(k) is not None
                for k in (
                    "marketCap",
                    "trailingPE",
                    "forwardPE",
                    "trailingEps",
                    "targetMeanPrice",
                    "recommendationKey",
                    "shortName",
                    "longName",
                )
            )

            if useful >= 4:
                best = info
                break

        except Exception:
            pass

        # Retry immediately; caching limits repeated requests across app runs.

    info = best or {}

    market_cap = safe_num(info.get("marketCap"))
    trailing_pe = safe_num(info.get("trailingPE"))
    forward_pe = safe_num(info.get("forwardPE"))
    eps = safe_num(info.get("trailingEps"))
    dividend_rate = safe_num(info.get("dividendRate"))
    rating = rating_label(info)

    target_mean = safe_num(info.get("targetMeanPrice"))
    target_high = safe_num(info.get("targetHighPrice"))
    target_low = safe_num(info.get("targetLowPrice"))

    # Only make the extra analyst-target request if the main info payload
    # did not include a mean target.
    if pd.isna(target_mean):
        try:
            t = yf.Ticker(symbol)
            targets = t.get_analyst_price_targets()
            if isinstance(targets, dict):
                target_mean = safe_num(targets.get("mean"))
                target_high = safe_num(targets.get("high"))
                target_low = safe_num(targets.get("low"))
        except Exception:
            pass

    company = info.get("shortName") or info.get("longName") or symbol

    return {
        "Ticker": symbol,
        "Company": company,
        "Market Cap": market_cap,
        "P/E": trailing_pe,
        "Forward P/E": forward_pe,
        "EPS": eps,
        "Dividend Rate": dividend_rate,
        "Analyst Rating": rating,
        "Analyst Target": target_mean,
        "Analyst Target Low": target_low,
        "Analyst Target High": target_high,
    }


@st.cache_data(ttl=1800, show_spinner=False)
def load_sector(symbols_tuple, max_stocks=MAX_STOCKS):
    """
    Load a sector without liquidity filtering.
    Prices are fetched in one batch; fundamentals are then fetched sequentially
    to avoid overwhelming Yahoo with concurrent requests.
    """
    symbols = list(symbols_tuple)[:int(max_stocks)]

    market = batch_market_data(tuple(symbols))
    if market.empty:
        market = pd.DataFrame({"Ticker": symbols, "Batch Price": [np.nan] * len(symbols)})

    fundamentals = []

    for idx, symbol in enumerate(symbols):
        try:
            fundamentals.append(fetch_fundamentals(symbol))
        except Exception:
            fundamentals.append({"Ticker": symbol, "Company": symbol})


    fdf = pd.DataFrame(fundamentals)
    out = market.merge(fdf, on="Ticker", how="outer")

    order_map = {symbol: i for i, symbol in enumerate(symbols)}
    out["_order"] = out["Ticker"].map(order_map)
    out = out.sort_values("_order").reset_index(drop=True)

    out["Price"] = out["Batch Price"]

    out["Forward Dividend Yield"] = np.nan
    valid_div = (
        out.get("Dividend Rate", pd.Series(index=out.index, dtype=float)).notna()
        & out["Price"].notna()
        & (out["Price"] > 0)
    )
    out.loc[valid_div, "Forward Dividend Yield"] = (
        out.loc[valid_div, "Dividend Rate"] / out.loc[valid_div, "Price"]
    )

    out.loc[
        (out["Forward Dividend Yield"] < 0)
        | (out["Forward Dividend Yield"] > 0.20),
        "Forward Dividend Yield"
    ] = np.nan

    out["Analyst Upside"] = np.where(
        out["Analyst Target"].notna()
        & out["Price"].notna()
        & (out["Price"] != 0),
        out["Analyst Target"] / out["Price"] - 1,
        np.nan,
    )

    return out.drop(
        columns=["Batch Price", "Dividend Rate", "_order"],
        errors="ignore",
    )

def inverse_percentile(series):
    """Lower is better; returns roughly 0..100 within current peer group."""
    s = pd.to_numeric(series, errors="coerce")
    return (1 - s.rank(pct=True, method="average")) * 100

def direct_percentile(series):
    """Higher is better; returns roughly 0..100 within current peer group."""
    s = pd.to_numeric(series, errors="coerce")
    return s.rank(pct=True, method="average") * 100

def analyst_rating_score(series):
    mapping = {
        "Strong Buy": 100,
        "Buy": 80,
        "Hold": 50,
        "Sell": 20,
        "Strong Sell": 0,
    }
    return series.map(mapping).astype(float)

def add_internal_score(df):
    out = df.copy()

    # P/E <= 0 means earnings are zero/negative and should not be rewarded as "cheap".
    pe_for_score = out["P/E"].where(out["P/E"] > 0, np.nan)
    fwd_pe_for_score = out["Forward P/E"].where(out["Forward P/E"] > 0, np.nan)

    pe_score = inverse_percentile(pe_for_score)
    fwd_pe_score = inverse_percentile(fwd_pe_for_score)
    upside_score = direct_percentile(out["Analyst Upside"])
    rating_score = analyst_rating_score(out["Analyst Rating"])
    dividend_score = direct_percentile(out["Forward Dividend Yield"])

    # Missing values are replaced by the neutral midpoint.
    pe_score = pe_score.fillna(50)
    fwd_pe_score = fwd_pe_score.fillna(50)
    upside_score = upside_score.fillna(50)
    rating_score = rating_score.fillna(50)
    dividend_score = dividend_score.fillna(50)

    out["Internal Rating"] = (
        0.15 * pe_score
        + 0.30 * fwd_pe_score
        + 0.30 * upside_score
        + 0.15 * rating_score
        + 0.10 * dividend_score
    ).round(0)

    return out

def style_forward_pe(v, median):
    if pd.isna(v) or pd.isna(median) or median == 0:
        return ""
    rel = v / median - 1
    if rel <= -0.10:
        return "background-color: rgba(46,160,67,0.16); font-weight:600;"
    if rel >= 0.15:
        return "background-color: rgba(248,81,73,0.16); font-weight:600;"
    return ""

def style_upside(v):
    if pd.isna(v):
        return ""
    if v >= 0.10:
        return "background-color: rgba(46,160,67,0.16); font-weight:600;"
    if v < 0:
        return "background-color: rgba(248,81,73,0.16); font-weight:600;"
    return ""

def style_internal_rating(v):
    if pd.isna(v):
        return ""
    if v >= 70:
        return "background-color: rgba(46,160,67,0.20); font-weight:700;"
    if v < 45:
        return "background-color: rgba(248,81,73,0.18); font-weight:700;"
    return "font-weight:700;"

view = st.radio("View", ["Home", "Sector Detail"], horizontal=True, label_visibility="collapsed")

if view == "Home":
    st.title("Whatsupstock")
    st.caption("Simple stock comparison by sector")

    refresh_col, _ = st.columns([1.1, 5])
    with refresh_col:
        if st.button("↻ Refresh Yahoo data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    all_rows, sector_rows, sector_data = [], [], {}
    progress = st.progress(0, text="Loading sectors...")

    for i, (sector_name, symbols) in enumerate(SECTORS.items(), start=1):
        sector_df = load_sector(
            tuple(symbols),
            max_stocks=5
        )
        if not sector_df.empty:
                sector_df = add_internal_score(sector_df)
                sector_df.loc[sector_df["P/E"] <= 0, "P/E"] = np.nan
                sector_df.loc[sector_df["Forward P/E"] <= 0, "Forward P/E"] = np.nan
                sector_df["Sector"] = sector_name
                sector_df["Sector Forward P/E"] = sector_df["Forward P/E"].median(skipna=True)
                all_rows.append(sector_df)
                sector_data[sector_name] = sector_df.copy()
                sector_rows.append({
                    "Sector": sector_name,
                    "Sector Rating": sector_df["Internal Rating"].mean(),
                    "Median P/E": sector_df["P/E"].median(skipna=True),
                    "Median Forward P/E": sector_df["Forward P/E"].median(skipna=True),
                    "Avg Analyst Upside": sector_df["Analyst Upside"].mean(skipna=True),
                    "Avg Dividend Yield": sector_df["Forward Dividend Yield"].mean(skipna=True),
                })
        progress.progress(i / len(SECTORS), text=f"Loading sectors... {i}/{len(SECTORS)}")
    progress.empty()

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)

        # 1. All Sectors
        st.subheader("Top 10 — All Sectors")
        st.caption("Highest Internal Ratings across all covered sectors.")
        top10 = combined.sort_values(["Internal Rating", "Analyst Upside"], ascending=[False, False], na_position="last").head(10).reset_index(drop=True)
        top_display = top10[["Ticker", "Company", "Sector", "Price", "Forward P/E", "Sector Forward P/E", "Forward Dividend Yield", "Analyst Upside", "Internal Rating"]].copy()
        top_display["Forward Dividend Yield"] *= 100
        top_display["Analyst Upside"] *= 100
        top_display.rename(columns={"Sector Forward P/E":"Sector Fwd P/E", "Forward Dividend Yield":"Dividend Yield"}, inplace=True)
        st.dataframe(top_display.style.format({"Price":"${:,.2f}", "Forward P/E":"{:.1f}x", "Sector Fwd P/E":"{:.1f}x", "Dividend Yield":"{:.1f}%", "Analyst Upside":"{:+.1f}%", "Internal Rating":"{:.0f}"}, na_rep="—"), use_container_width=True, hide_index=True, height=390)

        st.divider()

        # 2. Sector Overview
        st.subheader("Sector Overview")
        st.caption("Sectors ordered by average analyst upside, highest first.")
        sector_summary = pd.DataFrame(sector_rows).sort_values("Avg Analyst Upside", ascending=False, na_position="last").reset_index(drop=True)
        left, right = st.columns([1.15, 1.35], gap="large")
        with left:
            chart_df = sector_summary[["Sector", "Avg Analyst Upside"]].copy()
            chart_df["Avg Analyst Upside"] *= 100

            valid_upside = chart_df["Avg Analyst Upside"].dropna()
            max_abs = max(float(valid_upside.abs().max()), 1.0) if not valid_upside.empty else 1.0

            static_rows = ""
            for _, chart_row in chart_df.iterrows():
                sector_label = html.escape(str(chart_row["Sector"]))
                value = chart_row["Avg Analyst Upside"]

                if pd.isna(value):
                    value_text = "—"
                    width = 0
                    bar_color = "#94a3b8"
                else:
                    value_text = f"{value:+.1f}%"
                    width = min(abs(float(value)) / max_abs * 100, 100)
                    bar_color = "#2563eb" if value >= 0 else "#dc2626"

                static_rows += (
                    f'<div style="display:grid;grid-template-columns:150px minmax(80px,1fr) 58px;'
                    f'align-items:center;gap:8px;margin:8px 0;">'
                    f'<div style="font-size:12px;color:#334155;white-space:nowrap;overflow:hidden;'
                    f'text-overflow:ellipsis;" title="{sector_label}">{sector_label}</div>'
                    f'<div style="height:18px;background:#eef2f7;border-radius:4px;overflow:hidden;">'
                    f'<div style="height:100%;width:{width:.1f}%;background:{bar_color};'
                    f'border-radius:4px;"></div></div>'
                    f'<div style="font-size:12px;font-weight:650;text-align:right;color:#334155;">'
                    f'{value_text}</div></div>'
                )

            static_chart_html = (
                '<div style="border:1px solid #e2e8f0;border-radius:8px;padding:12px 14px;'
                'background:white;min-height:500px;">'
                '<div style="font-size:12px;color:#64748b;margin-bottom:10px;">'
                'Average analyst upside by sector</div>'
                + static_rows +
                '</div>'
            )

            st.markdown(static_chart_html, unsafe_allow_html=True)
        with right:
            sector_table = sector_summary[["Sector", "Median P/E", "Median Forward P/E", "Avg Analyst Upside", "Avg Dividend Yield"]].copy()
            sector_table["Avg Analyst Upside"] *= 100
            sector_table["Avg Dividend Yield"] *= 100
            sector_table.rename(columns={"Median Forward P/E":"Median Fwd P/E", "Avg Analyst Upside":"Analyst Upside", "Avg Dividend Yield":"Dividend Yield"}, inplace=True)
            st.dataframe(sector_table.style.format({"Median P/E":"{:.1f}x", "Median Fwd P/E":"{:.1f}x", "Analyst Upside":"{:+.1f}%", "Dividend Yield":"{:.1f}%"}, na_rep="—"), use_container_width=True, hide_index=True, height=520)

        st.divider()
        st.caption("Data: Yahoo Finance via yfinance · For informational purposes only · Market and analyst data may be delayed or temporarily unavailable. Cached ticker snapshots are used to improve stability.")
    else:
        st.warning("No companies were available for this view.")
    st.stop()

st.title("Whatsupstock")
st.caption("Simple stock comparison by sector")

top1, top2 = st.columns([2.2, 1.4])

with top1:
    sector = st.selectbox("Sector", list(SECTORS.keys()), index=0)

with top2:
    max_stocks = st.number_input(
        "Max. stocks",
        min_value=3,
        max_value=10,
        value=MAX_STOCKS,
        step=1,
    )

with st.spinner("Loading market and analyst data..."):
    raw = load_sector(
        tuple(SECTORS[sector]),
        max_stocks=int(max_stocks)
    )

if raw.empty:
    st.error("No data returned.")
    st.stop()

eligible = raw.copy().reset_index(drop=True)

if eligible.empty:
    st.warning("No companies were available for this sector.")
    st.stop()

eligible = add_internal_score(eligible)

# Display non-positive P/E values as unavailable rather than as meaningful valuation multiples.
eligible.loc[eligible["P/E"] <= 0, "P/E"] = np.nan
eligible.loc[eligible["Forward P/E"] <= 0, "Forward P/E"] = np.nan

def internal_rating_label(score):
    if pd.isna(score):
        return "—"
    score = int(round(score))
    if score >= 70:
        return f"🟢 {score} — Attractive"
    if score >= 45:
        return f"🟡 {score} — Neutral"
    return f"🔴 {score} — Unattractive"

eligible["Internal Rating Label"] = eligible["Internal Rating"].map(internal_rating_label)

median_pe = eligible["P/E"].median(skipna=True)
median_forward_pe = eligible["Forward P/E"].median(skipna=True)

c1, c2, c3 = st.columns(3)
c1.metric("Companies", len(eligible))
c2.metric("Median P/E", "—" if pd.isna(median_pe) else f"{median_pe:.1f}x")
c3.metric("Median Forward P/E", "—" if pd.isna(median_forward_pe) else f"{median_forward_pe:.1f}x")

st.divider()

display = eligible[
    [
        "Ticker",
        "Company",
        "Price",
        "Market Cap",
        "P/E",
        "Forward P/E",
        "EPS",
        "Forward Dividend Yield",
        "Analyst Rating",
        "Analyst Target",
        "Analyst Upside",
        "Internal Rating",
        "Internal Rating Label",
    ]
].copy()

# Keep numeric values numeric, so Streamlit sorts them numerically.
display["Market Cap"] = display["Market Cap"] / 1_000_000_000
display["Forward Dividend Yield"] = display["Forward Dividend Yield"] * 100
display["Analyst Upside"] = display["Analyst Upside"] * 100

# Keep target visible as a separate sortable numeric field only in tooltip-like compact form
# is not possible while retaining pure numeric sorting, so we keep a small Target column.
display = display.drop(columns=["Internal Rating"])
display.rename(columns={
    "Analyst Target": "Target",
    "Internal Rating Label": "Internal Rating"
}, inplace=True)

def style_table(df):
    styles = pd.DataFrame("", index=df.index, columns=df.columns)
    for idx in df.index:
        styles.loc[idx, "Forward P/E"] = style_forward_pe(
            eligible.loc[idx, "Forward P/E"], median_forward_pe
        )
        styles.loc[idx, "Analyst Upside"] = style_upside(
            eligible.loc[idx, "Analyst Upside"]
        )
        styles.loc[idx, "Internal Rating"] = style_internal_rating(
            eligible.loc[idx, "Internal Rating"]
        )
    return styles

styled = (
    display.style
    .apply(style_table, axis=None)
    .format({
        "Price": "${:,.2f}",
        "Market Cap": "${:,.1f}B",
        "P/E": "{:.1f}x",
        "Forward P/E": "{:.1f}x",
        "EPS": "${:,.2f}",
        "Forward Dividend Yield": "{:.1f}%",
        "Target": "${:,.2f}",
        "Analyst Upside": "{:.1f}%",
    }, na_rep="—")
)

table_event = st.dataframe(
    styled,
    use_container_width=True,
    hide_index=True,
    height=420,
    on_select="rerun",
    selection_mode="single-row",
)

st.caption(
    "Click any column header to sort. Select one row to open its analyst price-target range below."
)

# Analyst target detail, inspired by the Yahoo-style low / average / high / current view.
selected_rows = []
try:
    selected_rows = table_event.selection.rows
except Exception:
    selected_rows = []

if selected_rows:
    selected_idx = selected_rows[0]
    row = eligible.iloc[selected_idx]

    low = row.get("Analyst Target Low", np.nan)
    avg = row.get("Analyst Target", np.nan)
    high = row.get("Analyst Target High", np.nan)
    current = row.get("Price", np.nan)

    st.subheader(f'{row["Ticker"]} — Analyst Price Targets')

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Low", "—" if pd.isna(low) else f"${low:,.2f}")
    m2.metric("Average", "—" if pd.isna(avg) else f"${avg:,.2f}")
    m3.metric("High", "—" if pd.isna(high) else f"${high:,.2f}")
    m4.metric("Current", "—" if pd.isna(current) else f"${current:,.2f}")

    if not any(pd.isna(v) for v in [low, avg, high, current]) and high > low:
        def pos(v):
            return max(0, min(100, (v - low) / (high - low) * 100))

        avg_pos = pos(avg)
        current_pos = pos(current)

        st.markdown(
            f"""
            <div style="margin:8px 4px 2px 4px;">
              <div style="position:relative;height:42px;">
                <div style="position:absolute;left:0;right:0;top:20px;height:6px;background:#aab2bd;border-radius:6px;"></div>
                <div style="position:absolute;left:{avg_pos}%;top:14px;width:14px;height:14px;background:#2f80ed;border:2px solid white;border-radius:50%;transform:translateX(-50%);"></div>
                <div style="position:absolute;left:{current_pos}%;top:14px;width:14px;height:14px;background:#2d3436;border:2px solid white;border-radius:50%;transform:translateX(-50%);"></div>
              </div>
              <div style="display:flex;justify-content:space-between;font-size:0.9rem;">
                <span><b>${low:,.2f}</b><br>Low</span>
                <span style="text-align:center;"><span style="color:#2f80ed;">●</span> Average ${avg:,.2f}</span>
                <span style="text-align:center;"><span style="color:#2d3436;">●</span> Current ${current:,.2f}</span>
                <span style="text-align:right;"><b>${high:,.2f}</b><br>High</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.caption(
    "Internal Rating (0–100): P/E 15% + Forward P/E 30% + Analyst Upside 30% + "
    "Analyst Rating 15% + Dividend Yield 10%. "
    "It is a relative score within the selected peer group, not an investment recommendation."
)

with st.expander("How the Internal Rating works"):
    st.markdown(
        """
        **Higher score = more attractive relative to the current peer group.**

        - **P/E — 15%:** lower than peers scores better.
        - **Forward P/E — 30%:** lower than peers scores better.
        - **Analyst Upside — 30%:** higher consensus upside scores better.
        - **Analyst Rating — 15%:** Strong Buy > Buy > Hold > Sell > Strong Sell.
        - **Dividend Yield — 10%:** higher than peers scores better.

        Price, Market Cap and EPS are shown for context but do not enter the score.
        The dividend component is relative to the selected peer group, so a higher yield earns more points.
        """
    )


st.divider()
st.caption(
    "Data: Yahoo Finance via yfinance · For informational purposes only · "
    "Market and analyst data may be delayed or unavailable."
)
