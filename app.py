import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import html
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(page_title="Whatsupstock", page_icon="📊", layout="wide")

SECTORS = {
    "Restaurants": ["MCD","SBUX","CMG","YUM","QSR","DRI","TXRH","DPZ","WING","CAVA","CAKE","WEN","JACK","SHAK"],
    "Semiconductors": ["NVDA","AVGO","AMD","QCOM","MU","TXN","INTC","ADI","NXPI","MCHP","ON","MPWR","MRVL","SWKS"],
    "Software": ["MSFT","ORCL","CRM","ADBE","NOW","INTU","PANW","SNOW","PLTR","DDOG","MDB","TEAM","ZS","HUBS"],
    "Technology Hardware": ["AAPL","DELL","HPE","HPQ","ANET","SMCI","NTAP","WDC","STX","PSTG","LOGI","ZBRA","FFIV","JNPR"],
    "Internet & Digital Platforms": ["GOOGL","META","NFLX","UBER","ABNB","SPOT","DASH","PINS","RDDT","SNAP","RBLX","TTD","MTCH","IAC"],
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
    "Consumer Brands": ["PG","KO","PEP","PM","MDLZ","MO","CL","KMB","KHC","GIS","HSY","KDP","STZ","MNST"],
}

HOME_UNIVERSE = {
    "Restaurants": ["MCD","SBUX","CMG","YUM","QSR","DRI","TXRH"],
    "Semiconductors": ["NVDA","AVGO","AMD","QCOM","MU","TXN","INTC"],
    "Software": ["MSFT","ORCL","CRM","ADBE","INTU","NOW","PANW"],
    "Technology Hardware": ["AAPL","DELL","HPE","HPQ","ANET","SMCI","NTAP"],
    "Internet & Digital Platforms": ["GOOGL","META","NFLX","UBER","ABNB","SPOT","DASH"],
    "Banks": ["JPM","BAC","WFC","C","GS","MS","USB"],
    "Pharma": ["LLY","JNJ","ABBV","MRK","PFE","BMY","AMGN"],
    "Energy": ["XOM","CVX","COP","EOG","SLB","MPC","PSX"],
    "Consumer Brands": ["PG","KO","PEP","PM","MDLZ","MO","CL"],
}

HOME_SECTORS = list(HOME_UNIVERSE.keys())
MAX_DETAIL_STOCKS = 10

WEIGHTS = {
    "P/E": 0.15,
    "Forward P/E": 0.30,
    "Analyst Upside": 0.30,
    "Analyst Rating": 0.15,
    "Dividend Yield": 0.10,
}

def safe_num(value):
    try:
        if value is None:
            return np.nan
        return float(value)
    except Exception:
        return np.nan

def rating_label(raw):
    key = str(raw or "").strip().lower()
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

def analyst_rating_score(series):
    mapping = {
        "Strong Buy": 100,
        "Buy": 80,
        "Hold": 50,
        "Sell": 20,
        "Strong Sell": 0,
    }
    return series.map(mapping).astype(float)

def inverse_percentile(series):
    s = pd.to_numeric(series, errors="coerce")
    valid = s.notna()
    out = pd.Series(np.nan, index=s.index, dtype=float)
    if valid.sum() >= 2:
        out.loc[valid] = (1 - s.loc[valid].rank(pct=True, method="average")) * 100
    elif valid.sum() == 1:
        out.loc[valid] = 50
    return out

def direct_percentile(series):
    s = pd.to_numeric(series, errors="coerce")
    valid = s.notna()
    out = pd.Series(np.nan, index=s.index, dtype=float)
    if valid.sum() >= 2:
        out.loc[valid] = s.loc[valid].rank(pct=True, method="average") * 100
    elif valid.sum() == 1:
        out.loc[valid] = 50
    return out

def add_internal_score(df):
    out = df.copy()

    pe_for_score = out["P/E"].where(out["P/E"] > 0, np.nan)
    fwd_pe_for_score = out["Forward P/E"].where(out["Forward P/E"] > 0, np.nan)

    components = pd.DataFrame(index=out.index)
    components["P/E"] = inverse_percentile(pe_for_score)
    components["Forward P/E"] = inverse_percentile(fwd_pe_for_score)
    components["Analyst Upside"] = direct_percentile(out["Analyst Upside"])
    components["Analyst Rating"] = analyst_rating_score(out["Analyst Rating"])
    components["Dividend Yield"] = direct_percentile(out["Dividend Yield"])

    weighted_sum = pd.Series(0.0, index=out.index)
    available_weight = pd.Series(0.0, index=out.index)
    component_count = pd.Series(0, index=out.index, dtype=int)

    for name, weight in WEIGHTS.items():
        valid = components[name].notna()
        weighted_sum.loc[valid] += components.loc[valid, name] * weight
        available_weight.loc[valid] += weight
        component_count.loc[valid] += 1

    out["Internal Rating"] = np.where(
        available_weight > 0,
        weighted_sum / available_weight,
        np.nan
    )
    out["Internal Rating"] = pd.Series(out["Internal Rating"], index=out.index).round(0)
    out["Score Components"] = component_count
    out.loc[out["Score Components"] < 2, "Internal Rating"] = np.nan

    return out

@st.cache_data(ttl=1800, show_spinner=False)
def batch_prices(symbols_tuple):
    symbols = list(symbols_tuple)
    result = {s: np.nan for s in symbols}

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

        for symbol in symbols:
            try:
                sdf = hist if len(symbols) == 1 else hist[symbol]
                closes = pd.to_numeric(sdf["Close"], errors="coerce").dropna()
                if not closes.empty:
                    result[symbol] = safe_num(closes.iloc[-1])
            except Exception:
                pass
    except Exception:
        pass

    return result

@st.cache_data(ttl=21600, show_spinner=False)
def fetch_snapshot(symbol):
    """
    One and only one fundamental Yahoo request per ticker.
    Target fields are taken from the same info payload.
    """
    try:
        info = yf.Ticker(symbol).get_info() or {}
    except Exception:
        info = {}

    return {
        "Ticker": symbol,
        "Company": info.get("shortName") or info.get("longName"),
        "Market Cap": safe_num(info.get("marketCap")),
        "P/E": safe_num(info.get("trailingPE")),
        "Forward P/E": safe_num(info.get("forwardPE")),
        "EPS": safe_num(info.get("trailingEps")),
        "Dividend Rate": safe_num(info.get("dividendRate")),
        "Analyst Rating": rating_label(info.get("recommendationKey")),
        "Analyst Target": safe_num(info.get("targetMeanPrice")),
        "Analyst Target Low": safe_num(info.get("targetLowPrice")),
        "Analyst Target High": safe_num(info.get("targetHighPrice")),
    }

@st.cache_data(ttl=1800, show_spinner=False)
def build_dataset(sector_map_tuple):
    """
    Build one dataset. All UI sections reuse it.
    """
    sector_map = [(name, list(symbols)) for name, symbols in sector_map_tuple]
    all_symbols = []
    sector_for_symbol = {}

    for sector_name, symbols in sector_map:
        for symbol in symbols:
            if symbol not in all_symbols:
                all_symbols.append(symbol)
            sector_for_symbol[symbol] = sector_name

    prices = batch_prices(tuple(all_symbols))

    snapshots = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(fetch_snapshot, s): s for s in all_symbols}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                row = future.result()
            except Exception:
                row = {"Ticker": symbol}
            snapshots.append(row)

    df = pd.DataFrame(snapshots)
    if df.empty:
        return df

    for col in [
        "Company","Market Cap","P/E","Forward P/E","EPS","Dividend Rate",
        "Analyst Rating","Analyst Target","Analyst Target Low","Analyst Target High"
    ]:
        if col not in df.columns:
            df[col] = np.nan

    df["Sector"] = df["Ticker"].map(sector_for_symbol)
    df["Price"] = df["Ticker"].map(prices)
    df["Company"] = df["Company"].where(df["Company"].notna(), df["Ticker"])

    df["Dividend Yield"] = np.where(
        df["Dividend Rate"].notna()
        & df["Price"].notna()
        & (df["Price"] > 0),
        df["Dividend Rate"] / df["Price"],
        np.nan,
    )
    df.loc[
        (df["Dividend Yield"] < 0) | (df["Dividend Yield"] > 0.20),
        "Dividend Yield"
    ] = np.nan

    df["Analyst Upside"] = np.where(
        df["Analyst Target"].notna()
        & df["Price"].notna()
        & (df["Price"] != 0),
        df["Analyst Target"] / df["Price"] - 1,
        np.nan,
    )

    scored = []
    for sector_name, _ in sector_map:
        sdf = df[df["Sector"] == sector_name].copy().reset_index(drop=True)
        if sdf.empty:
            continue

        sdf = add_internal_score(sdf)
        sdf.loc[sdf["P/E"] <= 0, "P/E"] = np.nan
        sdf.loc[sdf["Forward P/E"] <= 0, "Forward P/E"] = np.nan
        sdf["Sector Forward P/E"] = sdf["Forward P/E"].median(skipna=True)
        scored.append(sdf)

    return pd.concat(scored, ignore_index=True) if scored else pd.DataFrame()

def home_dataset():
    return build_dataset(
        tuple((sector, tuple(HOME_UNIVERSE[sector])) for sector in HOME_SECTORS)
    )

def detail_dataset(sector, max_stocks):
    symbols = SECTORS[sector][:int(max_stocks)]
    return build_dataset(((sector, tuple(symbols)),))

def rating_text(score):
    if pd.isna(score):
        return "—"
    score = int(round(score))
    if score >= 70:
        return f"🟢 {score} — Attractive"
    if score >= 45:
        return f"🟡 {score} — Neutral"
    return f"🔴 {score} — Unattractive"

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

def static_sector_overview(sector_summary):
    chart_df = sector_summary[["Sector", "Avg Analyst Upside"]].copy()
    chart_df["Avg Analyst Upside"] *= 100
    valid = chart_df["Avg Analyst Upside"].dropna()
    max_abs = max(float(valid.abs().max()), 1.0) if not valid.empty else 1.0

    rows_html = ""
    for _, row in chart_df.iterrows():
        sector = html.escape(str(row["Sector"]))
        value = row["Avg Analyst Upside"]

        if pd.isna(value):
            value_text = "—"
            width = 0
            color = "#94a3b8"
        else:
            value_text = f"{value:+.1f}%"
            width = min(abs(float(value)) / max_abs * 100, 100)
            color = "#2563eb" if value >= 0 else "#dc2626"

        rows_html += (
            f'<div style="display:grid;grid-template-columns:160px minmax(80px,1fr) 60px;'
            f'align-items:center;gap:8px;margin:9px 0;">'
            f'<div style="font-size:12px;color:#334155;white-space:nowrap;overflow:hidden;'
            f'text-overflow:ellipsis;" title="{sector}">{sector}</div>'
            f'<div style="height:18px;background:#eef2f7;border-radius:4px;overflow:hidden;">'
            f'<div style="height:100%;width:{width:.1f}%;background:{color};border-radius:4px;"></div></div>'
            f'<div style="font-size:12px;font-weight:650;text-align:right;color:#334155;">'
            f'{value_text}</div></div>'
        )

    return (
        '<div style="border:1px solid #e2e8f0;border-radius:8px;padding:12px 14px;'
        'background:white;min-height:420px;">'
        '<div style="font-size:12px;color:#64748b;margin-bottom:10px;">'
        'Average analyst upside by sector</div>'
        + rows_html +
        '</div>'
    )


TOP10_SORT_OPTIONS = {
    "Internal Rating": {"column": "Internal Rating", "default_ascending": False},
    "Analyst Upside": {"column": "Analyst Upside", "default_ascending": False},
    "Forward P/E": {"column": "Forward P/E", "default_ascending": True},
    "Dividend Yield": {"column": "Dividend Yield", "default_ascending": False},
    "P/E": {"column": "P/E", "default_ascending": True},
}

def init_top10_sort_state():
    if "top10_sort_metric" not in st.session_state:
        st.session_state["top10_sort_metric"] = "Internal Rating"
    if "top10_sort_ascending" not in st.session_state:
        st.session_state["top10_sort_ascending"] = TOP10_SORT_OPTIONS["Internal Rating"]["default_ascending"]

def set_top10_sort(metric):
    init_top10_sort_state()

    if st.session_state["top10_sort_metric"] == metric:
        st.session_state["top10_sort_ascending"] = not st.session_state["top10_sort_ascending"]
    else:
        st.session_state["top10_sort_metric"] = metric
        st.session_state["top10_sort_ascending"] = TOP10_SORT_OPTIONS[metric]["default_ascending"]

view = st.radio(
    "View",
    ["Home", "Sector Detail"],
    horizontal=True,
    label_visibility="collapsed"
)

if view == "Home":
    st.title("Whatsupstock")
    st.caption("Simple stock comparison by sector")

    refresh_col, _ = st.columns([1.1, 5])
    with refresh_col:
        if st.button("↻ Refresh Home data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    with st.spinner("Loading Home data..."):
        combined = home_dataset()

    if combined.empty:
        st.error("Yahoo Finance did not return enough data for the Home page.")
        st.stop()

    usable = (
        combined[["P/E","Forward P/E","EPS","Analyst Target"]]
        .notna()
        .any(axis=1)
        .sum()
    )
    st.caption(f"Usable fundamentals loaded for {usable} of {len(combined)} Home stocks.")

    st.subheader("Top 10 — All Sectors")
    st.caption("Click a criterion to rebuild the Top 10 from all Home stocks. Click the same criterion again to reverse the order.")

    init_top10_sort_state()

    metric_cols = st.columns(5, gap="small")
    metric_names = list(TOP10_SORT_OPTIONS.keys())

    for col, metric in zip(metric_cols, metric_names):
        with col:
            is_active = st.session_state["top10_sort_metric"] == metric
            arrow = ""
            if is_active:
                arrow = " ↑" if st.session_state["top10_sort_ascending"] else " ↓"

            if st.button(
                f"{metric}{arrow}",
                key=f"top10_sort_{metric}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                set_top10_sort(metric)
                st.rerun()

    sort_metric = st.session_state["top10_sort_metric"]
    sort_column = TOP10_SORT_OPTIONS[sort_metric]["column"]
    sort_ascending = st.session_state["top10_sort_ascending"]

    top10_source = combined[combined[sort_column].notna()].copy()

    # Secondary sort keeps results deterministic when values tie.
    secondary_column = "Internal Rating" if sort_column != "Internal Rating" else "Analyst Upside"
    secondary_ascending = False

    top10 = (
        top10_source
        .sort_values(
            [sort_column, secondary_column],
            ascending=[sort_ascending, secondary_ascending],
            na_position="last"
        )
        .head(10)
        .reset_index(drop=True)
    )

    direction_text = "lowest first" if sort_ascending else "highest first"
    st.caption(f"Current ranking: **{sort_metric} — {direction_text}**")

    top_display = top10[
        [
            "Ticker","Company","Sector","Price","Forward P/E",
            "Sector Forward P/E","Dividend Yield","Analyst Upside","Internal Rating"
        ]
    ].copy()

    top_display["Dividend Yield"] *= 100
    top_display["Analyst Upside"] *= 100
    top_display.rename(columns={"Sector Forward P/E":"Sector Fwd P/E"}, inplace=True)

    st.dataframe(
        top_display.style.format(
            {
                "Price":"${:,.2f}",
                "Forward P/E":"{:.1f}x",
                "Sector Fwd P/E":"{:.1f}x",
                "Dividend Yield":"{:.1f}%",
                "Analyst Upside":"{:+.1f}%",
                "Internal Rating":"{:.0f}",
            },
            na_rep="—"
        ),
        use_container_width=True,
        hide_index=True,
        height=390
    )

    st.divider()
    st.subheader("Top 3 — Each Sector")
    st.caption("No new Yahoo requests are made here.")

    sector_icons = {
        "Restaurants":"🍽️",
        "Semiconductors":"◈",
        "Software":"⌘",
        "Technology Hardware":"▣",
        "Internet & Digital Platforms":"◎",
        "Banks":"🏦",
        "Pharma":"⚕",
        "Energy":"⚡",
        "Consumer Brands":"◉",
    }

    colors = [
        "#4f46e5","#16a34a","#7c3aed",
        "#ea580c","#0891b2","#2563eb",
        "#059669","#ca8a04","#0f766e"
    ]

    for row_start in range(0, len(HOME_SECTORS), 3):
        row_sectors = HOME_SECTORS[row_start:row_start+3]
        cols = st.columns(3, gap="small")

        for offset, (col, sector_name) in enumerate(zip(cols, row_sectors)):
            sdf = combined[combined["Sector"] == sector_name].copy()
            sdf = (
                sdf[sdf["Internal Rating"].notna()]
                .sort_values(
                    ["Internal Rating","Analyst Upside"],
                    ascending=[False,False],
                    na_position="last"
                )
                .head(3)
                .reset_index(drop=True)
            )

            accent = colors[(row_start + offset) % len(colors)]
            icon = sector_icons.get(sector_name, "●")

            rows_html = ""
            if sdf.empty:
                rows_html = (
                    '<div style="font-size:11px;color:#94a3b8;padding:12px 0;">'
                    'Insufficient Yahoo data</div>'
                )
            else:
                for rank, (_, stock) in enumerate(sdf.iterrows(), start=1):
                    ticker = html.escape(str(stock["Ticker"]))
                    company = html.escape(str(stock["Company"]))
                    rating = f'{stock["Internal Rating"]:.0f}'
                    rows_html += (
                        f'<div style="display:grid;grid-template-columns:22px 52px minmax(0,1fr) 38px;'
                        f'align-items:center;gap:6px;padding:6px 0;font-size:11px;">'
                        f'<span style="color:#64748b;">{rank}</span>'
                        f'<span style="font-weight:650;color:#0f172a;">{ticker}</span>'
                        f'<span style="color:#334155;white-space:nowrap;overflow:hidden;'
                        f'text-overflow:ellipsis;" title="{company}">{company}</span>'
                        f'<span style="font-weight:700;color:#16a34a;text-align:right;">{rating}</span>'
                        f'</div>'
                    )

            card_html = (
                f'<div style="border:1px solid #dbe2ea;border-radius:8px;padding:11px 12px 9px 12px;'
                f'background:white;min-height:155px;margin-bottom:10px;">'
                f'<div style="font-size:13px;font-weight:750;color:{accent};margin-bottom:9px;">'
                f'<span style="margin-right:5px;">{icon}</span>{html.escape(sector_name)}</div>'
                f'<div style="display:grid;grid-template-columns:22px 52px minmax(0,1fr) 38px;'
                f'gap:6px;color:#64748b;font-size:9px;padding-bottom:4px;'
                f'border-bottom:1px solid #eef2f6;">'
                f'<span>#</span><span>Ticker</span><span>Company</span>'
                f'<span style="text-align:right;">Rating</span></div>'
                f'{rows_html}</div>'
            )

            with col:
                st.markdown(card_html, unsafe_allow_html=True)

    st.divider()
    st.subheader("Sector Overview")
    st.caption("Built from the same Home dataset.")

    rows = []
    for sector_name in HOME_SECTORS:
        sdf = combined[combined["Sector"] == sector_name].copy()
        rows.append({
            "Sector":sector_name,
            "Median P/E":sdf["P/E"].median(skipna=True),
            "Median Forward P/E":sdf["Forward P/E"].median(skipna=True),
            "Avg Analyst Upside":sdf["Analyst Upside"].mean(skipna=True),
            "Avg Dividend Yield":sdf["Dividend Yield"].mean(skipna=True),
        })

    sector_summary = (
        pd.DataFrame(rows)
        .sort_values("Avg Analyst Upside", ascending=False, na_position="last")
        .reset_index(drop=True)
    )

    left, right = st.columns([1.15, 1.35], gap="large")

    with left:
        st.markdown(static_sector_overview(sector_summary), unsafe_allow_html=True)

    with right:
        sector_table = sector_summary.copy()
        sector_table["Avg Analyst Upside"] *= 100
        sector_table["Avg Dividend Yield"] *= 100
        sector_table.rename(
            columns={
                "Median Forward P/E":"Median Fwd P/E",
                "Avg Analyst Upside":"Analyst Upside",
                "Avg Dividend Yield":"Dividend Yield",
            },
            inplace=True
        )

        st.dataframe(
            sector_table.style.format(
                {
                    "Median P/E":"{:.1f}x",
                    "Median Fwd P/E":"{:.1f}x",
                    "Analyst Upside":"{:+.1f}%",
                    "Dividend Yield":"{:.1f}%",
                },
                na_rep="—"
            ),
            use_container_width=True,
            hide_index=True,
            height=440
        )

    st.divider()
    st.caption(
        "Data: Yahoo Finance via yfinance · For informational purposes only · "
        "One shared Home data collection is reused by every Home section."
    )
    st.stop()

st.title("Whatsupstock")
st.caption("Sector Detail")

top1, top2 = st.columns([2.2, 1.4])

with top1:
    sector = st.selectbox("Sector", list(SECTORS.keys()), index=0)

with top2:
    max_stocks = st.number_input(
        "Max. stocks",
        min_value=3,
        max_value=10,
        value=10,
        step=1,
    )

with st.spinner("Loading sector data..."):
    eligible = detail_dataset(sector, max_stocks)

if eligible.empty:
    st.error("No data returned for this sector.")
    st.stop()

median_pe = eligible["P/E"].median(skipna=True)
median_forward_pe = eligible["Forward P/E"].median(skipna=True)

c1, c2, c3 = st.columns(3)
c1.metric("Companies", len(eligible))
c2.metric("Median P/E", "—" if pd.isna(median_pe) else f"{median_pe:.1f}x")
c3.metric("Median Forward P/E", "—" if pd.isna(median_forward_pe) else f"{median_forward_pe:.1f}x")

st.divider()

eligible["Internal Rating Label"] = eligible["Internal Rating"].map(rating_text)

display = eligible[
    [
        "Ticker","Company","Price","Market Cap","P/E","Forward P/E","EPS",
        "Dividend Yield","Analyst Rating","Analyst Target","Analyst Upside",
        "Internal Rating","Internal Rating Label"
    ]
].copy()

display["Market Cap"] = display["Market Cap"] / 1_000_000_000
display["Dividend Yield"] *= 100
display["Analyst Upside"] *= 100

display = display.drop(columns=["Internal Rating"])
display.rename(
    columns={
        "Analyst Target":"Target",
        "Internal Rating Label":"Internal Rating"
    },
    inplace=True
)

def style_table(df):
    styles = pd.DataFrame("", index=df.index, columns=df.columns)
    for idx in df.index:
        styles.loc[idx, "Forward P/E"] = style_forward_pe(
            eligible.loc[idx, "Forward P/E"], median_forward_pe
        )
        styles.loc[idx, "Analyst Upside"] = style_upside(
            eligible.loc[idx, "Analyst Upside"]
        )
    return styles

styled = (
    display.style
    .apply(style_table, axis=None)
    .format(
        {
            "Price":"${:,.2f}",
            "Market Cap":"${:,.1f}B",
            "P/E":"{:.1f}x",
            "Forward P/E":"{:.1f}x",
            "EPS":"${:,.2f}",
            "Dividend Yield":"{:.1f}%",
            "Target":"${:,.2f}",
            "Analyst Upside":"{:.1f}%",
        },
        na_rep="—"
    )
)

table_event = st.dataframe(
    styled,
    use_container_width=True,
    hide_index=True,
    height=420,
    on_select="rerun",
    selection_mode="single-row"
)

st.caption(
    "Click any column header to sort. Select one row to open its analyst price-target range below."
)

selected_rows = []
try:
    selected_rows = table_event.selection.rows
except Exception:
    pass

if selected_rows:
    row = eligible.iloc[selected_rows[0]]

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

    values = [v for v in [low, avg, high, current] if not pd.isna(v)]
    if len(values) >= 2:
        scale_min = min(values)
        scale_max = max(values)

        if scale_max > scale_min:
            def pos(v):
                return (v - scale_min) / (scale_max - scale_min) * 100

            avg_pos = pos(avg) if not pd.isna(avg) else None
            current_pos = pos(current) if not pd.isna(current) else None

            avg_dot = (
                f'<div style="position:absolute;left:{avg_pos:.1f}%;top:14px;width:14px;height:14px;'
                f'background:#2f80ed;border:2px solid white;border-radius:50%;transform:translateX(-50%);"></div>'
                if avg_pos is not None else ""
            )
            current_dot = (
                f'<div style="position:absolute;left:{current_pos:.1f}%;top:14px;width:14px;height:14px;'
                f'background:#2d3436;border:2px solid white;border-radius:50%;transform:translateX(-50%);"></div>'
                if current_pos is not None else ""
            )

            st.markdown(
                f"""
                <div style="margin:8px 4px 2px 4px;">
                  <div style="position:relative;height:42px;">
                    <div style="position:absolute;left:0;right:0;top:20px;height:6px;background:#aab2bd;border-radius:6px;"></div>
                    {avg_dot}
                    {current_dot}
                  </div>
                </div>
                """,
                unsafe_allow_html=True
            )

st.caption(
    "Internal Rating: P/E 15% + Forward P/E 30% + Analyst Upside 30% + "
    "Analyst Rating 15% + Dividend Yield 10%. Missing metrics are ignored and "
    "remaining weights are re-normalized; at least two score components are required."
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

        Missing metrics are not converted automatically into a neutral 50.
        The available weights are re-normalized.
        """
    )

st.divider()
st.caption(
    "Data: Yahoo Finance via yfinance · For informational purposes only · "
    "Market and analyst fields may occasionally be unavailable or delayed."
)
