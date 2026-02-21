import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta

# --- CONFIG ---
st.set_page_config(page_title="Dashboard", layout="wide")

TICKERS = ["EEM", "BRK-B", "META", "MSFT", "ASML", "SPY", "TSM", "VEA"]
ETF_TICKERS = {"EEM", "SPY", "VEA"}

# Sector / industria / geografia -- enriquecimiento manual
TICKER_META = {
    "EEM":   {"sector": "ETF",            "industry": "Emerging Markets Equity",         "geo": "Global EM"},
    "SPY":   {"sector": "ETF",            "industry": "US Large Cap Blend",              "geo": "USA"},
    "VEA":   {"sector": "ETF",            "industry": "Developed Markets ex-US Equity",  "geo": "Intl DM"},
    "BRK-B": {"sector": "Financials",     "industry": "Insurance / Diversified Holdings","geo": "USA"},
    "META":  {"sector": "Comm. Services", "industry": "Social Media / Digital Ads",      "geo": "USA"},
    "MSFT":  {"sector": "Technology",     "industry": "Cloud / Enterprise Software",     "geo": "USA"},
    "ASML":  {"sector": "Technology",     "industry": "Semiconductor Equipment",         "geo": "Netherlands"},
    "TSM":   {"sector": "Technology",     "industry": "Semiconductor Manufacturing",     "geo": "Taiwan"},
}
BENCHMARK = "SPY"
RISK_FREE_RATE = 0.05

FMP_BASE = "https://financialmodelingprep.com/api/v3"

# ─── FMP HELPERS ──────────────────────────────────────────────────────────────

def _fmp_key():
    """Get FMP key from secrets or session state."""
    try:
        k = st.secrets.get("FMP_API_KEY", "")
        if k:
            return k
    except Exception:
        pass
    return st.session_state.get("fmp_api_key", "")

@st.cache_data(ttl=3600)
def fmp_get(endpoint, ticker, api_key, params=None):
    if not api_key:
        return []
    url = f"{FMP_BASE}/{endpoint}/{ticker}"
    p = {"apikey": api_key}
    if params:
        p.update(params)
    try:
        r = requests.get(url, params=p, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return []

def get_key_metrics_history(ticker, api_key, limit=10):
    data = fmp_get("key-metrics", ticker, api_key, {"limit": limit, "period": "annual"})
    if not data or isinstance(data, dict):
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if df.empty or "date" not in df.columns:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")

def get_ratios_history(ticker, api_key, limit=10):
    data = fmp_get("ratios", ticker, api_key, {"limit": limit, "period": "annual"})
    if not data or isinstance(data, dict):
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if df.empty or "date" not in df.columns:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")

def get_analyst_estimates(ticker, api_key, limit=5):
    data = fmp_get("analyst-estimates", ticker, api_key, {"limit": limit, "period": "annual"})
    if not data or isinstance(data, dict):
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if df.empty or "date" not in df.columns:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")

def get_price_target(ticker, api_key):
    data = fmp_get("price-target-consensus", ticker, api_key)
    if not data or isinstance(data, dict):
        return {}
    return data[0] if isinstance(data, list) and data else {}

def get_price_target_history(ticker, api_key, limit=15):
    data = fmp_get("price-target", ticker, api_key, {"limit": limit})
    if not data or isinstance(data, dict):
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if df.empty:
        return pd.DataFrame()
    if "publishedDate" in df.columns:
        df["date"] = pd.to_datetime(df["publishedDate"])
    return df

def get_income_history(ticker, api_key, limit=10):
    data = fmp_get("income-statement", ticker, api_key, {"limit": limit, "period": "annual"})
    if not data or isinstance(data, dict):
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if df.empty or "date" not in df.columns:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")

def get_cashflow_history(ticker, api_key, limit=10):
    data = fmp_get("cash-flow-statement", ticker, api_key, {"limit": limit, "period": "annual"})
    if not data or isinstance(data, dict):
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if df.empty or "date" not in df.columns:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")

# ─── YFINANCE ─────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_data(tickers, period="2y"):
    return yf.download(tickers, period=period, auto_adjust=True)

@st.cache_data(ttl=3600)
def get_info(tickers):
    info = {}
    for t in tickers:
        try:
            i = yf.Ticker(t).info
            info[t] = {
                "sector": i.get("sector", "N/A"),
                "dividend_yield": i.get("dividendYield", 0) or 0,
                "pe_ratio": i.get("trailingPE", None),
                "pe_forward": i.get("forwardPE", None),
                "peg_ratio": i.get("pegRatio", None),
                "ps_ratio": i.get("priceToSalesTrailing12Months", None),
                "pb_ratio": i.get("priceToBook", None),
                "ev_ebitda": i.get("enterpriseToEbitda", None),
                "market_cap": i.get("marketCap", None),
                "revenue_growth": i.get("revenueGrowth", None),
                "earnings_growth": i.get("earningsGrowth", None),
                "operating_margins": i.get("operatingMargins", None),
                "profit_margins": i.get("profitMargins", None),
                "return_on_equity": i.get("returnOnEquity", None),
                "return_on_assets": i.get("returnOnAssets", None),
                "debt_to_equity": i.get("debtToEquity", None),
                "buyback_yield": i.get("buybackYield", None),
            }
        except Exception:
            info[t] = {k: None for k in [
                "sector", "pe_ratio", "pe_forward", "peg_ratio", "ps_ratio", "pb_ratio",
                "ev_ebitda", "market_cap", "revenue_growth", "earnings_growth",
                "operating_margins", "profit_margins", "return_on_equity", "return_on_assets",
                "debt_to_equity", "buyback_yield"
            ]}
            info[t]["dividend_yield"] = 0
    return info

# ─── PRICE METRICS ────────────────────────────────────────────────────────────

def calc_return(prices, ticker, days=None, start_date=None):
    s = prices[ticker].dropna()
    if s.empty:
        return None
    if start_date is not None:
        mask = s.index >= pd.Timestamp(start_date)
        prev = s[mask].iloc[0] if mask.any() else None
    elif days is not None:
        mask = s.index >= s.index[-1] - timedelta(days=days)
        prev = s[mask].iloc[0] if mask.any() else s.iloc[0]
    else:
        prev = s.iloc[0]
    if prev is None or prev == 0:
        return None
    return (s.iloc[-1] / prev) - 1

def calc_mtd(prices, ticker):
    s = prices[ticker].dropna()
    if s.empty:
        return None
    mask = s.index < s.index[-1].replace(day=1)
    prev = s[mask].iloc[-1] if mask.any() else s.iloc[0]
    return (s.iloc[-1] / prev) - 1

def calc_ytd(prices, ticker):
    s = prices[ticker].dropna()
    if s.empty:
        return None
    return calc_return(prices, ticker, start_date=datetime(s.index[-1].year, 1, 1))

def calc_sharpe(prices, ticker, days=252):
    s = prices[ticker].dropna()
    if len(s) < 2:
        return None
    r = s.pct_change().dropna().tail(days)
    vol = r.std() * np.sqrt(252)
    return (r.mean() * 252 - RISK_FREE_RATE) / vol if vol else None

def calc_sortino(prices, ticker, days=252):
    s = prices[ticker].dropna()
    if len(s) < 2:
        return None
    r = s.pct_change().dropna().tail(days)
    down = r[r < 0].std() * np.sqrt(252)
    return (r.mean() * 252 - RISK_FREE_RATE) / down if down else None

def calc_max_drawdown(prices, ticker):
    s = prices[ticker].dropna()
    if s.empty:
        return None
    return ((s - s.cummax()) / s.cummax()).min()

def calc_current_drawdown(prices, ticker):
    s = prices[ticker].dropna()
    if s.empty:
        return None
    peak = s.cummax().iloc[-1]
    return (s.iloc[-1] - peak) / peak if peak else None

def calc_calmar(prices, ticker):
    r = calc_return(prices, ticker, days=252)
    dd = calc_max_drawdown(prices, ticker)
    if r is None or dd is None or dd == 0:
        return None
    return r / abs(dd)

def calc_beta(prices, ticker, days=252):
    if ticker == BENCHMARK:
        return 1.0
    s = prices[[ticker, BENCHMARK]].dropna()
    if len(s) < 2:
        return None
    r = s.pct_change().dropna().tail(days)
    cov = r.cov()
    var = cov.loc[BENCHMARK, BENCHMARK]
    return cov.loc[ticker, BENCHMARK] / var if var else None

def calc_alpha(prices, ticker, days=252):
    if ticker == BENCHMARK:
        return 0.0
    beta = calc_beta(prices, ticker, days)
    ra = calc_return(prices, ticker, days=days)
    rb = calc_return(prices, BENCHMARK, days=days)
    if any(v is None for v in [beta, ra, rb]):
        return None
    return ra - (RISK_FREE_RATE + beta * (rb - RISK_FREE_RATE))

# ─── FORMAT ───────────────────────────────────────────────────────────────────

def fp(x):
    return f"{x:.2%}" if pd.notna(x) and x is not None else "N/A"

def fn(x, d=2):
    return f"{x:.{d}f}" if pd.notna(x) and x is not None else "N/A"

def fpr(x):
    return f"${x:,.2f}" if pd.notna(x) and x is not None else "N/A"

def fmc(x):
    if x is None or not pd.notna(x):
        return "N/A"
    if x >= 1e12:
        return f"${x/1e12:.1f}T"
    if x >= 1e9:
        return f"${x/1e9:.1f}B"
    return f"${x/1e6:.0f}M"

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.subheader("⚙️ Configuración")
    fmp_input = st.text_input("FMP API Key", type="password",
                               value=st.session_state.get("fmp_api_key", ""),
                               help="Gratis en financialmodelingprep.com — 250 req/día")
    if fmp_input:
        st.session_state["fmp_api_key"] = fmp_input
        st.success("Key cargada ✓")
    elif not _fmp_key():
        st.warning("Sin API Key: tabs Fundamentals y Analistas no disponibles.")

FMP_KEY = _fmp_key()

# ─── LOAD DATA ────────────────────────────────────────────────────────────────

with st.spinner("Cargando datos de mercado..."):
    data = load_data(TICKERS, period="2y")
    prices = data["Close"]
    info = get_info(TICKERS)

st.title("📊 Dashboard")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Resumen", "Valuación", "Fundamentals", "Analistas", "Rendimiento & Corr"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — RESUMEN
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    rows = []
    for t in TICKERS:
        s = prices[t].dropna()
        if s.empty:
            continue
        rows.append({
            "Ticker": t,
            "Sector": TICKER_META.get(t, {}).get("sector", "N/A"),
            "Industria": TICKER_META.get(t, {}).get("industry", "N/A"),
            "Geo": TICKER_META.get(t, {}).get("geo", "N/A"),
            "Precio": s.iloc[-1],
            "Var. Diaria": s.pct_change().iloc[-1] if len(s) > 1 else 0,
            "MTD": calc_mtd(prices, t),
            "1M": calc_return(prices, t, days=30),
            "3M": calc_return(prices, t, days=90),
            "YTD": calc_ytd(prices, t),
            "1Y": calc_return(prices, t, days=365),
            "Sharpe": calc_sharpe(prices, t),
            "Sortino": calc_sortino(prices, t),
            "Calmar": calc_calmar(prices, t),
            "Max DD": calc_max_drawdown(prices, t),
            "DD Actual": calc_current_drawdown(prices, t),
            "Beta": calc_beta(prices, t),
            "Alpha (1Y)": calc_alpha(prices, t),
        })

    df_summary = pd.DataFrame(rows)
    df_disp = df_summary.copy()
    for col in ["Var. Diaria", "MTD", "1M", "3M", "YTD", "1Y", "Max DD", "DD Actual", "Alpha (1Y)"]:
        df_disp[col] = df_disp[col].apply(fp)
    df_disp["Precio"] = df_disp["Precio"].apply(fpr)
    for col in ["Sharpe", "Sortino", "Calmar", "Beta"]:
        df_disp[col] = df_disp[col].apply(fn)
    st.dataframe(df_disp.set_index("Ticker"), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — VALUACIÓN (snapshot yfinance)
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Múltiplos — snapshot actual")
    val_rows = []
    for t in TICKERS:
        i = info[t]
        sh = (i["dividend_yield"] or 0) + (i["buyback_yield"] or 0) if i["buyback_yield"] else i["dividend_yield"]
        val_rows.append({
            "Ticker": t,
            "Sector": TICKER_META.get(t, {}).get("sector", "N/A"),
            "Industria": TICKER_META.get(t, {}).get("industry", "N/A"),
            "Geo": TICKER_META.get(t, {}).get("geo", "N/A"),
            "Market Cap": i["market_cap"],
            "P/E Trailing": i["pe_ratio"],
            "P/E Forward": i["pe_forward"],
            "PEG": i["peg_ratio"],
            "P/S": i["ps_ratio"],
            "P/B": i["pb_ratio"],
            "EV/EBITDA": i["ev_ebitda"],
            "Rev. Growth": i["revenue_growth"],
            "EPS Growth": i["earnings_growth"],
            "Mg. Operativo": i["operating_margins"],
            "Mg. Neto": i["profit_margins"],
            "ROE": i["return_on_equity"],
            "ROA": i["return_on_assets"],
            "Deuda/Equity": i["debt_to_equity"],
            "Div. Yield": i["dividend_yield"],
            "Shareholder Yield": sh,
        })

    df_val = pd.DataFrame(val_rows)
    df_vd = df_val.copy()
    df_vd["Market Cap"] = df_vd["Market Cap"].apply(fmc)
    for col in ["P/E Trailing", "P/E Forward", "PEG", "P/S", "P/B", "EV/EBITDA", "Deuda/Equity"]:
        df_vd[col] = df_vd[col].apply(lambda x: fn(x, 1))
    for col in ["Rev. Growth", "EPS Growth", "Mg. Operativo", "Mg. Neto", "ROE", "ROA",
                "Div. Yield", "Shareholder Yield"]:
        df_vd[col] = df_vd[col].apply(fp)
    st.dataframe(df_vd.set_index("Ticker"), use_container_width=True)

    # Scatter
    st.subheader("P/E Forward vs Crecimiento EPS")
    sc = df_val.dropna(subset=["P/E Forward", "EPS Growth"]).copy()
    if not sc.empty:
        fig_s = px.scatter(sc, x="EPS Growth", y="P/E Forward", text="Ticker",
                           template="plotly_white",
                           labels={"EPS Growth": "EPS Growth YoY", "P/E Forward": "P/E Forward"})
        fig_s.update_traces(textposition="top center", marker=dict(size=12))
        fig_s.update_layout(height=420)
        st.plotly_chart(fig_s, use_container_width=True)

    ca, cb = st.columns(2)
    with ca:
        st.subheader("Márgenes")
        mg = df_val[["Ticker", "Mg. Operativo", "Mg. Neto"]].dropna(subset=["Mg. Operativo"])
        if not mg.empty:
            mg_m = mg.melt(id_vars="Ticker", var_name="Margen", value_name="Valor")
            mg_m["Valor"] *= 100
            fig_mg = px.bar(mg_m, x="Ticker", y="Valor", color="Margen",
                            barmode="group", template="plotly_white", labels={"Valor": "%"})
            fig_mg.update_layout(height=350)
            st.plotly_chart(fig_mg, use_container_width=True)
    with cb:
        st.subheader("ROE y ROA")
        roa = df_val[["Ticker", "ROE", "ROA"]].dropna(subset=["ROE"])
        if not roa.empty:
            roa_m = roa.melt(id_vars="Ticker", var_name="Ratio", value_name="Valor")
            roa_m["Valor"] *= 100
            fig_roa = px.bar(roa_m, x="Ticker", y="Valor", color="Ratio",
                             barmode="group", template="plotly_white", labels={"Valor": "%"})
            fig_roa.update_layout(height=350)
            st.plotly_chart(fig_roa, use_container_width=True)

    # Sector / Geo breakdown
    st.subheader("Composición por sector y geografía")
    col_sec, col_geo = st.columns(2)
    with col_sec:
        sec_df = pd.DataFrame([{"Ticker": t, "Sector": TICKER_META.get(t, {}).get("sector", "N/A")} for t in TICKERS])
        fig_sec = px.pie(sec_df, names="Sector", title="Por sector",
                         template="plotly_white", hole=0.4)
        fig_sec.update_layout(height=320)
        st.plotly_chart(fig_sec, use_container_width=True)
    with col_geo:
        geo_df = pd.DataFrame([{"Ticker": t, "Geo": TICKER_META.get(t, {}).get("geo", "N/A")} for t in TICKERS])
        fig_geo = px.pie(geo_df, names="Geo", title="Por geografía",
                         template="plotly_white", hole=0.4)
        fig_geo.update_layout(height=320)
        st.plotly_chart(fig_geo, use_container_width=True)

    # Industria detail table
    ind_df = pd.DataFrame([{
        "Ticker": t,
        "Sector": TICKER_META.get(t, {}).get("sector", "N/A"),
        "Industria": TICKER_META.get(t, {}).get("industry", "N/A"),
        "Geo": TICKER_META.get(t, {}).get("geo", "N/A"),
    } for t in TICKERS])
    st.dataframe(ind_df.set_index("Ticker"), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — FUNDAMENTALS HISTÓRICOS (FMP)
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    if not FMP_KEY:
        st.warning("Ingresá tu FMP API Key en el sidebar para ver fundamentals históricos.")
        st.stop()

    equity = [t for t in TICKERS if t not in ETF_TICKERS]
    sel = st.multiselect("Activos", equity, default=equity[:4])

    if sel:
        # ROIC / ROE / ROA histórico
        st.subheader("ROIC, ROE y ROA histórico")
        metric_sel = st.selectbox("Métrica", ["ROIC", "ROE", "ROA"])
        metric_cfg = {
            "ROIC": ("key-metrics", "roic"),
            "ROE":  ("ratios",       "returnOnEquity"),
            "ROA":  ("ratios",       "returnOnAssets"),
        }
        ep, col_k = metric_cfg[metric_sel]
        fig_h = go.Figure()
        for t in sel:
            df_h = get_key_metrics_history(t, FMP_KEY) if ep == "key-metrics" else get_ratios_history(t, FMP_KEY)
            if df_h.empty or col_k not in df_h.columns:
                continue
            s = df_h[["date", col_k]].dropna()
            fig_h.add_trace(go.Scatter(x=s["date"], y=s[col_k] * 100,
                                        name=t, mode="lines+markers"))
        fig_h.update_layout(yaxis_title=f"{metric_sel} (%)", height=420,
                             template="plotly_white", hovermode="x unified")
        st.plotly_chart(fig_h, use_container_width=True)

        # Múltiplos históricos
        st.subheader("Múltiplos históricos")
        mult_sel = st.selectbox("Múltiplo",
                                 ["priceEarningsRatio", "enterpriseValueOverEBITDA",
                                  "priceToBookRatio", "priceToSalesRatio"],
                                 format_func=lambda x: {
                                     "priceEarningsRatio": "P/E",
                                     "enterpriseValueOverEBITDA": "EV/EBITDA",
                                     "priceToBookRatio": "P/B",
                                     "priceToSalesRatio": "P/S"
                                 }[x])
        fig_m = go.Figure()
        for t in sel:
            df_r = get_ratios_history(t, FMP_KEY)
            if df_r.empty or mult_sel not in df_r.columns:
                continue
            s = df_r[["date", mult_sel]].dropna()
            fig_m.add_trace(go.Scatter(x=s["date"], y=s[mult_sel],
                                        name=t, mode="lines+markers"))
        fig_m.update_layout(height=400, template="plotly_white", hovermode="x unified")
        st.plotly_chart(fig_m, use_container_width=True)

        # FCF histórico
        st.subheader("Free Cash Flow histórico")
        fig_fcf = go.Figure()
        for t in sel:
            df_cf = get_cashflow_history(t, FMP_KEY)
            if df_cf.empty or "freeCashFlow" not in df_cf.columns:
                continue
            s = df_cf[["date", "freeCashFlow"]].dropna()
            fig_fcf.add_trace(go.Bar(x=s["date"].dt.year.astype(str),
                                      y=s["freeCashFlow"] / 1e9, name=t))
        fig_fcf.update_layout(yaxis_title="USD Bn", barmode="group",
                               height=380, template="plotly_white")
        st.plotly_chart(fig_fcf, use_container_width=True)

        # Buybacks + Dividendos
        st.subheader("Retorno al accionista — Buybacks y Dividendos")
        sel_sh = st.selectbox("Activo", sel, key="sh")
        df_cf2 = get_cashflow_history(sel_sh, FMP_KEY)
        if not df_cf2.empty:
            df_sh = df_cf2[["date"]].copy()
            if "commonStockRepurchased" in df_cf2.columns:
                df_sh["Buybacks"] = df_cf2["commonStockRepurchased"].abs() / 1e9
            if "dividendsPaid" in df_cf2.columns:
                df_sh["Dividendos"] = df_cf2["dividendsPaid"].abs() / 1e9
            df_sh["Año"] = df_sh["date"].dt.year.astype(str)
            melt_c = [c for c in ["Buybacks", "Dividendos"] if c in df_sh.columns]
            if melt_c:
                sh_m = df_sh[["Año"] + melt_c].melt(id_vars="Año", var_name="Tipo", value_name="USD Bn")
                fig_sh = px.bar(sh_m, x="Año", y="USD Bn", color="Tipo",
                                barmode="stack", template="plotly_white",
                                title=f"{sel_sh} — Retorno al accionista")
                fig_sh.update_layout(height=360)
                st.plotly_chart(fig_sh, use_container_width=True)

        # Revenue + Net Income
        st.subheader("Revenue y Net Income histórico")
        sel_inc = st.selectbox("Activo", sel, key="inc")
        df_inc = get_income_history(sel_inc, FMP_KEY)
        if not df_inc.empty:
            ci1, ci2 = st.columns(2)
            with ci1:
                if "revenue" in df_inc.columns:
                    fig_rv = px.bar(df_inc, x=df_inc["date"].dt.year.astype(str),
                                    y=df_inc["revenue"] / 1e9, template="plotly_white",
                                    labels={"x": "Año", "y": "USD Bn"}, title="Revenue")
                    fig_rv.update_layout(height=320, showlegend=False)
                    st.plotly_chart(fig_rv, use_container_width=True)
            with ci2:
                if "netIncome" in df_inc.columns:
                    fig_ni = px.bar(df_inc, x=df_inc["date"].dt.year.astype(str),
                                    y=df_inc["netIncome"] / 1e9, template="plotly_white",
                                    labels={"x": "Año", "y": "USD Bn"}, title="Net Income",
                                    color_discrete_sequence=["#00b09b"])
                    fig_ni.update_layout(height=320, showlegend=False)
                    st.plotly_chart(fig_ni, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ANALISTAS (FMP)
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    if not FMP_KEY:
        st.warning("Ingresá tu FMP API Key en el sidebar para ver estimaciones de analistas.")
        st.stop()

    equity = [t for t in TICKERS if t not in ETF_TICKERS]

    # Price Target Consensus
    st.subheader("Precio objetivo — consenso de analistas")
    pt_rows = []
    for t in equity:
        pt = get_price_target(t, FMP_KEY)
        s = prices[t].dropna()
        current = float(s.iloc[-1]) if not s.empty else None
        tc = pt.get("targetConsensus", None)
        upside = ((tc / current) - 1) if (tc and current) else None
        pt_rows.append({
            "Ticker": t,
            "Precio Actual": current,
            "Target Bajo": pt.get("targetLow", None),
            "Target Consenso": tc,
            "Target Alto": pt.get("targetHigh", None),
            "Upside Consenso": upside,
        })

    df_pt = pd.DataFrame(pt_rows)
    if not df_pt.empty:
        df_ptd = df_pt.copy()
        for col in ["Precio Actual", "Target Bajo", "Target Consenso", "Target Alto"]:
            df_ptd[col] = df_ptd[col].apply(fpr)
        df_ptd["Upside Consenso"] = df_ptd["Upside Consenso"].apply(fp)
        st.dataframe(df_ptd.set_index("Ticker"), use_container_width=True)

        uc = df_pt.dropna(subset=["Upside Consenso"]).sort_values("Upside Consenso")
        if not uc.empty:
            colors = ["#ef5350" if v < 0 else "#26a69a" for v in uc["Upside Consenso"]]
            fig_up = go.Figure(go.Bar(
                x=uc["Upside Consenso"] * 100, y=uc["Ticker"], orientation="h",
                marker_color=colors,
                text=[f"{v:.1f}%" for v in uc["Upside Consenso"] * 100],
                textposition="outside"
            ))
            fig_up.update_layout(xaxis_title="%", height=380, template="plotly_white",
                                  title="Upside implícito vs consenso")
            st.plotly_chart(fig_up, use_container_width=True)

    # Analyst Estimates
    st.subheader("Estimaciones forward")
    sel_est = st.selectbox("Activo", equity, key="est")
    df_est = get_analyst_estimates(sel_est, FMP_KEY)
    if not df_est.empty:
        years = df_est["date"].dt.year.astype(str)
        ce1, ce2 = st.columns(2)
        with ce1:
            rev_cols = ["estimatedRevenueLow", "estimatedRevenueAvg", "estimatedRevenueHigh"]
            if all(c in df_est.columns for c in rev_cols):
                fig_re = go.Figure()
                fig_re.add_trace(go.Bar(x=years, y=df_est["estimatedRevenueAvg"] / 1e9,
                                         name="Consenso", marker_color="#42a5f5"))
                fig_re.add_trace(go.Scatter(x=years, y=df_est["estimatedRevenueHigh"] / 1e9,
                                             name="Alto", line=dict(dash="dot", color="green"), mode="lines"))
                fig_re.add_trace(go.Scatter(x=years, y=df_est["estimatedRevenueLow"] / 1e9,
                                             name="Bajo", line=dict(dash="dot", color="red"), mode="lines"))
                fig_re.update_layout(title="Revenue estimado (USD Bn)", height=350, template="plotly_white")
                st.plotly_chart(fig_re, use_container_width=True)
        with ce2:
            eps_cols = ["estimatedEpsLow", "estimatedEpsAvg", "estimatedEpsHigh"]
            if all(c in df_est.columns for c in eps_cols):
                fig_ep = go.Figure()
                fig_ep.add_trace(go.Bar(x=years, y=df_est["estimatedEpsAvg"],
                                         name="Consenso", marker_color="#ab47bc"))
                fig_ep.add_trace(go.Scatter(x=years, y=df_est["estimatedEpsHigh"],
                                             name="Alto", line=dict(dash="dot", color="green"), mode="lines"))
                fig_ep.add_trace(go.Scatter(x=years, y=df_est["estimatedEpsLow"],
                                             name="Bajo", line=dict(dash="dot", color="red"), mode="lines"))
                fig_ep.update_layout(title="EPS estimado (USD)", height=350, template="plotly_white")
                st.plotly_chart(fig_ep, use_container_width=True)

        if "estimatedEbitdaAvg" in df_est.columns:
            fig_eb = px.bar(df_est, x=years, y=df_est["estimatedEbitdaAvg"] / 1e9,
                            template="plotly_white", labels={"y": "USD Bn"},
                            title="EBITDA estimado")
            fig_eb.update_layout(height=320, showlegend=False)
            st.plotly_chart(fig_eb, use_container_width=True)
    else:
        st.info(f"Sin estimaciones disponibles para {sel_est}.")

    # Historial price targets individuales
    st.subheader("Historial de price targets por analista")
    sel_pth = st.selectbox("Activo", equity, key="pth")
    df_pth = get_price_target_history(sel_pth, FMP_KEY)
    if not df_pth.empty:
        show = [c for c in ["date", "analystCompany", "priceTarget", "adjPriceTarget"] if c in df_pth.columns]
        st.dataframe(df_pth[show].sort_values("date", ascending=False).head(15),
                     use_container_width=True)
    else:
        st.info("Sin historial de price targets.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — RENDIMIENTO & CORRELACIÓN
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Rendimiento acumulado")
    period_opt = {"3M": 90, "6M": 180, "YTD": "ytd", "1Y": 365}
    sel_per = st.selectbox("Período", list(period_opt.keys()), index=2)

    if sel_per == "YTD":
        filt = prices[prices.index >= pd.Timestamp(datetime(datetime.now().year, 1, 1))]
    else:
        filt = prices.tail(period_opt[sel_per])

    if not filt.empty:
        cum = (filt / filt.iloc[0] - 1) * 100
        fig_cum = go.Figure()
        for t in TICKERS:
            if t in cum.columns:
                s = cum[t].dropna()
                is_b = t == BENCHMARK
                fig_cum.add_trace(go.Scatter(
                    x=s.index, y=s.values, name=t,
                    line=dict(width=3 if is_b else 1.5, dash="dash" if is_b else "solid"),
                    opacity=1 if is_b else 0.8
                ))
        fig_cum.update_layout(yaxis_title="Rendimiento (%)", hovermode="x unified",
                               height=500, template="plotly_white",
                               legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_cum, use_container_width=True)

    # Detalle
    st.subheader("Detalle por activo")
    sel_t = st.selectbox("Activo", TICKERS)
    if sel_t:
        row = df_summary[df_summary["Ticker"] == sel_t].iloc[0]
        i = info[sel_t]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Precio", fpr(row["Precio"]), fp(row["Var. Diaria"]))
        c2.metric("YTD", fp(row["YTD"]))
        c3.metric("Sharpe", fn(row["Sharpe"]))
        c4.metric("Max DD", fp(row["Max DD"]))
        c5.metric("DD Actual", fp(row["DD Actual"]))
        v1, v2, v3, v4, v5, v6 = st.columns(6)
        v1.metric("P/E Trailing", fn(i["pe_ratio"], 1))
        v2.metric("P/E Forward", fn(i["pe_forward"], 1))
        v3.metric("EV/EBITDA", fn(i["ev_ebitda"], 1))
        v4.metric("ROE", fp(i["return_on_equity"]))
        v5.metric("Mg. Neto", fp(i["profit_margins"]))
        v6.metric("Deuda/Equity", fn(i["debt_to_equity"], 1))

        s = prices[sel_t].dropna().tail(365)
        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(x=s.index, y=s.values, fill="tozeroy",
                                    fillcolor="rgba(99,110,250,0.1)"))
        fig_p.update_layout(yaxis_title="Precio (USD)", height=400,
                              template="plotly_white", showlegend=False)
        st.plotly_chart(fig_p, use_container_width=True)

        dd = (s - s.cummax()) / s.cummax() * 100
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=dd.index, y=dd.values, fill="tozeroy",
                                     fillcolor="rgba(239,85,59,0.2)",
                                     line=dict(color="rgba(239,85,59,0.8)")))
        fig_dd.update_layout(yaxis_title="Drawdown (%)", height=280,
                               template="plotly_white", showlegend=False)
        st.plotly_chart(fig_dd, use_container_width=True)

    # Correlación
    st.subheader("Matriz de correlación")
    cc, _ = st.columns([1, 3])
    with cc:
        cp = st.selectbox("Período", ["3M", "6M", "1Y"], index=2, key="corr")
    cd = {"3M": 90, "6M": 180, "1Y": 365}[cp]
    corr = prices.pct_change().dropna().tail(cd).corr()
    fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                          zmin=-1, zmax=1, aspect="auto")
    fig_corr.update_layout(height=500, template="plotly_white")
    st.plotly_chart(fig_corr, use_container_width=True)
