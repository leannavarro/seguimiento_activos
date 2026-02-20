import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- CONFIG ---
st.set_page_config(page_title="Dashboard", layout="wide")

TICKERS = ["EEM", "BRK-B", "META", "MSFT", "ASML", "SPY", "TSM", "VEA", "EA"]
BENCHMARK = "SPY"
RISK_FREE_RATE = 0.05

# --- DATA ---
@st.cache_data(ttl=3600)
def load_data(tickers, period="2y"):
    data = yf.download(tickers, period=period, auto_adjust=True)
    return data

@st.cache_data(ttl=3600)
def get_info(tickers):
    info = {}
    for t in tickers:
        try:
            tk = yf.Ticker(t)
            i = tk.info
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
                "buyback_yield": None,  # calculated below
                "shares_outstanding": i.get("sharesOutstanding", None),
                "float_shares": i.get("floatShares", None),
            }
            # Approximate buyback yield via share count change (not always available)
            # Use trailing12MonthsSharesBuyback if available
            buyback = i.get("buybackYield", None)
            if buyback is None:
                # yfinance doesn't expose this directly, mark N/A
                info[t]["buyback_yield"] = None
            else:
                info[t]["buyback_yield"] = buyback
        except Exception:
            info[t] = {
                "sector": "N/A",
                "dividend_yield": 0,
                "pe_ratio": None, "pe_forward": None, "peg_ratio": None,
                "ps_ratio": None, "pb_ratio": None, "ev_ebitda": None,
                "market_cap": None, "revenue_growth": None, "earnings_growth": None,
                "operating_margins": None, "profit_margins": None,
                "return_on_equity": None, "return_on_assets": None,
                "debt_to_equity": None, "buyback_yield": None,
                "shares_outstanding": None, "float_shares": None,
            }
    return info

def calc_return(prices, ticker, days=None, start_date=None):
    s = prices[ticker].dropna()
    if s.empty:
        return None
    current = s.iloc[-1]
    if start_date is not None:
        mask = s.index >= pd.Timestamp(start_date)
        prev = s[mask].iloc[0] if mask.any() else None
    elif days is not None:
        target_date = s.index[-1] - timedelta(days=days)
        mask = s.index >= target_date
        prev = s[mask].iloc[0] if mask.any() else s.iloc[0]
    else:
        prev = s.iloc[0]
    if prev is None or prev == 0:
        return None
    return (current / prev) - 1

def calc_mtd(prices, ticker):
    s = prices[ticker].dropna()
    if s.empty:
        return None
    month_start = s.index[-1].replace(day=1)
    mask = s.index < month_start
    prev = s[mask].iloc[-1] if mask.any() else s.iloc[0]
    return (s.iloc[-1] / prev) - 1

def calc_ytd(prices, ticker):
    s = prices[ticker].dropna()
    if s.empty:
        return None
    year_start = datetime(s.index[-1].year, 1, 1)
    return calc_return(prices, ticker, start_date=year_start)

def calc_sharpe(prices, ticker, days=252, rf=RISK_FREE_RATE):
    s = prices[ticker].dropna()
    if len(s) < 2:
        return None
    returns = s.pct_change().dropna().tail(days)
    excess = returns.mean() * 252 - rf
    vol = returns.std() * np.sqrt(252)
    return excess / vol if vol != 0 else None

def calc_sortino(prices, ticker, days=252, rf=RISK_FREE_RATE):
    s = prices[ticker].dropna()
    if len(s) < 2:
        return None
    returns = s.pct_change().dropna().tail(days)
    excess = returns.mean() * 252 - rf
    downside = returns[returns < 0].std() * np.sqrt(252)
    return excess / downside if downside != 0 else None

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
    current = s.iloc[-1]
    return (current - peak) / peak if peak != 0 else None

def calc_calmar(prices, ticker):
    s = prices[ticker].dropna()
    if s.empty or len(s) < 2:
        return None
    annual_return = calc_return(prices, ticker, days=252)
    max_dd = calc_max_drawdown(prices, ticker)
    if annual_return is None or max_dd is None or max_dd == 0:
        return None
    return annual_return / abs(max_dd)

def calc_beta(prices, ticker, benchmark=BENCHMARK, days=252):
    if ticker == benchmark:
        return 1.0
    s = prices[[ticker, benchmark]].dropna()
    if len(s) < 2:
        return None
    returns = s.pct_change().dropna().tail(days)
    cov = returns.cov()
    var_bench = cov.loc[benchmark, benchmark]
    return cov.loc[ticker, benchmark] / var_bench if var_bench != 0 else None

def calc_alpha(prices, ticker, benchmark=BENCHMARK, days=252, rf=RISK_FREE_RATE):
    if ticker == benchmark:
        return 0.0
    beta = calc_beta(prices, ticker, benchmark, days)
    if beta is None:
        return None
    r_asset = calc_return(prices, ticker, days=days)
    r_bench = calc_return(prices, benchmark, days=days)
    if r_asset is None or r_bench is None:
        return None
    return r_asset - (rf + beta * (r_bench - rf))

def fmt_pct(x):
    return f"{x:.2%}" if pd.notna(x) and x is not None else "N/A"

def fmt_num(x, dec=2):
    return f"{x:.{dec}f}" if pd.notna(x) and x is not None else "N/A"

def fmt_price(x):
    return f"${x:,.2f}" if pd.notna(x) and x is not None else "N/A"

def fmt_mcap(x):
    if x is None or not pd.notna(x):
        return "N/A"
    if x >= 1e12:
        return f"${x/1e12:.1f}T"
    if x >= 1e9:
        return f"${x/1e9:.1f}B"
    return f"${x/1e6:.0f}M"

# --- MAIN ---
st.title("📊 Dashboard")

with st.spinner("Cargando datos..."):
    data = load_data(TICKERS, period="2y")
    prices = data["Close"]
    info = get_info(TICKERS)

# ─── TAB LAYOUT ───────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["Resumen", "Valuación", "Rendimiento", "Detalle"])

# ─── TAB 1: RESUMEN ───────────────────────────────────────────────────────────
with tab1:
    rows = []
    for t in TICKERS:
        s = prices[t].dropna()
        if s.empty:
            continue
        daily_chg = s.pct_change().iloc[-1] if len(s) > 1 else 0
        rows.append({
            "Ticker": t,
            "Precio": s.iloc[-1],
            "Var. Diaria": daily_chg,
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

    pct_cols = ["Var. Diaria", "MTD", "1M", "3M", "YTD", "1Y", "Max DD", "DD Actual"]
    for col in pct_cols:
        df_disp[col] = df_disp[col].apply(fmt_pct)
    df_disp["Precio"] = df_disp["Precio"].apply(fmt_price)
    for col in ["Sharpe", "Sortino", "Calmar", "Beta"]:
        df_disp[col] = df_disp[col].apply(fmt_num)
    df_disp["Alpha (1Y)"] = df_disp["Alpha (1Y)"].apply(fmt_pct)

    st.dataframe(df_disp.set_index("Ticker"), use_container_width=True)

# ─── TAB 2: VALUACIÓN ─────────────────────────────────────────────────────────
with tab2:
    st.subheader("Múltiplos de valuación")

    val_rows = []
    for t in TICKERS:
        i = info[t]
        s = prices[t].dropna()
        current = s.iloc[-1] if not s.empty else None
        shareholder_yield = None
        if i["dividend_yield"] is not None and i["buyback_yield"] is not None:
            shareholder_yield = i["dividend_yield"] + i["buyback_yield"]
        elif i["dividend_yield"] is not None:
            shareholder_yield = i["dividend_yield"]  # buyback not available

        val_rows.append({
            "Ticker": t,
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
            "Shareholder Yield": shareholder_yield,
        })

    df_val = pd.DataFrame(val_rows)
    df_val_disp = df_val.copy()
    df_val_disp["Market Cap"] = df_val_disp["Market Cap"].apply(fmt_mcap)
    for col in ["P/E Trailing", "P/E Forward", "PEG", "P/S", "P/B", "EV/EBITDA", "Deuda/Equity"]:
        df_val_disp[col] = df_val_disp[col].apply(lambda x: fmt_num(x, 1))
    for col in ["Rev. Growth", "EPS Growth", "Mg. Operativo", "Mg. Neto", "ROE", "ROA", "Div. Yield", "Shareholder Yield"]:
        df_val_disp[col] = df_val_disp[col].apply(fmt_pct)

    st.dataframe(df_val_disp.set_index("Ticker"), use_container_width=True)

    # ─── Scatter: P/E vs EPS Growth ───────────────────────────────────────────
    st.subheader("P/E Forward vs Crecimiento EPS")
    scatter_df = df_val.dropna(subset=["pe_forward", "EPS Growth"]).copy()
    if not scatter_df.empty:
        fig_scatter = px.scatter(
            scatter_df,
            x="EPS Growth", y="P/E Forward",
            text="Ticker",
            template="plotly_white",
            labels={"EPS Growth": "Crecimiento EPS (YoY)", "P/E Forward": "P/E Forward"},
        )
        fig_scatter.update_traces(textposition="top center", marker=dict(size=12))
        fig_scatter.update_layout(height=450)
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("Datos insuficientes para el scatter (ETFs no tienen earnings data).")

    # ─── Bar: Márgenes comparativos ────────────────────────────────────────────
    st.subheader("Márgenes operativo y neto")
    margin_df = df_val[["Ticker", "Mg. Operativo", "Mg. Neto"]].dropna(subset=["Mg. Operativo"])
    if not margin_df.empty:
        margin_melt = margin_df.melt(id_vars="Ticker", var_name="Margen", value_name="Valor")
        margin_melt["Valor"] = margin_melt["Valor"] * 100
        fig_margins = px.bar(
            margin_melt,
            x="Ticker", y="Valor", color="Margen",
            barmode="group",
            template="plotly_white",
            labels={"Valor": "%"},
        )
        fig_margins.update_layout(height=400)
        st.plotly_chart(fig_margins, use_container_width=True)

    # ─── Bar: ROE / ROA ────────────────────────────────────────────────────────
    st.subheader("ROE y ROA")
    roe_df = df_val[["Ticker", "ROE", "ROA"]].dropna(subset=["ROE"])
    if not roe_df.empty:
        roe_melt = roe_df.melt(id_vars="Ticker", var_name="Ratio", value_name="Valor")
        roe_melt["Valor"] = roe_melt["Valor"] * 100
        fig_roe = px.bar(
            roe_melt,
            x="Ticker", y="Valor", color="Ratio",
            barmode="group",
            template="plotly_white",
            labels={"Valor": "%"},
        )
        fig_roe.update_layout(height=400)
        st.plotly_chart(fig_roe, use_container_width=True)

# ─── TAB 3: RENDIMIENTO ───────────────────────────────────────────────────────
with tab3:
    st.subheader("Rendimiento acumulado")
    period_options = {"3M": 90, "6M": 180, "YTD": "ytd", "1Y": 365}
    selected_period = st.selectbox("Período", list(period_options.keys()), index=2)

    if selected_period == "YTD":
        filtered = prices[prices.index >= pd.Timestamp(datetime(datetime.now().year, 1, 1))]
    else:
        filtered = prices.tail(period_options[selected_period])

    if not filtered.empty:
        cum_returns = (filtered / filtered.iloc[0] - 1) * 100
        fig_cum = go.Figure()
        for t in TICKERS:
            if t in cum_returns.columns:
                s = cum_returns[t].dropna()
                is_bench = t == BENCHMARK
                fig_cum.add_trace(go.Scatter(
                    x=s.index, y=s.values, name=t,
                    line=dict(width=3 if is_bench else 1.5, dash="dash" if is_bench else "solid"),
                    opacity=1 if is_bench else 0.8
                ))
        fig_cum.update_layout(
            yaxis_title="Rendimiento (%)", hovermode="x unified",
            height=500, template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_cum, use_container_width=True)

    # ─── Correlation matrix ────────────────────────────────────────────────────
    st.subheader("Matriz de correlación")
    col_corr, _ = st.columns([1, 3])
    with col_corr:
        corr_period = st.selectbox("Período correlación", ["3M", "6M", "1Y"], index=2, key="corr")
    corr_days = {"3M": 90, "6M": 180, "1Y": 365}[corr_period]
    corr_matrix = prices.pct_change().dropna().tail(corr_days).corr()
    fig_corr = px.imshow(
        corr_matrix, text_auto=".2f",
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto"
    )
    fig_corr.update_layout(height=500, template="plotly_white")
    st.plotly_chart(fig_corr, use_container_width=True)

# ─── TAB 4: DETALLE ───────────────────────────────────────────────────────────
with tab4:
    selected_ticker = st.selectbox("Seleccionar activo", TICKERS)

    if selected_ticker:
        row = df_summary[df_summary["Ticker"] == selected_ticker].iloc[0]
        i = info[selected_ticker]

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Precio", fmt_price(row["Precio"]), fmt_pct(row["Var. Diaria"]))
        c2.metric("YTD", fmt_pct(row["YTD"]))
        c3.metric("Sharpe", fmt_num(row["Sharpe"]))
        c4.metric("Max DD", fmt_pct(row["Max DD"]))
        c5.metric("DD Actual", fmt_pct(row["DD Actual"]))

        # Quick valuation strip
        v1, v2, v3, v4, v5, v6 = st.columns(6)
        v1.metric("P/E Trailing", fmt_num(i["pe_ratio"], 1))
        v2.metric("P/E Forward", fmt_num(i["pe_forward"], 1))
        v3.metric("EV/EBITDA", fmt_num(i["ev_ebitda"], 1))
        v4.metric("ROE", fmt_pct(i["return_on_equity"]))
        v5.metric("Mg. Neto", fmt_pct(i["profit_margins"]))
        v6.metric("Deuda/Equity", fmt_num(i["debt_to_equity"], 1))

        # Price chart
        s = prices[selected_ticker].dropna().tail(365)
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(
            x=s.index, y=s.values, name=selected_ticker,
            fill="tozeroy", fillcolor="rgba(99,110,250,0.1)"
        ))
        fig_price.update_layout(yaxis_title="Precio (USD)", height=400,
                                 template="plotly_white", showlegend=False)
        st.plotly_chart(fig_price, use_container_width=True)

        # Drawdown chart
        cummax = s.cummax()
        dd = (s - cummax) / cummax * 100
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=dd.index, y=dd.values, name="Drawdown",
            fill="tozeroy", fillcolor="rgba(239,85,59,0.2)",
            line=dict(color="rgba(239,85,59,0.8)")
        ))
        fig_dd.update_layout(yaxis_title="Drawdown (%)", height=300,
                               template="plotly_white", showlegend=False)
        st.plotly_chart(fig_dd, use_container_width=True)
