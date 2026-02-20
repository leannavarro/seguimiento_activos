import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# --- CONFIG ---
st.set_page_config(page_title="Portfolio Dashboard", layout="wide")

TICKERS = ["EEM", "BRK-B", "META", "MSFT", "ASML", "SPY", "TSM", "VEA", "EA"]
BENCHMARK = "SPY"
RISK_FREE_RATE = 0.05  # Approximate annual risk-free rate

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
                "name": i.get("shortName", t),
                "sector": i.get("sector", "N/A"),
                "dividend_yield": i.get("dividendYield", 0) or 0,
                "pe_ratio": i.get("trailingPE", None),
                "market_cap": i.get("marketCap", None),
            }
        except Exception:
            info[t] = {"name": t, "sector": "N/A", "dividend_yield": 0, "pe_ratio": None, "market_cap": None}
    return info

def calc_return(prices, ticker, days=None, start_date=None):
    """Calculate return for a ticker over a period."""
    s = prices[ticker].dropna()
    if s.empty:
        return None
    current = s.iloc[-1]
    if start_date is not None:
        mask = s.index >= pd.Timestamp(start_date)
        if mask.any():
            prev = s[mask].iloc[0]
        else:
            return None
    elif days is not None:
        target_date = s.index[-1] - timedelta(days=days)
        mask = s.index >= target_date
        if mask.any():
            prev = s[mask].iloc[0]
        else:
            prev = s.iloc[0]
    else:
        prev = s.iloc[0]
    if prev == 0:
        return None
    return (current / prev) - 1

def calc_mtd(prices, ticker):
    """Month-to-date return."""
    s = prices[ticker].dropna()
    if s.empty:
        return None
    last_date = s.index[-1]
    month_start = last_date.replace(day=1)
    mask = s.index < month_start
    if mask.any():
        prev = s[mask].iloc[-1]
    else:
        prev = s.iloc[0]
    return (s.iloc[-1] / prev) - 1

def calc_ytd(prices, ticker):
    """Year-to-date return."""
    s = prices[ticker].dropna()
    if s.empty:
        return None
    last_date = s.index[-1]
    year_start = datetime(last_date.year, 1, 1)
    return calc_return(prices, ticker, start_date=year_start)

def calc_volatility(prices, ticker, days=252):
    """Annualized volatility."""
    s = prices[ticker].dropna()
    if len(s) < 2:
        return None
    returns = s.pct_change().dropna().tail(days)
    return returns.std() * np.sqrt(252)

def calc_sharpe(prices, ticker, days=252, rf=RISK_FREE_RATE):
    """Annualized Sharpe ratio."""
    s = prices[ticker].dropna()
    if len(s) < 2:
        return None
    returns = s.pct_change().dropna().tail(days)
    excess = returns.mean() * 252 - rf
    vol = returns.std() * np.sqrt(252)
    if vol == 0:
        return None
    return excess / vol

def calc_sortino(prices, ticker, days=252, rf=RISK_FREE_RATE):
    """Annualized Sortino ratio."""
    s = prices[ticker].dropna()
    if len(s) < 2:
        return None
    returns = s.pct_change().dropna().tail(days)
    excess = returns.mean() * 252 - rf
    downside = returns[returns < 0].std() * np.sqrt(252)
    if downside == 0:
        return None
    return excess / downside

def calc_max_drawdown(prices, ticker):
    """Maximum drawdown."""
    s = prices[ticker].dropna()
    if s.empty:
        return None
    cummax = s.cummax()
    drawdown = (s - cummax) / cummax
    return drawdown.min()

def calc_beta(prices, ticker, benchmark=BENCHMARK, days=252):
    """Beta vs benchmark."""
    if ticker == benchmark:
        return 1.0
    s = prices[[ticker, benchmark]].dropna()
    if len(s) < 2:
        return None
    returns = s.pct_change().dropna().tail(days)
    cov = returns.cov()
    var_bench = cov.loc[benchmark, benchmark]
    if var_bench == 0:
        return None
    return cov.loc[ticker, benchmark] / var_bench

# --- MAIN ---
st.title("📊 Portfolio Dashboard")

with st.spinner("Cargando datos..."):
    data = load_data(TICKERS, period="2y")
    prices = data["Close"]
    info = get_info(TICKERS)

# --- SUMMARY TABLE ---
st.header("Resumen")

rows = []
for t in TICKERS:
    s = prices[t].dropna()
    if s.empty:
        continue
    current_price = s.iloc[-1]
    daily_chg = s.pct_change().iloc[-1] if len(s) > 1 else 0

    rows.append({
        "Ticker": t,
        "Nombre": info[t]["name"],
        "Precio": current_price,
        "Var. Diaria": daily_chg,
        "MTD": calc_mtd(prices, t),
        "1M": calc_return(prices, t, days=30),
        "3M": calc_return(prices, t, days=90),
        "YTD": calc_ytd(prices, t),
        "1Y": calc_return(prices, t, days=365),
        "Vol. Anual": calc_volatility(prices, t),
        "Sharpe": calc_sharpe(prices, t),
        "Sortino": calc_sortino(prices, t),
        "Max DD": calc_max_drawdown(prices, t),
        "Beta": calc_beta(prices, t),
        "Div. Yield": info[t]["dividend_yield"],
    })

df_summary = pd.DataFrame(rows)

# Format for display
df_display = df_summary.copy()
fmt_pct = ["Var. Diaria", "MTD", "1M", "3M", "YTD", "1Y", "Vol. Anual", "Max DD", "Div. Yield"]
for col in fmt_pct:
    df_display[col] = df_display[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
df_display["Precio"] = df_display["Precio"].apply(lambda x: f"${x:,.2f}")
df_display["Sharpe"] = df_display["Sharpe"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
df_display["Sortino"] = df_display["Sortino"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
df_display["Beta"] = df_display["Beta"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")

st.dataframe(df_display.set_index("Ticker"), use_container_width=True)

# --- CUMULATIVE RETURN CHART ---
st.header("Rendimiento acumulado")

period_options = {"3M": 90, "6M": 180, "YTD": "ytd", "1Y": 365}
selected_period = st.selectbox("Período", list(period_options.keys()), index=2)

if selected_period == "YTD":
    start = datetime(datetime.now().year, 1, 1)
    filtered = prices[prices.index >= pd.Timestamp(start)]
else:
    days = period_options[selected_period]
    filtered = prices.tail(days)

if not filtered.empty:
    cum_returns = (filtered / filtered.iloc[0] - 1) * 100
    fig_cum = go.Figure()
    for t in TICKERS:
        if t in cum_returns.columns:
            s = cum_returns[t].dropna()
            is_bench = t == BENCHMARK
            fig_cum.add_trace(go.Scatter(
                x=s.index, y=s.values,
                name=t,
                line=dict(width=3 if is_bench else 1.5, dash="dash" if is_bench else "solid"),
                opacity=1 if is_bench else 0.8
            ))
    fig_cum.update_layout(
        yaxis_title="Rendimiento (%)",
        hovermode="x unified",
        height=500,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_cum, use_container_width=True)

# --- CORRELATION MATRIX ---
st.header("Matriz de correlación")

col_corr_period, _ = st.columns([1, 3])
with col_corr_period:
    corr_period = st.selectbox("Período correlación", ["3M", "6M", "1Y"], index=2, key="corr")

corr_days = {"3M": 90, "6M": 180, "1Y": 365}[corr_period]
returns_corr = prices.pct_change().dropna().tail(corr_days)
corr_matrix = returns_corr.corr()

fig_corr = px.imshow(
    corr_matrix,
    text_auto=".2f",
    color_continuous_scale="RdBu_r",
    zmin=-1, zmax=1,
    aspect="auto"
)
fig_corr.update_layout(height=500, template="plotly_white")
st.plotly_chart(fig_corr, use_container_width=True)

# --- INDIVIDUAL ASSET DETAIL ---
st.header("Detalle por activo")

selected_ticker = st.selectbox("Seleccionar activo", TICKERS)

if selected_ticker:
    col1, col2, col3, col4 = st.columns(4)
    idx = df_summary[df_summary["Ticker"] == selected_ticker].index[0]
    row = df_summary.iloc[idx]

    col1.metric("Precio", f"${row['Precio']:,.2f}",
                f"{row['Var. Diaria']:.2%}" if pd.notna(row['Var. Diaria']) else None)
    col2.metric("YTD", f"{row['YTD']:.2%}" if pd.notna(row['YTD']) else "N/A")
    col3.metric("Sharpe", f"{row['Sharpe']:.2f}" if pd.notna(row['Sharpe']) else "N/A")
    col4.metric("Max Drawdown", f"{row['Max DD']:.2%}" if pd.notna(row['Max DD']) else "N/A")

    # Price chart
    s = prices[selected_ticker].dropna().tail(365)
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=s.index, y=s.values, name=selected_ticker,
                                    fill="tozeroy", fillcolor="rgba(99,110,250,0.1)"))
    fig_price.update_layout(
        yaxis_title="Precio (USD)",
        height=400,
        template="plotly_white",
        showlegend=False
    )
    st.plotly_chart(fig_price, use_container_width=True)

    # Drawdown chart
    cummax = s.cummax()
    dd = (s - cummax) / cummax * 100
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=dd.index, y=dd.values, name="Drawdown",
                                 fill="tozeroy", fillcolor="rgba(239,85,59,0.2)",
                                 line=dict(color="rgba(239,85,59,0.8)")))
    fig_dd.update_layout(
        yaxis_title="Drawdown (%)",
        height=300,
        template="plotly_white",
        showlegend=False
    )
    st.plotly_chart(fig_dd, use_container_width=True)
