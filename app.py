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

FMP_BASE = "https://financialmodelingprep.com/stable"

# ─── FMP HELPERS (price target consensus only — other endpoints are premium) ──

def _fmp_key():
    """Always reads fresh — never cached."""
    try:
        k = st.secrets.get("FMP_API_KEY", "")
        if k:
            return k
    except Exception:
        pass
    return st.session_state.get("fmp_api_key", "")

def fmp_get(endpoint, api_key, params_tuple=()):
    """FMP stable API. params_tuple = tuple of (key, value) pairs for hashability."""
    if not api_key:
        return []
    url = f"{FMP_BASE}/{endpoint}"
    p = {"apikey": api_key, **dict(params_tuple)}
    try:
        r = requests.get(url, params=p, timeout=15)
        if r.status_code != 200:
            st.session_state.setdefault("fmp_errors", []).append(
                f"{endpoint} {dict(params_tuple)}: HTTP {r.status_code} - {r.text[:150]}"
            )
            return []
        data = r.json()
        if isinstance(data, dict) and ("Error Message" in data or "message" in data or "error" in data):
            msg = data.get("Error Message") or data.get("message") or data.get("error", "Unknown")
            st.session_state.setdefault("fmp_errors", []).append(f"{endpoint}: {msg}")
            return []
        return data if isinstance(data, list) else []
    except Exception as e:
        st.session_state.setdefault("fmp_errors", []).append(f"{endpoint}: {str(e)}")
        return []

def get_price_target_fmp(ticker, api_key):
    """Price target consensus from FMP — one of the few free endpoints."""
    data = fmp_get("price-target-consensus", api_key, (("symbol", ticker),))
    if data and isinstance(data, list):
        return data[0]
    return {}

# ─── YFINANCE PRICE DATA ─────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_data(tickers, period="2y"):
    return yf.download(tickers, period=period, auto_adjust=True)

# ─── YFINANCE FUNDAMENTALS ────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def get_info_yf(tickers):
    """Snapshot fundamentals from yfinance .info"""
    info = {}
    for t in tickers:
        try:
            i = yf.Ticker(t).info
            info[t] = {
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
                # Margins
                "gross_margins": i.get("grossMargins", None),
                "ebitda_margins": i.get("ebitdaMargins", None),
                # Cash flow
                "free_cashflow": i.get("freeCashflow", None),
                "operating_cashflow": i.get("operatingCashflow", None),
                # Balance sheet
                "total_cash": i.get("totalCash", None),
                "total_debt": i.get("totalDebt", None),
                "total_cash_per_share": i.get("totalCashPerShare", None),
                # EPS
                "trailing_eps": i.get("trailingEps", None),
                "forward_eps": i.get("forwardEps", None),
                # Enterprise value
                "enterprise_value": i.get("enterpriseValue", None),
                "ev_revenue": i.get("enterpriseToRevenue", None),
                # Ownership
                "insider_pct": i.get("heldPercentInsiders", None),
                "institution_pct": i.get("heldPercentInstitutions", None),
                # Short interest
                "short_pct": i.get("sharesPercentSharesOut", None),
                # ROIC: yfinance exposes returnOnCapital (some tickers)
                "roic": i.get("returnOnCapital", None),
                # FCF Yield = freeCashflow / marketCap
                "fcf_yield": (i["freeCashflow"] / i["marketCap"])
                              if i.get("freeCashflow") and i.get("marketCap")
                              else None,
                # Net debt = total_debt - total_cash
                "net_debt": (i.get("totalDebt", 0) or 0) - (i.get("totalCash", 0) or 0)
                             if i.get("totalDebt") else None,
            }
        except Exception:
            info[t] = {k: None for k in [
                "pe_ratio", "pe_forward", "peg_ratio", "ps_ratio", "pb_ratio",
                "ev_ebitda", "market_cap", "revenue_growth", "earnings_growth",
                "operating_margins", "profit_margins", "return_on_equity",
                "return_on_assets", "debt_to_equity", "buyback_yield", "roic", "fcf_yield",
                "free_cashflow", "operating_cashflow", "total_cash", "total_debt",
                "total_cash_per_share", "trailing_eps", "forward_eps", "enterprise_value",
                "ev_revenue", "insider_pct", "institution_pct", "short_pct",
                "gross_margins", "ebitda_margins", "net_debt"
            ]}
            info[t]["dividend_yield"] = 0
    return info

@st.cache_data(ttl=3600)
def get_financials_yf(ticker):
    """Annual income statement history from yfinance."""
    try:
        tk = yf.Ticker(ticker)
        df = tk.financials  # rows=metrics, cols=dates
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.T.copy()
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df.index.name = "date"
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_cashflow_yf(ticker):
    """Annual cash flow from yfinance."""
    try:
        tk = yf.Ticker(ticker)
        df = tk.cashflow
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.T.copy()
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df.index.name = "date"
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_balance_yf(ticker):
    """Annual balance sheet from yfinance."""
    try:
        tk = yf.Ticker(ticker)
        df = tk.balance_sheet
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.T.copy()
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df.index.name = "date"
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_analyst_yf(ticker):
    """Analyst estimates: earnings + revenue forward."""
    try:
        tk = yf.Ticker(ticker)
        return {
            "price_targets": tk.analyst_price_targets,   # dict: low/mean/high/current
            "recommendations": tk.recommendations_summary, # df: buy/hold/sell
            "earnings_est": tk.earnings_estimate,          # df: forward EPS
            "revenue_est": tk.revenue_estimate,            # df: forward revenue
        }
    except Exception:
        return {}

@st.cache_data(ttl=3600)
def compute_roic_history(ticker):
    """Compute ROIC from yfinance financials + balance sheet."""
    try:
        fin = get_financials_yf(ticker)
        bal = get_balance_yf(ticker)
        if fin.empty or bal.empty:
            return pd.DataFrame()
        # NOPAT = EBIT * (1 - tax_rate), approximated as Operating Income * (1 - effective_tax)
        # Invested Capital = Total Assets - Current Liabilities - Cash
        rows = []
        for date in fin.index:
            if date not in bal.index:
                continue
            ebit = fin.loc[date].get("EBIT", fin.loc[date].get("Operating Income", None))
            tax_prov = fin.loc[date].get("Tax Provision", None)
            pretax = fin.loc[date].get("Pretax Income", None)
            tax_rate = (tax_prov / pretax) if (tax_prov and pretax and pretax != 0) else 0.21
            nopat = ebit * (1 - tax_rate) if ebit else None

            total_assets = bal.loc[date].get("Total Assets", None)
            curr_liab = bal.loc[date].get("Current Liabilities", None)
            cash = bal.loc[date].get("Cash And Cash Equivalents", bal.loc[date].get("Cash", 0)) or 0
            invested_capital = (total_assets - curr_liab - cash) if (total_assets and curr_liab) else None

            roic = (nopat / invested_capital) if (nopat and invested_capital and invested_capital != 0) else None
            rows.append({"date": date, "roic": roic})
        return pd.DataFrame(rows).dropna()
    except Exception:
        return pd.DataFrame()

def get_info(tickers, fmp_key=""):
    return get_info_yf(tickers)

# ─── STOCKANALYSIS SCRAPER ────────────────────────────────────────────────────

@st.cache_data(ttl=86400)  # cache 24 hours - financials don't change daily
def scrape_stockanalysis(ticker, statement="financials"):
    """
    Scrape annual fundamentals from stockanalysis.com.
    Returns DataFrame with years as columns, metrics as rows.
    Falls back to empty DataFrame on any error.
    statement: 'financials' | 'cash-flow-statement' | 'balance-sheet'
    """
    try:
        from bs4 import BeautifulSoup

        # Map tickers to stockanalysis URL format
        sa_ticker = ticker.replace("-", ".").lower()  # BRK-B -> brk.b
        url = f"https://stockanalysis.com/stocks/{sa_ticker}/{statement}/?p=annual"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://stockanalysis.com/",
        }
        r = requests.get(url, headers=headers, timeout=20)
        if r.status_code != 200:
            return pd.DataFrame()

        soup = BeautifulSoup(r.text, "lxml")
        table = soup.find("table")
        if not table:
            return pd.DataFrame()

        # Parse header (years)
        headers_row = table.find("thead")
        if not headers_row:
            return pd.DataFrame()
        cols = [th.get_text(strip=True) for th in headers_row.find_all("th")]

        # Parse rows
        rows = []
        for tr in table.find("tbody").find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            if cells:
                rows.append(cells)

        if not rows or not cols:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=cols[:len(rows[0])] if cols else None)
        if df.empty:
            return pd.DataFrame()

        # First column is metric name
        df = df.set_index(df.columns[0])

        # Clean numeric values: remove $, %, B, M, commas
        def clean_val(v):
            if not isinstance(v, str):
                return np.nan
            v = v.strip().replace(",", "").replace("$", "").replace("%", "")
            if v in ("-", "", "—", "N/A"):
                return np.nan
            mult = 1
            if v.endswith("B"):
                mult = 1e9
                v = v[:-1]
            elif v.endswith("M"):
                mult = 1e6
                v = v[:-1]
            elif v.endswith("T"):
                mult = 1e12
                v = v[:-1]
            try:
                return float(v) * mult
            except ValueError:
                return np.nan

        df = df.applymap(clean_val)
        return df

    except ImportError:
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=86400)
def get_sa_key_metrics(ticker):
    """
    Get key metrics from stockanalysis: ROIC, FCF, margins, EPS history.
    Returns dict with metric name -> {year: value} or empty dict on failure.
    """
    if ticker in ETF_TICKERS:
        return {}

    results = {}

    # Income statement: revenue, net income, EPS, margins
    df_inc = scrape_stockanalysis(ticker, "financials")
    if not df_inc.empty:
        results["income"] = df_inc

    # Cash flow: FCF, capex, buybacks, dividends
    df_cf = scrape_stockanalysis(ticker, "cash-flow-statement")
    if not df_cf.empty:
        results["cashflow"] = df_cf

    # Balance sheet: assets, debt, equity
    df_bs = scrape_stockanalysis(ticker, "balance-sheet")
    if not df_bs.empty:
        results["balance"] = df_bs

    return results

def sa_available():
    """Check if stockanalysis data is available (cached result)."""
    return st.session_state.get("sa_available", None)

def get_sa_roic_series(ticker):
    """Extract ROIC history from stockanalysis data."""
    data = get_sa_key_metrics(ticker)
    if not data:
        return pd.DataFrame()

    # Try to find ROIC row directly
    for key in ["income", "cashflow", "balance"]:
        df = data.get(key, pd.DataFrame())
        if df.empty:
            continue
        roic_row = next((idx for idx in df.index
                         if "roic" in idx.lower() or "return on invested" in idx.lower()), None)
        if roic_row:
            s = df.loc[roic_row].dropna()
            return pd.DataFrame({"year": s.index, "roic": s.values / 100})

    # Calculate ROIC from components if not directly available
    try:
        df_inc = data.get("income", pd.DataFrame())
        df_bs  = data.get("balance", pd.DataFrame())
        df_cf  = data.get("cashflow", pd.DataFrame())
        if df_inc.empty or df_bs.empty:
            return pd.DataFrame()

        # Find operating income
        op_inc_row = next((idx for idx in df_inc.index
                           if "operating income" in idx.lower()), None)
        # Find tax rate from income statement
        tax_row = next((idx for idx in df_inc.index
                        if "income tax" in idx.lower() or "tax provision" in idx.lower()), None)
        pretax_row = next((idx for idx in df_inc.index
                           if "pretax" in idx.lower() or "pre-tax" in idx.lower()), None)

        if not op_inc_row:
            return pd.DataFrame()

        # Invested capital: Total Assets - Current Liabilities - Cash
        assets_row = next((idx for idx in df_bs.index if "total assets" in idx.lower()), None)
        curr_liab_row = next((idx for idx in df_bs.index
                              if "current liabilities" in idx.lower()), None)
        cash_row = next((idx for idx in df_bs.index
                         if idx.lower() in ["cash", "cash & equivalents",
                                            "cash and equivalents"]), None)

        if not (assets_row and curr_liab_row):
            return pd.DataFrame()

        common_years = [y for y in df_inc.columns if y in df_bs.columns]
        rows = []
        for yr in common_years:
            ebit = df_inc.loc[op_inc_row, yr]
            tax_rate = 0.21  # default
            if tax_row and pretax_row:
                tax = df_inc.loc[tax_row, yr]
                pretax = df_inc.loc[pretax_row, yr]
                if pretax and pretax != 0:
                    tax_rate = abs(tax / pretax)
            nopat = ebit * (1 - tax_rate) if pd.notna(ebit) else None

            assets = df_bs.loc[assets_row, yr]
            curr_l = df_bs.loc[curr_liab_row, yr]
            cash = df_bs.loc[cash_row, yr] if cash_row and cash_row in df_bs.index else 0
            cash = cash if pd.notna(cash) else 0
            inv_cap = assets - curr_l - cash if pd.notna(assets) and pd.notna(curr_l) else None

            roic = nopat / inv_cap if (nopat and inv_cap and inv_cap != 0) else None
            if roic is not None:
                rows.append({"year": yr, "roic": roic})

        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()

def get_sa_fcf_series(ticker):
    """Extract FCF history from stockanalysis cashflow data."""
    data = get_sa_key_metrics(ticker)
    df_cf = data.get("cashflow", pd.DataFrame())
    if df_cf.empty:
        return pd.DataFrame()

    fcf_row = next((idx for idx in df_cf.index
                    if "free cash flow" in idx.lower() or idx.lower() == "fcf"), None)
    if fcf_row:
        s = df_cf.loc[fcf_row].dropna()
        return pd.DataFrame({"year": s.index, "fcf": s.values})
    return pd.DataFrame()

def get_sa_margins_series(ticker):
    """Extract margin history from stockanalysis income data."""
    data = get_sa_key_metrics(ticker)
    df_inc = data.get("income", pd.DataFrame())
    if df_inc.empty:
        return pd.DataFrame()

    result = {}
    margin_map = {
        "gross_margin": ["gross margin"],
        "operating_margin": ["operating margin", "operating income margin"],
        "net_margin": ["net margin", "net profit margin", "profit margin"],
    }
    for key, candidates in margin_map.items():
        row = next((idx for idx in df_inc.index
                    if any(c in idx.lower() for c in candidates)), None)
        if row:
            result[key] = df_inc.loc[row]

    if result:
        df = pd.DataFrame(result)
        df.index.name = "year"
        return df / 100  # convert % to decimal
    return pd.DataFrame()


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
        st.session_state["fmp_errors"] = []  # clear stale errors on new key entry
        st.success("Key cargada ✓")
    elif not _fmp_key():
        st.warning("Sin API Key: tabs Fundamentals y Analistas no disponibles.")

    st.divider()
    st.subheader("📅 Período de análisis")
    
    DATA_PERIOD_MAP = {"1Y": "1y", "2Y": "2y", "3Y": "3y", "5Y": "5y", "10Y": "10y"}
    data_period_label = st.select_slider(
        "Datos históricos (descarga)",
        options=list(DATA_PERIOD_MAP.keys()),
        value="2Y",
        help="Cuántos años de precios descarga. Afecta Max DD, volatilidad histórica y rolling Sharpe."
    )
    st.session_state["data_period"] = DATA_PERIOD_MAP[data_period_label]

    CALC_PERIOD_MAP = {"6M": 126, "1Y": 252, "2Y": 504, "3Y": 756, "Todo": None}
    calc_period_label = st.select_slider(
        "Ventana de cálculo (métricas)",
        options=list(CALC_PERIOD_MAP.keys()),
        value="1Y",
        help="Ventana para Sharpe, Sortino, Calmar, Beta, Alpha. Independiente de los datos descargados."
    )
    st.session_state["calc_days"] = CALC_PERIOD_MAP[calc_period_label]

    st.caption(f"Datos: {data_period_label} · Métricas: {calc_period_label}")

    st.divider()
    if st.button("🗑️ Limpiar caché", help="Forzar recarga de datos"):
        st.cache_data.clear()
        st.session_state["fmp_errors"] = []
        st.rerun()

    # Debug panel
    if st.checkbox("🔍 Mostrar errores FMP", value=False):
        errors = st.session_state.get("fmp_errors", [])
        if errors:
            st.error("Errores FMP detectados:")
            for e in errors[-10:]:
                st.code(e)
        else:
            st.success("Sin errores FMP registrados")

# FMP_KEY is read fresh via _fmp_key() at each call site
# Note: FMP is only used for price-target-consensus in Analistas tab.
# All fundamentals (P/E, margins, ROE etc) come from yfinance .info

# ─── LOAD DATA ────────────────────────────────────────────────────────────────

_data_period = st.session_state.get("data_period", "2y")
_calc_days   = st.session_state.get("calc_days", 252)

with st.spinner("Cargando datos de mercado..."):
    data = load_data(TICKERS, period=_data_period)
    prices = data["Close"]
    info = get_info(TICKERS, _fmp_key())

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
        cd = _calc_days  # use selected window
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
            "Sharpe": calc_sharpe(prices, t, days=cd) if cd else calc_sharpe(prices, t, days=len(s)),
            "Sortino": calc_sortino(prices, t, days=cd) if cd else calc_sortino(prices, t, days=len(s)),
            "Calmar": calc_calmar(prices, t),
            "Max DD": calc_max_drawdown(prices, t),
            "DD Actual": calc_current_drawdown(prices, t),
            "Beta": calc_beta(prices, t, days=cd) if cd else calc_beta(prices, t, days=len(s)),
            "Alpha (1Y)": calc_alpha(prices, t, days=cd) if cd else calc_alpha(prices, t, days=len(s)),
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
        # PEG: use yfinance value if available, else calculate P/E Forward / EPS Growth
        peg = i["peg_ratio"]
        if peg is None and i["pe_forward"] and i["earnings_growth"] and i["earnings_growth"] > 0:
            peg = i["pe_forward"] / (i["earnings_growth"] * 100)
        elif peg is None and i["pe_ratio"] and i["earnings_growth"] and i["earnings_growth"] > 0:
            peg = i["pe_ratio"] / (i["earnings_growth"] * 100)

        val_rows.append({
            "Ticker": t,
            "Sector": TICKER_META.get(t, {}).get("sector", "N/A"),
            "Industria": TICKER_META.get(t, {}).get("industry", "N/A"),
            "Geo": TICKER_META.get(t, {}).get("geo", "N/A"),
            "Market Cap": i["market_cap"],
            "P/E Trailing": i["pe_ratio"],
            "P/E Forward": i["pe_forward"],
            "PEG": peg,
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
            "ROIC": i.get("roic", None),
            "FCF Yield": i.get("fcf_yield", None),
        })

    df_val = pd.DataFrame(val_rows)
    df_vd = df_val.copy()
    df_vd["Market Cap"] = df_vd["Market Cap"].apply(fmc)
    for col in ["P/E Trailing", "P/E Forward", "PEG", "P/S", "P/B", "EV/EBITDA", "Deuda/Equity"]:
        df_vd[col] = df_vd[col].apply(lambda x: fn(x, 1))
    for col in ["Rev. Growth", "EPS Growth", "Mg. Operativo", "Mg. Neto", "ROE", "ROA",
                "Div. Yield", "Shareholder Yield", "ROIC", "FCF Yield"]:
        df_vd[col] = df_vd[col].apply(fp)
    st.dataframe(df_vd.set_index("Ticker"), use_container_width=True)

    # Extended metrics table
    with st.expander("📋 Métricas adicionales (cash, deuda, EPS, ownership)"):
        ext_rows = []
        for t in TICKERS:
            i = info[t]
            mcap = i.get("market_cap") or 1
            ext_rows.append({
                "Ticker": t,
                "EPS Trailing": fn(i.get("trailing_eps"), 2),
                "EPS Forward": fn(i.get("forward_eps"), 2),
                "Mg. Bruto": fp(i.get("gross_margins")),
                "Mg. EBITDA": fp(i.get("ebitda_margins")),
                "FCF Yield": fp(i.get("fcf_yield")),
                "FCF (TTM)": fmc(i.get("free_cashflow")),
                "Op. CF (TTM)": fmc(i.get("operating_cashflow")),
                "Cash": fmc(i.get("total_cash")),
                "Deuda Total": fmc(i.get("total_debt")),
                "Deuda Neta": fmc(i.get("net_debt")),
                "EV": fmc(i.get("enterprise_value")),
                "EV/Rev": fn(i.get("ev_revenue"), 1),
                "Insiders %": fp(i.get("insider_pct")),
                "Instituciones %": fp(i.get("institution_pct")),
                "Short %": fp(i.get("short_pct")),
            })
        df_ext = pd.DataFrame(ext_rows)
        st.dataframe(df_ext.set_index("Ticker"), use_container_width=True)

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

    # PEG bar chart
    st.subheader("PEG Ratio — barato vs caro ajustado por crecimiento")
    peg_df = df_val[["Ticker", "PEG"]].dropna(subset=["PEG"]).copy()
    if not peg_df.empty:
        peg_df = peg_df.sort_values("PEG")
        colors_peg = ["#26a69a" if v < 1 else "#ef9a9a" if v < 2 else "#ef5350"
                      for v in peg_df["PEG"]]
        fig_peg = go.Figure(go.Bar(
            x=peg_df["Ticker"], y=peg_df["PEG"],
            marker_color=colors_peg,
            text=[f"{v:.2f}" for v in peg_df["PEG"]],
            textposition="outside"
        ))
        fig_peg.add_hline(y=1, line_dash="dot", line_color="white", opacity=0.5,
                           annotation_text="PEG = 1 (referencia justo precio)",
                           annotation_position="right")
        fig_peg.update_layout(
            height=380, template="plotly_dark" if True else "plotly_white",
            template="plotly_white",
            yaxis_title="PEG",
            showlegend=False
        )
        st.plotly_chart(fig_peg, use_container_width=True)
        st.caption("Verde < 1: barato dado su crecimiento · Naranja 1–2: valuación razonable · Rojo > 2: caro")
    else:
        st.info("Sin datos de PEG suficientes (requiere P/E y EPS Growth).")

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
# TAB 3 — FUNDAMENTALS HISTÓRICOS (yfinance)
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    equity = [t for t in TICKERS if t not in ETF_TICKERS]
    sel = st.multiselect("Activos", equity, default=equity[:4])

    if sel:
        # ── Data source indicator ────────────────────────────────────────────
        # Try stockanalysis for first ticker to check availability
        sa_test = get_sa_key_metrics(sel[0])
        sa_ok = bool(sa_test)
        if sa_ok:
            st.session_state["sa_available"] = True
            st.success("📊 Fuente: Stockanalysis.com (datos históricos completos)", icon="✅")
        else:
            st.session_state["sa_available"] = False
            st.info("📊 Fuente: yfinance (histórico limitado a ~4 años). "
                    "Stockanalysis no disponible desde este servidor.", icon="ℹ️")

        # ── ROIC histórico ───────────────────────────────────────────────────
        st.subheader("ROIC histórico")
        fig_roic = go.Figure()
        for t in sel:
            # Try stockanalysis first
            df_r = get_sa_roic_series(t) if sa_ok else pd.DataFrame()
            if not df_r.empty:
                fig_roic.add_trace(go.Scatter(
                    x=df_r["year"], y=df_r["roic"] * 100,
                    name=t, mode="lines+markers"
                ))
            else:
                # Fallback: compute from yfinance financials
                df_r = compute_roic_history(t)
                if not df_r.empty:
                    fig_roic.add_trace(go.Scatter(
                        x=df_r["date"], y=df_r["roic"] * 100,
                        name=f"{t} (yf)", mode="lines+markers",
                        line=dict(dash="dot")
                    ))
        if fig_roic.data:
            fig_roic.update_layout(yaxis_title="ROIC (%)", height=420,
                                    template="plotly_white", hovermode="x unified")
            st.plotly_chart(fig_roic, use_container_width=True)
        else:
            st.info("Sin datos de ROIC disponibles para los activos seleccionados.")

        # ── Márgenes históricos ──────────────────────────────────────────────
        st.subheader("Márgenes históricos")
        sel_mg = st.selectbox("Activo", sel, key="mg_hist")
        if sa_ok:
            df_mg = get_sa_margins_series(sel_mg)
            if not df_mg.empty:
                fig_mg_h = go.Figure()
                color_map = {"gross_margin": "#42a5f5", "operating_margin": "#ab47bc", "net_margin": "#26a69a"}
                label_map = {"gross_margin": "Mg. Bruto", "operating_margin": "Mg. Operativo", "net_margin": "Mg. Neto"}
                for col in df_mg.columns:
                    s = df_mg[col].dropna()
                    fig_mg_h.add_trace(go.Scatter(
                        x=s.index, y=s.values * 100,
                        name=label_map.get(col, col),
                        mode="lines+markers",
                        line=dict(color=color_map.get(col))
                    ))
                fig_mg_h.update_layout(yaxis_title="%", height=380,
                                        template="plotly_white", hovermode="x unified")
                st.plotly_chart(fig_mg_h, use_container_width=True)

        # ── Revenue + Net Income ─────────────────────────────────────────────
        st.subheader("Revenue y Net Income histórico")
        sel_inc = st.selectbox("Activo", sel, key="inc")
        df_fin = get_financials_yf(sel_inc)
        if not df_fin.empty:
            ci1, ci2 = st.columns(2)
            rev_col = next((c for c in ["Total Revenue", "Revenue"] if c in df_fin.columns), None)
            ni_col  = next((c for c in ["Net Income", "Net Income Common Stockholders"] if c in df_fin.columns), None)
            with ci1:
                if rev_col:
                    fig_rv = px.bar(
                        x=df_fin.index.year.astype(str), y=df_fin[rev_col] / 1e9,
                        template="plotly_white", labels={"x": "Año", "y": "USD Bn"},
                        title="Revenue"
                    )
                    fig_rv.update_layout(height=320, showlegend=False)
                    st.plotly_chart(fig_rv, use_container_width=True)
            with ci2:
                if ni_col:
                    fig_ni = px.bar(
                        x=df_fin.index.year.astype(str), y=df_fin[ni_col] / 1e9,
                        template="plotly_white", labels={"x": "Año", "y": "USD Bn"},
                        title="Net Income", color_discrete_sequence=["#00b09b"]
                    )
                    fig_ni.update_layout(height=320, showlegend=False)
                    st.plotly_chart(fig_ni, use_container_width=True)
        else:
            st.info(f"Sin datos de income statement para {sel_inc}.")

        # ── EBITDA histórico ─────────────────────────────────────────────────
        st.subheader("EBITDA histórico")
        fig_ebitda = go.Figure()
        for t in sel:
            df_fin2 = get_financials_yf(t)
            ebitda_col = next((c for c in ["EBITDA", "Normalized EBITDA"] if c in df_fin2.columns), None)
            if df_fin2.empty or not ebitda_col:
                continue
            fig_ebitda.add_trace(go.Bar(
                x=df_fin2.index.year.astype(str),
                y=df_fin2[ebitda_col] / 1e9,
                name=t
            ))
        fig_ebitda.update_layout(yaxis_title="USD Bn", barmode="group",
                                  height=380, template="plotly_white")
        st.plotly_chart(fig_ebitda, use_container_width=True)

        # ── Free Cash Flow ───────────────────────────────────────────────────
        st.subheader("Free Cash Flow histórico")
        fig_fcf = go.Figure()
        for t in sel:
            added = False
            # Try stockanalysis first
            if sa_ok:
                df_sa_fcf = get_sa_fcf_series(t)
                if not df_sa_fcf.empty:
                    fig_fcf.add_trace(go.Bar(
                        x=df_sa_fcf["year"],
                        y=df_sa_fcf["fcf"] / 1e9,
                        name=t
                    ))
                    added = True
            # Fallback: yfinance cashflow
            if not added:
                df_cf = get_cashflow_yf(t)
                if df_cf.empty:
                    continue
                fcf_col = next((c for c in ["Free Cash Flow", "FreeCashFlow"] if c in df_cf.columns), None)
                if not fcf_col:
                    ocf = next((c for c in ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"] if c in df_cf.columns), None)
                    capex = next((c for c in ["Capital Expenditure", "Capital Expenditures"] if c in df_cf.columns), None)
                    if ocf and capex:
                        df_cf = df_cf.copy()
                        df_cf["Free Cash Flow"] = df_cf[ocf] + df_cf[capex]
                        fcf_col = "Free Cash Flow"
                if fcf_col:
                    fig_fcf.add_trace(go.Bar(
                        x=df_cf.index.year.astype(str),
                        y=df_cf[fcf_col] / 1e9,
                        name=f"{t} (yf)"
                    ))
        if fig_fcf.data:
            fig_fcf.update_layout(yaxis_title="USD Bn", barmode="group",
                                   height=380, template="plotly_white")
            st.plotly_chart(fig_fcf, use_container_width=True)

        # ── Buybacks + Dividendos ────────────────────────────────────────────
        st.subheader("Retorno al accionista — Buybacks y Dividendos")
        sel_sh = st.selectbox("Activo", sel, key="sh")
        df_cf2 = get_cashflow_yf(sel_sh)
        if not df_cf2.empty:
            buyback_col = next((c for c in [
                "Repurchase Of Capital Stock", "Common Stock Repurchased",
                "Repurchase Of Common Stock", "Purchase Of Business"
            ] if c in df_cf2.columns), None)
            div_col = next((c for c in [
                "Payment Of Dividends", "Cash Dividends Paid",
                "Common Stock Dividend Paid", "Dividends Paid"
            ] if c in df_cf2.columns), None)

            sh_data = {"Año": df_cf2.index.year.astype(str)}
            if buyback_col:
                sh_data["Buybacks"] = df_cf2[buyback_col].abs() / 1e9
            if div_col:
                sh_data["Dividendos"] = df_cf2[div_col].abs() / 1e9

            melt_cols = [c for c in ["Buybacks", "Dividendos"] if c in sh_data]
            if melt_cols:
                df_sh = pd.DataFrame(sh_data)
                sh_m = df_sh.melt(id_vars="Año", var_name="Tipo", value_name="USD Bn")
                fig_sh = px.bar(sh_m, x="Año", y="USD Bn", color="Tipo",
                                barmode="stack", template="plotly_white",
                                title=f"{sel_sh} — Retorno al accionista")
                fig_sh.update_layout(height=360)
                st.plotly_chart(fig_sh, use_container_width=True)
            else:
                st.info("Sin datos de buybacks/dividendos.")
        else:
            st.info(f"Sin datos de cashflow para {sel_sh}.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ANALISTAS (yfinance + FMP price targets)
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    equity = [t for t in TICKERS if t not in ETF_TICKERS]

    # ── Price targets ────────────────────────────────────────────────────────
    st.subheader("Precio objetivo — estimaciones de analistas")
    pt_rows = []
    for t in equity:
        s = prices[t].dropna()
        current = float(s.iloc[-1]) if not s.empty else None
        # yfinance analyst_price_targets
        try:
            apt = yf.Ticker(t).analyst_price_targets or {}
        except Exception:
            apt = {}
        target_mean = apt.get("mean", None)
        target_low  = apt.get("low", None)
        target_high = apt.get("high", None)
        upside = ((target_mean / current) - 1) if (target_mean and current) else None
        pt_rows.append({
            "Ticker": t,
            "Precio Actual": current,
            "Target Bajo": target_low,
            "Target Medio": target_mean,
            "Target Alto": target_high,
            "Upside": upside,
        })

    df_pt = pd.DataFrame(pt_rows)
    if not df_pt.empty:
        df_ptd = df_pt.copy()
        for col in ["Precio Actual", "Target Bajo", "Target Medio", "Target Alto"]:
            df_ptd[col] = df_ptd[col].apply(fpr)
        df_ptd["Upside"] = df_ptd["Upside"].apply(fp)
        st.dataframe(df_ptd.set_index("Ticker"), use_container_width=True)

        uc = df_pt.dropna(subset=["Upside"]).sort_values("Upside")
        if not uc.empty:
            colors = ["#ef5350" if v < 0 else "#26a69a" for v in uc["Upside"]]
            fig_up = go.Figure(go.Bar(
                x=uc["Upside"] * 100, y=uc["Ticker"], orientation="h",
                marker_color=colors,
                text=[f"{v:.1f}%" for v in uc["Upside"] * 100],
                textposition="outside"
            ))
            fig_up.update_layout(xaxis_title="%", height=350, template="plotly_white",
                                  title="Upside implícito vs target medio")
            st.plotly_chart(fig_up, use_container_width=True)

    # ── Recomendaciones buy/hold/sell ────────────────────────────────────────
    st.subheader("Recomendaciones de analistas")
    rec_rows = []
    for t in equity:
        try:
            rec = yf.Ticker(t).recommendations_summary
            if rec is not None and not rec.empty:
                row = rec.iloc[0].to_dict() if len(rec) > 0 else {}
                row["Ticker"] = t
                rec_rows.append(row)
        except Exception:
            pass

    if rec_rows:
        df_rec = pd.DataFrame(rec_rows).set_index("Ticker")
        # Normalize column names
        df_rec.columns = [c.replace("strongBuy", "Strong Buy").replace("buy", "Buy")
                           .replace("hold", "Hold").replace("sell", "Sell")
                           .replace("strongSell", "Strong Sell") for c in df_rec.columns]
        st.dataframe(df_rec, use_container_width=True)

        # Stacked bar
        rec_cols = [c for c in ["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"] if c in df_rec.columns]
        if rec_cols:
            rec_melt = df_rec[rec_cols].reset_index().melt(id_vars="Ticker",
                                                            var_name="Rating", value_name="Count")
            color_map = {
                "Strong Buy": "#1b5e20", "Buy": "#43a047",
                "Hold": "#f9a825", "Sell": "#e53935", "Strong Sell": "#b71c1c"
            }
            fig_rec = px.bar(rec_melt, x="Ticker", y="Count", color="Rating",
                             barmode="stack", template="plotly_white",
                             color_discrete_map=color_map,
                             title="Distribución de recomendaciones")
            fig_rec.update_layout(height=380)
            st.plotly_chart(fig_rec, use_container_width=True)

    # ── Forward estimates ────────────────────────────────────────────────────
    st.subheader("Estimaciones forward — EPS y Revenue")
    sel_est = st.selectbox("Activo", equity, key="est")
    try:
        tk_est = yf.Ticker(sel_est)
        eps_est    = tk_est.earnings_estimate
        rev_est    = tk_est.revenue_estimate

        ce1, ce2 = st.columns(2)
        with ce1:
            if eps_est is not None and not eps_est.empty:
                fig_eps = go.Figure()
                fig_eps.add_trace(go.Bar(
                    x=eps_est.index.astype(str), y=eps_est.get("avg", eps_est.iloc[:, 0]),
                    name="Consenso EPS", marker_color="#ab47bc"
                ))
                if "low" in eps_est.columns and "high" in eps_est.columns:
                    fig_eps.add_trace(go.Scatter(
                        x=eps_est.index.astype(str), y=eps_est["high"],
                        name="Alto", line=dict(dash="dot", color="green"), mode="lines"
                    ))
                    fig_eps.add_trace(go.Scatter(
                        x=eps_est.index.astype(str), y=eps_est["low"],
                        name="Bajo", line=dict(dash="dot", color="red"), mode="lines"
                    ))
                fig_eps.update_layout(title="EPS estimado", height=350, template="plotly_white")
                st.plotly_chart(fig_eps, use_container_width=True)
            else:
                st.info("Sin estimaciones de EPS.")

        with ce2:
            if rev_est is not None and not rev_est.empty:
                avg_col = next((c for c in ["avg", "mean"] if c in rev_est.columns), rev_est.columns[0])
                fig_rev = go.Figure()
                fig_rev.add_trace(go.Bar(
                    x=rev_est.index.astype(str), y=rev_est[avg_col] / 1e9,
                    name="Consenso Revenue", marker_color="#42a5f5"
                ))
                if "low" in rev_est.columns and "high" in rev_est.columns:
                    fig_rev.add_trace(go.Scatter(
                        x=rev_est.index.astype(str), y=rev_est["high"] / 1e9,
                        name="Alto", line=dict(dash="dot", color="green"), mode="lines"
                    ))
                    fig_rev.add_trace(go.Scatter(
                        x=rev_est.index.astype(str), y=rev_est["low"] / 1e9,
                        name="Bajo", line=dict(dash="dot", color="red"), mode="lines"
                    ))
                fig_rev.update_layout(title="Revenue estimado (USD Bn)", height=350, template="plotly_white")
                st.plotly_chart(fig_rev, use_container_width=True)
            else:
                st.info("Sin estimaciones de Revenue.")
    except Exception as e:
        st.info(f"Sin estimaciones disponibles para {sel_est}. ({e})")

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

    # ── Rolling Sharpe ────────────────────────────────────────────────────────
    st.subheader("Rolling Sharpe ratio")
    rs_col1, rs_col2 = st.columns([1, 3])
    with rs_col1:
        roll_window = st.selectbox("Ventana rolling", ["60 días", "90 días", "180 días"], index=1, key="roll_w")
        roll_tickers = st.multiselect("Activos", TICKERS, default=TICKERS, key="roll_t")
    roll_days = {"60 días": 60, "90 días": 90, "180 días": 180}[roll_window]

    if roll_tickers:
        fig_rs = go.Figure()
        daily_returns = prices.pct_change().dropna()
        for t in roll_tickers:
            if t not in daily_returns.columns:
                continue
            r = daily_returns[t].dropna()
            # Rolling annualized Sharpe
            roll_mean = r.rolling(roll_days).mean() * 252
            roll_std  = r.rolling(roll_days).std() * np.sqrt(252)
            roll_sharpe = (roll_mean - RISK_FREE_RATE) / roll_std
            is_b = t == BENCHMARK
            fig_rs.add_trace(go.Scatter(
                x=roll_sharpe.index, y=roll_sharpe.values, name=t,
                line=dict(width=2.5 if is_b else 1.5, dash="dash" if is_b else "solid"),
                opacity=1 if is_b else 0.85
            ))
        # Reference line at 0 and 1
        fig_rs.add_hline(y=0, line_dash="dot", line_color="red", opacity=0.4)
        fig_rs.add_hline(y=1, line_dash="dot", line_color="green", opacity=0.4,
                          annotation_text="Sharpe = 1", annotation_position="left")
        fig_rs.update_layout(
            yaxis_title="Sharpe (rolling)", hovermode="x unified",
            height=450, template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_rs, use_container_width=True)
        st.caption(f"Línea roja = Sharpe 0 (no cubre la tasa libre de riesgo). "
                   f"Línea verde = Sharpe 1 (referencia 'bueno'). Ventana: {roll_window}.")

    # ── Rolling Volatilidad ────────────────────────────────────────────────────
    st.subheader("Volatilidad rolling (anualizada)")
    vol_tickers = st.multiselect("Activos", TICKERS, default=TICKERS[:4], key="vol_t")
    if vol_tickers:
        fig_vol = go.Figure()
        for t in vol_tickers:
            if t not in daily_returns.columns:
                continue
            r = daily_returns[t].dropna()
            roll_vol = r.rolling(roll_days).std() * np.sqrt(252) * 100
            fig_vol.add_trace(go.Scatter(
                x=roll_vol.index, y=roll_vol.values, name=t,
                line=dict(width=1.5)
            ))
        fig_vol.update_layout(
            yaxis_title="Volatilidad anualizada (%)", hovermode="x unified",
            height=380, template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_vol, use_container_width=True)

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
