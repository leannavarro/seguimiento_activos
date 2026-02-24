import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta, date

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

# ─── MAE API ──────────────────────────────────────────────────────────────────

MAE_BASE = "https://api.mae.com.ar/MarketData/v1"
MAE_SEG_SOBERANOS_PPT = "4"   # AL30D/CI, AE38D/CI, GD38D/CI …
MAE_SEG_ONS           = "5"   # Obligaciones Negociables corporativas
MAE_SEG_SOBERANOS_MAE = "2"   # AL29, AL30, GD29 … (sin sufijo plazo)

MAE_HEADERS_EXTRA = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "es-AR,es;q=0.9,en;q=0.8",
    "Referer": "https://www.mae.com.ar/",
    "Origin": "https://www.mae.com.ar",
}

def _mae_key():
    try:
        k = st.secrets.get("MAE_API_KEY", "")
        if k:
            return k
    except Exception:
        pass
    return st.session_state.get("mae_api_key", "")

D912_BASE = "https://data912.com"

@st.cache_data(ttl=120)
def d912_live_bonds():
    """
    Cotizaciones en tiempo real — data912.com/live/arg_bonds.
    Sin autenticacion. Campos: symbol, px_bid, px_ask, c, pct_change, v, q_op.
    TTL 2 min.
    """
    try:
        r = requests.get(f"{D912_BASE}/live/arg_bonds", timeout=15)
        r.raise_for_status()
        raw = r.json()
        if isinstance(raw, dict):
            raw = raw.get("value", raw.get("data", []))
        df = pd.DataFrame(raw)
        if df.empty:
            return df
        df = df[df["symbol"].notna() & (df["symbol"] != "")]
        return df
    except Exception as e:
        st.session_state.setdefault("d912_errors", []).append(str(e))
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def d912_historical(symbol: str):
    """
    Serie historica OHLCV para un simbolo.
    Campos: date, o, h, l, c, v, dr (daily return), sa.
    """
    try:
        r = requests.get(f"{D912_BASE}/historical/bonds/{symbol}", timeout=20)
        r.raise_for_status()
        raw = r.json()
        if isinstance(raw, dict):
            raw = raw.get("value", raw.get("data", []))
        df = pd.DataFrame(raw)
        if df.empty:
            return df
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        return df
    except Exception as e:
        st.session_state.setdefault("d912_errors", []).append(str(e))
        return pd.DataFrame()

def mae_cotizaciones_hoy():
    """Wrapper legacy — deprecado, usar d912_live_bonds()."""
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def mae_boletin(fecha: str):
    """
    Boletín diario MAE. fecha = 'YYYY-MM-DD'.
    Devuelve dict {seg_codigo: DataFrame} con columnas estandarizadas.
    Usa precioCierreHoy (plazo 000 = CI) como precio de referencia.
    """
    key = _mae_key()
    if not key:
        return {}
    try:
        r = requests.get(
            f"{MAE_BASE}/mercado/boletin/ReporteResumenFinal",
            headers={"x-api-key": key, **MAE_HEADERS_EXTRA},
            params={"fecha": fecha},
            timeout=20,
        )
        r.raise_for_status()
        data = r.json()
        segs = data.get("segmento", [])
        result = {}
        for seg in segs:
            cod = str(seg.get("segmentoCodigo", ""))
            titulos = seg.get("titulos", {})
            # titulos puede ser lista o dict con "value"
            if isinstance(titulos, dict):
                titulos = titulos.get("value", [])
            df = pd.DataFrame(titulos)
            if df.empty:
                continue
            # Conservar solo filas con ticker real
            df = df[df["ticker"].notna() & (df["ticker"] != "")]
            # Separar ticker base y sufijo
            df["ticker_base"] = df["ticker"].str.replace(r"/.*$", "", regex=True)
            df["plazo_sfx"]   = df["ticker"].str.extract(r"/(.*)")
            df["plazo_sfx"]   = df["plazo_sfx"].fillna("")
            # Precio normalizado: para ONs, precioCierreHoy está en % (ej 151020 = 1510.20%)
            # Para soberanos en D, está también escalado — usamos precioCierreHoy / 100
            df["precio_cierre"] = pd.to_numeric(df.get("precioCierreHoy", 0), errors="coerce")
            df["precio_ayer"]   = pd.to_numeric(df.get("precioCierreAyer", 0), errors="coerce")
            df["variacion"]     = pd.to_numeric(df.get("variacion", 0), errors="coerce")
            df["cantidad"]      = pd.to_numeric(df.get("cantidad", 0), errors="coerce")
            df["monto"]         = pd.to_numeric(df.get("monto", 0), errors="coerce")
            result[cod] = df
        return result
    except Exception as e:
        st.session_state.setdefault("mae_errors", []).append(str(e))
        return {}

def mae_boletin_historico(ticker_base: str, dias: int = 60, plazo: str = "000"):
    """
    Descarga cierres diarios para un ticker iterando el boletín.
    plazo '000' = CI, '001' = 24hs.
    CUIDADO: genera N llamadas HTTP. Usar con caché externa o poco frecuente.
    """
    key = _mae_key()
    if not key:
        return pd.DataFrame()
    records = []
    today = datetime.today()
    fechas = pd.bdate_range(end=today, periods=dias)
    for f in fechas:
        fecha_str = f.strftime("%Y-%m-%d")
        try:
            r = requests.get(
                f"{MAE_BASE}/mercado/boletin/ReporteResumenFinal",
                headers={"x-api-key": key, **MAE_HEADERS_EXTRA},
                params={"fecha": fecha_str},
                timeout=15,
            )
            if r.status_code != 200:
                continue
            segs = r.json().get("segmento", [])
            for seg in segs:
                titulos = seg.get("titulos", {})
                if isinstance(titulos, dict):
                    titulos = titulos.get("value", [])
                for t in titulos:
                    tb = str(t.get("ticker", "")).replace(f"/{plazo}", "").split("/")[0]
                    p_sfx = str(t.get("plazo", ""))
                    if tb == ticker_base and p_sfx == plazo:
                        records.append({
                            "fecha": f.date(),
                            "precio": float(t.get("precioCierreHoy", 0) or 0),
                            "variacion": float(t.get("variacion", 0) or 0),
                            "monto": float(t.get("monto", 0) or 0),
                        })
        except Exception:
            continue
    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values("fecha").drop_duplicates("fecha")
    return df

# ─── BASE DE BONOS (hardcodeada + verificable con MAE) ────────────────────────
# Estructura: ticker_mae = ticker sin sufijo como aparece en MAE (ej "AL30D", "GD38D")
# Para soberanos en USD: moneda = "USD", precio MAE en D (dólares cable / MEP)
# duration y ytm se calculan on-the-fly si hay precio de mercado

from scipy.optimize import brentq

# ─── BOND MATH — base 30/360, prospectos MECON Dto 391/2020 ──────────────────

from dateutil.relativedelta import relativedelta

def _days_30_360(d1, d2):
    """Convención 30/360 US Bond Basis."""
    y1, m1, day1 = d1.year, d1.month, min(d1.day, 30)
    y2, m2, day2 = d2.year, d2.month, d2.day
    if day2 == 31 and day1 >= 30:
        day2 = 30
    return 360*(y2-y1) + 30*(m2-m1) + (day2-day1)

def _frac_30_360(d1, d2):
    return _days_30_360(d1, d2) / 360.0

def _build_sovereign_flows(amort_sched, rate_sched, coupon_start, end_date):
    """
    Genera cash flows completos para bonos soberanos canje 2020.
    amort_sched: [(date, pct_vn_original), ...]
    rate_sched:  [(date_from, date_to, tna), ...]
    Retorna: [(date, cupon_pct, amort_pct), ...] sobre VN original=100
    """
    from datetime import date as _date
    amort_map = {d: a for d, a in amort_sched}
    # Generar todas las fechas de cupón
    coupon_dates = []
    d = coupon_start
    while d <= end_date:
        coupon_dates.append(d)
        d += relativedelta(months=6)
    vn = 100.0
    flows = []
    prev = coupon_start - relativedelta(months=6)
    for cd in coupon_dates:
        tna = next((r for df, dt, r in rate_sched if df <= prev < dt), rate_sched[-1][2])
        frac = _frac_30_360(prev, cd)
        cup  = vn * tna * frac
        am   = amort_map.get(cd, 0.0)
        flows.append((cd, round(cup, 8), round(am, 8)))
        vn  -= am
        prev = cd
    return flows

# ─ Helpers de amortización ───────────────────────────────────────────────────
def _amort_equal(first_date, n):
    """n cuotas iguales de 100/n% semestrales."""
    return [(first_date + relativedelta(months=6*i), round(100/n, 8)) for i in range(n)]

def _amort_step(first_date, first_pct, rest_pct, end_date):
    """Primera cuota = first_pct, restantes = rest_pct cada 6 meses hasta end_date."""
    sched = [(first_date, first_pct)]
    d = first_date + relativedelta(months=6)
    while d <= end_date:
        sched.append((d, rest_pct))
        d += relativedelta(months=6)
    return sched

# ─ Fecha inicio cupones (común a todos los bonos del canje) ──────────────────
_CS = date(2021, 7, 9)

# ─ Tramos de tasas por familia ───────────────────────────────────────────────
_RATES_GD29 = [(date(2020,9,4), date(2030,1,1), 0.01)]
_RATES_GD30 = [
    (date(2020,9,4),  date(2021,7,9),  0.00125),
    (date(2021,7,9),  date(2023,7,9),  0.0050),
    (date(2023,7,9),  date(2027,7,9),  0.0075),
    (date(2027,7,9),  date(2031,1,1),  0.0175),
]
_RATES_GD35 = [
    (date(2020,9,4),  date(2021,7,9),  0.00125),
    (date(2021,7,9),  date(2022,7,9),  0.01125),
    (date(2022,7,9),  date(2023,7,9),  0.0150),
    (date(2023,7,9),  date(2024,7,9),  0.03625),
    (date(2024,7,9),  date(2027,7,9),  0.04125),
    (date(2027,7,9),  date(2028,7,9),  0.0475),
    (date(2028,7,9),  date(2036,1,1),  0.0500),
]
_RATES_GD38 = [
    (date(2020,9,4),  date(2021,7,9),  0.00125),
    (date(2021,7,9),  date(2022,7,9),  0.0200),
    (date(2022,7,9),  date(2023,7,9),  0.03875),
    (date(2023,7,9),  date(2024,7,9),  0.0425),
    (date(2024,7,9),  date(2038,7,1),  0.0500),
]
_RATES_GD41 = [
    (date(2020,9,4),  date(2021,7,9),  0.00125),
    (date(2021,7,9),  date(2022,7,9),  0.0250),
    (date(2022,7,9),  date(2029,7,9),  0.0350),
    (date(2029,7,9),  date(2042,1,1),  0.04875),
]
_RATES_GD46 = [
    (date(2020,9,4),  date(2021,7,9),  0.00125),
    (date(2021,7,9),  date(2022,7,9),  0.01125),
    (date(2022,7,9),  date(2023,7,9),  0.0150),
    (date(2023,7,9),  date(2024,7,9),  0.03625),
    (date(2024,7,9),  date(2027,7,9),  0.04125),
    (date(2027,7,9),  date(2028,7,9),  0.04375),
    (date(2028,7,9),  date(2047,1,1),  0.0500),
]

# ─── BONDS_DB ────────────────────────────────────────────────────────────────
BONDS_DB = {
    # ── GD29 / AL29 — 1% 2029 ────────────────────────────────────────────────
    "GD29": {
        "nombre": "Global USD 1% 2029 (L.NY)", "ley": "NY", "tipo": "Soberano",
        "moneda": "USD", "vencimiento": date(2029,7,9),
        "cash_flows": _build_sovereign_flows(_amort_equal(date(2025,1,9), 10), _RATES_GD29, _CS, date(2029,7,9)),
    },
    "AL29": {
        "nombre": "Bonar USD 1% 2029 (L.AR)", "ley": "AR", "tipo": "Soberano",
        "moneda": "USD", "vencimiento": date(2029,7,9),
        "cash_flows": _build_sovereign_flows(_amort_equal(date(2025,1,9), 10), _RATES_GD29, _CS, date(2029,7,9)),
    },
    # ── GD30 / AL30 — Step Up 2030 ───────────────────────────────────────────
    "GD30": {
        "nombre": "Global USD Step Up 2030 (L.NY)", "ley": "NY", "tipo": "Soberano",
        "moneda": "USD", "vencimiento": date(2030,7,9),
        "cash_flows": _build_sovereign_flows(_amort_step(date(2024,7,9), 4.0, 8.0, date(2030,7,9)), _RATES_GD30, _CS, date(2030,7,9)),
    },
    "AL30": {
        "nombre": "Bonar USD Step Up 2030 (L.AR)", "ley": "AR", "tipo": "Soberano",
        "moneda": "USD", "vencimiento": date(2030,7,9),
        "cash_flows": _build_sovereign_flows(_amort_step(date(2024,7,9), 4.0, 8.0, date(2030,7,9)), _RATES_GD30, _CS, date(2030,7,9)),
    },
    # ── GD35 / AL35 — Step Up 2035 ───────────────────────────────────────────
    "GD35": {
        "nombre": "Global USD Step Up 2035 (L.NY)", "ley": "NY", "tipo": "Soberano",
        "moneda": "USD", "vencimiento": date(2035,7,9),
        "cash_flows": _build_sovereign_flows(_amort_step(date(2029,7,9), 4.0, 8.0, date(2035,7,9)), _RATES_GD35, _CS, date(2035,7,9)),
    },
    "AL35": {
        "nombre": "Bonar USD Step Up 2035 (L.AR)", "ley": "AR", "tipo": "Soberano",
        "moneda": "USD", "vencimiento": date(2035,7,9),
        "cash_flows": _build_sovereign_flows(_amort_equal(date(2031,1,9), 10), _RATES_GD35, _CS, date(2035,7,9)),
    },
    # ── GD38 / AL38 — Step Up 2038 ───────────────────────────────────────────
    "GD38": {
        "nombre": "Global USD Step Up 2038 (L.NY)", "ley": "NY", "tipo": "Soberano",
        "moneda": "USD", "vencimiento": date(2038,1,9),
        "cash_flows": _build_sovereign_flows(_amort_equal(date(2027,7,9), 22), _RATES_GD38, _CS, date(2038,1,9)),
    },
    "AL38": {
        "nombre": "Bonar USD Step Up 2038 (L.AR)", "ley": "AR", "tipo": "Soberano",
        "moneda": "USD", "vencimiento": date(2038,1,9),
        "cash_flows": _build_sovereign_flows(_amort_equal(date(2027,7,9), 22), _RATES_GD38, _CS, date(2038,1,9)),
    },
    # ── GD41 / AL41 — Step Up 2041 ───────────────────────────────────────────
    "GD41": {
        "nombre": "Global USD Step Up 2041 (L.NY)", "ley": "NY", "tipo": "Soberano",
        "moneda": "USD", "vencimiento": date(2041,7,9),
        "cash_flows": _build_sovereign_flows(_amort_equal(date(2028,1,9), 28), _RATES_GD41, _CS, date(2041,7,9)),
    },
    "AL41": {
        "nombre": "Bonar USD Step Up 2041 (L.AR)", "ley": "AR", "tipo": "Soberano",
        "moneda": "USD", "vencimiento": date(2041,7,9),
        "cash_flows": _build_sovereign_flows(_amort_equal(date(2028,1,9), 28), _RATES_GD41, _CS, date(2041,7,9)),
    },
    # ── GD46 / AL46 — Step Up 2046 ───────────────────────────────────────────
    "GD46": {
        "nombre": "Global USD Step Up 2046 (L.NY)", "ley": "NY", "tipo": "Soberano",
        "moneda": "USD", "vencimiento": date(2046,7,9),
        "cash_flows": _build_sovereign_flows(_amort_equal(date(2025,1,9), 44), _RATES_GD46, _CS, date(2046,7,9)),
    },
    "AL46": {
        "nombre": "Bonar USD Step Up 2046 (L.AR)", "ley": "AR", "tipo": "Soberano",
        "moneda": "USD", "vencimiento": date(2046,7,9),
        "cash_flows": _build_sovereign_flows(_amort_equal(date(2025,1,9), 44), _RATES_GD46, _CS, date(2046,7,9)),
    },
}

# ─── BOND MATH FUNCTIONS ──────────────────────────────────────────────────────

def _bond_future_cfs(bond_key: str, settlement=None):
    """
    Retorna [(date, total_flujo_pct_vn)] para flujos futuros al settlement.
    total = cupón + amortización.
    """
    if settlement is None:
        settlement = datetime.today().date()
    elif hasattr(settlement, 'date'):
        settlement = settlement.date()
    b = BONDS_DB[bond_key]
    return [(cd, round(cup+am, 8))
            for cd, cup, am in b["cash_flows"] if cd > settlement]

def _current_vn(bond_key: str, settlement=None):
    """VN residual como % del VN original en la fecha de settlement."""
    if settlement is None:
        settlement = datetime.today().date()
    elif hasattr(settlement, 'date'):
        settlement = settlement.date()
    b = BONDS_DB[bond_key]
    paid = sum(am for cd, cup, am in b["cash_flows"] if cd <= settlement)
    return 100.0 - paid

def _cupon_corrido(bond_key: str, settlement=None):
    """
    Cupón corrido en % del VN original. Convención 30/360.
    """
    if settlement is None:
        settlement = datetime.today().date()
    elif hasattr(settlement, 'date'):
        settlement = settlement.date()
    b = BONDS_DB[bond_key]
    flows = b["cash_flows"]
    pasados = [(cd, cup, am) for cd, cup, am in flows if cd <= settlement]
    futuros = [(cd, cup, am) for cd, cup, am in flows if cd > settlement]
    if not futuros:
        return 0.0
    prev_cd = pasados[-1][0] if pasados else flows[0][0] - relativedelta(months=6)
    next_cd, cup_prox, am_prox = futuros[0]
    # VN al inicio del período actual
    vn_inicio = _current_vn(bond_key, prev_cd)
    # Tasa vigente — recalculamos interpolando con la duración del período
    frac_total = _frac_30_360(prev_cd, next_cd)
    if frac_total <= 0:
        return 0.0
    tna_vigente = cup_prox / (vn_inicio * frac_total) if vn_inicio > 0 else 0.0
    frac_corrida = _frac_30_360(prev_cd, settlement)
    return vn_inicio * tna_vigente * frac_corrida

def _valor_tecnico(bond_key: str, settlement=None):
    """Valor técnico = VN_residual * (1 + CC/VN_residual) = VN_residual + CC"""
    cc = _cupon_corrido(bond_key, settlement)
    vn = _current_vn(bond_key, settlement)
    return vn + cc

def _ytm(bond_key: str, precio_sucio: float, settlement=None):
    """YTM anualizada con base 30/360. Precio sucio en % VN original."""
    if settlement is None:
        settlement = datetime.today().date()
    elif hasattr(settlement, 'date'):
        settlement = settlement.date()
    cfs = _bond_future_cfs(bond_key, settlement)
    if not cfs or precio_sucio <= 0:
        return None
    def f(y):
        return sum(tot / (1+y)**_frac_30_360(settlement, cd) for cd, tot in cfs) - precio_sucio
    try:
        return brentq(f, -0.5, 10.0, maxiter=300)
    except Exception:
        return None

def _duration_macaulay(bond_key: str, precio_sucio: float, settlement=None):
    """Duration Macaulay en años (30/360)."""
    if settlement is None:
        settlement = datetime.today().date()
    elif hasattr(settlement, 'date'):
        settlement = settlement.date()
    cfs = _bond_future_cfs(bond_key, settlement)
    if not cfs or precio_sucio <= 0:
        return None
    ytm = _ytm(bond_key, precio_sucio, settlement)
    if ytm is None:
        return None
    pv_tot = sum(tot/(1+ytm)**_frac_30_360(settlement, cd) for cd, tot in cfs)
    if pv_tot <= 0:
        return None
    return sum(_frac_30_360(settlement, cd) * tot/(1+ytm)**_frac_30_360(settlement, cd) for cd, tot in cfs) / pv_tot

def _convexity(bond_key: str, precio_sucio: float, settlement=None):
    """Convexidad (30/360)."""
    if settlement is None:
        settlement = datetime.today().date()
    elif hasattr(settlement, 'date'):
        settlement = settlement.date()
    cfs = _bond_future_cfs(bond_key, settlement)
    if not cfs or precio_sucio <= 0:
        return None
    ytm = _ytm(bond_key, precio_sucio, settlement)
    if ytm is None:
        return None
    pv_tot = sum(tot/(1+ytm)**_frac_30_360(settlement, cd) for cd, tot in cfs)
    if pv_tot <= 0:
        return None
    return sum(_frac_30_360(settlement,cd)**2 * tot/(1+ytm)**(_frac_30_360(settlement,cd)+2)
               for cd, tot in cfs) / pv_tot

def _current_yield(bond_key: str, precio_sucio: float, settlement=None):
    """Current yield = flujos próximos 365 días / precio sucio."""
    if settlement is None:
        settlement = datetime.today().date()
    elif hasattr(settlement, 'date'):
        settlement = settlement.date()
    cfs = _bond_future_cfs(bond_key, settlement)
    cf_12m = sum(tot for cd, tot in cfs if _days_30_360(settlement, cd) <= 360)
    return cf_12m / precio_sucio if precio_sucio > 0 else None

def _paridad(bond_key: str, precio_sucio: float, settlement=None):
    """Paridad = precio_sucio / valor_técnico."""
    vt = _valor_tecnico(bond_key, settlement)
    return precio_sucio / vt if vt > 0 else None

def _tna_from_ytm(ytm: float):
    """TNA semestral compuesta desde YTM anual: TNA = 2*((1+YTM)^0.5 - 1)."""
    return 2 * ((1 + ytm)**0.5 - 1) if ytm is not None else None

def _precio_sucio_from_ytm(bond_key: str, ytm_target: float, settlement=None):
    """Precio sucio dado un YTM objetivo."""
    if settlement is None:
        settlement = datetime.today().date()
    elif hasattr(settlement, 'date'):
        settlement = settlement.date()
    cfs = _bond_future_cfs(bond_key, settlement)
    return sum(tot/(1+ytm_target)**_frac_30_360(settlement, cd) for cd, tot in cfs)



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

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Resumen", "Valuación", "Fundamentals", "Analistas", "Rendimiento & Corr", "🏦 Renta Fija"
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
            height=380,
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

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — RENTA FIJA (MAE API)
# ══════════════════════════════════════════════════════════════════════════════

with tab6:
    st.subheader("🏦 Renta Fija Argentina")

    # ── API Key MAE ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.divider()
        st.subheader("🏦 MAE API Key")
        mae_input = st.text_input(
            "MAE API Key (A3 Mercados)",
            type="password",
            value=st.session_state.get("mae_api_key", ""),
            help="API Key de A3 Mercados / MAE para datos de renta fija",
        )
        if mae_input:
            st.session_state["mae_api_key"] = mae_input
            st.success("MAE Key cargada ✓")
        elif not _mae_key():
            st.warning("Sin MAE Key: datos de renta fija no disponibles.")

        if st.checkbox("🔍 Errores MAE", value=False):
            errs = st.session_state.get("mae_errors", [])
            if errs:
                for e in errs[-5:]:
                    st.code(e)
            else:
                st.success("Sin errores MAE")

    has_mae = bool(_mae_key())

    rf_tab1, rf_tab2, rf_tab3, rf_tab4 = st.tabs([
        "📡 Mercado Hoy", "📊 Análisis de Bonos", "📈 ONs Corporativas", "💼 Mi Cartera RF"
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # RF TAB 1 — MERCADO HOY (cotizaciones intraday)
    # ══════════════════════════════════════════════════════════════════════════
    with rf_tab1:
        st.markdown("#### Cotizaciones en tiempo real — data912.com")

        df_live = d912_live_bonds()

        if df_live.empty:
            st.warning("Sin datos. data912.com puede estar fuera de servicio.")
            errs = st.session_state.get("d912_errors", [])
            if errs:
                st.code(errs[-1])
        else:
            def _clasificar(sym):
                s = str(sym).upper()
                if s.startswith("GD"):  return "Global USD"
                if s.startswith("AL"):  return "Bonar USD"
                if s.startswith("AE"):  return "AE USD"
                if s.startswith("TX"):  return "CER ARS"
                if s.startswith("X"):   return "Dollar-linked"
                if any(s.startswith(p) for p in ["S2","S3","T2","T3","BL","BU","BG","PM"]):
                    return "Lecap/Boncap"
                return "Otros"

            df_live["clase"] = df_live["symbol"].apply(_clasificar)
            activos = df_live[df_live["v"] > 0]

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total", len(df_live))
            m2.metric("Con operaciones", len(activos))
            m3.metric("Sob. USD", len(df_live[df_live["clase"].isin(["Global USD","Bonar USD","AE USD"])]))
            m4.metric("Tasa/ARS", len(df_live[df_live["clase"].isin(["Lecap/Boncap","CER ARS"])]))

            st.divider()

            fc1, fc2 = st.columns(2)
            with fc1:
                clases_disp = sorted(df_live["clase"].unique())
                clases_sel = st.multiselect(
                    "Clase", clases_disp,
                    default=[c for c in ["Global USD","Bonar USD","AE USD"] if c in clases_disp],
                    key="rf_clase_filter",
                )
            with fc2:
                solo_op = st.checkbox("Solo con operaciones hoy", value=True, key="rf_op_filter")

            df_show = df_live.copy()
            if clases_sel:
                df_show = df_show[df_show["clase"].isin(clases_sel)]
            if solo_op:
                df_show = df_show[df_show["v"] > 0]

            if df_show.empty:
                st.info("Sin instrumentos con los filtros seleccionados.")
            else:
                cols_map = {"symbol":"Ticker","clase":"Clase","px_bid":"Bid","px_ask":"Ask",
                            "c":"Último","pct_change":"Var. %","v":"Volumen (VN)","q_op":"Operaciones"}
                df_disp = df_show[[c for c in cols_map if c in df_show.columns]].rename(columns=cols_map)
                df_disp = df_disp.sort_values("Var. %", ascending=False)

                def _color_var(val):
                    if pd.isna(val): return ""
                    return "color: #00cc88" if val > 0 else ("color: #ff4b4b" if val < 0 else "")

                st.dataframe(
                    df_disp.style.applymap(_color_var, subset=["Var. %"])
                        .format({"Bid":"{:.4f}","Ask":"{:.4f}","Último":"{:.4f}",
                                 "Var. %":"{:.2f}%","Volumen (VN)":"{:,.0f}","Operaciones":"{:.0f}"}),
                    use_container_width=True, height=430,
                )
                st.caption(f"Fuente: data912.com · {len(df_show)} instrumentos · Actualizado c/2 min")

                df_var = df_show[df_show["pct_change"] != 0].copy()
                if len(df_var) >= 3:
                    df_bar = df_var.reindex(df_var["pct_change"].abs().nlargest(20).index).sort_values("pct_change")
                    fig_var = go.Figure(go.Bar(
                        x=df_bar["pct_change"], y=df_bar["symbol"], orientation="h",
                        marker_color=["#00cc88" if v >= 0 else "#ff4b4b" for v in df_bar["pct_change"]],
                        text=df_bar["pct_change"].round(2).astype(str) + "%",
                        textposition="outside",
                    ))
                    fig_var.update_layout(
                        title="Top 20 variaciones del día", xaxis_title="Variación %",
                        height=max(350, len(df_bar)*22), template="plotly_white", margin=dict(l=100),
                    )
                    st.plotly_chart(fig_var, use_container_width=True)

                st.divider()
                st.markdown("**Serie histórica**")
                sym_sel = st.selectbox("Ticker", df_show["symbol"].tolist(), key="rf_sym_hist")
                if sym_sel:
                    df_hist = d912_historical(sym_sel)
                    if df_hist.empty:
                        st.warning(f"Sin histórico para {sym_sel}")
                    else:
                        periodo = st.select_slider("Período", ["3M","6M","1Y","2Y","Todo"],
                                                    value="1Y", key="rf_hist_period")
                        dias_map = {"3M":90,"6M":180,"1Y":252,"2Y":504,"Todo":99999}
                        df_p = df_hist.tail(dias_map[periodo])
                        fig_h = go.Figure()
                        fig_h.add_trace(go.Scatter(
                            x=df_p["date"], y=df_p["c"], mode="lines", name="Cierre",
                            line=dict(color="#636EFA", width=2),
                            fill="tozeroy", fillcolor="rgba(99,110,250,0.08)",
                        ))
                        fig_h.update_layout(
                            title=f"{sym_sel} — Precio histórico", yaxis_title="Precio",
                            height=350, template="plotly_white", hovermode="x unified",
                        )
                        st.plotly_chart(fig_h, use_container_width=True)
                        st.caption(f"{len(df_p)} ruedas · {df_p['date'].iloc[0].date()} → {df_p['date'].iloc[-1].date()}")

    # ══════════════════════════════════════════════════════════════════════════
    # RF TAB 2 — ANÁLISIS DE BONOS (soberanos con math)
    # ══════════════════════════════════════════════════════════════════════════
    with rf_tab2:
        st.markdown("#### Análisis cuantitativo — Soberanos USD")

        soberanos = {k: v for k, v in BONDS_DB.items() if v["tipo"] == "Soberano"}
        bond_sel = st.selectbox(
            "Seleccionar bono",
            list(soberanos.keys()),
            format_func=lambda k: f"{k} — {BONDS_DB[k]['nombre']}",
            key="rf_bond_sel",
        )

        b   = BONDS_DB[bond_sel]
        hoy = datetime.today().date()

        # Precargar precio desde data912 live — buscar ticker USD (sufijo D primero)
        precio_mercado = None
        df_live_tab2 = d912_live_bonds()
        if not df_live_tab2.empty:
            syms_to_try = [bond_sel + "D", bond_sel]  # GD29D primero, luego GD29
            for sym in syms_to_try:
                row = df_live_tab2[df_live_tab2["symbol"].str.upper() == sym.upper()]
                if not row.empty:
                    raw = float(row.iloc[0].get("c", 0) or 0)
                    if raw <= 0:
                        continue
                    # Normalizar: ratio (<5) → *100, precio razonable (<200) → usar
                    if raw < 5:
                        raw *= 100
                    if raw <= 200:
                        precio_mercado = raw
                        break  # precio válido encontrado

        # Input precio
        c_inp, c_metr = st.columns([1, 3])
        with c_inp:
            precio_input = st.number_input(
                "Precio limpio (% VN)",
                min_value=0.1, max_value=200.0,
                value=float(round(min(max(precio_mercado, 0.1), 199.0), 2)) if precio_mercado and precio_mercado > 0 else 63.0,
                step=0.01, format="%.2f",
                key="rf_precio_input",
                help="Precio limpio = cotización de mercado (sin cupón corrido)"
            )
            if precio_mercado:
                st.caption(f"📡 data912: {precio_mercado:.2f}")

        # Calcular métricas
        cc        = _cupon_corrido(bond_sel, hoy)
        ps        = precio_input + cc
        ytm       = _ytm(bond_sel, ps, hoy)
        vt        = _valor_tecnico(bond_sel, hoy)
        paridad   = _paridad(bond_sel, ps, hoy)
        dur_mac   = _duration_macaulay(bond_sel, ps, hoy)
        dur_mod   = (dur_mac / (1 + ytm)) if (dur_mac and ytm is not None) else None
        convex    = _convexity(bond_sel, ps, hoy)
        cy        = _current_yield(bond_sel, ps, hoy)
        tna       = _tna_from_ytm(ytm)
        vn_res    = _current_vn(bond_sel, hoy)
        dias_venc = (b["vencimiento"] - hoy).days

        with c_metr:
            r1c1, r1c2, r1c3, r1c4 = st.columns(4)
            r1c1.metric("Precio limpio",    f"{precio_input:.4f}")
            r1c2.metric("Cupón corrido",    f"{cc:.5f}")
            r1c3.metric("Precio sucio",     f"{ps:.4f}")
            r1c4.metric("VN residual",      f"{vn_res:.2f}%")

            r2c1, r2c2, r2c3, r2c4 = st.columns(4)
            r2c1.metric("Valor técnico",    f"{vt:.5f}")
            r2c2.metric("Paridad",          f"{paridad:.6f}" if paridad else "N/D")
            r2c3.metric("TIR (YTM)",        f"{ytm*100:.4f}%" if ytm is not None else "N/D")
            r2c4.metric("TNA semestral",    f"{tna*100:.4f}%" if tna is not None else "N/D")

            r3c1, r3c2, r3c3, r3c4 = st.columns(4)
            r3c1.metric("Duration Mac.",    f"{dur_mac:.6f}" if dur_mac else "N/D")
            r3c2.metric("Duration Mod.",    f"{dur_mod:.6f}" if dur_mod else "N/D")
            r3c3.metric("Convexidad",       f"{convex:.6f}" if convex else "N/D")
            r3c4.metric("Current yield",    f"{cy*100:.4f}%" if cy else "N/D")

        st.divider()

        # ── Cash flows ─────────────────────────────────────────────────────
        st.markdown("**Flujos de fondos**")
        rows = []
        vn_iter = 100.0
        for cd, cup, am in b["cash_flows"]:
            frac = _frac_30_360(hoy, cd) if cd > hoy else None
            vp   = (cup+am)/(1+ytm)**frac if (frac and ytm) else None
            rows.append({
                "Fecha": cd.strftime("%Y-%m-%d"),
                "VN inicio": round(vn_iter, 4),
                "Cupón": round(cup, 6),
                "Amort": round(am, 4),
                "Total": round(cup+am, 6),
                "VP": round(vp, 6) if vp else "",
                "Pagado": "✓" if cd <= hoy else "",
            })
            vn_iter -= am
        df_cf = pd.DataFrame(rows)

        def _grey_paid(row):
            if row.get("Pagado") == "✓":
                return ["color: #999"] * len(row)
            return [""] * len(row)

        st.dataframe(
            df_cf.style.apply(_grey_paid, axis=1),
            use_container_width=True, height=280,
        )
        st.caption(f"Σ VP futuros = {sum(r['VP'] for r in rows if isinstance(r['VP'], float)):.4f} — Precio sucio ingresado: {ps:.4f}")

        st.divider()

        # ── Curva precio-YTM ───────────────────────────────────────────────
        st.markdown("**Sensibilidad precio / TIR**")
        ytm_range = np.linspace(0.02, 0.25, 100)
        precios_c = [_precio_sucio_from_ytm(bond_sel, y, hoy) for y in ytm_range]
        fig_sens = go.Figure()
        fig_sens.add_trace(go.Scatter(
            x=ytm_range*100, y=precios_c, mode="lines",
            line=dict(color="#636EFA", width=2), name="Precio sucio",
        ))
        if ytm is not None:
            fig_sens.add_vline(x=ytm*100, line_dash="dash", line_color="red",
                annotation_text=f"TIR: {ytm*100:.2f}%", annotation_position="top right")
            fig_sens.add_hline(y=ps, line_dash="dot", line_color="orange",
                annotation_text=f"PS: {ps:.2f}", annotation_position="right")
        fig_sens.update_layout(
            xaxis_title="TIR (%)", yaxis_title="Precio sucio (% VN)",
            height=350, template="plotly_white",
        )
        st.plotly_chart(fig_sens, use_container_width=True)

        st.divider()

        # ── Curva de TIRs comparativa (arbitraje GD vs AL) ────────────────
        st.markdown("**Curva de TIRs — Soberanos USD** (ingresá precios para actualizar)")

        st.caption("Ingresá precios limpios actuales para cada bono. Precios en 0 se omiten.")
        n_bonds = len(soberanos)
        cols_curve = st.columns(min(n_bonds, 6))
        precios_curve = {}
        for i, (tk, bv) in enumerate(soberanos.items()):
            # Precargar desde data912
            pm = None
            if not df_live_tab2.empty:
                for sym in [tk + "D", tk]:
                    row_c = df_live_tab2[df_live_tab2["symbol"].str.upper() == sym.upper()]
                    if not row_c.empty:
                        raw_c = float(row_c.iloc[0].get("c", 0) or 0)
                        if raw_c <= 0:
                            continue
                        if raw_c < 5:
                            raw_c *= 100
                        if raw_c <= 200:
                            pm = raw_c
                            break
            default_p = round(min(max(pm, 0.0), 199.0), 2) if pm and pm > 0 else 0.0
            with cols_curve[i % 6]:
                precios_curve[tk] = st.number_input(
                    tk, min_value=0.0, max_value=200.0,
                    value=default_p, step=0.01, format="%.2f",
                    key=f"rf_curve_{tk}",
                )

        curve_data = []
        for tk, pl in precios_curve.items():
            if pl <= 0:
                continue
            cc_k = _cupon_corrido(tk, hoy)
            ps_k = pl + cc_k
            y_k  = _ytm(tk, ps_k, hoy)
            dm_k = _duration_macaulay(tk, ps_k, hoy)
            vt_k = _valor_tecnico(tk, hoy)
            par_k = ps_k / vt_k if vt_k > 0 else None
            if y_k and dm_k:
                curve_data.append({
                    "ticker": tk, "ley": BONDS_DB[tk]["ley"],
                    "duration": dm_k, "ytm": y_k*100,
                    "paridad": par_k, "precio_limpio": pl,
                    "vto": str(BONDS_DB[tk]["vencimiento"]),
                })

        if len(curve_data) >= 2:
            df_curve = pd.DataFrame(curve_data)
            gd = df_curve[df_curve["ley"] == "NY"]
            al = df_curve[df_curve["ley"] == "AR"]

            fig_curve = go.Figure()
            if not gd.empty:
                fig_curve.add_trace(go.Scatter(
                    x=gd["duration"], y=gd["ytm"],
                    mode="markers+lines+text",
                    name="Ley NY (GD)", marker=dict(color="#636EFA", size=10),
                    line=dict(dash="solid", width=1.5),
                    text=gd["ticker"], textposition="top center",
                ))
            if not al.empty:
                fig_curve.add_trace(go.Scatter(
                    x=al["duration"], y=al["ytm"],
                    mode="markers+lines+text",
                    name="Ley AR (AL)", marker=dict(color="#EF553B", size=10),
                    line=dict(dash="dot", width=1.5),
                    text=al["ticker"], textposition="bottom center",
                ))
            fig_curve.update_layout(
                title="Curva de TIRs — GD (azul) vs AL (rojo)",
                xaxis_title="Duration Macaulay (años)",
                yaxis_title="TIR (%)",
                height=420, template="plotly_white",
                hovermode="x unified",
            )
            st.plotly_chart(fig_curve, use_container_width=True)

            # Tabla con spread GD-AL para arbitraje
            st.markdown("**Spread GD−AL por vencimiento**")
            spread_rows = []
            for yr in ["2029","2030","2035","2038","2041","2046"]:
                gd_tk = f"GD{yr[2:]}"
                al_tk = f"AL{yr[2:]}"
                gd_r = next((r for r in curve_data if r["ticker"]==gd_tk), None)
                al_r = next((r for r in curve_data if r["ticker"]==al_tk), None)
                if gd_r and al_r:
                    spread = gd_r["ytm"] - al_r["ytm"]
                    spread_rows.append({
                        "Vcto": yr, "TIR GD": f"{gd_r['ytm']:.2f}%",
                        "TIR AL": f"{al_r['ytm']:.2f}%",
                        "Spread GD−AL (bps)": round(spread*100, 1),
                        "Par. GD": f"{gd_r['paridad']:.4f}" if gd_r['paridad'] else "",
                        "Par. AL": f"{al_r['paridad']:.4f}" if al_r['paridad'] else "",
                    })
            if spread_rows:
                df_spread = pd.DataFrame(spread_rows)
                def _color_spread(val):
                    try:
                        v = float(val)
                        return "color: #00cc88" if v > 0 else "color: #ff4b4b"
                    except: return ""
                st.dataframe(
                    df_spread.style.applymap(_color_spread, subset=["Spread GD−AL (bps)"]),
                    use_container_width=True, hide_index=True,
                )
                st.caption("Spread positivo → GD rinde más (trade: comprar GD, vender AL)")
        else:
            st.info("Ingresá precios en al menos 2 bonos para ver la curva.")

    # ══════════════════════════════════════════════════════════════════════════
    # RF TAB 3 — ONs CORPORATIVAS
    # ══════════════════════════════════════════════════════════════════════════
    with rf_tab3:
        st.markdown("#### Obligaciones Negociables Corporativas — MAE Segmento 5")

        if not has_mae:
            st.info("Ingresá la MAE API Key en el sidebar.")
        else:
            hoy_str = datetime.today().strftime("%Y-%m-%d")
            dow = datetime.today().weekday()
            if dow == 0:
                fetch_date = (datetime.today() - timedelta(days=3)).strftime("%Y-%m-%d")
            elif dow == 6:
                fetch_date = (datetime.today() - timedelta(days=2)).strftime("%Y-%m-%d")
            else:
                fetch_date = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")

            boletin_data = mae_boletin(fetch_date)

            if MAE_SEG_ONS not in boletin_data:
                st.warning(f"Sin datos de ONs para {fetch_date}. Intentá con otra fecha.")
                fecha_manual = st.date_input("Fecha alternativa", key="rf_fecha_on")
                if st.button("Cargar", key="rf_load_on"):
                    boletin_data = mae_boletin(str(fecha_manual))
            
            if MAE_SEG_ONS in boletin_data:
                df_ons = boletin_data[MAE_SEG_ONS].copy()

                # Filtrar: solo plazo CI (000) y precio > 0
                df_ons_ci = df_ons[
                    (df_ons["plazo"].astype(str) == "000") &
                    (df_ons["precio_cierre"] > 0)
                ].copy()

                # Normalizar precio: en segmento 5 viene como 100x (ej. 151020 = 1510.20 = 15.102% ??)
                # Revisando los datos: VSCOO precioCierreHoy=148550, precioPromedioPonderado=1485.5
                # → precioCierreHoy = precioPromedioPonderado * 100
                # → precio real en % VN = precioPromedioPonderado (ej. 1485.5 = 1485.5% = 14.855 sobre VN 100?)
                # Para ONs USD bullet: precio en % VN típicamente 90-110
                # precioCierreHoy=102 para T662OD → precio=1.02 → 102% VN ✓ (moneda D)
                # precioCierreHoy=148550 para VSCOO (moneda $) → 148550/100 = 1485.5 = precio en pesos?
                # Concluimos: para moneda D → precioCierreHoy = precio % VN directamente
                #             para moneda $ → precio en pesos nominales
                def _precio_on(row):
                    if row["monedaCodigo"] == "D":
                        return row["precio_cierre"]   # ya en % VN
                    else:
                        return row["precio_cierre"]   # en pesos — no normalizable sin TC

                df_ons_ci["precio_norm"] = df_ons_ci.apply(_precio_on, axis=1)

                # Filtros UI
                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    mon_filter = st.multiselect(
                        "Moneda", df_ons_ci["monedaCodigo"].unique().tolist(),
                        default=["D"],
                        key="rf_on_moneda",
                    )
                with col_f2:
                    min_monto = st.number_input(
                        "Monto mínimo (USD / ARS)", value=0.0, step=1000.0, key="rf_on_monto"
                    )

                df_f = df_ons_ci[df_ons_ci["monedaCodigo"].isin(mon_filter)] if mon_filter else df_ons_ci
                if min_monto > 0:
                    df_f = df_f[df_f["monto"] >= min_monto]

                if df_f.empty:
                    st.warning("Sin datos con los filtros seleccionados.")
                else:
                    cols_on = ["ticker_base", "monedaCodigo", "precio_norm", "precio_ayer",
                               "variacion", "cantidad", "monto"]
                    cols_on_ex = [c for c in cols_on if c in df_f.columns]
                    df_on_disp = df_f[cols_on_ex].rename(columns={
                        "ticker_base": "Ticker",
                        "monedaCodigo": "Moneda",
                        "precio_norm": "Precio",
                        "precio_ayer": "Cierre Ayer",
                        "variacion": "Var. %",
                        "cantidad": "Cantidad (VN)",
                        "monto": "Monto",
                    }).sort_values("Monto", ascending=False)

                    st.dataframe(
                        df_on_disp.style.applymap(_color_var, subset=["Var. %"])
                                  .format({"Precio": "{:.4f}", "Var. %": "{:.2f}%"}),
                        use_container_width=True,
                        height=420,
                    )
                    st.caption(f"Fuente: MAE Boletín {fetch_date} · {len(df_f)} ONs · Segmento 5")

                    # Buscar TPCO (Tecpetrol) si está
                    tpco_row = df_f[df_f["ticker_base"].str.upper().str.contains("TPCO|TPC", na=False)]
                    if not tpco_row.empty:
                        st.success(f"✅ TPCO encontrado en MAE: precio {tpco_row.iloc[0]['precio_norm']:.4f}")

                    # Gráfico: distribución de precios ONs USD
                    df_usd = df_f[df_f["monedaCodigo"] == "D"].copy()
                    if len(df_usd) > 3:
                        fig_hist = go.Figure(go.Histogram(
                            x=df_usd["precio_norm"],
                            nbinsx=20,
                            marker_color="#636EFA",
                            opacity=0.75,
                        ))
                        fig_hist.update_layout(
                            title="Distribución de precios — ONs USD",
                            xaxis_title="Precio (% VN)",
                            yaxis_title="Cantidad de ONs",
                            height=320,
                            template="plotly_white",
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # RF TAB 4 — MI CARTERA RF
    # ══════════════════════════════════════════════════════════════════════════
    with rf_tab4:
        st.markdown("#### Cartera de Renta Fija")
        st.caption("Registrá tu posición en bonos y ONs para tracking de P&L y analytics.")

        # Inicializar cartera RF en session state
        if "cartera_rf" not in st.session_state:
            st.session_state["cartera_rf"] = []

        # ── Agregar posición ──────────────────────────────────────────────────
        with st.expander("➕ Agregar posición", expanded=len(st.session_state["cartera_rf"]) == 0):
            ca1, ca2, ca3, ca4, ca5 = st.columns(5)
            with ca1:
                bond_options = list(BONDS_DB.keys()) + ["OTRO"]
                pos_ticker = st.selectbox("Instrumento", bond_options, key="rf_pos_ticker")
            with ca2:
                pos_vn = st.number_input("VN ($)", min_value=100.0, value=10000.0, step=100.0, key="rf_pos_vn")
            with ca3:
                pos_precio_compra = st.number_input("Precio compra (% VN)", min_value=0.1, value=65.0, step=0.01, key="rf_pos_pc")
            with ca4:
                pos_fecha = st.date_input("Fecha compra", key="rf_pos_fecha")
            with ca5:
                pos_moneda = st.selectbox("Moneda", ["USD", "ARS"], key="rf_pos_moneda")

            if st.button("Agregar posición", key="rf_add_pos"):
                st.session_state["cartera_rf"].append({
                    "ticker": pos_ticker,
                    "vn": pos_vn,
                    "precio_compra": pos_precio_compra,
                    "fecha_compra": str(pos_fecha),
                    "moneda": pos_moneda,
                    "costo_total": pos_vn * pos_precio_compra / 100,
                })
                st.success(f"✅ {pos_ticker} agregado — VN {pos_vn:,.0f} a {pos_precio_compra:.2f}%")
                st.rerun()

        # ── Tabla de posiciones ───────────────────────────────────────────────
        cartera = st.session_state["cartera_rf"]
        if not cartera:
            st.info("Sin posiciones registradas. Usá el panel de arriba para agregar.")
        else:
            rows = []
            for i, pos in enumerate(cartera):
                tk = pos["ticker"]
                b_data = BONDS_DB.get(tk, {})
                costo = pos["costo_total"]

                # Precio actual: intentar desde MAE si disponible
                precio_actual = None
                if has_mae and b_data:
                    fetch_d = (datetime.today() - timedelta(days=1))
                    if fetch_d.weekday() >= 5:
                        fetch_d -= timedelta(days=fetch_d.weekday() - 4)
                    boletin_c = mae_boletin(fetch_d.strftime("%Y-%m-%d"))
                    for seg_id in [MAE_SEG_SOBERANOS_MAE, MAE_SEG_SOBERANOS_PPT, MAE_SEG_ONS]:
                        if seg_id not in boletin_c:
                            continue
                        df_s = boletin_c[seg_id]
                        fila = df_s[
                            (df_s["ticker_base"].str.upper() == tk.upper()) &
                            (df_s["plazo"].astype(str) == "000") &
                            (df_s["precio_cierre"] > 0)
                        ]
                        if not fila.empty:
                            raw = float(fila.iloc[0]["precio_cierre"])
                            if seg_id == MAE_SEG_SOBERANOS_PPT and raw < 5:
                                precio_actual = raw * 100
                            else:
                                precio_actual = raw
                            break

                precio_actual = precio_actual or pos["precio_compra"]
                cc = _cupon_corrido(tk, datetime.today().date()) if tk in BONDS_DB else 0.0
                ps_actual = precio_actual + cc
                valor_actual = pos["vn"] * ps_actual / 100
                pnl_abs = valor_actual - costo
                pnl_pct = pnl_abs / costo * 100 if costo > 0 else 0

                ytm_actual = _ytm(tk, ps_actual, datetime.today().date()) if tk in BONDS_DB else None
                dur = _duration_macaulay(tk, ps_actual, datetime.today().date()) if tk in BONDS_DB else None

                rows.append({
                    "Ticker": tk,
                    "VN": pos["vn"],
                    "P. Compra": pos["precio_compra"],
                    "P. Actual": round(precio_actual, 2),
                    "CC": round(cc, 4),
                    "Costo Total": round(costo, 2),
                    "Valor Actual": round(valor_actual, 2),
                    "P&L ($)": round(pnl_abs, 2),
                    "P&L (%)": round(pnl_pct, 2),
                    "YTM (%)": round(ytm_actual * 100, 2) if ytm_actual else None,
                    "Duration": round(dur, 2) if dur else None,
                    "Moneda": pos["moneda"],
                    "Origen precio": "MAE" if precio_actual != pos["precio_compra"] else "Manual",
                })
                rows[-1]["_idx"] = i

            df_cartera = pd.DataFrame(rows)
            idx_col = "_idx"

            def _color_pnl(val):
                if pd.isna(val): return ""
                return "color: #00cc88" if val > 0 else ("color: #ff4b4b" if val < 0 else "")

            st.dataframe(
                df_cartera.drop(columns=[idx_col]).style
                    .applymap(_color_pnl, subset=["P&L ($)", "P&L (%)"])
                    .format({
                        "VN": "{:,.0f}",
                        "P. Compra": "{:.2f}",
                        "P. Actual": "{:.2f}",
                        "CC": "{:.4f}",
                        "Costo Total": "{:,.2f}",
                        "Valor Actual": "{:,.2f}",
                        "P&L ($)": "{:,.2f}",
                        "P&L (%)": "{:.2f}%",
                    }),
                use_container_width=True,
                height=300,
            )

            # Resumen cartera
            total_costo  = df_cartera["Costo Total"].sum()
            total_valor  = df_cartera["Valor Actual"].sum()
            total_pnl    = total_valor - total_costo
            total_pnl_pct = total_pnl / total_costo * 100 if total_costo > 0 else 0
            dur_pond = (
                (df_cartera["Duration"] * df_cartera["Valor Actual"]).sum() /
                df_cartera["Valor Actual"].sum()
            ) if df_cartera["Duration"].notna().any() else None

            st.divider()
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Costo total",     f"${total_costo:,.2f}")
            r2.metric("Valor actual",    f"${total_valor:,.2f}", f"{total_pnl_pct:+.2f}%")
            r3.metric("P&L total",       f"${total_pnl:+,.2f}")
            r4.metric("Duration pond.",  f"{dur_pond:.2f} años" if dur_pond else "N/D")

            # Eliminar posición
            st.divider()
            st.markdown("**Eliminar posición**")
            del_idx = st.selectbox(
                "Seleccionar posición a eliminar",
                options=list(range(len(cartera))),
                format_func=lambda i: f"{cartera[i]['ticker']} — VN {cartera[i]['vn']:,.0f}",
                key="rf_del_sel",
            )
            if st.button("🗑️ Eliminar posición seleccionada", key="rf_del_btn"):
                st.session_state["cartera_rf"].pop(del_idx)
                st.success("Posición eliminada.")
                st.rerun()
