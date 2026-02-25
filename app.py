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
    # ── Dark theme CSS para sección RF ────────────────────────────────────────
    st.markdown("""
    <style>
    /* Scope dark theme to RF section via class injection */
    .rf-terminal {
        background: #0d1117;
        border-radius: 8px;
        padding: 0;
    }
    /* Override Streamlit dataframe background in RF */
    section[data-testid="stMain"] .stDataFrame {
        background: transparent;
    }
    /* Tab styling — terminal look */
    .stTabs [data-baseweb="tab-list"] {
        background: #0d1117;
        border-bottom: 1px solid #21262d;
        gap: 0;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #8b949e;
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        font-size: 12px;
        letter-spacing: 0.05em;
        border-radius: 0;
        padding: 10px 20px;
        border-bottom: 2px solid transparent;
        text-transform: uppercase;
    }
    .stTabs [aria-selected="true"] {
        background: transparent !important;
        color: #58a6ff !important;
        border-bottom: 2px solid #58a6ff !important;
    }
    .stTabs [data-baseweb="tab-panel"] {
        background: #0d1117;
        padding: 16px 0 0 0;
    }
    /* Metric cards */
    [data-testid="stMetric"] {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 6px;
        padding: 12px 16px;
    }
    [data-testid="stMetricLabel"] {
        color: #8b949e !important;
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    [data-testid="stMetricValue"] {
        color: #e6edf3 !important;
        font-family: 'JetBrains Mono', monospace;
        font-size: 20px !important;
    }
    [data-testid="stMetricDelta"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px !important;
    }
    /* Section headers */
    .rf-section-header {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #58a6ff;
        border-left: 3px solid #58a6ff;
        padding-left: 10px;
        margin: 20px 0 12px 0;
    }
    /* Asset class badge */
    .asset-badge {
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.1em;
        padding: 2px 8px;
        border-radius: 3px;
        text-transform: uppercase;
    }
    .badge-hard-usd { background: #1f3a1f; color: #3fb950; border: 1px solid #3fb950; }
    .badge-cer      { background: #1a2f4a; color: #58a6ff; border: 1px solid #58a6ff; }
    .badge-tasa     { background: #3a2a1a; color: #e3b341; border: 1px solid #e3b341; }
    .badge-dl       { background: #2a1a3a; color: #bc8cff; border: 1px solid #bc8cff; }
    /* Divider */
    .rf-divider {
        border: none;
        border-top: 1px solid #21262d;
        margin: 16px 0;
    }
    /* Number input dark */
    .stNumberInput input {
        background: #161b22 !important;
        color: #e6edf3 !important;
        border-color: #30363d !important;
        font-family: 'JetBrains Mono', monospace;
    }
    /* Selectbox dark */
    .stSelectbox [data-baseweb="select"] {
        background: #161b22 !important;
        border-color: #30363d !important;
    }
    </style>
    """, unsafe_allow_html=True)

    has_mae = bool(_mae_key())

    st.markdown(
        '<div style="display:flex;align-items:center;gap:12px;margin-bottom:4px;">'
        '<span style="font-family:JetBrains Mono,monospace;font-size:22px;'
        'font-weight:700;color:#e6edf3;letter-spacing:-0.02em;">RENTA FIJA</span>'
        '<span style="font-family:JetBrains Mono,monospace;font-size:11px;'
        'color:#58a6ff;letter-spacing:0.1em;text-transform:uppercase;'
        'border:1px solid #58a6ff;padding:2px 8px;border-radius:3px;">ARG</span>'
        '<span style="font-family:JetBrains Mono,monospace;font-size:11px;'
        'color:#8b949e;margin-left:auto;">data912.com</span>'
        '</div>',
        unsafe_allow_html=True
    )

    rf_tab1, rf_tab2, rf_tab3, rf_tab4, rf_tab5, rf_tab6 = st.tabs([
        "MERCADO", "ANÁLISIS", "ONs", "CARTERA", "CURVA & ESCENARIOS", "CALENDARIO"
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # RF TAB 1 — MERCADO HOY
    # ══════════════════════════════════════════════════════════════════════════
    with rf_tab1:
        df_live = d912_live_bonds()

        if df_live.empty:
            st.warning("Sin datos. data912.com puede estar fuera de servicio.")
        else:
            # ── Clasificación extendida ───────────────────────────────────
            def _clase(sym):
                s = str(sym).upper()
                # Strip suffix
                base = s.rstrip("CD")
                # Hard dollar soberanos
                if any(base == t for t in ["GD29","GD30","GD35","GD38","GD41","GD46",
                                            "AL29","AL30","AL35","AL38","AL41","AL46","AE38"]):
                    return "Hard Dollar"
                if s.startswith("TX") or s.startswith("CER") or base in ["TC25","TC26","TZ26","TZ28"]:
                    return "CER"
                if s.startswith("BP") or s.startswith("BPOA") or s.startswith("BPOB") or s.startswith("BPOC"):
                    return "BOPREAL"
                if s.startswith("TV") or s.startswith("TDG") or s.startswith("T2V") or s.startswith("T3V"):
                    return "Dollar-linked"
                if any(s.startswith(p) for p in ["S","BL","BU","BG","PM","T4","T6","TTM","TZX"]):
                    return "Tasa Fija / Lecap"
                if s.startswith("T") and len(s) <= 5:
                    return "Tasa Fija / Lecap"
                return "Otros"

            def _ley(sym):
                s = str(sym).upper().rstrip("CD")
                if any(s == t for t in ["GD29","GD30","GD35","GD38","GD41","GD46"]):
                    return "NY"
                if any(s == t for t in ["AL29","AL30","AL35","AL38","AL41","AL46","AE38"]):
                    return "AR"
                return ""

            df_live["clase"] = df_live["symbol"].apply(_clase)
            df_live["ley"]   = df_live["symbol"].apply(_ley)

            # ── Métricas header ───────────────────────────────────────────
            activos = df_live[df_live["v"] > 0]
            hd = df_live[df_live["clase"] == "Hard Dollar"]
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Instrumentos", len(df_live))
            m2.metric("Con operaciones", len(activos))
            m3.metric("Hard Dollar", len(hd))
            m4.metric("CER / Tasa", len(df_live[df_live["clase"].isin(["CER","Tasa Fija / Lecap"])]))
            m5.metric("Última act.", datetime.now().strftime("%H:%M:%S"))

            st.markdown('<hr class="rf-divider">', unsafe_allow_html=True)

            # ── Sección HARD DOLLAR — tabla con D/C separados ─────────────
            st.markdown('<div class="rf-section-header">Hard Dollar — Soberanos USD</div>', unsafe_allow_html=True)

            SOBERANOS_BASE = ["GD29","GD30","GD35","GD38","GD41","GD46",
                               "AL29","AL30","AL35","AL38","AL41","AL46","AE38"]

            def _get_price(df, ticker, suffix=""):
                sym = (ticker + suffix).upper()
                row = df[df["symbol"].str.upper() == sym]
                if row.empty:
                    return None, None, None
                r = row.iloc[0]
                return (float(r.get("c",0) or 0),
                        float(r.get("pct_change",0) or 0),
                        float(r.get("v",0) or 0))

            hoy_t1 = datetime.today().date()
            hard_rows = []
            for tk in SOBERANOS_BASE:
                p_ars, var_ars, _ = _get_price(df_live, tk, "")
                p_d,   var_d,   _ = _get_price(df_live, tk, "D")  # cable
                p_c,   var_c,   _ = _get_price(df_live, tk, "C")  # MEP/CCL
                vol, _, _         = _get_price(df_live, tk, "")

                # Calcular TIR y Duration si tenemos precio D
                tir, dur, par = None, None, None
                if p_d and p_d > 0 and tk in BONDS_DB:
                    pl_d = p_d if p_d > 5 else p_d * 100
                    cc   = _cupon_corrido(tk, hoy_t1)
                    ps_d = pl_d + cc
                    tir  = _ytm(tk, ps_d, hoy_t1)
                    dur  = _duration_macaulay(tk, ps_d, hoy_t1)
                    par  = _paridad(tk, ps_d, hoy_t1)

                ley = "NY" if tk.startswith("G") or tk == "AE38" else "AR"
                hard_rows.append({
                    "Ticker": tk,
                    "Ley": ley,
                    "ARS": f"{p_ars:,.0f}" if p_ars else "—",
                    "Cable (D)": round(p_d, 2) if p_d else None,
                    "Var D %": round(var_d, 2) if var_d is not None else None,
                    "MEP/CCL (C)": round(p_c, 2) if p_c else None,
                    "Var C %": round(var_c, 2) if var_c is not None else None,
                    "TIR": round(tir * 100, 3) if tir else None,
                    "Duration": round(dur, 2) if dur else None,
                    "Paridad": round(par, 4) if par else None,
                })

            df_hard = pd.DataFrame(hard_rows)

            def _color_var_hard(val):
                if val is None or (isinstance(val, float) and pd.isna(val)):
                    return "color: #8b949e"
                try:
                    v = float(val)
                    if v > 0:   return "color: #3fb950; font-family: JetBrains Mono, monospace"
                    if v < 0:   return "color: #f85149; font-family: JetBrains Mono, monospace"
                    return "color: #8b949e; font-family: JetBrains Mono, monospace"
                except:
                    return "font-family: JetBrains Mono, monospace"

            def _mono(val):
                return "font-family: JetBrains Mono, monospace; font-size: 13px"

            styled = (df_hard.style
                .applymap(_color_var_hard, subset=["Var D %", "Var C %"])
                .applymap(_mono, subset=["Cable (D)", "MEP/CCL (C)", "TIR", "Duration", "Paridad"])
                .format({
                    "Cable (D)":   lambda x: f"{x:.2f}" if x else "—",
                    "Var D %":     lambda x: f"{x:+.2f}%" if x is not None and not pd.isna(x) else "—",
                    "MEP/CCL (C)": lambda x: f"{x:.2f}" if x else "—",
                    "Var C %":     lambda x: f"{x:+.2f}%" if x is not None and not pd.isna(x) else "—",
                    "TIR":         lambda x: f"{x:.3f}%" if x else "—",
                    "Duration":    lambda x: f"{x:.2f}" if x else "—",
                    "Paridad":     lambda x: f"{x:.4f}" if x else "—",
                }, na_rep="—")
            )
            st.dataframe(styled, use_container_width=True, hide_index=True, height=530)
            st.caption("D = Cable USD · C = MEP/CCL · TIR calculada sobre precio cable · Fuente: data912.com")

            st.markdown('<hr class="rf-divider">', unsafe_allow_html=True)

            # ── Otras clases (colapsables) ────────────────────────────────
            otras_clases = [c for c in ["CER","Tasa Fija / Lecap","Dollar-linked","BOPREAL","Otros"]
                             if c in df_live["clase"].values]

            for clase in otras_clases:
                df_clase = df_live[(df_live["clase"] == clase) & (df_live["v"] > 0)].copy()
                if df_clase.empty:
                    continue
                badge_map = {
                    "CER":             ("badge-cer",  "CER"),
                    "Tasa Fija / Lecap":("badge-tasa","TASA FIJA"),
                    "Dollar-linked":   ("badge-dl",   "DOLLAR-LINKED"),
                    "BOPREAL":         ("badge-hard-usd","BOPREAL"),
                    "Otros":           ("badge-tasa", "OTROS"),
                }
                badge_cls, badge_txt = badge_map.get(clase, ("badge-tasa", clase.upper()))
                with st.expander(f"**{clase}** — {len(df_clase)} instrumentos con operaciones", expanded=False):
                    cols_show = ["symbol","c","pct_change","v","q_op"]
                    cols_show = [c for c in cols_show if c in df_clase.columns]
                    df_oc = df_clase[cols_show].rename(columns={
                        "symbol":"Ticker","c":"Último","pct_change":"Var %",
                        "v":"Volumen","q_op":"Operaciones"
                    }).sort_values("Var %", ascending=False)
                    def _cv(val):
                        try:
                            return "color: #3fb950" if float(val) > 0 else ("color: #f85149" if float(val) < 0 else "")
                        except: return ""
                    st.dataframe(
                        df_oc.style.applymap(_cv, subset=["Var %"])
                             .format({"Último":"{:.4f}","Var %":"{:+.2f}%",
                                      "Volumen":"{:,.0f}","Operaciones":"{:.0f}"}, na_rep="—"),
                        use_container_width=True, hide_index=True, height=250,
                    )

            st.markdown('<hr class="rf-divider">', unsafe_allow_html=True)

            # ── Serie histórica ───────────────────────────────────────────
            st.markdown('<div class="rf-section-header">Serie histórica</div>', unsafe_allow_html=True)
            col_sym, col_per = st.columns([2,1])
            with col_sym:
                all_syms = sorted(df_live["symbol"].tolist())
                default_sym = "GD30D" if "GD30D" in all_syms else (all_syms[0] if all_syms else None)
                sym_sel = st.selectbox("Ticker", all_syms,
                                        index=all_syms.index(default_sym) if default_sym in all_syms else 0,
                                        key="rf_sym_hist")
            with col_per:
                periodo = st.select_slider("Período", ["3M","6M","1Y","2Y","Todo"], value="1Y", key="rf_hist_period")

            if sym_sel:
                df_hist = d912_historical(sym_sel)
                if not df_hist.empty:
                    dias_map = {"3M":90,"6M":180,"1Y":252,"2Y":504,"Todo":99999}
                    df_p = df_hist.tail(dias_map[periodo])
                    fig_h = go.Figure()
                    fig_h.add_trace(go.Scatter(
                        x=df_p["date"], y=df_p["c"], mode="lines", name="Cierre",
                        line=dict(color="#58a6ff", width=1.8),
                        fill="tozeroy", fillcolor="rgba(88,166,255,0.06)",
                    ))
                    fig_h.update_layout(
                        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                        font=dict(color="#8b949e", family="JetBrains Mono, monospace", size=11),
                        title=dict(text=f"{sym_sel}", font=dict(color="#e6edf3", size=14)),
                        xaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
                        yaxis=dict(gridcolor="#21262d", linecolor="#30363d", title="Precio"),
                        height=320, hovermode="x unified",
                        margin=dict(l=10, r=10, t=40, b=10),
                    )
                    st.plotly_chart(fig_h, use_container_width=True)
                    st.caption(f"{len(df_p)} ruedas · {df_p['date'].iloc[0].date()} → {df_p['date'].iloc[-1].date()}")

    # ══════════════════════════════════════════════════════════════════════════
    # RF TAB 2 — ANÁLISIS DE BONOS
    # ══════════════════════════════════════════════════════════════════════════
    with rf_tab2:
        st.markdown('<div class="rf-section-header">Análisis cuantitativo — Soberanos USD</div>', unsafe_allow_html=True)

        soberanos = {k: v for k, v in BONDS_DB.items() if v["tipo"] == "Soberano"}
        bond_sel = st.selectbox(
            "Seleccionar bono",
            list(soberanos.keys()),
            format_func=lambda k: f"{k} — {BONDS_DB[k]['nombre']}",
            key="rf_bond_sel",
        )

        b   = BONDS_DB[bond_sel]
        hoy = datetime.today().date()

        # Precargar precio D
        precio_mercado = None
        df_live_tab2 = d912_live_bonds()
        if not df_live_tab2.empty:
            for sym in [bond_sel + "D", bond_sel]:
                row = df_live_tab2[df_live_tab2["symbol"].str.upper() == sym.upper()]
                if not row.empty:
                    raw = float(row.iloc[0].get("c", 0) or 0)
                    if raw <= 0: continue
                    if raw < 5: raw *= 100
                    if raw <= 200:
                        precio_mercado = raw
                        break

        c_inp, c_metr = st.columns([1, 3])
        with c_inp:
            precio_input = st.number_input(
                "Precio limpio (% VN)",
                min_value=0.1, max_value=200.0,
                value=float(round(min(max(precio_mercado, 0.1), 199.0), 2)) if precio_mercado else 63.0,
                step=0.01, format="%.2f",
                key="rf_precio_input",
            )
            if precio_mercado:
                st.caption(f"📡 data912 cable: {precio_mercado:.2f}")

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

        with c_metr:
            r1c1, r1c2, r1c3, r1c4 = st.columns(4)
            r1c1.metric("Precio limpio",  f"{precio_input:.4f}")
            r1c2.metric("Cupón corrido",  f"{cc:.5f}")
            r1c3.metric("Precio sucio",   f"{ps:.4f}")
            r1c4.metric("VN residual",    f"{vn_res:.2f}%")
            r2c1, r2c2, r2c3, r2c4 = st.columns(4)
            r2c1.metric("Valor técnico",  f"{vt:.5f}")
            r2c2.metric("Paridad",        f"{paridad:.6f}" if paridad else "N/D")
            r2c3.metric("TIR (YTM)",      f"{ytm*100:.4f}%" if ytm is not None else "N/D")
            r2c4.metric("TNA semestral",  f"{tna*100:.4f}%" if tna is not None else "N/D")
            r3c1, r3c2, r3c3, r3c4 = st.columns(4)
            r3c1.metric("Duration Mac.",  f"{dur_mac:.6f}" if dur_mac else "N/D")
            r3c2.metric("Duration Mod.",  f"{dur_mod:.6f}" if dur_mod else "N/D")
            r3c3.metric("Convexidad",     f"{convex:.6f}" if convex else "N/D")
            r3c4.metric("Current yield",  f"{cy*100:.4f}%" if cy else "N/D")

        st.markdown('<hr class="rf-divider">', unsafe_allow_html=True)
        st.markdown('<div class="rf-section-header">Flujos de fondos</div>', unsafe_allow_html=True)

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
                "✓": "✓" if cd <= hoy else "",
            })
            vn_iter -= am
        df_cf = pd.DataFrame(rows)

        def _grey_paid(row):
            if row.get("✓") == "✓":
                return ["color: #4a5568"] * len(row)
            return ["color: #e6edf3; font-family: JetBrains Mono, monospace"] * len(row)

        st.dataframe(df_cf.style.apply(_grey_paid, axis=1),
                     use_container_width=True, height=280)
        st.caption(f"Σ VP futuros = {sum(r['VP'] for r in rows if isinstance(r['VP'], float)):.4f}")

        st.markdown('<hr class="rf-divider">', unsafe_allow_html=True)
        st.markdown('<div class="rf-section-header">Sensibilidad precio / TIR</div>', unsafe_allow_html=True)

        ytm_range = np.linspace(0.02, 0.25, 100)
        precios_c = [_precio_sucio_from_ytm(bond_sel, y, hoy) for y in ytm_range]
        fig_sens = go.Figure()
        fig_sens.add_trace(go.Scatter(
            x=ytm_range*100, y=precios_c, mode="lines",
            line=dict(color="#58a6ff", width=2), name="Precio sucio",
        ))
        if ytm is not None:
            fig_sens.add_vline(x=ytm*100, line_dash="dash", line_color="#f85149",
                annotation_text=f"TIR: {ytm*100:.2f}%",
                annotation_font=dict(color="#f85149", family="JetBrains Mono"),
                annotation_position="top right")
            fig_sens.add_hline(y=ps, line_dash="dot", line_color="#e3b341",
                annotation_text=f"PS: {ps:.2f}",
                annotation_font=dict(color="#e3b341", family="JetBrains Mono"),
                annotation_position="right")
        fig_sens.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            font=dict(color="#8b949e", family="JetBrains Mono, monospace", size=11),
            xaxis=dict(title="TIR (%)", gridcolor="#21262d", linecolor="#30363d"),
            yaxis=dict(title="Precio sucio (% VN)", gridcolor="#21262d", linecolor="#30363d"),
            height=340, margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig_sens, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # RF TAB 3 — ONs CORPORATIVAS
    # ══════════════════════════════════════════════════════════════════════════
    with rf_tab3:
        st.markdown('<div class="rf-section-header">Obligaciones Negociables Corporativas — MAE Segmento 5</div>', unsafe_allow_html=True)

        if not has_mae:
            st.info("Ingresá la MAE API Key en Streamlit Secrets para acceder a datos de ONs.")
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
                st.warning(f"Sin datos de ONs para {fetch_date}.")
                fecha_manual = st.date_input("Fecha alternativa", key="rf_fecha_on")
                if st.button("Cargar", key="rf_load_on"):
                    boletin_data = mae_boletin(str(fecha_manual))

            if MAE_SEG_ONS in boletin_data:
                df_ons = boletin_data[MAE_SEG_ONS].copy()
                df_ons_ci = df_ons[
                    (df_ons["plazo"].astype(str) == "000") &
                    (df_ons["precio_cierre"] > 0)
                ].copy()

                if df_ons_ci.empty:
                    st.info("Sin ONs con operaciones en CI.")
                else:
                    cols_disp = ["ticker_base","precio_cierre","precio_promedio","nominal_operado","cantidad_operaciones"]
                    cols_disp = [c for c in cols_disp if c in df_ons_ci.columns]
                    df_ons_disp = df_ons_ci[cols_disp].rename(columns={
                        "ticker_base":"Ticker","precio_cierre":"Precio cierre",
                        "precio_promedio":"P. promedio","nominal_operado":"Vol. nominal",
                        "cantidad_operaciones":"Operaciones",
                    }).sort_values("Vol. nominal", ascending=False)
                    st.dataframe(df_ons_disp, use_container_width=True, hide_index=True, height=450)
                    st.caption(f"Boletín MAE · {fetch_date} · {len(df_ons_ci)} ONs con operaciones en CI")

    # ══════════════════════════════════════════════════════════════════════════
    # RF TAB 4 — CARTERA
    # ══════════════════════════════════════════════════════════════════════════
    with rf_tab4:
        st.markdown('<div class="rf-section-header">Mi Cartera RF</div>', unsafe_allow_html=True)

        CARTERA_RF_KEY = "cartera_rf_v2"
        if CARTERA_RF_KEY not in st.session_state:
            st.session_state[CARTERA_RF_KEY] = []

        with st.expander("➕ Agregar posición", expanded=False):
            ca1, ca2, ca3, ca4 = st.columns(4)
            with ca1:
                bond_options = list(BONDS_DB.keys()) + ["OTRO"]
                tk_new = st.selectbox("Ticker", bond_options, key="rf_tk_new")
            with ca2:
                vn_new = st.number_input("VN nominal", min_value=1, value=1000, step=100, key="rf_vn_new")
            with ca3:
                pc_new = st.number_input("Precio compra (% VN)", min_value=0.01, max_value=300.0,
                                          value=63.0, step=0.01, format="%.2f", key="rf_pc_new")
            with ca4:
                fecha_compra = st.date_input("Fecha compra", value=date.today(), key="rf_fc_new")

            if st.button("Agregar", key="rf_add_pos"):
                st.session_state[CARTERA_RF_KEY].append({
                    "ticker": tk_new, "vn": vn_new,
                    "precio_compra": pc_new, "fecha_compra": str(fecha_compra),
                })
                st.rerun()

        posiciones = st.session_state[CARTERA_RF_KEY]
        if not posiciones:
            st.info("Sin posiciones. Usá el panel de arriba para agregar.")
        else:
            df_live_c = d912_live_bonds()
            cartera_rows = []
            for i, pos in enumerate(posiciones):
                tk = pos["ticker"]
                vn = pos["vn"]
                pc = pos["precio_compra"]

                p_actual = None
                if not df_live_c.empty:
                    for sym in [tk+"D", tk]:
                        row_c = df_live_c[df_live_c["symbol"].str.upper() == sym.upper()]
                        if not row_c.empty:
                            raw = float(row_c.iloc[0].get("c",0) or 0)
                            if raw <= 0: continue
                            if raw < 5: raw *= 100
                            if raw <= 200:
                                p_actual = raw
                                break

                cc_val = _cupon_corrido(tk, datetime.today().date()) if tk in BONDS_DB else 0.0
                ps_actual = (p_actual + cc_val) if p_actual else None
                ytm_actual = _ytm(tk, ps_actual, datetime.today().date()) if (ps_actual and tk in BONDS_DB) else None
                dur = _duration_macaulay(tk, ps_actual, datetime.today().date()) if (ps_actual and tk in BONDS_DB) else None

                costo = vn * pc / 100
                val_actual = vn * p_actual / 100 if p_actual else None
                pnl = (val_actual - costo) if val_actual else None
                pnl_pct = (pnl / costo * 100) if (pnl and costo) else None

                idx_col = f"_idx_{i}"
                cartera_rows.append({
                    idx_col: i,
                    "Ticker": tk, "VN": vn,
                    "P. Compra": pc,
                    "P. Actual": round(p_actual,2) if p_actual else None,
                    "CC": round(cc_val,4),
                    "Costo Total": round(costo,2),
                    "Valor Actual": round(val_actual,2) if val_actual else None,
                    "P&L ($)": round(pnl,2) if pnl else None,
                    "P&L (%)": round(pnl_pct,2) if pnl_pct else None,
                    "TIR": f"{ytm_actual*100:.3f}%" if ytm_actual else "—",
                    "Duration": f"{dur:.2f}" if dur else "—",
                })

            if cartera_rows:
                idx_col = [c for c in cartera_rows[0] if c.startswith("_idx_")][0]
                df_cartera = pd.DataFrame(cartera_rows)

                def _color_pnl(val):
                    if pd.isna(val): return ""
                    return "color: #3fb950" if val > 0 else ("color: #f85149" if val < 0 else "")

                st.dataframe(
                    df_cartera.drop(columns=[idx_col]).style
                        .applymap(_color_pnl, subset=["P&L ($)", "P&L (%)"])
                        .format({
                            "VN": "{:,.0f}", "P. Compra": "{:.2f}", "P. Actual": "{:.2f}",
                            "CC": "{:.4f}", "Costo Total": "{:,.2f}",
                            "Valor Actual": "{:,.2f}", "P&L ($)": "{:,.2f}", "P&L (%)": "{:.2f}%",
                        }, na_rep="—"),
                    use_container_width=True, height=300,
                )

                to_del = st.number_input("N° fila a eliminar (0-based)", min_value=0,
                                          max_value=len(posiciones)-1, value=0, step=1, key="rf_del_idx")
                if st.button("🗑️ Eliminar posición", key="rf_del_btn"):
                    st.session_state[CARTERA_RF_KEY].pop(int(to_del))
                    st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # RF TAB 5 — CURVA & ESCENARIOS
    # ══════════════════════════════════════════════════════════════════════════
    with rf_tab5:
        st.markdown('<div class="rf-section-header">Curva de TIRs & Escenarios de shock</div>', unsafe_allow_html=True)

        hoy_t5 = datetime.today().date()
        df_live_t5 = d912_live_bonds()
        soberanos_t5 = {k: v for k, v in BONDS_DB.items() if v["tipo"] == "Soberano"}

        precios_actuales = {}
        if not df_live_t5.empty:
            for tk in soberanos_t5:
                for sym in [tk + "D", tk]:
                    row = df_live_t5[df_live_t5["symbol"].str.upper() == sym.upper()]
                    if not row.empty:
                        raw = float(row.iloc[0].get("c", 0) or 0)
                        if raw <= 0: continue
                        if raw < 5: raw *= 100
                        if raw <= 200:
                            precios_actuales[tk] = raw
                            break

        metricas = {}
        for tk, bv in soberanos_t5.items():
            pl = precios_actuales.get(tk)
            if not pl: continue
            cc  = _cupon_corrido(tk, hoy_t5)
            ps  = pl + cc
            ytm = _ytm(tk, ps, hoy_t5)
            dm  = _duration_macaulay(tk, ps, hoy_t5)
            dmod = (dm / (1 + ytm)) if (dm and ytm) else None
            conv = _convexity(tk, ps, hoy_t5)
            par  = _paridad(tk, ps, hoy_t5)
            if ytm and dm:
                metricas[tk] = {
                    "ley": bv["ley"], "vto": bv["vencimiento"],
                    "precio_limpio": pl, "precio_sucio": ps,
                    "ytm": ytm, "duration": dm, "dur_mod": dmod,
                    "convexity": conv, "paridad": par,
                }

        ct5a, ct5b = st.tabs(["📈 Curva & Spreads", "⚡ Escenarios de shock"])

        with ct5a:
            if len(metricas) < 2:
                st.info("Sin suficientes precios desde data912.")
            else:
                eje_y = st.radio("Eje Y", ["TIR (%)", "Paridad"], horizontal=True, key="t5_eje")
                df_m = pd.DataFrame([
                    {"Ticker": tk, "Ley": v["ley"], "Duration": v["duration"],
                     "TIR (%)": round(v["ytm"]*100, 3),
                     "Paridad": round(v["paridad"], 4) if v["paridad"] else None,
                     "Precio": round(v["precio_limpio"], 2),
                     "Vcto": str(v["vto"])}
                    for tk, v in metricas.items()
                ]).dropna(subset=["Duration"])

                y_col = "TIR (%)" if eje_y == "TIR (%)" else "Paridad"
                gd_df = df_m[df_m["Ley"] == "NY"].sort_values("Duration")
                al_df = df_m[df_m["Ley"] == "AR"].sort_values("Duration")

                fig_c = go.Figure()
                for df_sub, color, name in [(gd_df, "#58a6ff", "Ley NY (GD)"),
                                             (al_df, "#f85149", "Ley AR (AL)")]:
                    if df_sub.empty: continue
                    fig_c.add_trace(go.Scatter(
                        x=df_sub["Duration"], y=df_sub[y_col],
                        mode="markers+lines+text", name=name,
                        marker=dict(color=color, size=11),
                        line=dict(width=1.5, color=color),
                        text=df_sub["Ticker"], textposition="top center",
                        textfont=dict(family="JetBrains Mono", size=10, color=color),
                        hovertemplate=f"<b>%{{text}}</b><br>{y_col}: %{{y:.3f}}<br>Duration: %{{x:.2f}}y<extra></extra>",
                    ))
                fig_c.update_layout(
                    paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                    font=dict(color="#8b949e", family="JetBrains Mono, monospace", size=11),
                    xaxis=dict(title="Duration Macaulay (años)", gridcolor="#21262d", linecolor="#30363d"),
                    yaxis=dict(title=y_col, gridcolor="#21262d", linecolor="#30363d"),
                    legend=dict(bgcolor="#161b22", bordercolor="#30363d"),
                    height=400, margin=dict(l=10, r=10, t=20, b=10),
                )
                st.plotly_chart(fig_c, use_container_width=True)

                # Tabla spreads
                st.markdown('<div class="rf-section-header">Spreads GD − AL</div>', unsafe_allow_html=True)
                spread_rows = []
                for yr in ["29","30","35","38","41","46"]:
                    gd_tk, al_tk = f"GD{yr}", f"AL{yr}"
                    if gd_tk in metricas and al_tk in metricas:
                        sp_bps = (metricas[gd_tk]["ytm"] - metricas[al_tk]["ytm"]) * 10000
                        spread_rows.append({
                            "Vcto": f"20{yr}",
                            "TIR GD": f"{metricas[gd_tk]['ytm']*100:.3f}%",
                            "TIR AL": f"{metricas[al_tk]['ytm']*100:.3f}%",
                            "Spread bps": round(sp_bps, 1),
                            "Par. GD": f"{metricas[gd_tk]['paridad']:.4f}",
                            "Par. AL": f"{metricas[al_tk]['paridad']:.4f}",
                            "Dif. Par.": round((metricas[gd_tk]["paridad"] or 0)-(metricas[al_tk]["paridad"] or 0), 4),
                        })
                if spread_rows:
                    df_sp = pd.DataFrame(spread_rows)
                    def _cs(val):
                        try:
                            v = float(val)
                            return "color: #3fb950" if v > 0 else "color: #f85149"
                        except: return ""
                    st.dataframe(
                        df_sp.style.applymap(_cs, subset=["Spread bps","Dif. Par."]),
                        use_container_width=True, hide_index=True,
                    )
                    st.caption("Spread > 0 → GD rinde más → long GD / short AL")

                st.markdown('<hr class="rf-divider">', unsafe_allow_html=True)
                st.markdown('<div class="rf-section-header">Spread histórico de precios</div>', unsafe_allow_html=True)

                col_par, col_per = st.columns([1,2])
                with col_par:
                    par_sel = st.selectbox("Par", [f"GD{yr}D / AL{yr}D" for yr in ["29","30","35","38","41","46"]], index=1, key="t5_par_hist")
                with col_per:
                    periodo_hist = st.select_slider("Período", ["3M","6M","1Y","2Y","Todo"], value="1Y", key="t5_periodo_hist")

                yr_sel = par_sel[2:4]
                gd_sym, al_sym = f"GD{yr_sel}D", f"AL{yr_sel}D"
                dias_map2 = {"3M":90,"6M":180,"1Y":252,"2Y":504,"Todo":9999}
                n_dias = dias_map2[periodo_hist]

                df_gd_h = d912_historical(gd_sym).tail(n_dias)
                df_al_h = d912_historical(al_sym).tail(n_dias)

                if not df_gd_h.empty and not df_al_h.empty:
                    df_merge = pd.merge(
                        df_gd_h[["date","c"]].rename(columns={"c":"gd"}),
                        df_al_h[["date","c"]].rename(columns={"c":"al"}),
                        on="date", how="inner"
                    )
                    df_merge["spread"] = df_merge["gd"] - df_merge["al"]
                    df_merge["ratio"]  = df_merge["gd"] / df_merge["al"]

                    spread_mean = df_merge["spread"].mean()
                    spread_std  = df_merge["spread"].std()
                    spread_now  = df_merge["spread"].iloc[-1]
                    zscore_now  = (spread_now - spread_mean) / spread_std if spread_std > 0 else 0

                    ms1, ms2, ms3, ms4 = st.columns(4)
                    ms1.metric("Spread actual", f"{spread_now:.3f}")
                    ms2.metric("Media", f"{spread_mean:.3f}")
                    ms3.metric("1σ", f"{spread_std:.3f}")
                    z_label = "↑ caro" if zscore_now > 1 else ("↓ barato" if zscore_now < -1 else "en rango")
                    ms4.metric("Z-score", f"{zscore_now:.2f}", delta=z_label)

                    fig_sp = go.Figure()
                    fig_sp.add_trace(go.Scatter(
                        x=df_merge["date"], y=df_merge["spread"],
                        mode="lines", name="Spread",
                        line=dict(color="#58a6ff", width=1.5),
                    ))
                    fig_sp.add_hline(y=spread_mean, line_dash="dash", line_color="#8b949e",
                                      annotation_text="μ", annotation_font=dict(color="#8b949e"))
                    fig_sp.add_hrect(y0=spread_mean-spread_std, y1=spread_mean+spread_std,
                                      fillcolor="rgba(88,166,255,0.06)", line_width=0)
                    fig_sp.add_hrect(y0=spread_mean-2*spread_std, y1=spread_mean+2*spread_std,
                                      fillcolor="rgba(88,166,255,0.03)", line_width=0)
                    fig_sp.add_scatter(
                        x=[df_merge["date"].iloc[-1]], y=[spread_now],
                        mode="markers", name="Hoy",
                        marker=dict(color="#f85149", size=9, symbol="circle"),
                    )
                    fig_sp.update_layout(
                        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                        font=dict(color="#8b949e", family="JetBrains Mono, monospace", size=11),
                        xaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
                        yaxis=dict(title="Spread precio (GD−AL)", gridcolor="#21262d", linecolor="#30363d"),
                        height=320, hovermode="x unified", margin=dict(l=10, r=10, t=20, b=10),
                        legend=dict(bgcolor="#161b22", bordercolor="#30363d"),
                    )
                    st.plotly_chart(fig_sp, use_container_width=True)
                    st.caption(f"Z-score: {zscore_now:.2f} | Ratio GD/AL actual: {df_merge['ratio'].iloc[-1]:.4f} vs μ {df_merge['ratio'].mean():.4f}")
                else:
                    st.warning(f"Sin histórico para {gd_sym} o {al_sym}")

        with ct5b:
            st.markdown('<div class="rf-section-header">Impacto en precio ante cambio en TIR</div>', unsafe_allow_html=True)
            st.caption("ΔP ≈ −DurMod × ΔY + ½ × Convexidad × ΔY²")

            if not metricas:
                st.info("Sin precios actuales de data912.")
            else:
                c_shock1, c_shock2 = st.columns(2)
                with c_shock1:
                    shocks_input = st.text_input("Shocks en bps (separados por coma)",
                                                  value="-200,-100,-50,+50,+100,+200", key="t5_shocks")
                with c_shock2:
                    bonos_shock = st.multiselect("Bonos", list(metricas.keys()),
                                                  default=["GD30","AL30","GD35","AL35","GD38","GD41"],
                                                  key="t5_bonos_shock")
                try:
                    shocks_bps = [int(s.strip().replace("+","")) for s in shocks_input.split(",")]
                except:
                    shocks_bps = [-200,-100,-50,50,100,200]

                if bonos_shock:
                    shock_rows = []
                    for tk in bonos_shock:
                        if tk not in metricas: continue
                        m = metricas[tk]
                        row = {"Ticker": tk,
                               "TIR": f"{m['ytm']*100:.3f}%",
                               "Precio": f"{m['precio_limpio']:.2f}",
                               "DurMod": f"{m['dur_mod']:.3f}" if m['dur_mod'] else "N/D"}
                        for bps in shocks_bps:
                            dy = bps/10000
                            dm = m["dur_mod"] or 0
                            cv = m["convexity"] or 0
                            dp = (-dm*dy + 0.5*cv*dy**2)*100
                            new_p = m["precio_limpio"]*(1+dp/100)
                            row[f"{bps:+d}bps"] = f"{dp:+.2f}%\n({new_p:.2f})"
                        shock_rows.append(row)

                    df_shock = pd.DataFrame(shock_rows)
                    st.dataframe(df_shock, use_container_width=True, hide_index=True)

                    st.markdown('<hr class="rf-divider">', unsafe_allow_html=True)
                    shock_sel = st.select_slider("Shock para visualizar", options=shocks_bps,
                                                  value=shocks_bps[len(shocks_bps)//2],
                                                  key="t5_shock_vis", format_func=lambda x: f"{x:+d} bps")
                    dy_sel = shock_sel/10000
                    tks_bar = [tk for tk in bonos_shock if tk in metricas]
                    dp_vals = [(-metricas[tk]["dur_mod"]*dy_sel + 0.5*(metricas[tk]["convexity"] or 0)*dy_sel**2)*100
                                for tk in tks_bar]

                    fig_bar = go.Figure(go.Bar(
                        x=tks_bar, y=dp_vals,
                        marker_color=["#3fb950" if v >= 0 else "#f85149" for v in dp_vals],
                        text=[f"{v:+.2f}%" for v in dp_vals], textposition="outside",
                        textfont=dict(family="JetBrains Mono", size=11),
                    ))
                    fig_bar.update_layout(
                        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                        font=dict(color="#8b949e", family="JetBrains Mono, monospace", size=11),
                        title=dict(text=f"ΔP ante shock {shock_sel:+d} bps",
                                   font=dict(color="#e6edf3", family="JetBrains Mono", size=13)),
                        xaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
                        yaxis=dict(title="ΔP (%)", gridcolor="#21262d", linecolor="#30363d"),
                        height=360, showlegend=False, margin=dict(l=10, r=10, t=40, b=10),
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

                    st.markdown('<hr class="rf-divider">', unsafe_allow_html=True)
                    st.markdown('<div class="rf-section-header">Perfil continuo ±400 bps</div>', unsafe_allow_html=True)
                    bono_perfil = st.selectbox("Bono", bonos_shock, key="t5_bono_perfil")
                    if bono_perfil in metricas:
                        m = metricas[bono_perfil]
                        dm = m["dur_mod"] or 0
                        cv = m["convexity"] or 0
                        dy_r = np.linspace(-0.04, 0.04, 200)
                        dp_r = (-dm*dy_r + 0.5*cv*dy_r**2)*100
                        new_p_r = m["precio_limpio"]*(1+dp_r/100)

                        fig_perf = go.Figure()
                        fig_perf.add_trace(go.Scatter(
                            x=dy_r*10000, y=new_p_r, mode="lines",
                            line=dict(color="#58a6ff", width=2), name="Precio estimado",
                            fill="tozeroy", fillcolor="rgba(88,166,255,0.05)",
                        ))
                        fig_perf.add_vline(x=0, line_dash="dash", line_color="#30363d")
                        fig_perf.add_hline(y=m["precio_limpio"], line_dash="dot", line_color="#e3b341",
                                            annotation_text=f"{m['precio_limpio']:.2f}",
                                            annotation_font=dict(color="#e3b341", family="JetBrains Mono"))
                        fig_perf.update_layout(
                            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                            font=dict(color="#8b949e", family="JetBrains Mono, monospace", size=11),
                            xaxis=dict(title="Shock (bps)", gridcolor="#21262d", linecolor="#30363d"),
                            yaxis=dict(title="Precio estimado", gridcolor="#21262d", linecolor="#30363d"),
                            height=340, margin=dict(l=10, r=10, t=20, b=10),
                        )
                        st.plotly_chart(fig_perf, use_container_width=True)
                        st.caption(f"DurMod: {dm:.4f} · Convex: {cv:.4f} · TIR: {m['ytm']*100:.3f}%")

    # ══════════════════════════════════════════════════════════════════════════
    # RF TAB 6 — CALENDARIO DE PAGOS
    # ══════════════════════════════════════════════════════════════════════════
    with rf_tab6:
        st.markdown('<div class="rf-section-header">Próximos pagos — Soberanos USD (canje 2020)</div>', unsafe_allow_html=True)

        hoy_cal = datetime.today().date()

        cal_rows = []
        for tk, bv in BONDS_DB.items():
            if bv["tipo"] != "Soberano":
                continue
            for cd, cup, am in bv["cash_flows"]:
                if cd < hoy_cal:
                    continue
                dias = (cd - hoy_cal).days
                tipo = ("C+A" if am > 0 and cup > 0 else
                        "A"   if am > 0 else "C")
                cal_rows.append({
                    "Fecha":    cd,
                    "Días":     dias,
                    "Ticker":   tk,
                    "Ley":      bv["ley"],
                    "Tipo":     tipo,
                    "Cupón (% VN)":  round(cup, 5),
                    "Amort (% VN)":  round(am, 4) if am > 0 else "",
                    "Total (% VN)":  round(cup + am, 5),
                })

        if cal_rows:
            df_cal = pd.DataFrame(cal_rows).sort_values(["Fecha","Ticker"])

            # Filtros
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                ventana = st.select_slider("Ventana", ["30d","90d","180d","1Y","2Y","Todo"],
                                            value="1Y", key="cal_ventana")
            with fc2:
                tickers_cal = st.multiselect("Bonos", sorted(df_cal["Ticker"].unique()),
                                              default=[], key="cal_tickers",
                                              placeholder="Todos")
            with fc3:
                tipo_cal = st.multiselect("Tipo pago", ["C","A","C+A"],
                                           default=[], key="cal_tipo",
                                           placeholder="Todos")

            dias_ventana = {"30d":30,"90d":90,"180d":180,"1Y":365,"2Y":730,"Todo":99999}
            max_dias = dias_ventana[ventana]
            df_cal_f = df_cal[df_cal["Días"] <= max_dias].copy()
            if tickers_cal:
                df_cal_f = df_cal_f[df_cal_f["Ticker"].isin(tickers_cal)]
            if tipo_cal:
                df_cal_f = df_cal_f[df_cal_f["Tipo"].isin(tipo_cal)]

            # Métricas
            proximos_30 = df_cal_f[df_cal_f["Días"] <= 30]
            cm1, cm2, cm3 = st.columns(3)
            cm1.metric("Pagos en ventana", len(df_cal_f))
            cm2.metric("Próximos 30 días", len(proximos_30))
            if not proximos_30.empty:
                prox = proximos_30.iloc[0]
                cm3.metric("Próximo pago", f"{prox['Ticker']} — {prox['Fecha'].strftime('%d/%m/%Y')}",
                            delta=f"{prox['Días']} días")

            st.markdown('<hr class="rf-divider">', unsafe_allow_html=True)

            # Tabla principal
            df_cal_disp = df_cal_f.copy()
            df_cal_disp["Fecha"] = df_cal_disp["Fecha"].apply(lambda x: x.strftime("%Y-%m-%d"))

            def _color_dias(val):
                try:
                    v = int(val)
                    if v <= 30:  return "color: #f85149; font-weight: bold"
                    if v <= 90:  return "color: #e3b341"
                    return "color: #8b949e"
                except: return ""

            def _color_tipo(val):
                if val == "C+A": return "color: #3fb950; font-weight:bold"
                if val == "A":   return "color: #e3b341"
                return "color: #8b949e"

            styled_cal = (df_cal_disp.style
                .applymap(_color_dias, subset=["Días"])
                .applymap(_color_tipo, subset=["Tipo"])
                .format({"Cupón (% VN)": "{:.5f}", "Total (% VN)": "{:.5f}"}, na_rep="")
            )
            st.dataframe(styled_cal, use_container_width=True, hide_index=True,
                         height=min(600, 40 + len(df_cal_f)*36))

            # Timeline visual — próximos 90 días
            df_tl = df_cal_f[df_cal_f["Días"] <= 90].copy()
            if not df_tl.empty:
                st.markdown('<hr class="rf-divider">', unsafe_allow_html=True)
                st.markdown('<div class="rf-section-header">Timeline próximos 90 días</div>', unsafe_allow_html=True)
                fig_tl = go.Figure()
                color_map = {"C": "#8b949e", "A": "#e3b341", "C+A": "#3fb950"}
                for tipo_t in ["C","A","C+A"]:
                    df_sub = df_tl[df_tl["Tipo"] == tipo_t]
                    if df_sub.empty: continue
                    fig_tl.add_trace(go.Scatter(
                        x=df_sub["Fecha"].apply(lambda x: x.strftime("%Y-%m-%d") if not isinstance(x, str) else x),
                        y=df_sub["Total (% VN)"],
                        mode="markers+text",
                        name=tipo_t,
                        marker=dict(color=color_map[tipo_t], size=12,
                                    line=dict(color="#0d1117", width=2)),
                        text=df_sub["Ticker"],
                        textposition="top center",
                        textfont=dict(family="JetBrains Mono", size=10),
                        hovertemplate="<b>%{text}</b><br>Fecha: %{x}<br>Total: %{y:.4f}%<extra></extra>",
                    ))
                fig_tl.update_layout(
                    paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                    font=dict(color="#8b949e", family="JetBrains Mono, monospace", size=11),
                    xaxis=dict(title="Fecha", gridcolor="#21262d", linecolor="#30363d"),
                    yaxis=dict(title="Total (% VN)", gridcolor="#21262d", linecolor="#30363d"),
                    legend=dict(bgcolor="#161b22", bordercolor="#30363d"),
                    height=320, margin=dict(l=10, r=10, t=20, b=10),
                )
                st.plotly_chart(fig_tl, use_container_width=True)
                st.caption("Verde = C+A (cupón + amortización) · Amarillo = A · Gris = C")
