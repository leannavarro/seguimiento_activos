import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta, date
from scipy.optimize import brentq
from dateutil.relativedelta import relativedelta

st.set_page_config(
    page_title='Bond Terminal ARG',
    page_icon='📊',
    layout='wide',
    initial_sidebar_state='collapsed',
)

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

# ─── CATÁLOGO TESORO ──────────────────────────────────────────────────────────
CATALOG_TESORO = {
    "T30J6": {"familia":"BONCAP","desc":"BONCAP $/30-06-2026","vto":"2026-06-30","moneda":"ARP"},
    "T15E7": {"familia":"BONCAP","desc":"BONCAP $/15-01-2027","vto":"2027-01-15","moneda":"ARP"},
    "T30A7": {"familia":"BONCAP","desc":"BONCAP $/30-04-2027","vto":"2027-04-30","moneda":"ARP"},
    "T31Y7": {"familia":"BONCAP","desc":"BONCAP $/31-05-2027","vto":"2027-05-31","moneda":"ARP"},
    "T30J7": {"familia":"BONCAP","desc":"BONCAP $/30-06-2027","vto":"2027-06-30","moneda":"ARP"},
    "TZXM6": {"familia":"BONCER","desc":"BONCER CER/31-03-2026","vto":"2026-03-31","moneda":"UCP"},
    "TZX26": {"familia":"BONCER","desc":"BONCER CER/30-06-2026","vto":"2026-06-30","moneda":"UCP"},
    "TZXO6": {"familia":"BONCER","desc":"BONCER CER/31-10-2026","vto":"2026-10-31","moneda":"UCP"},
    "TZXD6": {"familia":"BONCER","desc":"BONCER CER/15-12-2026","vto":"2026-12-15","moneda":"UCP"},
    "TZXM7": {"familia":"BONCER","desc":"BONCER CER/31-03-2027","vto":"2027-03-31","moneda":"UCP"},
    "TZXA7": {"familia":"BONCER","desc":"BONCER CER/30-04-2027","vto":"2027-04-30","moneda":"UCP"},
    "TZXY7": {"familia":"BONCER","desc":"BONCER CER/31-05-2027","vto":"2027-05-31","moneda":"UCP"},
    "TZX27": {"familia":"BONCER","desc":"BONCER CER/30-06-2027","vto":"2027-06-30","moneda":"UCP"},
    "TZXD7": {"familia":"BONCER","desc":"BONCER CER/15-12-2027","vto":"2027-12-15","moneda":"UCP"},
    "TZX28": {"familia":"BONCER","desc":"BONCER CER/30-06-2028","vto":"2028-06-30","moneda":"UCP"},
    "TV26":  {"familia":"DL",    "desc":"BONTE DL/30-06-2026","vto":"2026-06-30","moneda":"DLK"},
    "BOPRE": {"familia":"BONTE", "desc":"BONTE USD 6.5%/30-11-2029","vto":"2029-11-30","moneda":"USD"},
    "TB30":  {"familia":"BONTE", "desc":"BONTE $/29.5%/30-05-2030","vto":"2030-05-30","moneda":"ARP"},
    "TTJ26": {"familia":"DUAL",  "desc":"DUAL $/30-06-2026","vto":"2026-06-30","moneda":"ARP"},
    "TTS26": {"familia":"DUAL",  "desc":"DUAL $/15-09-2026","vto":"2026-09-15","moneda":"ARP"},
    "TTD26": {"familia":"DUAL",  "desc":"DUAL $/15-12-2026","vto":"2026-12-15","moneda":"ARP"},
    "AO27":  {"familia":"Hard Dollar","desc":"BONAR USD 6%/29-10-2027","vto":"2027-10-29","moneda":"USD"},
    "S17A6": {"familia":"LECAP", "desc":"LECAP $/17-04-2026","vto":"2026-04-17","moneda":"ARP"},
    "S30A6": {"familia":"LECAP", "desc":"LECAP $/30-04-2026","vto":"2026-04-30","moneda":"ARP"},
    "S15Y6": {"familia":"LECAP", "desc":"LECAP $/15-05-2026","vto":"2026-05-15","moneda":"ARP"},
    "S29Y6": {"familia":"LECAP", "desc":"LECAP $/29-05-2026","vto":"2026-05-29","moneda":"ARP"},
    "S16M6": {"familia":"LECAP", "desc":"LECAP $/16-03-2026","vto":"2026-03-16","moneda":"ARP"},
    "S27F6": {"familia":"LECAP", "desc":"LECAP $/27-02-2026","vto":"2026-02-27","moneda":"ARP"},
    "S31L6": {"familia":"LECAP", "desc":"LECAP $/31-07-2026","vto":"2026-07-31","moneda":"ARP"},
    "S31G6": {"familia":"LECAP", "desc":"LECAP $/31-08-2026","vto":"2026-08-31","moneda":"ARP"},
    "S30S6": {"familia":"LECAP", "desc":"LECAP $/30-09-2026","vto":"2026-09-30","moneda":"ARP"},
    "S30O6": {"familia":"LECAP", "desc":"LECAP $/30-10-2026","vto":"2026-10-30","moneda":"ARP"},
    "S30N6": {"familia":"LECAP", "desc":"LECAP $/30-11-2026","vto":"2026-11-30","moneda":"ARP"},
    "M30A6": {"familia":"TAMAR", "desc":"TAMAR+4%/30-04-2026","vto":"2026-04-30","moneda":"ARP"},
    "M31G6": {"familia":"TAMAR", "desc":"TAMAR+5%/31-08-2026","vto":"2026-08-31","moneda":"ARP"},
    "M15D5": {"familia":"TAMAR", "desc":"TAMAR+2%/15-12-2026","vto":"2026-12-15","moneda":"ARP"},
    "TMF27": {"familia":"TAMAR", "desc":"BONO TAMAR/26-02-2027","vto":"2027-02-26","moneda":"ARP"},
    "X15Y6": {"familia":"LECER", "desc":"LECER CER/15-05-2026","vto":"2026-05-15","moneda":"ARP"},
    "X29Y6": {"familia":"LECER", "desc":"LECER CER/29-05-2026","vto":"2026-05-29","moneda":"ARP"},
    "X31L6": {"familia":"LECER", "desc":"LECER CER/31-07-2026","vto":"2026-07-31","moneda":"ARP"},
    "X30S6": {"familia":"LECER", "desc":"LECER CER/30-09-2026","vto":"2026-09-30","moneda":"ARP"},
    "X30N6": {"familia":"LECER", "desc":"LECER CER/30-11-2026","vto":"2026-11-30","moneda":"ARP"},
    "D30A6": {"familia":"LELINK","desc":"LELINK USD/30-04-2026","vto":"2026-04-30","moneda":"USD"},
    "D30S6": {"familia":"LELINK","desc":"LELINK USD/30-09-2026","vto":"2026-09-30","moneda":"USD"},
}
FAMILIA_CLASE = {
    "Hard Dollar":"Hard Dollar","BONCAP":"Tasa Fija / Lecap","LECAP":"Tasa Fija / Lecap",
    "TAMAR":"TAMAR","DUAL":"Dual","BONCER":"CER","LECER":"CER",
    "DL":"Dollar-linked","LELINK":"Dollar-linked","BONTE":"Otros","BOPREAL":"BOPREAL",
}
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

rf_tab1, rf_tab2, rf_tab3, rf_tab4, rf_tab5, rf_tab6, rf_tab7 = st.tabs([
    "MERCADO", "ANÁLISIS", "ONs", "CARTERA", "CURVA & ESCENARIOS", "CALENDARIO", "TIR HISTÓRICA"
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


# ══════════════════════════════════════════════════════════════════════════
# RF TAB 7 — TIR HISTÓRICA
# ══════════════════════════════════════════════════════════════════════════
with rf_tab7:
    st.markdown('<div class="rf-section-header">TIR / Rendimiento histórico por instrumento</div>', unsafe_allow_html=True)

    hoy_th = datetime.today().date()

    # ── Helpers de rendimiento por tipo ───────────────────────────────────

    def _calc_rendimiento_soberano(sym_d, n_dias):
        """YTM histórica para GD/AL usando BONDS_DB. Devuelve DataFrame date/rend."""
        tk = sym_d.replace("D","").replace("C","")
        if tk not in BONDS_DB:
            return pd.DataFrame()
        df_h = d912_historical(sym_d).tail(n_dias).copy()
        if df_h.empty:
            return pd.DataFrame()
        results = []
        for _, row in df_h.iterrows():
            precio_limpio = float(row["c"])
            if precio_limpio <= 0:
                continue
            settle = row["date"].date() if hasattr(row["date"], "date") else row["date"]
            cc  = _cupon_corrido(tk, settle)
            ps  = precio_limpio + cc
            ytm = _ytm(tk, ps, settle)
            if ytm is not None:
                results.append({"date": row["date"], "rend": ytm * 100})
        return pd.DataFrame(results)

    def _calc_tem_bullet(sym, vto_str, n_dias):
        """TEM implícita para LECAP/BONCAP bullet. rend = TEM % mensual."""
        df_h = d912_historical(sym).tail(n_dias).copy()
        if df_h.empty:
            return pd.DataFrame()
        vto = datetime.strptime(vto_str, "%Y-%m-%d").date()
        results = []
        for _, row in df_h.iterrows():
            precio = float(row["c"])
            if precio <= 0:
                continue
            settle = row["date"].date() if hasattr(row["date"], "date") else row["date"]
            dias   = (vto - settle).days
            if dias <= 0:
                continue
            meses  = dias / 30.4375
            # TEM: precio = 1000 / (1+TEM)^meses  →  TEM = (1000/precio)^(1/meses) - 1
            # precio viene normalizado sobre 1000 en data912 para letras
            # si precio < 10 asumimos ratio, si > 100 es sobre 1000
            if precio > 100:
                ratio = 1000 / precio
            elif precio > 10:
                ratio = 100 / precio
            else:
                ratio = 1 / precio
            tem = ratio ** (1 / meses) - 1
            results.append({"date": row["date"], "rend": tem * 100})
        return pd.DataFrame(results)

    def _calc_tir_cer(sym, vto_str, n_dias):
        """Tasa real CER implícita anualizada para BONCER/LECER zero-coupon."""
        df_h = d912_historical(sym).tail(n_dias).copy()
        if df_h.empty:
            return pd.DataFrame()
        vto = datetime.strptime(vto_str, "%Y-%m-%d").date()
        results = []
        for _, row in df_h.iterrows():
            precio = float(row["c"])
            if precio <= 0:
                continue
            settle = row["date"].date() if hasattr(row["date"], "date") else row["date"]
            dias   = (vto - settle).days
            if dias <= 0:
                continue
            # precio sobre 100 o sobre 1 — normalizar
            if precio > 100:
                p = precio / 1000   # si viene sobre 1000
            elif precio > 1:
                p = precio / 100
            else:
                p = precio
            tir_real = (1 / p) ** (365 / dias) - 1
            results.append({"date": row["date"], "rend": tir_real * 100})
        return pd.DataFrame(results)

    def _calc_precio_norm(sym, n_dias):
        """Precio normalizado base 100 para instrumentos sin TIR directa."""
        df_h = d912_historical(sym).tail(n_dias).copy()
        if df_h.empty:
            return pd.DataFrame()
        if df_h["c"].iloc[0] <= 0:
            return pd.DataFrame()
        base = df_h["c"].iloc[0]
        df_h["rend"] = df_h["c"] / base * 100
        return df_h[["date","rend"]]

    # ── Clasificación de qué métrica usar ─────────────────────────────────
    SOBERANOS_BASE = ["GD29","GD30","GD35","GD38","GD41","GD46",
                       "AL29","AL30","AL35","AL38","AL41","AL46","AE38"]

    def _tipo_metrica(tk_base):
        if tk_base in SOBERANOS_BASE:
            return "ytm"
        cat = CATALOG_TESORO.get(tk_base, {})
        fam = cat.get("familia","")
        if fam in ("LECAP","BONCAP"):
            return "tem"
        if fam in ("BONCER","LECER"):
            return "cer"
        if fam in ("DUAL","TAMAR","BONTE","DL","LELINK"):
            return "norm"
        return "norm"

    METRICA_LABEL = {
        "ytm":  "TIR % anual (YTM, 30/360)",
        "tem":  "TEM % mensual implícita",
        "cer":  "Tasa real CER % anual implícita",
        "norm": "Precio base 100",
    }

    METRICA_COLOR = {
        "ytm":  "#58a6ff",
        "tem":  "#e3b341",
        "cer":  "#3fb950",
        "norm": "#bc8cff",
    }

    # ── UI ────────────────────────────────────────────────────────────────
    cl1, cl2, cl3 = st.columns([2, 3, 1])

    with cl1:
        clases_disp = ["Hard Dollar", "CER", "Tasa Fija / Lecap", "TAMAR", "Dual",
                        "Dollar-linked", "BOPREAL"]
        clase_sel = st.selectbox("Clase de instrumento", clases_disp, key="th_clase")

    # Armar lista de tickers disponibles para la clase
    tickers_clase = []
    for tk in SOBERANOS_BASE:
        if clase_sel == "Hard Dollar":
            tickers_clase.append((tk, tk + "D", "ytm"))
    for tk, meta in CATALOG_TESORO.items():
        fam = meta["familia"]
        clase_cat = FAMILIA_CLASE.get(fam, "Otros")
        if clase_cat == clase_sel:
            tipo = _tipo_metrica(tk)
            # Usar símbolo directo (sin sufijo) para estos instrumentos
            tickers_clase.append((tk, tk, tipo))

    with cl2:
        if not tickers_clase:
            st.warning("Sin instrumentos para esta clase.")
            st.stop()
        opts = [t[0] for t in tickers_clase]
        default_opts = opts[:min(3, len(opts))]
        tickers_sel = st.multiselect("Tickers", opts, default=default_opts, key="th_tickers")

    with cl3:
        periodo_th = st.select_slider("Período", ["1M","3M","6M","1Y","2Y","Todo"],
                                       value="1Y", key="th_periodo")

    dias_map_th = {"1M":21,"3M":63,"6M":126,"1Y":252,"2Y":504,"Todo":9999}
    n_dias_th   = dias_map_th[periodo_th]

    if not tickers_sel:
        st.info("Seleccioná al menos un ticker.")
    else:
        # ── Calcular series ───────────────────────────────────────────────
        # Cache en session_state para no recalcular en cada interacción
        cache_key = f"th_cache_{clase_sel}_{periodo_th}"
        if cache_key not in st.session_state:
            st.session_state[cache_key] = {}

        series = {}
        tipo_metrica_global = None

        with st.spinner("Calculando rendimientos históricos..."):
            for tk in tickers_sel:
                # Encontrar sym y tipo para este ticker
                match = next((t for t in tickers_clase if t[0] == tk), None)
                if not match:
                    continue
                _, sym, tipo = match
                tipo_metrica_global = tipo  # todos de la misma clase tienen el mismo tipo

                ck = f"{sym}_{n_dias_th}"
                if ck in st.session_state[cache_key]:
                    df_s = st.session_state[cache_key][ck]
                else:
                    if tipo == "ytm":
                        df_s = _calc_rendimiento_soberano(sym, n_dias_th)
                    elif tipo == "tem":
                        vto = CATALOG_TESORO.get(tk, {}).get("vto","2099-01-01")
                        df_s = _calc_tem_bullet(sym, vto, n_dias_th)
                    elif tipo == "cer":
                        vto = CATALOG_TESORO.get(tk, {}).get("vto","2099-01-01")
                        df_s = _calc_tir_cer(sym, vto, n_dias_th)
                    else:
                        df_s = _calc_precio_norm(sym, n_dias_th)
                    st.session_state[cache_key][ck] = df_s

                if not df_s.empty:
                    series[tk] = df_s

        if not series:
            st.warning("Sin datos históricos para los tickers seleccionados.")
        else:
            metrica_label = METRICA_LABEL.get(tipo_metrica_global, "Rendimiento")

            # ── Toggle: nivel vs compresión ───────────────────────────────
            c_tog1, c_tog2, _ = st.columns([1,1,4])
            with c_tog1:
                modo = st.radio("Vista", ["Nivel", "Compresión vs inicio"],
                                 horizontal=True, key="th_modo")
            with c_tog2:
                suavizado = st.checkbox("Suavizado 5d", value=False, key="th_suav")

            # ── Gráfico principal ─────────────────────────────────────────
            fig_th = go.Figure()

            palette = ["#58a6ff","#3fb950","#e3b341","#f85149","#bc8cff",
                        "#79c0ff","#56d364","#ffa657","#ff7b72","#d2a8ff"]

            for i, (tk, df_s) in enumerate(series.items()):
                color = palette[i % len(palette)]
                y_vals = df_s["rend"].copy()
                if suavizado:
                    y_vals = y_vals.rolling(5, min_periods=1).mean()
                if modo == "Compresión vs inicio":
                    base_val = y_vals.iloc[0]
                    y_vals   = y_vals - base_val
                fig_th.add_trace(go.Scatter(
                    x=df_s["date"], y=y_vals,
                    mode="lines", name=tk,
                    line=dict(color=color, width=1.8),
                    hovertemplate=f"<b>{tk}</b><br>%{{x|%Y-%m-%d}}<br>{metrica_label}: %{{y:.3f}}<extra></extra>",
                ))

            y_title = (f"Δ {metrica_label} (vs inicio período)"
                        if modo == "Compresión vs inicio" else metrica_label)

            fig_th.update_layout(
                paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                font=dict(color="#8b949e", family="JetBrains Mono, monospace", size=11),
                xaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
                yaxis=dict(title=y_title, gridcolor="#21262d", linecolor="#30363d"),
                legend=dict(bgcolor="#161b22", bordercolor="#30363d",
                             orientation="h", yanchor="bottom", y=1.02),
                height=440, hovermode="x unified",
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_th, use_container_width=True)

            # ── Tabla resumen: primer / último / cambio / min / max ───────
            st.markdown('<div class="rf-section-header">Resumen del período</div>', unsafe_allow_html=True)
            resumen_rows = []
            for tk, df_s in series.items():
                y = df_s["rend"]
                resumen_rows.append({
                    "Ticker":  tk,
                    "Métrica": metrica_label.split(" ")[0],
                    "Inicio":  round(y.iloc[0], 3),
                    "Actual":  round(y.iloc[-1], 3),
                    "Cambio":  round(y.iloc[-1] - y.iloc[0], 3),
                    "Mín":     round(y.min(), 3),
                    "Máx":     round(y.max(), 3),
                    "Avg":     round(y.mean(), 3),
                })
            df_res = pd.DataFrame(resumen_rows)

            def _color_cambio(val):
                try:
                    v = float(val)
                    # Para TIR/TEM: compresión (negativo) es bueno (precio subió)
                    # Para precio norm: positivo es bueno
                    if tipo_metrica_global in ("ytm","tem","cer"):
                        return "color: #3fb950" if v < 0 else ("color: #f85149" if v > 0 else "")
                    else:
                        return "color: #3fb950" if v > 0 else ("color: #f85149" if v < 0 else "")
                except: return ""

            st.dataframe(
                df_res.style.applymap(_color_cambio, subset=["Cambio"]),
                use_container_width=True, hide_index=True,
            )
            unidad = {"ytm":"bps","tem":"bps TEM","cer":"bps","norm":"%"}
            u = unidad.get(tipo_metrica_global,"")
            st.caption(
                f"Cambio expresado en {u} · "
                f"{'Negativo = compresión (precio subió)' if tipo_metrica_global in ('ytm','tem','cer') else 'Positivo = precio subió'} · "
                f"Fuente: data912.com"
            )
