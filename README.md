# Portfolio Dashboard

Dashboard de seguimiento de activos con Streamlit + yfinance.

## Activos
EEM, BRK-B, META, MSFT, ASML, SPY, TSM, VEA

## Métricas
- Precio actual y variación diaria
- Rendimiento: MTD, 1M, 3M, YTD, 1Y
- Volatilidad anualizada, Sharpe, Sortino
- Max Drawdown, Beta vs SPY
- Dividend Yield
- Gráfico de rendimiento acumulado comparativo
- Matriz de correlación
- Detalle por activo con gráfico de precio y drawdown

## Deploy en Streamlit Cloud

1. Crear repo en GitHub y subir `app.py` y `requirements.txt`
2. Ir a [share.streamlit.io](https://share.streamlit.io)
3. Conectar tu cuenta de GitHub
4. Seleccionar el repo y `app.py` como archivo principal
5. Click en "Deploy"

La app se actualiza automáticamente con cada push al repo.

## Correr local

```bash
pip install -r requirements.txt
streamlit run app.py
```
