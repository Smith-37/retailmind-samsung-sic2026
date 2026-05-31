from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
CSV_DIR = BASE_DIR / "outputs" / "csv"
IMG_DIR = BASE_DIR / "outputs" / "imagenes"
OTHER_DIR = BASE_DIR / "outputs" / "otros"

BLUE = "#1251B5"
INK = "#0F1923"
INK2 = "#1A2535"
GREEN = "#0A8A5C"
RED = "#D93025"
AMBER = "#D97706"
PURPLE = "#6D28D9"

st.set_page_config(
    page_title="RetailMind Challenge - Samsung SIC2026",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');
        :root {
            --ink:#0F1923; --ink2:#1A2535; --surface:#FFFFFF; --muted:#F1F5FB;
            --border:#DDE4EF; --accent1:#1251B5; --accent2:#D93025;
            --accent3:#0A8A5C; --accent4:#D97706; --accent5:#6D28D9;
            --txt-sec:#4A5568; --bg:#EEF2F9;
            --grad-hero:linear-gradient(140deg,#081E52 0%,#1251B5 60%,#2563EB 100%);
            --r:12px;
        }
        html, body, [class*="css"] {
            font-family:'IBM Plex Sans', system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
        }
        .stApp { background:var(--bg); color:var(--ink); }
        .block-container { padding-top:1.4rem; padding-bottom:2rem; max-width:1440px; }
        section[data-testid="stSidebar"] { background:var(--ink2); border-right:1px solid rgba(255,255,255,.08); }
        section[data-testid="stSidebar"] * { color:#EAF1FF; }
        section[data-testid="stSidebar"] .stRadio > label { font-size:.78rem; color:#96A4B8; text-transform:uppercase; letter-spacing:.08em; }
        section[data-testid="stSidebar"] [role="radiogroup"] label {
            padding:.48rem .55rem; border-radius:10px; margin:.12rem 0;
        }
        section[data-testid="stSidebar"] [role="radiogroup"] label:hover {
            background:rgba(255,255,255,.08);
        }
        h1,h2,h3 { color:var(--ink); letter-spacing:0; }
        div[data-testid="stMetric"] {
            background:#fff; border:1px solid var(--border); border-radius:12px;
            padding:1rem 1rem .8rem 1rem; box-shadow:0 8px 22px rgba(15,25,35,.05);
        }
        div[data-testid="stMetric"] label { color:#5B6778; font-size:.84rem; }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color:var(--ink); font-family:'IBM Plex Mono', monospace; font-size:1.6rem;
        }
        .hero {
            background:var(--grad-hero); color:white; border-radius:18px; padding:1.65rem 1.8rem;
            box-shadow:0 18px 42px rgba(18,81,181,.24); margin-bottom:1rem;
        }
        .hero h1 { color:white; font-size:2.2rem; line-height:1.05; margin:0 0 .45rem 0; }
        .hero p { color:#DDE9FF; margin:.2rem 0 1rem 0; max-width:980px; font-size:1.02rem; }
        .chip {
            display:inline-flex; align-items:center; gap:.35rem; padding:.25rem .55rem; margin:.14rem .2rem .14rem 0;
            border:1px solid rgba(255,255,255,.28); border-radius:999px; color:#EEF5FF; font-size:.8rem;
            background:rgba(255,255,255,.1);
        }
        .card {
            background:#fff; border:1px solid var(--border); border-radius:12px; padding:1rem;
            box-shadow:0 8px 22px rgba(15,25,35,.05); margin:.55rem 0 1rem 0;
        }
        .section-title {
            background:#fff; border:1px solid var(--border); border-radius:12px; padding:1rem 1.1rem;
            margin:1rem 0 .8rem 0; box-shadow:0 8px 22px rgba(15,25,35,.04);
        }
        .section-title h2 { margin:0; font-size:1.3rem; }
        .section-title p { margin:.25rem 0 0 0; color:var(--txt-sec); }
        .eyebrow { color:var(--accent1); font-size:.76rem; text-transform:uppercase; letter-spacing:.09em; font-weight:700; }
        .insight {
            border-radius:12px; padding:.85rem 1rem; margin:.55rem 0 .9rem 0;
            border:1px solid var(--border); background:#fff;
        }
        .insight b { display:block; margin-bottom:.18rem; }
        .insight.blue { background:#EAF2FF; border-color:#CFE0FF; }
        .insight.green { background:#EAF8F1; border-color:#C6EBD9; }
        .insight.yellow { background:#FFF7E6; border-color:#F4D9A6; }
        .insight.red { background:#FDEDEC; border-color:#F7C7C3; }
        .insight.purple { background:#F2ECFF; border-color:#D8C9FF; }
        .mini {
            color:#5B6778; font-size:.88rem; line-height:1.45;
        }
        .mono { font-family:'IBM Plex Mono', monospace; }
        .dataframe th { background:var(--ink2) !important; color:#fff !important; }
        div[data-testid="stDataFrame"] { border:1px solid var(--border); border-radius:12px; overflow:hidden; }
        .stTabs [data-baseweb="tab-list"] { gap:.35rem; }
        .stTabs [data-baseweb="tab"] {
            background:#fff; border:1px solid var(--border); border-radius:10px 10px 0 0;
            padding:.55rem .8rem;
        }
        .stTabs [aria-selected="true"] { border-bottom:3px solid var(--accent1); color:var(--accent1); }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_csv(filename: str, parse_dates: tuple[str, ...] = ()) -> pd.DataFrame:
    path = CSV_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    for col in parse_dates:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_model_winner() -> str:
    path = OTHER_DIR / "modelo_ganador.txt"
    if path.exists():
        return path.read_text(encoding="utf-8", errors="ignore").strip() or "No definido"
    return "No definido"


def fmt_num(value: float | int | None, digits: int = 0) -> str:
    if value is None or pd.isna(value):
        return "N/D"
    if abs(float(value)) >= 1_000_000:
        return f"{float(value) / 1_000_000:.1f} M"
    if abs(float(value)) >= 1_000:
        return f"{float(value) / 1_000:.1f} K"
    return f"{float(value):,.{digits}f}"


def fmt_pct(value: float | int | None, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "N/D"
    return f"{float(value):.{digits}f}%"


def section(title: str, subtitle: str = "", badge: str = "") -> None:
    badge_html = f'<div class="eyebrow">{html.escape(badge)}</div>' if badge else ""
    st.markdown(
        f"""
        <div class="section-title">
          {badge_html}
          <h2>{html.escape(title)}</h2>
          <p>{html.escape(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def insight(title: str, body: str, tone: str = "blue") -> None:
    st.markdown(
        f"""
        <div class="insight {tone}">
          <b>{html.escape(title)}</b>
          <span>{html.escape(body)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str, chips: list[str]) -> None:
    chips_html = "".join(f'<span class="chip">{html.escape(c)}</span>' for c in chips)
    st.markdown(
        f"""
        <div class="hero">
          <div class="eyebrow" style="color:#BFD5FF;">ASYJ Nexus IA</div>
          <h1>{html.escape(title)}</h1>
          <p>{html.escape(subtitle)}</p>
          <div>{chips_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def img_path(filename: str) -> Path:
    return IMG_DIR / filename


def image_card(filename: str, title: str, caption: str = "") -> None:
    path = img_path(filename)
    if not path.exists():
        st.info(f"No se encontro la imagen: {filename}")
        return
    st.markdown(f"**{title}**")
    st.image(str(path), width="stretch")
    if caption:
        st.caption(caption)


def safe_table(df: pd.DataFrame, columns: list[str] | None = None, n: int = 15) -> None:
    if df.empty:
        st.info("No hay datos disponibles para esta tabla.")
        return
    view = df.copy()
    if columns:
        view = view[[c for c in columns if c in view.columns]]
    st.dataframe(view.head(n), width="stretch", hide_index=True)


def compute_test_metrics(test_df: pd.DataFrame) -> dict[str, float]:
    if test_df.empty or "pred" not in test_df.columns:
        return {}
    y_col = "target_next_month" if "target_next_month" in test_df.columns else "target_demand"
    if y_col not in test_df.columns:
        return {}
    y = pd.to_numeric(test_df[y_col], errors="coerce").fillna(0).to_numpy()
    pred = pd.to_numeric(test_df["pred"], errors="coerce").fillna(0).to_numpy()
    error = np.abs(y - pred)
    mae = float(np.mean(error))
    rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
    denom = float(np.sum(np.abs(y)))
    wape = float(np.sum(error) / denom * 100) if denom else np.nan
    smape = float(np.mean(2 * error / np.maximum(np.abs(y) + np.abs(pred), 1e-9)) * 100)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot else np.nan
    return {"MAE": mae, "RMSE": rmse, "WAPE_%": wape, "sMAPE_%": smape, "R2": r2}


def count_dispatch_cases(priority: pd.DataFrame) -> int:
    if priority.empty or "recomendacion" not in priority.columns:
        return 0
    return int(priority["recomendacion"].astype(str).str.contains("Despachar", case=False, na=False).sum())


def plot_bar(df: pd.DataFrame, x: str, y: str, title: str, color: str = BLUE, orientation: str = "v"):
    if df.empty or x not in df.columns or y not in df.columns:
        return None
    fig = px.bar(df, x=x, y=y, title=title, orientation=orientation, color_discrete_sequence=[color])
    fig.update_layout(
        height=420,
        plot_bgcolor="white",
        paper_bgcolor="white",
        title_font_size=18,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def filter_monthly(df: pd.DataFrame, product_types: list[str], date_range) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "Product_Type" in out.columns and product_types:
        out = out[out["Product_Type"].isin(product_types)]
    if "Month_dt" in out.columns and date_range:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        out = out[(out["Month_dt"] >= start) & (out["Month_dt"] <= end)]
    return out


def app_sidebar(monthly: pd.DataFrame) -> tuple[str, list[str], tuple[pd.Timestamp, pd.Timestamp] | None]:
    st.sidebar.markdown("### RetailMind")
    st.sidebar.caption("Samsung SIC2026 Â· Dashboard ejecutivo")
    pages = [
        "Resumen del proyecto",
        "0. Exploracion y limpieza",
        "1. Analisis exploratorio - EDA",
        "2. Modelo predictivo supervisado",
        "3. Respuestas a las preguntas del reto",
        "4. Inventario y despacho",
        "5. Conclusiones por fase",
    ]
    page = st.sidebar.radio("Menu de fases", pages, label_visibility="collapsed")
    st.sidebar.markdown("---")
    product_types: list[str] = []
    date_range = None
    if not monthly.empty and "Product_Type" in monthly.columns:
        all_types = sorted(monthly["Product_Type"].dropna().astype(str).unique().tolist())
        product_types = st.sidebar.multiselect("Tipos de producto", all_types, default=all_types)
    if not monthly.empty and "Month_dt" in monthly.columns:
        min_date = monthly["Month_dt"].min().to_pydatetime()
        max_date = monthly["Month_dt"].max().to_pydatetime()
        date_range = st.sidebar.slider("Periodo", min_date, max_date, (min_date, max_date), format="YYYY-MM")
    st.sidebar.markdown("---")
    st.sidebar.caption("Fuente: notebook final + outputs exportados.")
    return page, product_types, date_range


def page_summary(monthly: pd.DataFrame, pred: pd.DataFrame, priority: pd.DataFrame, test_df: pd.DataFrame) -> None:
    winner = load_model_winner()
    metrics = compute_test_metrics(test_df)
    hero(
        "RetailMind Challenge - Samsung SIC2026",
        "Analisis predictivo de ventas e inventarios para anticipar demanda, detectar riesgos comerciales y priorizar despacho con criterios operativos.",
        ["Ventas", "Inventarios", "Demanda intermitente", "XGBoost", "Backtesting rolling-origin"],
    )
    if monthly.empty:
        st.warning("No se encontro el dataset mensual del dashboard.")
        return

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Ventas historicas", fmt_num(monthly["Cust_Sales"].sum()))
    c2.metric("Clientes", fmt_num(monthly["Cliente"].nunique()))
    c3.metric("Productos", fmt_num(monthly["Producto"].nunique()))
    c4.metric("Modelo campeon", winner)
    c5.metric("WAPE test", fmt_pct(metrics.get("WAPE_%")))

    zero_rate = (monthly["Cust_Sales"].fillna(0) == 0).mean() * 100
    inv_missing = monthly["Channel_Inv"].isna().mean() * 100 if "Channel_Inv" in monthly else np.nan
    negative_inv = int(monthly.get("channel_inv_negative_flag", pd.Series(dtype=float)).fillna(0).sum())
    risk_units = float(priority["gap_abs"].sum()) if "gap_abs" in priority else np.nan

    section("Lectura ejecutiva", "Que significa el proyecto para el director de ventas.", "Resumen")
    col_a, col_b = st.columns([1.15, 1])
    with col_a:
        insight(
            "Decision principal",
            "El tablero convierte el historico semanal en una lectura mensual accionable: donde crece la demanda, donde cae el cliente y donde el inventario no alcanza para cubrir la venta esperada.",
            "blue",
        )
        insight(
            "Riesgo operativo",
            f"La demanda es altamente intermitente: {fmt_pct(zero_rate)} de las combinaciones mensuales tienen venta cero. Por eso se evita borrar registros y se trabaja con banderas, lags y validacion temporal.",
            "yellow",
        )
        insight(
            "Inventario como senal",
            f"Channel_Inv conserva faltantes y negativos como informacion de negocio. Hay {fmt_num(negative_inv)} observaciones con inventario negativo y {fmt_pct(inv_missing)} sin dato de inventario.",
            "green",
        )
    with col_b:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Arquitectura del proyecto**")
        st.markdown(
            """
            1. Se despivota cada CSV semanal sin mezclar anos.
            2. Se agregan ventas e inventarios a nivel mensual.
            3. Se documentan ceros reales, devoluciones, faltantes y outliers.
            4. Se comparan modelos solo con validation.
            5. Se audita el campeon con backtesting y se reporta test al final.
            6. Se traduce la prediccion en recomendaciones de despacho.
            """
        )
        st.markdown(f"**Brecha total priorizada:** <span class='mono'>{fmt_num(risk_units)}</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    section("Indicadores rapidos", "Volumen, riesgo y desempeno del modelo final.", "KPIs")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Demanda predicha", fmt_num(pred["pred_ventas"].sum() if "pred_ventas" in pred else np.nan))
    p2.metric("Combinaciones a predecir", fmt_num(len(pred)))
    p3.metric("Casos a despachar", fmt_num(count_dispatch_cases(priority)))
    p4.metric("RMSE test", fmt_num(metrics.get("RMSE"), 1))


def page_cleaning(monthly: pd.DataFrame) -> None:
    hero(
        "0. Exploracion inicial, limpieza y preparacion",
        "Reporte de calidad de datos, conversion del formato ancho semanal a base mensual y reglas de tratamiento sin eliminar registros.",
        ["Sin eliminar registros", "Banderas explicitas", "Inventario negativo conservado", "Preparacion mensual"],
    )
    if monthly.empty:
        st.warning("No se encontro el dataset mensual.")
        return

    section("Reporte de limpieza", "Control de calidad antes de modelar.", "Datos")
    missing_inv = monthly["Channel_Inv"].isna().sum()
    negative_inv = monthly.get("channel_inv_negative_flag", pd.Series(dtype=float)).fillna(0).sum()
    stockout = monthly.get("stockout_flag", pd.Series(dtype=float)).fillna(0).mean() * 100
    outliers = monthly.get("outlier_sales_iqr", pd.Series(dtype=float)).fillna(0).sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros mensuales", fmt_num(len(monthly)))
    c2.metric("Meses analizados", fmt_num(monthly["Month_dt"].nunique() if "Month_dt" in monthly else np.nan))
    c3.metric("Inventario sin dato", fmt_num(missing_inv))
    c4.metric("Stockout historico", fmt_pct(stockout))

    insight(
        "Criterio metodologico",
        "La limpieza no fuerza inventario faltante a cero durante la preparacion. Un NaN indica ausencia de dato; cero indica inventario realmente agotado. Esta diferencia protege la interpretacion operativa.",
        "blue",
    )
    insight(
        "Senales conservadas",
        f"Se mantuvieron {fmt_num(negative_inv)} inventarios negativos como bandera explicita y {fmt_num(outliers)} posibles outliers de venta para auditoria, no como motivo de exclusion automatica.",
        "green",
    )

    col1, col2 = st.columns(2)
    with col1:
        by_year = monthly.groupby("Anio", as_index=False)["Cust_Sales"].sum() if "Anio" in monthly else pd.DataFrame()
        fig = plot_bar(by_year, "Anio", "Cust_Sales", "Ventas mensuales agregadas por ano", BLUE)
        if fig:
            st.plotly_chart(fig, width="stretch")
    with col2:
        if "Product_Type" in monthly:
            by_type = monthly.groupby("Product_Type", as_index=False)["Cust_Sales"].sum().sort_values("Cust_Sales", ascending=False)
            fig = plot_bar(by_type, "Product_Type", "Cust_Sales", "Ventas por tipo de producto", GREEN)
            if fig:
                st.plotly_chart(fig, width="stretch")

    section("Graficas de preparacion", "Evidencia visual de temporalidad y calidad.", "Outputs")
    g1, g2 = st.columns(2)
    with g1:
        image_card("eda_13_ventas_por_aÃ±o.jpg", "Ventas por ano", "Verifica la continuidad temporal del historico.")
        image_card("eda_14_ventas_por_semana.jpg", "Ventas por semana", "Muestra variaciones semanales antes de la agregacion mensual.")
    with g2:
        image_card("eda_12_outliers_boxplot.jpg", "Outliers de ventas", "Permite detectar ventas extremas sin eliminarlas automaticamente.")
        image_card("eda_05_correlacion.jpg", "Correlacion de variables", "Revisa relaciones entre ventas, sell-in, inventario y variables derivadas.")


def page_eda(monthly: pd.DataFrame, pareto_clients: pd.DataFrame, pareto_products: pd.DataFrame, inventory: pd.DataFrame, growth_clients: pd.DataFrame) -> None:
    hero(
        "1. Analisis exploratorio - EDA",
        "Patrones de venta, concentracion, estacionalidad, relacion sell-in versus venta y estado de inventario.",
        ["Tendencia temporal", "Pareto", "Stockout", "Estacionalidad", "Rotacion"],
    )
    if monthly.empty:
        st.warning("No se encontro el dataset mensual.")
        return

    section("Tendencia temporal", "Como se mueve la demanda agregada durante el periodo.", "EDA")
    trend = monthly.groupby("Month_dt", as_index=False).agg(Cust_Sales=("Cust_Sales", "sum"), Sell_in=("Sell_in", "sum"))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trend["Month_dt"], y=trend["Cust_Sales"], mode="lines+markers", name="Cust. Sales", line=dict(color=BLUE, width=3)))
    fig.add_trace(go.Scatter(x=trend["Month_dt"], y=trend["Sell_in"], mode="lines+markers", name="Sell-in", line=dict(color=AMBER, width=2)))
    fig.update_layout(height=430, title="Ventas y sell-in por mes", plot_bgcolor="white", paper_bgcolor="white", margin=dict(l=20, r=20, t=60, b=20))
    st.plotly_chart(fig, width="stretch")
    insight(
        "Lectura de negocio",
        "La comparacion entre sell-in y ventas ayuda a separar crecimiento real de acumulacion de inventario. Cuando sell-in sube sin venta equivalente, aparece riesgo de sobrestock o mala asignacion.",
        "blue",
    )

    section("EDA principal", "Graficas exportadas desde el notebook final.", "Graficas")
    a, b = st.columns(2)
    with a:
        image_card("eda_01_ventas_tipo_producto.jpg", "Ventas por tipo de producto")
        image_card("eda_03_top_clientes.jpg", "Top clientes por ventas")
        image_card("eda_07_estacionalidad.jpg", "Estacionalidad mensual")
        image_card("eda_09_heatmap_cliente_tipo.jpg", "Heatmap cliente x tipo")
    with b:
        image_card("eda_02_tendencia_temporal.jpg", "Tendencia temporal historica")
        image_card("eda_04_top_productos.jpg", "Top productos por ventas")
        image_card("eda_08_scatter_sellin_sales.jpg", "Sell-in versus ventas")
        image_card("eda_10_stockout_rate.jpg", "Stockout historico")

    section("1.1 EDA complementario", "Concentracion, clientes que cambian comportamiento e inventario.", "Complemento")
    pareto_n_clients = int(pareto_clients.get("en_pareto_80", pd.Series(dtype=bool)).sum()) if not pareto_clients.empty else 0
    pareto_n_products = int(pareto_products.get("en_pareto_80", pd.Series(dtype=bool)).sum()) if not pareto_products.empty else 0
    insight(
        "Concentracion comercial",
        f"El Pareto identifica {fmt_num(pareto_n_clients)} clientes y {fmt_num(pareto_n_products)} productos que explican aproximadamente el 80% de ventas. Sirve para priorizar cobertura, negociacion y monitoreo.",
        "yellow",
    )

    t1, t2, t3 = st.tabs(["Pareto", "Clientes", "Inventario"])
    with t1:
        c1, c2 = st.columns(2)
        with c1:
            safe_table(pareto_clients, ["Cliente", "Cust_Sales", "participacion_%", "participacion_acum_%", "en_pareto_80"], 20)
        with c2:
            safe_table(pareto_products, ["Producto", "Cust_Sales", "participacion_%", "participacion_acum_%", "en_pareto_80"], 20)
        image_card("eda_16_pareto_clientes_productos.jpg", "Pareto de clientes y productos")
    with t2:
        safe_table(growth_clients, ["Cliente", "ventas_prev_3m", "ventas_last_3m", "cambio_abs", "cambio_%", "estado_cliente"], 30)
        image_card("eda_15_clientes_crecen_disminuyen.jpg", "Clientes que crecen o disminuyen")
    with t3:
        inv_cols = ["Producto", "Product_Type", "ventas_total", "inventario_ultimo", "rotacion_prom", "cobertura_prom_sem", "stockout_rate", "clasificacion_inventario"]
        safe_table(inventory.sort_values("stockout_rate", ascending=False) if "stockout_rate" in inventory else inventory, inv_cols, 25)
        image_card("eda_17_inventario_rotacion_cobertura.jpg", "Rotacion, cobertura e inventario")


def page_model(backtesting: pd.DataFrame, test_df: pd.DataFrame) -> None:
    winner = load_model_winner()
    metrics = compute_test_metrics(test_df)
    hero(
        "2. Modelo predictivo supervisado",
        "Comparacion de candidatos, seleccion del campeon con validation y auditoria temporal del ganador antes del reporte final en test.",
        ["RF", "Arbol", "GBM", "XGBoost", "Test reservado"],
    )

    section("2.1 Comparacion multi-modelo", "La seleccion del campeon se hace sin mirar el test.", "Validation")
    insight(
        "Campeon elegido",
        f"El modelo ganador registrado es {winner}. La comparacion mantiene el test fuera de la seleccion para evitar contaminacion metodologica.",
        "green",
    )
    image_card("modelo_comparacion_multimodelo.jpg", "Comparacion multi-modelo: RF, Arbol, GBM y XGBoost")

    section("2.2 Auditoria estadistica y backtesting rolling-origin", "Evaluacion del campeon en 6 cortes temporales.", "Backtesting")
    if not backtesting.empty:
        mean_wape = backtesting["WAPE_%"].mean() if "WAPE_%" in backtesting else np.nan
        std_wape = backtesting["WAPE_%"].std() if "WAPE_%" in backtesting else np.nan
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Folds", fmt_num(backtesting["fold"].nunique() if "fold" in backtesting else len(backtesting)))
        c2.metric("WAPE promedio", fmt_pct(mean_wape))
        c3.metric("Desv. WAPE", fmt_pct(std_wape))
        c4.metric("R2 promedio", fmt_num(backtesting["R2"].mean() if "R2" in backtesting else np.nan, 3))
        if {"fold", "WAPE_%"}.issubset(backtesting.columns):
            fig = px.line(backtesting, x="fold", y="WAPE_%", color="modelo", markers=True, title="WAPE por fold de backtesting")
            fig.update_layout(height=420, plot_bgcolor="white", paper_bgcolor="white", margin=dict(l=20, r=20, t=60, b=20))
            st.plotly_chart(fig, width="stretch")
        safe_table(backtesting, n=24)
    image_card("modelo_backtesting_wape.jpg", "Backtesting rolling-origin del campeon")

    section("2.3 Entrenamiento final y validacion del campeon", "Una sola lectura final del test reservado.", "Test")
    if metrics:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("MAE", fmt_num(metrics["MAE"], 1))
        c2.metric("RMSE", fmt_num(metrics["RMSE"], 1))
        c3.metric("WAPE", fmt_pct(metrics["WAPE_%"]))
        c4.metric("sMAPE", fmt_pct(metrics["sMAPE_%"]))
        c5.metric("R2", fmt_num(metrics["R2"], 3))
    insight(
        "Interpretacion",
        "En demanda intermitente, el error debe leerse por segmentos: el WAPE resume impacto operativo agregado, mientras los errores altos suelen concentrarse en combinaciones de baja frecuencia o saltos de demanda.",
        "blue",
    )
    if not test_df.empty:
        sample = test_df.sample(min(7000, len(test_df)), random_state=42)
        y_col = "target_next_month" if "target_next_month" in sample else "target_demand"
        fig = px.scatter(sample, x=y_col, y="pred", color="volume_segment" if "volume_segment" in sample else None, opacity=.45, title="Prediccion versus venta real en test")
        fig.add_trace(go.Scatter(x=[0, sample[y_col].max()], y=[0, sample[y_col].max()], mode="lines", name="Linea ideal", line=dict(color=RED, dash="dash")))
        fig.update_layout(height=430, plot_bgcolor="white", paper_bgcolor="white", margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig, width="stretch")
    m1, m2 = st.columns(2)
    with m1:
        image_card("modelo_campeon_validacion.jpg", "Validacion del campeon")
        image_card("modelo_19_xgboost_importancia_variables.jpg", "Importancia de variables - XGBoost base")
    with m2:
        image_card("modelo_campeon_importancia_variables.jpg", "Importancia de variables - campeon final")


def build_churn_from_available(monthly: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    if monthly.empty or pred.empty:
        return pd.DataFrame()
    hist = monthly.groupby(["Cliente", "Month_dt"], as_index=False)["Cust_Sales"].sum()
    recent = hist.sort_values("Month_dt").groupby("Cliente").tail(6)
    rows = []
    pred_client = pred.groupby("Cliente", as_index=False)["pred_ventas"].sum() if "pred_ventas" in pred else pd.DataFrame()
    pred_map = dict(zip(pred_client.get("Cliente", []), pred_client.get("pred_ventas", [])))
    for client, grp in recent.groupby("Cliente"):
        y = grp["Cust_Sales"].to_numpy(dtype=float)
        slope = float(np.polyfit(np.arange(len(y)), y, 1)[0]) if len(y) >= 2 else 0.0
        rows.append(
            {
                "Cliente": client,
                "ventas_6m": float(y.sum()),
                "tendencia_6m": slope,
                "pred_proximo_mes": float(pred_map.get(client, 0)),
                "riesgo": "Alto" if slope < 0 and pred_map.get(client, 0) < np.mean(y) else "Medio" if slope < 0 else "Bajo",
            }
        )
    return pd.DataFrame(rows).sort_values(["riesgo", "tendencia_6m"], ascending=[True, True])


def page_questions(monthly: pd.DataFrame, pred: pd.DataFrame, top_rotation: pd.DataFrame, top_growth: pd.DataFrame, growth_clients: pd.DataFrame, priority: pd.DataFrame) -> None:
    hero(
        "3. Respuestas a las preguntas del reto",
        "Resultados listos para decision: rotacion, churn, crecimiento, caidas esperadas y despacho prioritario.",
        ["Preguntas de negocio", "Accion comercial", "Cliente-producto", "Forecast"],
    )
    tabs = st.tabs(["Rotacion", "Crecimiento", "Clientes en riesgo", "Despacho"])
    with tabs[0]:
        section("Productos de mayor rotacion esperada", "Donde la demanda futura debe monitorearse de cerca.", "Pregunta 1")
        safe_table(top_rotation, n=20)
        if not top_rotation.empty:
            fig = plot_bar(top_rotation.sort_values("pred_ventas", ascending=True), "pred_ventas", "Producto", "Top productos por venta predicha", BLUE, "h")
            if fig:
                st.plotly_chart(fig, width="stretch")
        insight("Decision", "Estos productos deben tener seguimiento de disponibilidad y reposicion porque concentran el mayor potencial de venta futura.", "green")
    with tabs[1]:
        section("Productos con crecimiento esperado", "Incremento frente al promedio reciente.", "Pregunta 3")
        safe_table(top_growth, n=20)
        if not top_growth.empty:
            fig = plot_bar(top_growth.sort_values("incremento_abs", ascending=True), "incremento_abs", "Producto", "Mayor incremento absoluto esperado", GREEN, "h")
            if fig:
                st.plotly_chart(fig, width="stretch")
        image_card("pred_productos.jpg", "Productos con oportunidad predicha")
    with tabs[2]:
        section("Clientes con reduccion o riesgo de churn", "Senales de caida para activar gestion comercial.", "Preguntas 2 y 4")
        churn = build_churn_from_available(monthly, pred)
        safe_table(growth_clients, n=25)
        safe_table(churn, n=25)
        image_card("pred_churn_clientes.jpg", "Clientes con riesgo de reduccion")
        insight("Accion comercial", "Los clientes con tendencia negativa no necesariamente estan perdidos: son candidatos para revisar surtido, frecuencia de abastecimiento, promociones o sustitutos.", "yellow")
    with tabs[3]:
        section("Prioridad de despacho por brecha absoluta", "Gap entre venta esperada e inventario disponible.", "Pregunta 5")
        safe_table(priority, ["Cliente", "Producto", "Product_Type", "pred_ventas", "inventario_disponible", "gap_abs", "cobertura_semanas", "prioridad_despacho", "recomendacion"], 30)
        image_card("pred_despacho_urgente.jpg", "Casos urgentes de despacho")
        insight("Regla aplicada", "La prioridad usa gap absoluto y pred_ventas como desempate. Asi se ordenan primero los casos donde la falta de inventario tiene mayor impacto operativo.", "blue")


def page_inventory(priority: pd.DataFrame, rec_prod: pd.DataFrame, rec_cp: pd.DataFrame) -> None:
    hero(
        "4. Recomendaciones de inventario y despacho",
        "Lista accionable de productos y combinaciones cliente-producto que requieren atencion de abastecimiento.",
        ["Gap absoluto", "Cobertura", "Prioridad alta", "Reabastecimiento"],
    )
    if priority.empty:
        st.warning("No se encontro prioridad_despacho.csv.")
        return
    high = priority[priority["recomendacion"].astype(str).str.contains("Despachar", case=False, na=False)] if "recomendacion" in priority else pd.DataFrame()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Casos evaluados", fmt_num(len(priority)))
    c2.metric("A despachar", fmt_num(len(high)))
    c3.metric("Gap absoluto total", fmt_num(priority["gap_abs"].sum() if "gap_abs" in priority else np.nan))
    c4.metric("Inventario negativo", fmt_num(priority.get("inventario_negativo_flag", pd.Series(dtype=float)).fillna(0).sum()))

    insight(
        "Postura conservadora",
        "Cuando no existe dato de inventario en la recomendacion final, se asume cero para priorizar despacho. No es imputacion de entrenamiento: es una regla operativa prudente para no dejar demanda sin atender.",
        "yellow",
    )
    t1, t2, t3 = st.tabs(["Productos a reabastecer", "Cliente-producto", "Mapa de accion"])
    with t1:
        safe_table(rec_prod, n=50)
        if not rec_prod.empty and "gap_total" in rec_prod:
            fig = plot_bar(rec_prod.sort_values("gap_total", ascending=True), "gap_total", "Producto", "Gap total por producto", RED, "h")
            if fig:
                st.plotly_chart(fig, width="stretch")
    with t2:
        safe_table(rec_cp, n=50)
        safe_table(priority.sort_values("gap_abs", ascending=False) if "gap_abs" in priority else priority, n=35)
    with t3:
        image_card("pred_20_recomendaciones_accionables.jpg", "Recomendaciones accionables")
        image_card("pred_despacho_urgente.jpg", "Despacho urgente")


def page_conclusions(interp: pd.DataFrame, monthly: pd.DataFrame, backtesting: pd.DataFrame, test_df: pd.DataFrame) -> None:
    hero(
        "5. Conclusiones e interpretaciones por fase",
        "Sintesis metodologica y recomendaciones finales para convertir el modelo en gestion comercial.",
        ["Interpretacion", "Auditoria", "Limitaciones", "Siguientes pasos"],
    )
    section("Interpretacion ejecutiva por fase", "Resumen exportado desde el notebook final.", "Conclusiones")
    safe_table(interp, n=20)
    metrics = compute_test_metrics(test_df)
    wape_bt = backtesting["WAPE_%"].mean() if not backtesting.empty and "WAPE_%" in backtesting else np.nan

    col1, col2 = st.columns(2)
    with col1:
        insight(
            "Metodologia",
            "El flujo es correcto para competencia de ML: comparacion en validation, auditoria temporal del campeon y test usado una unica vez para reporte final.",
            "green",
        )
        insight(
            "Resultado modelado",
            f"El desempeno debe reportarse con WAPE porque comunica error operativo agregado. Backtesting promedio: {fmt_pct(wape_bt)}. Test final: {fmt_pct(metrics.get('WAPE_%'))}.",
            "blue",
        )
    with col2:
        insight(
            "Uso recomendado",
            "El tablero no debe reemplazar al planeador comercial: debe ordenar prioridades, detectar excepciones y abrir conversaciones sobre abastecimiento, surtido y retencion.",
            "yellow",
        )
        insight(
            "Riesgos a vigilar",
            "La demanda intermitente, ventas cero reales y faltantes de inventario pueden sesgar conclusiones si se interpretan como ausencia de oportunidad. Por eso se conservan banderas y se segmentan las series.",
            "purple",
        )

    section("Auditoria final", "Chequeos de consistencia del proyecto.", "QA")
    rows = [
        ("No se eliminan registros base", "Cumplido: el flujo trabaja con banderas y agregacion mensual."),
        ("Inventario faltante no equivale a cero", "Cumplido en preparacion; cero se usa solo como postura conservadora en recomendacion final."),
        ("Test no contamina seleccion", "Cumplido: candidato se elige en validation y test se reserva para cierre."),
        ("Backtesting temporal", "Cumplido: rolling-origin permite ver estabilidad del campeon."),
        ("Salida accionable", "Cumplido: prioridad por gap absoluto y tablas de reabastecimiento."),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["Criterio", "Lectura"]), width="stretch", hide_index=True)


def main() -> None:
    inject_css()
    monthly_all = load_csv("dashboard_dataset_mensual.csv", parse_dates=("Month_dt",))
    page, product_types, date_range = app_sidebar(monthly_all)
    monthly = filter_monthly(monthly_all, product_types, date_range)

    pred = load_csv("predicciones_proximo_mes.csv")
    priority = load_csv("prioridad_despacho.csv")
    test_df = load_csv("validacion_predicciones_test.csv", parse_dates=("Month_dt",))
    backtesting = load_csv("backtesting_rolling_origin_detalle.csv")

    if page == "Resumen del proyecto":
        page_summary(monthly, pred, priority, test_df)
    elif page == "0. Exploracion y limpieza":
        page_cleaning(monthly)
    elif page == "1. Analisis exploratorio - EDA":
        page_eda(
            monthly,
            load_csv("pareto_clientes.csv"),
            load_csv("pareto_productos.csv"),
            load_csv("inventario_rotacion_cobertura_productos.csv"),
            load_csv("eda_clientes_crecen_disminuyen.csv"),
        )
    elif page == "2. Modelo predictivo supervisado":
        page_model(backtesting, test_df)
    elif page == "3. Respuestas a las preguntas del reto":
        page_questions(
            monthly,
            pred,
            load_csv("top_productos_rotacion_predicha.csv"),
            load_csv("top_productos_crecimiento_predicho.csv"),
            load_csv("eda_clientes_crecen_disminuyen.csv"),
            priority,
        )
    elif page == "4. Inventario y despacho":
        page_inventory(
            priority,
            load_csv("recomendaciones_productos_reabastecer.csv"),
            load_csv("recomendaciones_reabastecer_cliente_producto.csv"),
        )
    elif page == "5. Conclusiones por fase":
        page_conclusions(load_csv("interpretacion_ejecutiva_por_fase.csv"), monthly, backtesting, test_df)


if __name__ == "__main__":
    main()

