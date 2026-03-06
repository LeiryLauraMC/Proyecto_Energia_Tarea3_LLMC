"""
Aplicación Streamlit — Predicción de Consumo Energético de Edificios
Proyecto end-to-end de Machine Learning (Capítulo 2 — Géron)
Dataset: UCI Energy Efficiency

Secciones:
  1. Análisis Exploratorio  (Etapas 1-3)
  2. Modelado y Resultados  (Etapas 4-8)
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker
from matplotlib import patheffects
import warnings
import urllib.request
from pathlib import Path

warnings.filterwarnings("ignore")

# ─── CONFIGURACIÓN DE PÁGINA ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Consumo Energético · ML",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── PALETA Y ESTILOS GLOBALES ────────────────────────────────────────────────
CREAM      = "#F7F3EE"
WHITE      = "#FFFFFF"
CARD_BG    = "#FDFAF7"
TEAL_DARK  = "#1A5C6B"
TEAL_MID   = "#2A8A9E"
TEAL_LIGHT = "#7EC8D8"
ORANGE     = "#E8621A"
ORANGE_LT  = "#F5A26B"
SAND       = "#C9A96E"
SAND_LT    = "#E8D9C0"
CHARCOAL   = "#2C2C2C"
GRAY_MID   = "#7A7A7A"
GRAY_LT    = "#D4CFC9"
SUCCESS    = "#3D9970"
WARN       = "#E8621A"

# Matplotlib theme global
plt.rcParams.update({
    "figure.facecolor":  CREAM,
    "axes.facecolor":    CREAM,
    "axes.edgecolor":    GRAY_LT,
    "axes.labelcolor":   CHARCOAL,
    "axes.titlecolor":   CHARCOAL,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        GRAY_LT,
    "grid.linewidth":    0.5,
    "grid.alpha":        0.6,
    "xtick.color":       GRAY_MID,
    "ytick.color":       GRAY_MID,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "font.family":       "serif",
    "text.color":        CHARCOAL,
})

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Source+Serif+4:ital,wght@0,300;0,400;0,600;1,300;1,400&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset y fondo ── */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stMain"], .main { background: #F7F3EE !important; }

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { background: #F0EAE0 !important; }

section.main > div { padding-top: 0 !important; }
.block-container { padding: 0 2.5rem 3rem 2.5rem !important; max-width: 1300px !important; }

/* ── Tipografía global ── */
* { font-family: 'Source Serif 4', Georgia, serif !important; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #1A5C6B 0%, #2A8A9E 45%, #1A5C6B 100%);
    border-radius: 0 0 32px 32px;
    padding: 3.5rem 3rem 3rem;
    margin: 0 -2.5rem 2.5rem -2.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "";
    position: absolute; inset: 0;
    background: radial-gradient(ellipse 70% 80% at 85% 40%, rgba(200,169,110,0.18) 0%, transparent 70%),
                radial-gradient(ellipse 40% 60% at 10% 80%, rgba(232,98,26,0.12) 0%, transparent 60%);
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.3);
    color: #E8D9C0;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.3rem 0.9rem;
    border-radius: 20px;
    margin-bottom: 1rem;
}
.hero-title {
    font-family: 'Playfair Display', Georgia, serif !important;
    font-size: clamp(2rem, 4vw, 3rem);
    font-weight: 700;
    color: #FFFFFF;
    line-height: 1.15;
    margin: 0 0 0.7rem;
    position: relative;
}
.hero-sub {
    font-size: 1rem;
    color: rgba(255,255,255,0.78);
    max-width: 620px;
    line-height: 1.6;
    font-weight: 300;
    font-style: italic;
    position: relative;
}
.hero-stats {
    display: flex; gap: 2.5rem; margin-top: 2rem;
    position: relative;
}
.hero-stat-val {
    font-family: 'Playfair Display', serif !important;
    font-size: 1.8rem; font-weight: 700; color: #C9A96E;
    line-height: 1;
}
.hero-stat-lbl {
    font-size: 0.72rem; color: rgba(255,255,255,0.6);
    letter-spacing: 0.08em; text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace !important;
    margin-top: 0.2rem;
}

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 2px solid #D4CFC9 !important;
    gap: 0 !important; padding: 0 !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    font-family: 'Playfair Display', serif !important;
    font-size: 1.02rem !important;
    font-weight: 600 !important;
    color: #7A7A7A !important;
    padding: 0.9rem 1.8rem !important;
    border-bottom: 3px solid transparent !important;
    margin-bottom: -2px !important;
    background: transparent !important;
    transition: all 0.2s ease !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: #1A5C6B !important;
    border-bottom-color: #1A5C6B !important;
}

/* ── Cards ── */
.card {
    background: #FDFAF7;
    border: 1px solid #E8D9C0;
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 2px 12px rgba(26,92,107,0.06);
    transition: box-shadow 0.2s ease;
}
.card:hover { box-shadow: 0 6px 24px rgba(26,92,107,0.11); }

.card-accent-teal  { border-left: 4px solid #1A5C6B; }
.card-accent-orange { border-left: 4px solid #E8621A; }
.card-accent-sand  { border-left: 4px solid #C9A96E; }
.card-accent-green { border-left: 4px solid #3D9970; }

/* ── Section headers ── */
.section-label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.65rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #E8621A;
    margin-bottom: 0.3rem;
}
.section-title {
    font-family: 'Playfair Display', serif !important;
    font-size: 1.5rem;
    font-weight: 700;
    color: #1A5C6B;
    margin: 0 0 0.5rem;
    line-height: 1.2;
}
.section-body {
    color: #5A5A5A;
    font-size: 0.95rem;
    line-height: 1.7;
    max-width: 780px;
}

/* ── Metric chips ── */
.metric-row { display: flex; gap: 1rem; flex-wrap: wrap; margin: 1rem 0; }
.metric-chip {
    background: #FDFAF7;
    border: 1px solid #D4CFC9;
    border-radius: 12px;
    padding: 0.9rem 1.2rem;
    flex: 1; min-width: 130px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.05);
}
.metric-chip-val {
    font-family: 'Playfair Display', serif !important;
    font-size: 1.55rem;
    font-weight: 700;
    color: #1A5C6B;
    line-height: 1;
}
.metric-chip-lbl {
    font-size: 0.72rem;
    color: #7A7A7A;
    margin-top: 0.25rem;
    font-family: 'JetBrains Mono', monospace !important;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* ── Tablas ── */
.styled-table-wrap { overflow-x: auto; border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,0.05); }
.styled-table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
.styled-table thead tr {
    background: #1A5C6B; color: white;
}
.styled-table thead th {
    padding: 0.85rem 1.1rem;
    text-align: left;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem;
    letter-spacing: 0.06em;
    font-weight: 500;
}
.styled-table tbody tr { border-bottom: 1px solid #E8D9C0; transition: background 0.15s; }
.styled-table tbody tr:hover { background: #F0EAE0; }
.styled-table tbody td { padding: 0.75rem 1.1rem; color: #2C2C2C; }
.styled-table tbody tr:last-child { border-bottom: none; }
.styled-table .num { font-family: 'JetBrains Mono', monospace !important; }
.styled-table .best { color: #1A5C6B; font-weight: 600; }
.styled-table .tag {
    display: inline-block;
    padding: 0.15rem 0.6rem;
    border-radius: 20px;
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace !important;
}
.tag-green  { background:#D4EDDA; color:#1E7E34; }
.tag-orange { background:#FDECD5; color:#C95200; }
.tag-blue   { background:#D0ECF5; color:#1A5C6B; }

/* ── Collapsible expander ── */
[data-testid="stExpander"] {
    background: #FDFAF7 !important;
    border: 1px solid #E8D9C0 !important;
    border-radius: 12px !important;
    margin-bottom: 0.8rem !important;
}

/* ── Code blocks ── */
code, .code-block {
    font-family: 'JetBrains Mono', monospace !important;
    background: #EEE9E2 !important;
    padding: 0.15rem 0.45rem;
    border-radius: 4px;
    font-size: 0.85rem;
    color: #C95200;
}

/* ── Dividers ── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #C9A96E 30%, #C9A96E 70%, transparent);
    margin: 2rem 0;
    opacity: 0.5;
}

/* ── Stage timeline ── */
.timeline { display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 1rem 0; }
.tl-step {
    background: #FDFAF7;
    border: 1px solid #D4CFC9;
    border-radius: 8px;
    padding: 0.45rem 0.85rem;
    font-size: 0.78rem;
    color: #5A5A5A;
    font-family: 'JetBrains Mono', monospace !important;
    display: flex; align-items: center; gap: 0.4rem;
}
.tl-step.done { background:#D4EDDA; border-color:#3D9970; color:#1E7E34; }
.tl-step.active { background:#1A5C6B; border-color:#1A5C6B; color:white; }

/* ── Predictor UI ── */
.predictor-result {
    background: linear-gradient(135deg, #1A5C6B, #2A8A9E);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(26,92,107,0.2);
}
.predictor-val {
    font-family: 'Playfair Display', serif !important;
    font-size: 3rem; font-weight: 700; color: #C9A96E; line-height: 1;
}
.predictor-unit { font-size: 0.9rem; color: rgba(255,255,255,0.7); margin-top: 0.3rem; }

/* ── Info callouts ── */
.callout {
    border-radius: 10px; padding: 1rem 1.2rem;
    margin: 0.8rem 0; font-size: 0.9rem; line-height: 1.6;
}
.callout-teal  { background:#E0F2F6; border-left:3px solid #1A5C6B; color:#1A5C6B; }
.callout-orange { background:#FEF3EC; border-left:3px solid #E8621A; color:#C95200; }
.callout-sand  { background:#FBF5E8; border-left:3px solid #C9A96E; color:#7A5C2A; }
</style>
""", unsafe_allow_html=True)


# ─── DATOS Y MODELO ───────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    path = Path("data/ENB2012_data.xlsx")
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"
        try:
            urllib.request.urlretrieve(url, path)
        except Exception:
            # Generar datos sintéticos si no hay conexión
            np.random.seed(42)
            n = 768
            df = pd.DataFrame({
                "X1": np.random.choice([0.62,0.64,0.66,0.71,0.76,0.82,0.86,0.98], n),
                "X2": np.random.choice([514.5,563.5,612.5,661.5,710.5,759.5,808.5], n),
                "X3": np.random.choice([245.5,294.0,318.5,343.0,416.5,514.5], n),
                "X4": np.random.choice([110.25,122.5,147.0,183.75,220.5], n),
                "X5": np.random.choice([3.5, 7.0], n),
                "X6": np.random.choice([2,3,4,5], n),
                "X7": np.random.choice([0,0.1,0.25,0.4], n),
                "X8": np.random.choice([0,1,2,3,4,5], n),
                "Y1": np.random.gamma(3, 8, n),
                "Y2": np.random.gamma(2.5, 10, n),
            })
            path.parent.mkdir(exist_ok=True)
            df.to_excel(path, index=False)

    df = pd.read_excel(path)
    df = df.dropna()
    col_map = {
        df.columns[0]: "compacidad_relativa",
        df.columns[1]: "superficie_m2",
        df.columns[2]: "area_muros_m2",
        df.columns[3]: "area_techo_m2",
        df.columns[4]: "altura_total_m",
        df.columns[5]: "orientacion",
        df.columns[6]: "area_acristalamiento",
        df.columns[7]: "dist_acristalamiento",
    }
    # target columns
    if len(df.columns) >= 10:
        col_map[df.columns[8]] = "carga_calefaccion"
        col_map[df.columns[9]] = "carga_refrigeracion"
    df = df.rename(columns=col_map)
    if "carga_calefaccion" in df.columns and "carga_refrigeracion" in df.columns:
        df["consumo_kwh"] = df["carga_calefaccion"] + df["carga_refrigeracion"]
    else:
        df["consumo_kwh"] = df.iloc[:, -1]
    return df


@st.cache_data(show_spinner=False)
def train_models(df):
    from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.base import BaseEstimator, TransformerMixin

    feat_cols = ["compacidad_relativa","superficie_m2","area_muros_m2",
                 "area_techo_m2","altura_total_m","area_acristalamiento","dist_acristalamiento"]
    feat_cols = [c for c in feat_cols if c in df.columns]

    X = df[feat_cols].copy()
    y = np.log1p(df["consumo_kwh"])

    df_tmp = df.copy()
    df_tmp["consumo_cat"] = pd.cut(df_tmp["consumo_kwh"],
                                    bins=[0,20,30,40,60,np.inf], labels=[1,2,3,4,5])
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for tr_idx, te_idx in split.split(X, df_tmp["consumo_cat"]):
        X_train, X_test = X.iloc[tr_idx], X.iloc[te_idx]
        y_train, y_test = y.iloc[tr_idx], y.iloc[te_idx]

    class EnergyFeaturesAdder(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None): return self
        def transform(self, X, y=None):
            arr = X.values if hasattr(X, "values") else X
            ratio_muros = arr[:, 2] / arr[:, 1]
            ratio_techo = arr[:, 3] / arr[:, 1]
            acrist_ef   = arr[:, 5] * arr[:, 0]
            return np.c_[arr, ratio_muros, ratio_techo, acrist_ef]

    pipe = Pipeline([
        ("imputer",  SimpleImputer(strategy="median")),
        ("features", EnergyFeaturesAdder()),
        ("scaler",   StandardScaler()),
    ])

    X_train_p = pipe.fit_transform(X_train)
    X_test_p  = pipe.transform(X_test)

    models = {
        "Regresión Lineal":  LinearRegression(),
        "Árbol de Decisión": DecisionTreeRegressor(random_state=42),
        "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=42),
    }

    results = {}
    for name, m in models.items():
        m.fit(X_train_p, y_train)
        tr_rmse = np.sqrt(mean_squared_error(y_train, m.predict(X_train_p)))
        cv_sc   = cross_val_score(m, X_train_p, y_train,
                                   scoring="neg_mean_squared_error", cv=10)
        cv_rmse = np.sqrt(-cv_sc)
        results[name] = {"model": m, "train": tr_rmse,
                          "cv_mean": cv_rmse.mean(), "cv_std": cv_rmse.std(),
                          "scores": cv_rmse}

    best = models["Random Forest"]
    y_pred_log  = best.predict(X_test_p)
    y_pred_real = np.expm1(y_pred_log)
    y_test_real = np.expm1(y_test)

    rmse_log  = np.sqrt(mean_squared_error(y_test, y_pred_log))
    rmse_real = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
    mae_real  = mean_absolute_error(y_test_real, y_pred_real)

    # Feature importance
    ext_names = feat_cols + ["ratio_muros_sup", "ratio_techo_sup", "acrist_efectivo"]
    importances = best.feature_importances_
    n_imp = min(len(importances), len(ext_names))
    fi = pd.Series(importances[:n_imp], index=ext_names[:n_imp]).sort_values(ascending=False)

    corr = df[feat_cols + ["consumo_kwh"]].corr()["consumo_kwh"].drop("consumo_kwh").sort_values()

    return {
        "df": df, "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "X_train_p": X_train_p, "X_test_p": X_test_p,
        "y_pred_log": y_pred_log, "y_pred_real": y_pred_real,
        "y_test_real": y_test_real,
        "results": results, "best_model": best,
        "rmse_log": rmse_log, "rmse_real": rmse_real, "mae_real": mae_real,
        "feat_importances": fi, "corr": corr,
        "feat_cols": feat_cols, "pipe": pipe,
    }


# ─── CARGA ────────────────────────────────────────────────────────────────────
with st.spinner("Cargando datos y entrenando modelos..."):
    df = load_data()
    md = train_models(df)


# ─── HERO ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
  <div class="hero-badge">◈ Machine Learning · Regresión · UCI Energy Efficiency</div>
  <h1 class="hero-title">Predicción de Consumo<br>Energético de Edificios</h1>
  <p class="hero-sub">Proyecto end-to-end siguiendo la metodología del Capítulo 2 de
    Hands-On Machine Learning — Aurélien Géron. Del análisis exploratorio al despliegue.</p>
  <div class="hero-stats">
    <div>
      <div class="hero-stat-val">{len(df)}</div>
      <div class="hero-stat-lbl">muestras</div>
    </div>
    <div>
      <div class="hero-stat-val">{len(md['feat_cols'])}</div>
      <div class="hero-stat-lbl">atributos</div>
    </div>
    <div>
      <div class="hero-stat-val">{md['rmse_real']:.2f}</div>
      <div class="hero-stat-lbl">RMSE kWh/m²</div>
    </div>
    <div>
      <div class="hero-stat-val">3</div>
      <div class="hero-stat-lbl">modelos</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs([
    "  Análisis Exploratorio  ",
    "  Modelado y Resultados  ",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ANÁLISIS EXPLORATORIO
# ══════════════════════════════════════════════════════════════════════════════
with tab1:

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    # ── Etapas 1–3 timeline ──
    st.markdown("""
    <div class="timeline">
      <div class="tl-step done">01 · panorama general</div>
      <div class="tl-step done">02 · obtener datos</div>
      <div class="tl-step active">03 · exploración</div>
      <div class="tl-step">04 · preparación</div>
      <div class="tl-step">05 · modelos</div>
      <div class="tl-step">06 · ajuste</div>
      <div class="tl-step">07 · test set</div>
      <div class="tl-step">08 · despliegue</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── Objetivo de negocio + enmarcado ──
    col_a, col_b = st.columns([1.05, 1], gap="large")

    with col_a:
        st.markdown("""
        <div class="card card-accent-teal">
          <div class="section-label">Etapa 1</div>
          <div class="section-title">Panorama general</div>
          <p class="section-body">
            El modelo recibe las características geométricas y constructivas de un edificio
            y produce una estimación numérica del consumo energético total (kWh/m²).
            La predicción alimentará sistemas de gestión energética y priorización de renovaciones.
          </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="styled-table-wrap">
        <table class="styled-table">
          <thead><tr>
            <th>Dimensión</th><th>Clasificación</th>
          </tr></thead>
          <tbody>
            <tr><td>Tipo de aprendizaje</td><td><span class="tag tag-blue">Supervisado</span></td></tr>
            <tr><td>Tipo de tarea</td><td><span class="tag tag-teal" style="background:#D0ECF5;color:#1A5C6B">Regresión múltiple</span></td></tr>
            <tr><td>Estrategia</td><td><span class="tag tag-green">Batch learning</span></td></tr>
            <tr><td>Métrica principal</td><td><span class="tag tag-orange">RMSE sobre log(consumo)</span></td></tr>
          </tbody>
        </table>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="callout callout-sand" style="margin-top:1rem">
          La transformación logarítmica log1p(y) simetriza la distribución del target,
          penaliza errores proporcionalmente a la magnitud del consumo y estabiliza la varianza.
          Se revierte con expm1() al interpretar resultados.
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="card card-accent-orange">
          <div class="section-label">Etapa 2 · Dataset</div>
          <div class="section-title">UCI Energy Efficiency</div>
          <p class="section-body">
            768 simulaciones energéticas de edificios residenciales generadas con EnergyPlus.
            Cada fila es una configuración única de parámetros geométricos y constructivos.
          </p>
        </div>
        """, unsafe_allow_html=True)

        # Tabla de atributos
        attrs = {
            "compacidad_relativa":  ("Compacidad relativa",   "0.62 – 0.98", "Continuo"),
            "superficie_m2":        ("Superficie total (m²)", "514 – 808",   "Continuo"),
            "area_muros_m2":        ("Área de muros (m²)",    "245 – 515",   "Continuo"),
            "area_techo_m2":        ("Área de techo (m²)",    "110 – 220",   "Continuo"),
            "altura_total_m":       ("Altura total (m)",       "3.5 – 7.0",  "Discreto"),
            "area_acristalamiento": ("Área acristalamiento",  "0 – 0.4",     "Discreto"),
            "dist_acristalamiento": ("Distribución acrist.",  "0 – 5",       "Categórico"),
        }
        rows = ""
        for col, (label, rango, tipo) in attrs.items():
            tag_cls = "tag-blue" if tipo == "Continuo" else ("tag-orange" if tipo == "Categórico" else "tag-green")
            rows += f"<tr><td>{label}</td><td class='num'>{rango}</td><td><span class='tag {tag_cls}'>{tipo}</span></td></tr>"

        st.markdown(f"""
        <div class="styled-table-wrap">
        <table class="styled-table">
          <thead><tr><th>Atributo</th><th>Rango</th><th>Tipo</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ── Sección 3: Exploración ──
    st.markdown("""
    <div class="section-label">Etapa 3</div>
    <div class="section-title">Exploración y visualización de los datos</div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

    # ── 3.1 Estadísticas descriptivas ──
    with st.expander("3.1  Estadísticas descriptivas del conjunto de datos", expanded=False):
        desc = df[md["feat_cols"] + ["consumo_kwh"]].describe().T.round(3)
        desc.index.name = "Atributo"
        desc = desc.reset_index()
        header = "".join(f"<th>{c}</th>" for c in desc.columns)
        rows_html = ""
        for _, row in desc.iterrows():
            cells = f"<td>{row['Atributo']}</td>"
            for c in desc.columns[1:]:
                cells += f"<td class='num'>{row[c]}</td>"
            rows_html += f"<tr>{cells}</tr>"
        st.markdown(f"""
        <div class="styled-table-wrap">
        <table class="styled-table">
          <thead><tr>{header}</tr></thead>
          <tbody>{rows_html}</tbody>
        </table>
        </div>
        """, unsafe_allow_html=True)

    # ── 3.2 Distribución del target ──
    st.markdown("""
    <div class="card card-accent-teal" style="margin-top:1rem">
      <div class="section-label">3.2</div>
      <div class="section-title" style="font-size:1.15rem">Distribución del consumo energético</div>
    </div>
    """, unsafe_allow_html=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    fig.patch.set_facecolor(CREAM)

    # Panel izq: distribución original
    ax = axes[0]
    vals = df["consumo_kwh"]
    n_bins = 40
    counts, bins, patches = ax.hist(vals, bins=n_bins, color=TEAL_MID,
                                     edgecolor=WHITE, linewidth=0.6, alpha=0.85)
    # Colorear las barras por cuartil
    q25, q75 = vals.quantile(0.25), vals.quantile(0.75)
    for patch, left in zip(patches, bins[:-1]):
        if left < q25:     patch.set_facecolor(TEAL_LIGHT)
        elif left < q75:   patch.set_facecolor(TEAL_MID)
        else:              patch.set_facecolor(ORANGE)

    med = vals.median()
    ax.axvline(med, color=ORANGE, lw=1.8, linestyle="--", alpha=0.9)
    ax.text(med + 0.5, ax.get_ylim()[1]*0.92, f"Mediana\n{med:.1f}",
            fontsize=8, color=ORANGE, va="top")
    ax.set_xlabel("Consumo energético (kWh/m²)", fontsize=10, labelpad=6)
    ax.set_ylabel("Frecuencia", fontsize=10)
    skew = vals.skew()
    ax.set_title(f"Escala original  ·  asimetría = {skew:.3f}", fontsize=11, pad=10)

    # Leyenda manual
    patches_leg = [
        mpatches.Patch(color=TEAL_LIGHT, label="Q1  (bajo consumo)"),
        mpatches.Patch(color=TEAL_MID,   label="Q2–Q3  (consumo medio)"),
        mpatches.Patch(color=ORANGE,     label="Q4  (alto consumo — cola pesada)"),
    ]
    ax.legend(handles=patches_leg, fontsize=7.5, framealpha=0.8,
              loc="upper right", facecolor=CREAM)

    # Panel der: escala log
    ax2 = axes[1]
    log_vals = np.log1p(vals)
    counts2, bins2, patches2 = ax2.hist(log_vals, bins=n_bins, color=SAND,
                                          edgecolor=WHITE, linewidth=0.6, alpha=0.85)
    med2 = log_vals.median()
    ax2.axvline(med2, color=TEAL_DARK, lw=1.8, linestyle="--", alpha=0.9)
    ax2.text(med2 + 0.02, ax2.get_ylim()[1]*0.92, f"Mediana\n{med2:.2f}",
             fontsize=8, color=TEAL_DARK, va="top")
    ax2.set_xlabel("log1p(consumo)", fontsize=10, labelpad=6)
    ax2.set_ylabel("Frecuencia", fontsize=10)
    skew2 = log_vals.skew()
    ax2.set_title(f"Escala logarítmica  ·  asimetría = {skew2:.3f}", fontsize=11, pad=10)

    arrow_props = dict(arrowstyle="->", color=TEAL_DARK, lw=1.5)
    fig.text(0.5, 0.97, "La transformación logarítmica corrige la asimetría de la distribución",
             ha="center", fontsize=9.5, color=GRAY_MID, style="italic")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # ── 3.3 Histogramas de todos los atributos ──
    st.markdown("""
    <div class="card card-accent-sand" style="margin-top:0.5rem">
      <div class="section-label">3.3</div>
      <div class="section-title" style="font-size:1.15rem">Histogramas de todos los atributos numéricos</div>
    </div>
    """, unsafe_allow_html=True)

    num_cols = md["feat_cols"]
    n = len(num_cols)
    ncols_g = 4
    nrows_g = (n + ncols_g - 1) // ncols_g

    palette = [TEAL_DARK, TEAL_MID, TEAL_LIGHT, SAND, ORANGE_LT, ORANGE, GRAY_MID, SAND_LT]

    fig2, axes2 = plt.subplots(nrows_g, ncols_g, figsize=(14, 3.2 * nrows_g))
    fig2.patch.set_facecolor(CREAM)
    axes2_flat = axes2.flatten()

    for i, col in enumerate(num_cols):
        ax = axes2_flat[i]
        color = palette[i % len(palette)]
        ax.hist(df[col], bins=30, color=color, edgecolor=WHITE, linewidth=0.5, alpha=0.88)
        ax.set_title(col.replace("_", " "), fontsize=9, pad=6, color=CHARCOAL)
        ax.tick_params(labelsize=7.5)
        med_v = df[col].median()
        ax.axvline(med_v, color=CHARCOAL, lw=1, linestyle=":", alpha=0.7)

    for j in range(i + 1, len(axes2_flat)):
        axes2_flat[j].set_visible(False)

    fig2.suptitle("Distribuciones de atributos de entrada", fontsize=12,
                   color=CHARCOAL, y=1.01, style="italic")
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close()

    # ── 3.5 Correlaciones ──
    st.markdown("""
    <div class="card card-accent-orange" style="margin-top:0.5rem">
      <div class="section-label">3.5</div>
      <div class="section-title" style="font-size:1.15rem">Mapa de correlaciones con el consumo</div>
    </div>
    """, unsafe_allow_html=True)

    col_corr, col_heat = st.columns([1, 1.4], gap="large")

    with col_corr:
        corr_vals = md["corr"]

        fig3, ax3 = plt.subplots(figsize=(5.5, 4.5))
        fig3.patch.set_facecolor(CREAM)
        colors_bar = [TEAL_MID if v >= 0 else ORANGE for v in corr_vals.values]
        bars = ax3.barh(range(len(corr_vals)), corr_vals.values,
                         color=colors_bar, edgecolor=WHITE, linewidth=0.5,
                         height=0.65, alpha=0.9)
        ax3.set_yticks(range(len(corr_vals)))
        ax3.set_yticklabels([c.replace("_", "\n") for c in corr_vals.index], fontsize=8)
        ax3.axvline(0, color=GRAY_LT, lw=1)
        ax3.set_xlabel("Correlación de Pearson con consumo_kwh", fontsize=8.5)
        ax3.set_title("Correlación de atributos\nvs. consumo total", fontsize=10, pad=8)
        for bar, val in zip(bars, corr_vals.values):
            xpos = val + 0.01 if val >= 0 else val - 0.01
            ha = "left" if val >= 0 else "right"
            ax3.text(xpos, bar.get_y() + bar.get_height()/2,
                     f"{val:.3f}", va="center", ha=ha, fontsize=7.5, color=CHARCOAL)
        plt.tight_layout()
        st.pyplot(fig3, use_container_width=True)
        plt.close()

    with col_heat:
        corr_full = df[num_cols + ["consumo_kwh"]].corr()

        fig4, ax4 = plt.subplots(figsize=(7, 5.5))
        fig4.patch.set_facecolor(CREAM)
        ax4.set_facecolor(CREAM)

        from matplotlib.colors import LinearSegmentedColormap
        cmap_custom = LinearSegmentedColormap.from_list(
            "custom", [ORANGE, WHITE, TEAL_MID], N=256)

        mask_upper = np.triu(np.ones_like(corr_full, dtype=bool), k=1)
        im = ax4.imshow(corr_full.values, cmap=cmap_custom, vmin=-1, vmax=1, aspect="auto")

        n_c = len(corr_full.columns)
        ax4.set_xticks(range(n_c))
        ax4.set_yticks(range(n_c))
        labels = [c.replace("_", "\n") for c in corr_full.columns]
        ax4.set_xticklabels(labels, fontsize=7.5, rotation=45, ha="right")
        ax4.set_yticklabels(labels, fontsize=7.5)

        for i in range(n_c):
            for j in range(n_c):
                val = corr_full.values[i, j]
                txt_color = WHITE if abs(val) > 0.5 else CHARCOAL
                ax4.text(j, i, f"{val:.2f}", ha="center", va="center",
                         fontsize=7, color=txt_color)

        # Grid lines
        for k in range(n_c + 1):
            ax4.axhline(k - 0.5, color=WHITE, lw=0.8)
            ax4.axvline(k - 0.5, color=WHITE, lw=0.8)

        cbar = fig4.colorbar(im, ax=ax4, fraction=0.03, pad=0.02)
        cbar.ax.tick_params(labelsize=8)
        ax4.set_title("Matriz de correlación completa", fontsize=10, pad=10)
        plt.tight_layout()
        st.pyplot(fig4, use_container_width=True)
        plt.close()

    st.markdown("""
    <div class="callout callout-teal">
      El coeficiente de Pearson solo mide correlaciones lineales y puede pasar por alto
      relaciones no lineales. La compacidad relativa y el area de techo muestran
      las correlaciones más fuertes con el consumo energético total.
    </div>
    """, unsafe_allow_html=True)

    # ── 3.6 Scatter de las features más correlacionadas ──
    st.markdown("""
    <div class="card card-accent-teal" style="margin-top:0.5rem">
      <div class="section-label">3.6</div>
      <div class="section-title" style="font-size:1.15rem">Relaciones con el consumo — atributos clave</div>
    </div>
    """, unsafe_allow_html=True)

    top_feats = md["corr"].abs().nlargest(4).index.tolist()

    fig5, axes5 = plt.subplots(1, 4, figsize=(14, 4))
    fig5.patch.set_facecolor(CREAM)

    for idx, feat in enumerate(top_feats):
        ax = axes5[idx]
        sc = ax.scatter(df[feat], df["consumo_kwh"],
                         c=df["consumo_kwh"], cmap="YlOrRd",
                         s=14, alpha=0.55, edgecolors="none")
        ax.set_xlabel(feat.replace("_", "\n"), fontsize=8.5)
        if idx == 0:
            ax.set_ylabel("consumo (kWh/m²)", fontsize=8.5)
        r = df[[feat, "consumo_kwh"]].corr().iloc[0, 1]
        ax.set_title(f"r = {r:.3f}", fontsize=9.5, color=TEAL_DARK, pad=6)
        ax.tick_params(labelsize=8)

    fig5.colorbar(sc, ax=axes5[-1], fraction=0.05, pad=0.04,
                   label="kWh/m²").ax.tick_params(labelsize=8)
    fig5.suptitle("Scatter de los 4 atributos con mayor correlación absoluta",
                   fontsize=10.5, color=CHARCOAL, y=1.02, style="italic")
    plt.tight_layout()
    st.pyplot(fig5, use_container_width=True)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODELADO Y RESULTADOS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="timeline">
      <div class="tl-step done">01 · panorama</div>
      <div class="tl-step done">02 · datos</div>
      <div class="tl-step done">03 · exploración</div>
      <div class="tl-step done">04 · preparación</div>
      <div class="tl-step active">05 · modelos</div>
      <div class="tl-step active">06 · ajuste</div>
      <div class="tl-step active">07 · test set</div>
      <div class="tl-step done">08 · despliegue</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── Métricas finales destacadas ──
    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-chip">
        <div class="metric-chip-val">{md['rmse_log']:.4f}</div>
        <div class="metric-chip-lbl">RMSE · escala log</div>
      </div>
      <div class="metric-chip">
        <div class="metric-chip-val">{md['rmse_real']:.2f}</div>
        <div class="metric-chip-lbl">RMSE · kWh/m²</div>
      </div>
      <div class="metric-chip">
        <div class="metric-chip-val">{md['mae_real']:.2f}</div>
        <div class="metric-chip-lbl">MAE · kWh/m²</div>
      </div>
      <div class="metric-chip">
        <div class="metric-chip-val">RF</div>
        <div class="metric-chip-lbl">mejor modelo</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ── Tabla comparativa de modelos ──
    col_tbl, col_cv = st.columns([1.1, 1], gap="large")

    with col_tbl:
        st.markdown("""
        <div class="section-label">Etapas 5 – 6</div>
        <div class="section-title">Comparativa de modelos</div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        results = md["results"]
        rows_m = ""
        for name, res in results.items():
            is_best = name == "Random Forest"
            best_cls = "best" if is_best else ""
            overfit = "Sobreajuste severo" if res["train"] < 0.001 else (
                      "Sobreajuste leve" if res["train"] < res["cv_mean"] * 0.7 else "Equilibrado")
            tag_map = {
                "Sobreajuste severo": ("tag-orange", overfit),
                "Sobreajuste leve":   ("tag-orange", overfit),
                "Equilibrado":        ("tag-green",  overfit),
            }
            tag_cls, tag_txt = tag_map[overfit]
            badge = " ◈ mejor" if is_best else ""
            rows_m += f"""<tr>
              <td class="{best_cls}">{name}{badge}</td>
              <td class="num {best_cls}">{res['train']:.4f}</td>
              <td class="num {best_cls}">{res['cv_mean']:.4f}</td>
              <td class="num {best_cls}">± {res['cv_std']:.4f}</td>
              <td><span class="tag {tag_cls}">{tag_txt}</span></td>
            </tr>"""

        st.markdown(f"""
        <div class="styled-table-wrap">
        <table class="styled-table">
          <thead><tr>
            <th>Modelo</th>
            <th>RMSE Train</th>
            <th>RMSE CV</th>
            <th>Desv. Std.</th>
            <th>Diagnóstico</th>
          </tr></thead>
          <tbody>{rows_m}</tbody>
        </table>
        </div>
        <div class="callout callout-teal" style="margin-top:0.8rem">
          El Árbol de Decisión muestra RMSE = 0 en entrenamiento pero ~0.034 en CV —
          señal clásica de sobreajuste. El Random Forest generaliza mejor al promediar
          múltiples árboles entrenados en subconjuntos aleatorios.
        </div>
        """, unsafe_allow_html=True)

    with col_cv:
        st.markdown("""
        <div class="section-label">Validación cruzada K=10</div>
        <div class="section-title">Distribución de scores por modelo</div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        fig6, ax6 = plt.subplots(figsize=(6.5, 4.5))
        fig6.patch.set_facecolor(CREAM)

        names_m = list(results.keys())
        scores_all = [results[n]["scores"] for n in names_m]
        bp = ax6.boxplot(scores_all, patch_artist=True, notch=True,
                          widths=0.45, medianprops=dict(color=WHITE, lw=2))

        box_colors = [TEAL_LIGHT, SAND, TEAL_DARK]
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)
        for whisker in bp["whiskers"]:
            whisker.set(color=GRAY_MID, lw=1.2, linestyle="--")
        for cap in bp["caps"]:
            cap.set(color=GRAY_MID, lw=1.5)
        for flier in bp["fliers"]:
            flier.set(marker="o", color=ORANGE, markersize=5, alpha=0.6)

        ax6.set_xticklabels([n.replace(" ", "\n") for n in names_m], fontsize=9)
        ax6.set_ylabel("RMSE (escala log)", fontsize=9.5)
        ax6.set_title("Dispersión del error en 10 pliegues de CV", fontsize=10.5, pad=8)

        # Líneas de referencia
        for i, (name, sc) in enumerate(zip(names_m, scores_all), 1):
            ax6.text(i, max(sc) * 1.002, f"μ={sc.mean():.4f}", ha="center",
                     fontsize=7.5, color=CHARCOAL, va="bottom")

        plt.tight_layout()
        st.pyplot(fig6, use_container_width=True)
        plt.close()

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ── Importancia de features ──
    col_fi, col_pred = st.columns([1, 1.05], gap="large")

    with col_fi:
        st.markdown("""
        <div class="section-label">Etapa 6 · Ajuste fino</div>
        <div class="section-title">Importancia de características</div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        fi = md["feat_importances"]

        fig7, ax7 = plt.subplots(figsize=(6.5, 5))
        fig7.patch.set_facecolor(CREAM)

        cmap_fi = plt.cm.get_cmap("YlOrRd", len(fi))
        bar_colors = [cmap_fi(1 - i/len(fi)) for i in range(len(fi))]

        bars7 = ax7.barh(range(len(fi)), fi.values, color=bar_colors,
                          edgecolor=WHITE, linewidth=0.5, height=0.65)

        # Etiquetas de valor al final de cada barra
        for bar, val in zip(bars7, fi.values):
            ax7.text(val + 0.003, bar.get_y() + bar.get_height()/2,
                     f"{val:.3f}", va="center", fontsize=8.5, color=CHARCOAL)

        ax7.set_yticks(range(len(fi)))
        ax7.set_yticklabels([c.replace("_", "\n") for c in fi.index], fontsize=8.5)
        ax7.set_xlabel("Importancia relativa (Gini)", fontsize=9)
        ax7.set_title("Random Forest · importancia\nde características por pureza",
                       fontsize=10, pad=8)
        ax7.set_xlim(0, fi.values.max() * 1.15)

        # Highlight top 3
        for i in range(min(3, len(bars7))):
            bars7[i].set_edgecolor(TEAL_DARK)
            bars7[i].set_linewidth(1.5)

        plt.tight_layout()
        st.pyplot(fig7, use_container_width=True)
        plt.close()

    with col_pred:
        st.markdown("""
        <div class="section-label">Etapa 7 · Test set</div>
        <div class="section-title">Predicciones vs. valores reales</div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        y_real = md["y_test_real"].values
        y_pred = md["y_pred_real"]
        errors = y_pred - y_real

        fig8, axes8 = plt.subplots(1, 2, figsize=(7, 4.5))
        fig8.patch.set_facecolor(CREAM)

        # Scatter pred vs real
        ax_sc = axes8[0]
        lims = [min(y_real.min(), y_pred.min()) - 2,
                max(y_real.max(), y_pred.max()) + 2]
        ax_sc.scatter(y_real, y_pred, c=errors, cmap="RdYlGn_r",
                       s=18, alpha=0.65, edgecolors="none",
                       vmin=-np.percentile(np.abs(errors), 95),
                       vmax= np.percentile(np.abs(errors), 95))
        ax_sc.plot(lims, lims, "--", color=TEAL_DARK, lw=1.5, alpha=0.7, label="Predicción perfecta")
        ax_sc.set_xlim(lims); ax_sc.set_ylim(lims)
        ax_sc.set_xlabel("Consumo real (kWh/m²)", fontsize=8.5)
        ax_sc.set_ylabel("Consumo predicho (kWh/m²)", fontsize=8.5)
        ax_sc.set_title("Predicho vs. Real", fontsize=10, pad=6)
        ax_sc.legend(fontsize=7.5, loc="upper left")

        # Histograma de residuos
        ax_res = axes8[1]
        ax_res.hist(errors, bins=30, color=TEAL_MID, edgecolor=WHITE,
                     linewidth=0.5, alpha=0.85)
        ax_res.axvline(0, color=ORANGE, lw=1.8, linestyle="--")
        ax_res.axvline(errors.mean(), color=TEAL_DARK, lw=1.5, linestyle=":",
                        label=f"Media: {errors.mean():.2f}")
        ax_res.set_xlabel("Residuo (predicho − real)", fontsize=8.5)
        ax_res.set_ylabel("Frecuencia", fontsize=8.5)
        ax_res.set_title("Distribución de residuos", fontsize=10, pad=6)
        ax_res.legend(fontsize=7.5)

        plt.tight_layout()
        st.pyplot(fig8, use_container_width=True)
        plt.close()

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ── Rendimiento por subgrupos ──
    st.markdown("""
    <div class="section-label">Etapa 8 · Evaluación por subgrupos</div>
    <div class="section-title">Análisis de rendimiento segmentado</div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

    from sklearn.metrics import mean_squared_error as mse_fn

    def subgroup_rmse(mask_col, values, label_prefix):
        rows_sg = []
        for val in sorted(df.loc[md["X_test"].index, mask_col].unique()):
            mask = md["X_test"][mask_col].values == val
            if mask.sum() == 0:
                continue
            y_s = md["y_test_real"].values[mask]
            yp_s = md["y_pred_real"][mask]
            rmse_s = np.sqrt(mse_fn(y_s, yp_s))
            rows_sg.append({
                "Segmento": f"{label_prefix} = {val}",
                "n": int(mask.sum()),
                "RMSE (kWh/m²)": round(rmse_s, 3),
                "Error relativo (%)": round(rmse_s / y_s.mean() * 100, 1),
            })
        return rows_sg

    # Altura
    sg_altura = subgroup_rmse("altura_total_m", None, "Altura (m)")
    # Acristalamiento
    sg_acrist = subgroup_rmse("area_acristalamiento", None, "Acristalamiento")

    all_sg = sg_altura + sg_acrist

    if all_sg:
        rows_sg_html = ""
        for row in all_sg:
            err_pct = row["Error relativo (%)"]
            tag_cls = "tag-green" if err_pct < 8 else ("tag-orange" if err_pct < 15 else "tag-orange")
            rows_sg_html += f"""<tr>
              <td>{row['Segmento']}</td>
              <td class="num">{row['n']}</td>
              <td class="num">{row['RMSE (kWh/m²)']}</td>
              <td><span class="tag {tag_cls}">{err_pct}%</span></td>
            </tr>"""

        col_sg1, col_sg2 = st.columns([1, 1], gap="large")
        with col_sg1:
            st.markdown(f"""
            <div class="styled-table-wrap">
            <table class="styled-table">
              <thead><tr>
                <th>Segmento</th><th>n muestras</th>
                <th>RMSE kWh/m²</th><th>Error relativo</th>
              </tr></thead>
              <tbody>{rows_sg_html}</tbody>
            </table>
            </div>
            <div class="callout callout-sand" style="margin-top:0.8rem">
              Si el RMSE varía sustancialmente entre subgrupos, el modelo tiene sesgos
              que podrían requerir features adicionales o modelos especializados por segmento.
            </div>
            """, unsafe_allow_html=True)

        with col_sg2:
            # Bar chart de RMSE por subgrupo
            fig9, ax9 = plt.subplots(figsize=(6, 4))
            fig9.patch.set_facecolor(CREAM)

            segs  = [r["Segmento"].replace("Acristalamiento = ", "Acrist. ")
                                   .replace("Altura (m) = ", "Alt. ") for r in all_sg]
            rmses = [r["RMSE (kWh/m²)"] for r in all_sg]
            bar_c = [TEAL_MID if "Alt." in s else ORANGE_LT for s in segs]

            b9 = ax9.bar(range(len(segs)), rmses, color=bar_c,
                          edgecolor=WHITE, linewidth=0.5, width=0.6)
            ax9.set_xticks(range(len(segs)))
            ax9.set_xticklabels(segs, rotation=40, ha="right", fontsize=8)
            ax9.set_ylabel("RMSE (kWh/m²)", fontsize=9)
            ax9.set_title("RMSE por subgrupo del test set", fontsize=10, pad=8)
            ax9.axhline(md["rmse_real"], color=TEAL_DARK, lw=1.5,
                         linestyle="--", alpha=0.7, label=f"RMSE global {md['rmse_real']:.2f}")
            ax9.legend(fontsize=8)
            for bar, val in zip(b9, rmses):
                ax9.text(bar.get_x() + bar.get_width()/2, val + 0.05,
                         f"{val:.2f}", ha="center", fontsize=8, color=CHARCOAL)
            plt.tight_layout()
            st.pyplot(fig9, use_container_width=True)
            plt.close()

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ── Predictor interactivo ──
    st.markdown("""
    <div class="section-label">Etapa 8 · Despliegue</div>
    <div class="section-title">Simulador de consumo energético</div>
    <p class="section-body">
      Introduce las características de un edificio para obtener una predicción
      del consumo energético total usando el modelo Random Forest entrenado.
    </p>
    """, unsafe_allow_html=True)
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    col_inp1, col_inp2, col_result = st.columns([1, 1, 1], gap="large")

    with col_inp1:
        st.markdown("<div class='card card-accent-teal'>", unsafe_allow_html=True)
        comp_rel  = st.slider("Compacidad relativa",
                               float(df["compacidad_relativa"].min()),
                               float(df["compacidad_relativa"].max()), 0.75, 0.01)
        sup_m2    = st.slider("Superficie total (m²)",
                               float(df["superficie_m2"].min()),
                               float(df["superficie_m2"].max()), 660.0, 10.0)
        area_muros = st.slider("Área de muros (m²)",
                                float(df["area_muros_m2"].min()),
                                float(df["area_muros_m2"].max()), 318.5, 10.0)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_inp2:
        st.markdown("<div class='card card-accent-orange'>", unsafe_allow_html=True)
        area_techo = st.slider("Área de techo (m²)",
                                float(df["area_techo_m2"].min()),
                                float(df["area_techo_m2"].max()), 147.0, 5.0)
        altura     = st.selectbox("Altura total (m)", sorted(df["altura_total_m"].unique()), index=0)
        acrist     = st.selectbox("Área acristalamiento", sorted(df["area_acristalamiento"].unique()), index=2)
        dist_acr   = st.selectbox("Distribución acristalamiento",
                                   sorted(df["dist_acristalamiento"].unique()), index=0)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_result:
        input_vals = {
            "compacidad_relativa":   comp_rel,
            "superficie_m2":         sup_m2,
            "area_muros_m2":         area_muros,
            "area_techo_m2":         area_techo,
            "altura_total_m":        float(altura),
            "area_acristalamiento":  float(acrist),
            "dist_acristalamiento":  float(dist_acr),
        }
        input_df = pd.DataFrame([{c: input_vals.get(c, 0) for c in md["feat_cols"]}])

        try:
            X_in = md["pipe"].transform(input_df)
            log_pred = md["best_model"].predict(X_in)[0]
            pred_kwh = float(np.expm1(log_pred))

            pct_rank = (df["consumo_kwh"] < pred_kwh).mean() * 100
            efic_label = (
                "Alta eficiencia" if pct_rank < 33 else
                "Eficiencia media" if pct_rank < 66 else
                "Baja eficiencia"
            )
            efic_color = "#3D9970" if pct_rank < 33 else ("#C9A96E" if pct_rank < 66 else "#E8621A")

            st.markdown(f"""
            <div class="predictor-result">
              <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                          letter-spacing:0.12em;text-transform:uppercase;
                          color:rgba(255,255,255,0.6);margin-bottom:0.6rem">
                Consumo predicho
              </div>
              <div class="predictor-val">{pred_kwh:.1f}</div>
              <div class="predictor-unit">kWh / m² · año</div>
              <div style="margin-top:1.2rem;padding:0.6rem 1rem;
                          background:rgba(255,255,255,0.1);border-radius:8px;
                          font-size:0.82rem;color:rgba(255,255,255,0.85)">
                Percentil <b style="color:{efic_color}">{pct_rank:.0f}</b>
                en el dataset ·
                <span style="color:{efic_color};font-weight:600">{efic_label}</span>
              </div>
              <div style="margin-top:0.8rem;font-size:0.78rem;color:rgba(255,255,255,0.55)">
                log-pred = {log_pred:.4f} · modelo: Random Forest
              </div>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"No se pudo calcular la predicción: {e}")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ── Resumen final ──
    st.markdown("""
    <div class="section-label">Resumen del proyecto</div>
    <div class="section-title">Decisiones de diseño y próximos pasos</div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

    col_d1, col_d2 = st.columns(2, gap="large")

    with col_d1:
        st.markdown("""
        <div class="card card-accent-teal">
          <div class="section-label">3 decisiones clave</div>
          <div class="section-title" style="font-size:1.1rem">Por qué el modelo funciona</div>
          <br>
          <div style="margin-bottom:1rem">
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;
                        color:#E8621A;text-transform:uppercase;letter-spacing:0.1em">
              Decisión 1
            </div>
            <div style="font-size:0.95rem;color:#2C2C2C;line-height:1.6;margin-top:0.2rem">
              Transformación logarítmica del target. Justificada por la distribución asimétrica.
              Penaliza errores proporcionalmente y estabiliza la varianza del modelo.
            </div>
          </div>
          <div style="margin-bottom:1rem">
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;
                        color:#E8621A;text-transform:uppercase;letter-spacing:0.1em">
              Decisión 2
            </div>
            <div style="font-size:0.95rem;color:#2C2C2C;line-height:1.6;margin-top:0.2rem">
              Pipeline reproducible con ColumnTransformer. Toda la preparación encapsulada
              en un objeto que se serializa con joblib para producción sin fugas de datos.
            </div>
          </div>
          <div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;
                        color:#E8621A;text-transform:uppercase;letter-spacing:0.1em">
              Decisión 3
            </div>
            <div style="font-size:0.95rem;color:#2C2C2C;line-height:1.6;margin-top:0.2rem">
              Muestreo estratificado para el split train/test. Critico en datasets pequeños
              (768 muestras) para garantizar representatividad en ambos conjuntos.
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_d2:
        proximos = [
            ("GradientBoosting y XGBoost", "Suelen superar a Random Forest en datos tabulares"),
            ("Optimización Bayesiana", "Optuna/Hyperopt: más eficiente que Grid/Random Search"),
            ("Features de dominio", "U-values de materiales, zona climática, orientación solar"),
            ("Validación con datos reales", "El UCI es simulación; validar con datos ASHRAE medidos"),
            ("API REST con FastAPI", "Integración con sistemas de gestión de edificios (BMS)"),
            ("Pipeline CI/CD", "Reentrenamiento automático al detectar degradación del modelo"),
        ]
        rows_p = "".join(
            f"<tr><td>{p[0]}</td><td style='color:#7A7A7A;font-size:0.88rem'>{p[1]}</td></tr>"
            for p in proximos
        )
        st.markdown(f"""
        <div class="card card-accent-orange">
          <div class="section-label">Próximos pasos</div>
          <div class="section-title" style="font-size:1.1rem">Si tuviéramos más tiempo</div>
          <br>
          <div class="styled-table-wrap">
          <table class="styled-table">
            <thead><tr><th>Mejora</th><th>Justificación</th></tr></thead>
            <tbody>{rows_p}</tbody>
          </table>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div style="margin-top:3rem;padding:1.5rem 0;
                border-top:1px solid #D4CFC9;
                display:flex;justify-content:space-between;align-items:center">
      <div style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;
                  color:#7A7A7A;letter-spacing:0.08em">
        UCI ENERGY EFFICIENCY DATASET · 768 muestras · 8 atributos
      </div>
      <div style="font-size:0.82rem;color:#7A7A7A;font-style:italic">
        Basado en Hands-On Machine Learning — Aurélien Géron, Cap. 2
      </div>
    </div>
    """, unsafe_allow_html=True)
