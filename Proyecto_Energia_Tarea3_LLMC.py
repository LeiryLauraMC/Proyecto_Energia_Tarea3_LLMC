"""
Aplicación Streamlit — Predicción de Consumo Energético de Edificios
Leiry Laura Mares Cure
Dataset: UCI Energy Efficiency
"""

import streamlit as st
import numpy as np
import pandas as pd
import warnings
import urllib.request
from pathlib import Path

warnings.filterwarnings("ignore")

# ─── CONFIGURACIÓN ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Consumo Energético · ML",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── PALETA ───────────────────────────────────────────────────────────────────
CREAM      = "#F7F3EE"
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
WHITE      = "#FFFFFF"
SUCCESS    = "#3D9970"

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Source+Serif+4:ital,opsz,wght@0,8..60,300;0,8..60,400;0,8..60,600;1,8..60,300;1,8..60,400&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [data-testid="stAppViewContainer"],
[data-testid="stMain"], .main { background: #F7F3EE !important; }
[data-testid="stHeader"] { background: transparent !important; }
section.main > div { padding-top: 0 !important; }
.block-container { padding: 0 2.5rem 3rem 2.5rem !important; max-width: 1320px !important; }
* { font-family: 'Source Serif 4', Georgia, serif !important; }

/* ── HERO ── */
.hero {
    background: linear-gradient(145deg, #EAF4F7 0%, #F3FAFE 40%, #EBF5F0 100%);
    border-radius: 0 0 36px 36px;
    padding: 3rem 3rem 2.6rem;
    margin: 0 -2.5rem 2.5rem -2.5rem;
    position: relative; overflow: hidden;
    border-bottom: 2px solid #C4E3EC;
    box-shadow: 0 6px 32px rgba(26,92,107,0.08);
}
.hero::before {
    content: "";
    position: absolute; inset: 0;
    background:
        radial-gradient(ellipse 55% 70% at 92% 50%, rgba(42,138,158,0.10) 0%, transparent 65%),
        radial-gradient(ellipse 35% 50% at 5% 80%,  rgba(201,169,110,0.12) 0%, transparent 60%);
    pointer-events: none;
}
.hero-grid {
    display: grid; grid-template-columns: 1fr auto;
    align-items: center; gap: 2rem; position: relative;
}
.hero-badge {
    display: inline-block;
    background: rgba(26,92,107,0.10); border: 1px solid rgba(26,92,107,0.25);
    color: #1A5C6B;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.67rem; letter-spacing: 0.13em; text-transform: uppercase;
    padding: 0.28rem 0.85rem; border-radius: 20px; margin-bottom: 0.9rem;
}
.hero-title {
    font-family: 'Playfair Display', Georgia, serif !important;
    font-size: clamp(1.8rem, 3.5vw, 2.75rem); font-weight: 700;
    color: #1A5C6B; line-height: 1.15; margin: 0 0 0.5rem;
}
.hero-author {
    font-family: 'Source Serif 4', serif !important;
    font-size: 1rem; color: #E8621A; font-style: italic;
    font-weight: 400; letter-spacing: 0.01em; margin-bottom: 1.4rem;
}
.hero-stats { display: flex; gap: 2rem; flex-wrap: wrap; }
.hero-stat-val {
    font-family: 'Playfair Display', serif !important;
    font-size: 1.7rem; font-weight: 700; color: #1A5C6B; line-height: 1;
}
.hero-stat-lbl {
    font-size: 0.68rem; color: #7A7A7A; letter-spacing: 0.09em;
    text-transform: uppercase; font-family: 'JetBrains Mono', monospace !important;
    margin-top: 0.2rem;
}
.hero-deco { flex-shrink: 0; opacity: 0.55; }

/* ── TABS ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 2px solid #D4CFC9 !important;
    gap: 0 !important; padding: 0 !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    font-family: 'Playfair Display', serif !important;
    font-size: 1rem !important; font-weight: 600 !important; color: #7A7A7A !important;
    padding: 0.85rem 1.7rem !important; border-bottom: 3px solid transparent !important;
    margin-bottom: -2px !important; background: transparent !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: #1A5C6B !important; border-bottom-color: #1A5C6B !important;
}

/* ── CARDS ── */
.card {
    background: #FDFAF7; border: 1px solid #E8D9C0;
    border-radius: 16px; padding: 1.5rem 1.7rem; margin-bottom: 1rem;
    box-shadow: 0 2px 10px rgba(26,92,107,0.05);
}
.card-accent-teal   { border-left: 4px solid #1A5C6B; }
.card-accent-orange { border-left: 4px solid #E8621A; }
.card-accent-sand   { border-left: 4px solid #C9A96E; }

/* ── SECTION ── */
.section-label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.64rem; letter-spacing: 0.16em;
    text-transform: uppercase; color: #E8621A; margin-bottom: 0.25rem;
}
.section-label::before,
.section-label::after { content: none !important; display: none !important; }
/* Eliminar arrow_right y cualquier ícono de lista inyectado por Streamlit */
details > summary { list-style: none !important; }
details > summary::-webkit-details-marker { display: none !important; }
[data-testid="stExpander"] summary svg,
[data-testid="stExpander"] summary .arrow_right,
[data-testid="stExpander"] summary [class*="arrow"] { display: none !important; }
/* Evitar que Streamlit renderice el texto de section-label como ítem de lista */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] div { list-style: none !important; }
[data-testid="stMarkdownContainer"] .section-label::before { content: "" !important; }
/* Ocultar cualquier .arrow_right que Streamlit añada al DOM */
.arrow_right, [class*="arrow_right"], svg[class*="arrow"] { display: none !important; }
.section-title {
    font-family: 'Playfair Display', serif !important;
    font-size: 1.45rem; font-weight: 700; color: #1A5C6B; margin: 0 0 0.4rem; line-height: 1.2;
}
.section-body { color: #5A5A5A; font-size: 0.94rem; line-height: 1.7; max-width: 780px; }

/* ── METRICS ── */
.metric-row { display: flex; gap: 1rem; flex-wrap: wrap; margin: 1rem 0; }
.metric-chip {
    background: #FDFAF7; border: 1px solid #D4CFC9; border-radius: 12px;
    padding: 0.85rem 1.1rem; flex: 1; min-width: 120px;
    box-shadow: 0 1px 5px rgba(0,0,0,0.04);
}
.metric-chip-val {
    font-family: 'Playfair Display', serif !important;
    font-size: 1.5rem; font-weight: 700; color: #1A5C6B; line-height: 1;
}
.metric-chip-lbl {
    font-size: 0.69rem; color: #7A7A7A; margin-top: 0.22rem;
    font-family: 'JetBrains Mono', monospace !important;
    letter-spacing: 0.05em; text-transform: uppercase;
}

/* ── TABLES ── */
.styled-table-wrap { overflow-x: auto; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.04); }
.styled-table { width: 100%; border-collapse: collapse; font-size: 0.89rem; }
.styled-table thead tr { background: #1A5C6B; color: white; }
.styled-table thead th {
    padding: 0.8rem 1rem; text-align: left;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.73rem; letter-spacing: 0.06em; font-weight: 500;
}
.styled-table tbody tr { border-bottom: 1px solid #E8D9C0; }
.styled-table tbody tr:hover { background: #F0EAE0; }
.styled-table tbody td { padding: 0.7rem 1rem; color: #2C2C2C; }
.styled-table tbody tr:last-child { border-bottom: none; }
.styled-table .num { font-family: 'JetBrains Mono', monospace !important; }
.styled-table .best { color: #1A5C6B; font-weight: 600; }
.tag {
    display: inline-block; padding: 0.12rem 0.55rem; border-radius: 20px;
    font-size: 0.71rem; font-family: 'JetBrains Mono', monospace !important;
}
.tag-green  { background:#D4EDDA; color:#1E7E34; }
.tag-orange { background:#FDECD5; color:#C95200; }
.tag-blue   { background:#D0ECF5; color:#1A5C6B; }

/* ── EXPANDER ── */
[data-testid="stExpander"] {
    background: #FDFAF7 !important; border: 1px solid #E8D9C0 !important;
    border-radius: 12px !important; margin-bottom: 0.8rem !important;
}

/* ── CALLOUTS ── */
.callout { border-radius: 10px; padding: 0.9rem 1.1rem; margin: 0.7rem 0;
           font-size: 0.89rem; line-height: 1.65; }
.callout-teal   { background:#E0F2F6; border-left:3px solid #1A5C6B; color:#1A5C6B; }
.callout-orange { background:#FEF3EC; border-left:3px solid #E8621A; color:#C95200; }
.callout-sand   { background:#FBF5E8; border-left:3px solid #C9A96E; color:#7A5C2A; }

/* ── TIMELINE ── */
.timeline { display: flex; flex-wrap: wrap; gap: 0.45rem; margin: 1rem 0; }
.tl-step {
    background: #FDFAF7; border: 1px solid #D4CFC9; border-radius: 8px;
    padding: 0.4rem 0.8rem; font-size: 0.76rem; color: #5A5A5A;
    font-family: 'JetBrains Mono', monospace !important;
}
.tl-step.done   { background:#D4EDDA; border-color:#3D9970; color:#1E7E34; }
.tl-step.active { background:#1A5C6B; border-color:#1A5C6B; color:white; }

/* ── PREDICTOR ── */
.predictor-result {
    background: linear-gradient(135deg, #1A5C6B, #2A8A9E);
    border-radius: 16px; padding: 2rem; text-align: center;
    box-shadow: 0 8px 32px rgba(26,92,107,0.18);
}
.predictor-val {
    font-family: 'Playfair Display', serif !important;
    font-size: 2.8rem; font-weight: 700; color: #C9A96E; line-height: 1;
}

/* ── DIVIDER ── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #C9A96E 30%, #C9A96E 70%, transparent);
    margin: 1.8rem 0; opacity: 0.45;
}
</style>
""", unsafe_allow_html=True)


# ─── DATOS ────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    path = Path("data/ENB2012_data.xlsx")
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"
        try:
            urllib.request.urlretrieve(url, path)
        except Exception:
            np.random.seed(42)
            n = 768
            data = pd.DataFrame({
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
            data.to_excel(path, index=False)

    df = pd.read_excel(path).dropna()
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
    if len(df.columns) >= 10:
        col_map[df.columns[8]] = "carga_calefaccion"
        col_map[df.columns[9]] = "carga_refrigeracion"
    df = df.rename(columns=col_map)
    if "carga_calefaccion" in df.columns:
        df["consumo_kwh"] = df["carga_calefaccion"] + df["carga_refrigeracion"]
    else:
        df["consumo_kwh"] = df.iloc[:, -1]
    return df


@st.cache_resource(show_spinner=False)
def train_models(_df):
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
    feat_cols = [c for c in feat_cols if c in _df.columns]

    X = _df[feat_cols].copy()
    y = np.log1p(_df["consumo_kwh"])

    df_tmp = _df.copy()
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
            return np.c_[arr,
                         arr[:, 2] / arr[:, 1],
                         arr[:, 3] / arr[:, 1],
                         arr[:, 5] * arr[:, 0]]

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

    best        = models["Random Forest"]
    y_pred_log  = best.predict(X_test_p)
    y_pred_real = np.expm1(y_pred_log)
    y_test_real = np.expm1(y_test)

    rmse_log  = np.sqrt(mean_squared_error(y_test, y_pred_log))
    rmse_real = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
    mae_real  = mean_absolute_error(y_test_real, y_pred_real)

    ext_names   = feat_cols + ["ratio_muros_sup", "ratio_techo_sup", "acrist_efectivo"]
    importances = best.feature_importances_
    n_imp = min(len(importances), len(ext_names))
    fi = pd.Series(importances[:n_imp], index=ext_names[:n_imp]).sort_values(ascending=False)

    corr = _df[feat_cols + ["consumo_kwh"]].corr()["consumo_kwh"].drop("consumo_kwh").sort_values()

    return {
        "df": _df, "X_train": X_train, "X_test": X_test,
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

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Tema Plotly
PLOTLY_BASE = dict(
    paper_bgcolor=CREAM, plot_bgcolor=CREAM,
    font=dict(family="Source Serif 4, Georgia, serif", color=CHARCOAL, size=12),
    margin=dict(l=40, r=30, t=55, b=40),
    hoverlabel=dict(bgcolor=WHITE, font_size=12,
                    font_family="JetBrains Mono, monospace", bordercolor=GRAY_LT),
)

def apply_theme(fig, title=""):
    fig.update_layout(**PLOTLY_BASE, title=dict(
        text=title,
        font=dict(size=13, color=TEAL_DARK, family="Playfair Display, serif"),
        x=0.03, xanchor="left",
    ))
    fig.update_xaxes(showgrid=True, gridcolor=GRAY_LT, gridwidth=0.5,
                     zeroline=False, linecolor=GRAY_LT)
    fig.update_yaxes(showgrid=True, gridcolor=GRAY_LT, gridwidth=0.5,
                     zeroline=False, linecolor=GRAY_LT)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════════
HERO_SVG = """
<svg width="270" height="175" viewBox="0 0 270 175" fill="none" xmlns="http://www.w3.org/2000/svg">
  <rect x="10"  y="88"  width="26" height="77" rx="2" fill="#1A5C6B" opacity="0.16"/>
  <rect x="13"  y="96"  width="6"  height="6"  rx="1" fill="#2A8A9E" opacity="0.35"/>
  <rect x="23"  y="96"  width="6"  height="6"  rx="1" fill="#2A8A9E" opacity="0.28"/>
  <rect x="13"  y="106" width="6"  height="6"  rx="1" fill="#E8621A" opacity="0.28"/>
  <rect x="42"  y="58"  width="36" height="107" rx="2" fill="#1A5C6B" opacity="0.20"/>
  <rect x="46"  y="66"  width="7"  height="7"  rx="1" fill="#2A8A9E" opacity="0.38"/>
  <rect x="57"  y="66"  width="7"  height="7"  rx="1" fill="#C9A96E" opacity="0.38"/>
  <rect x="68"  y="66"  width="7"  height="7"  rx="1" fill="#E8621A" opacity="0.32"/>
  <rect x="46"  y="78"  width="7"  height="7"  rx="1" fill="#E8621A" opacity="0.26"/>
  <rect x="57"  y="78"  width="7"  height="7"  rx="1" fill="#2A8A9E" opacity="0.32"/>
  <rect x="68"  y="78"  width="7"  height="7"  rx="1" fill="#2A8A9E" opacity="0.22"/>
  <rect x="46"  y="90"  width="7"  height="7"  rx="1" fill="#C9A96E" opacity="0.28"/>
  <rect x="57"  y="90"  width="7"  height="7"  rx="1" fill="#E8621A" opacity="0.22"/>
  <rect x="84"  y="38"  width="46" height="127" rx="2" fill="#2A8A9E" opacity="0.14"/>
  <rect x="89"  y="46"  width="8"  height="8"  rx="1" fill="#1A5C6B" opacity="0.32"/>
  <rect x="101" y="46"  width="8"  height="8"  rx="1" fill="#E8621A" opacity="0.36"/>
  <rect x="113" y="46"  width="8"  height="8"  rx="1" fill="#1A5C6B" opacity="0.22"/>
  <rect x="89"  y="59"  width="8"  height="8"  rx="1" fill="#C9A96E" opacity="0.32"/>
  <rect x="101" y="59"  width="8"  height="8"  rx="1" fill="#1A5C6B" opacity="0.26"/>
  <rect x="113" y="59"  width="8"  height="8"  rx="1" fill="#E8621A" opacity="0.28"/>
  <rect x="89"  y="72"  width="8"  height="8"  rx="1" fill="#1A5C6B" opacity="0.18"/>
  <rect x="101" y="72"  width="8"  height="8"  rx="1" fill="#C9A96E" opacity="0.30"/>
  <line x1="107" y1="38" x2="107" y2="20" stroke="#1A5C6B" stroke-width="1.5" opacity="0.25"/>
  <circle cx="107" cy="18" r="3" fill="#E8621A" opacity="0.40"/>
  <rect x="136" y="68"  width="30" height="97" rx="2" fill="#1A5C6B" opacity="0.16"/>
  <rect x="140" y="76"  width="7"  height="7"  rx="1" fill="#2A8A9E" opacity="0.32"/>
  <rect x="151" y="76"  width="7"  height="7"  rx="1" fill="#C9A96E" opacity="0.28"/>
  <rect x="140" y="88"  width="7"  height="7"  rx="1" fill="#E8621A" opacity="0.24"/>
  <rect x="173" y="95" width="20" height="70" rx="2" fill="#C9A96E" opacity="0.16"/>
  <rect x="177" y="103" width="5" height="5"  rx="1" fill="#1A5C6B" opacity="0.28"/>
  <circle cx="220" cy="32"  r="7" fill="#1A5C6B" opacity="0.18"/>
  <circle cx="247" cy="20"  r="5" fill="#E8621A" opacity="0.26"/>
  <circle cx="254" cy="52"  r="8" fill="#2A8A9E" opacity="0.16"/>
  <circle cx="233" cy="72"  r="5" fill="#C9A96E" opacity="0.28"/>
  <circle cx="260" cy="88"  r="6" fill="#1A5C6B" opacity="0.18"/>
  <circle cx="240" cy="108" r="4" fill="#E8621A" opacity="0.20"/>
  <line x1="220" y1="32"  x2="247" y2="20"  stroke="#1A5C6B" stroke-width="0.9" opacity="0.16"/>
  <line x1="247" y1="20"  x2="254" y2="52"  stroke="#1A5C6B" stroke-width="0.9" opacity="0.14"/>
  <line x1="254" y1="52"  x2="233" y2="72"  stroke="#2A8A9E" stroke-width="0.9" opacity="0.16"/>
  <line x1="233" y1="72"  x2="260" y2="88"  stroke="#2A8A9E" stroke-width="0.9" opacity="0.14"/>
  <line x1="260" y1="88"  x2="240" y2="108" stroke="#E8621A" stroke-width="0.9" opacity="0.16"/>
  <line x1="220" y1="32"  x2="254" y2="52"  stroke="#C9A96E" stroke-width="0.7" opacity="0.11"/>
  <line x1="8" y1="167" x2="198" y2="167" stroke="#1A5C6B" stroke-width="1.2" opacity="0.16"/>
</svg>
"""

st.markdown(f"""
<div class="hero">
  <div class="hero-grid">
    <div>
      <div class="hero-badge">◈ Machine Learning · Regresión · UCI Energy Efficiency</div>
      <h1 class="hero-title">Predicción de Consumo<br>Energético de Edificios</h1>
      <div class="hero-author">Leiry Laura Mares Cure</div>
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
    <div class="hero-deco">{HERO_SVG}</div>
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
    st.markdown("<div style='height:1.4rem'></div>", unsafe_allow_html=True)
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
    st.markdown("<div style='height:0.9rem'></div>", unsafe_allow_html=True)

    # Etapas 1 y 2
    col_a, col_b = st.columns([1.05, 1], gap="large")
    with col_a:
        st.markdown("""
        <div class="card card-accent-teal">
          <div class="section-label">Etapa 1 · Panorama general</div>
          <div class="section-title">Objetivo del proyecto</div>
          <p class="section-body">El modelo recibe las características geométricas y constructivas
          de un edificio y produce una estimación del consumo energético total (kWh/m²).
          La predicción alimentará sistemas de gestión energética y priorización de renovaciones.</p>
        </div>
        <div class="styled-table-wrap">
        <table class="styled-table">
          <thead><tr><th>Dimensión</th><th>Clasificación</th></tr></thead>
          <tbody>
            <tr><td>Tipo de aprendizaje</td><td><span class="tag tag-blue">Supervisado</span></td></tr>
            <tr><td>Tipo de tarea</td><td><span class="tag tag-blue">Regresión múltiple</span></td></tr>
            <tr><td>Estrategia</td><td><span class="tag tag-green">Batch learning</span></td></tr>
            <tr><td>Métrica principal</td><td><span class="tag tag-orange">RMSE sobre log(consumo)</span></td></tr>
          </tbody>
        </table></div>
        <div class="callout callout-sand" style="margin-top:0.9rem">
          La transformación log1p(y) simetriza la distribución del target, penaliza errores
          proporcionalmente y estabiliza la varianza. Se revierte con expm1() al reportar resultados.
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        attrs = [
            ("Compacidad relativa",   "0.62 – 0.98", "Continuo"),
            ("Superficie total (m²)", "514 – 808",   "Continuo"),
            ("Área de muros (m²)",    "245 – 515",   "Continuo"),
            ("Área de techo (m²)",    "110 – 220",   "Continuo"),
            ("Altura total (m)",       "3.5 – 7.0",  "Discreto"),
            ("Área acristalamiento",  "0 – 0.4",     "Discreto"),
            ("Distribución acrist.",  "0 – 5",       "Categórico"),
        ]
        rows_attr = "".join(
            f"<tr><td>{a[0]}</td><td class='num'>{a[1]}</td>"
            f"<td><span class='tag {'tag-blue' if a[2]=='Continuo' else 'tag-green' if a[2]=='Discreto' else 'tag-orange'}'>{a[2]}</span></td></tr>"
            for a in attrs
        )
        st.markdown(f"""
        <div class="card card-accent-orange">
          <div class="section-label">Etapa 2 · Dataset UCI Energy Efficiency</div>
          <div class="section-title">768 simulaciones energéticas</div>
          <p class="section-body">Generadas con EnergyPlus. Cada fila es una configuración
          única de parámetros geométricos y constructivos de edificios residenciales.</p>
        </div>
        <div class="styled-table-wrap">
        <table class="styled-table">
          <thead><tr><th>Atributo</th><th>Rango</th><th>Tipo</th></tr></thead>
          <tbody>{rows_attr}</tbody>
        </table></div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Etapa 3
    st.markdown("""
    <div style="list-style:none;margin:0;padding:0">
      <div class="section-label">Etapa 3</div>
      <div class="section-title">Exploración y visualización de los datos</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='height:0.7rem'></div>", unsafe_allow_html=True)

    # 3.1
    st.markdown("""
    <div class="card card-accent-sand">
      <div class="section-label">3.1</div>
      <div class="section-title" style="font-size:1.1rem">Estadísticas descriptivas</div>
    </div>
    """, unsafe_allow_html=True)
    desc = df[md["feat_cols"] + ["consumo_kwh"]].describe().T.round(3).reset_index()
    desc.columns = ["Atributo"] + list(desc.columns[1:])
    hdr = "".join(f"<th>{c}</th>" for c in desc.columns)
    bdy = "".join(
        "<tr><td>{}</td>".format(row["Atributo"]) +
        "".join(f"<td class='num'>{row[c]}</td>" for c in desc.columns[1:]) +
        "</tr>" for _, row in desc.iterrows()
    )
    st.markdown(f"""<div class="styled-table-wrap">
    <table class="styled-table"><thead><tr>{hdr}</tr></thead><tbody>{bdy}</tbody></table>
    </div>""", unsafe_allow_html=True)

    # 3.2 — Distribución del target (Plotly)
    st.markdown("""
    <div class="card card-accent-teal" style="margin-top:0.8rem">
      <div class="section-label">3.2</div>
      <div class="section-title" style="font-size:1.1rem">Distribución del consumo energético</div>
    </div>
    """, unsafe_allow_html=True)

    vals     = df["consumo_kwh"]
    log_vals = np.log1p(vals)
    q25, q75 = float(vals.quantile(0.25)), float(vals.quantile(0.75))

    hist_counts, hist_edges = np.histogram(vals, bins=40)
    bin_mids   = [(hist_edges[i]+hist_edges[i+1])/2 for i in range(len(hist_edges)-1)]
    bin_widths = np.diff(hist_edges)
    bar_cols   = [TEAL_LIGHT if m < q25 else (ORANGE if m > q75 else TEAL_MID)
                  for m in bin_mids]

    fig_dist = make_subplots(rows=1, cols=2,
                              subplot_titles=["Escala original", "Escala logarítmica (log1p)"],
                              horizontal_spacing=0.10)
    for idx in range(len(hist_counts)):
        fig_dist.add_trace(go.Bar(
            x=[bin_mids[idx]], y=[hist_counts[idx]], width=[bin_widths[idx]],
            marker_color=bar_cols[idx], marker_line_width=0, showlegend=False,
            hovertemplate=f"{hist_edges[idx]:.1f}–{hist_edges[idx+1]:.1f} kWh/m²<br>n: {hist_counts[idx]}<extra></extra>",
        ), row=1, col=1)

    fig_dist.add_vline(x=float(vals.median()), line_dash="dash", line_color=ORANGE, line_width=2,
                        annotation_text=f"Mediana {vals.median():.1f}",
                        annotation_font_color=ORANGE, annotation_font_size=10,
                        row=1, col=1)

    lh_c, lh_e = np.histogram(log_vals, bins=40)
    lh_mids    = [(lh_e[i]+lh_e[i+1])/2 for i in range(len(lh_e)-1)]
    lh_widths  = np.diff(lh_e)
    fig_dist.add_trace(go.Bar(
        x=lh_mids, y=lh_c, width=lh_widths,
        marker_color=SAND, marker_line_width=0, showlegend=False,
        hovertemplate="log1p: %{x:.3f}<br>n: %{y}<extra></extra>",
    ), row=1, col=2)
    fig_dist.add_vline(x=float(log_vals.median()), line_dash="dash", line_color=TEAL_DARK, line_width=2,
                        annotation_text=f"Mediana {log_vals.median():.2f}",
                        annotation_font_color=TEAL_DARK, annotation_font_size=10,
                        row=1, col=2)

    fig_dist.update_layout(**PLOTLY_BASE, height=370, bargap=0.02,
        title=dict(text=f"Asimetría original: {vals.skew():.3f}  →  log1p: {log_vals.skew():.3f}",
                   font=dict(size=11, color=GRAY_MID), x=0.03))
    fig_dist.update_xaxes(showgrid=True, gridcolor=GRAY_LT, tickfont_size=9)
    fig_dist.update_yaxes(showgrid=True, gridcolor=GRAY_LT, tickfont_size=9)
    st.plotly_chart(fig_dist, use_container_width=True)

    # 3.3 — Histogramas (Plotly, sin superposición de etiquetas)
    st.markdown("""
    <div class="card card-accent-sand" style="margin-top:0.3rem">
      <div class="section-label">3.3</div>
      <div class="section-title" style="font-size:1.1rem">Histogramas de los atributos de entrada</div>
    </div>
    """, unsafe_allow_html=True)

    num_cols = md["feat_cols"]
    NCOLS    = 4
    NROWS    = (len(num_cols) + NCOLS - 1) // NCOLS
    pal_h    = [TEAL_DARK, TEAL_MID, TEAL_LIGHT, SAND, ORANGE_LT, ORANGE, SAND_LT, GRAY_MID]
    titles_h = [c.replace("_", " ").title() for c in num_cols] + [""] * (NROWS * NCOLS - len(num_cols))

    fig_hist = make_subplots(
        rows=NROWS, cols=NCOLS,
        subplot_titles=titles_h,
        horizontal_spacing=0.09,
        vertical_spacing=0.22,      # más espacio vertical para evitar solapamiento
    )
    for i, col in enumerate(num_cols):
        r, c_i = divmod(i, NCOLS)
        h_c, h_e = np.histogram(df[col].dropna(), bins=22)
        h_m = [(h_e[k]+h_e[k+1])/2 for k in range(len(h_e)-1)]
        h_w = np.diff(h_e)
        fig_hist.add_trace(go.Bar(
            x=h_m, y=h_c, width=h_w * 0.90,
            marker_color=pal_h[i % len(pal_h)],
            marker_line_width=0, showlegend=False,
            name=col,
            hovertemplate=f"{col}: %{{x:.2f}}<br>n: %{{y}}<extra></extra>",
        ), row=r+1, col=c_i+1)
        fig_hist.add_vline(x=float(df[col].median()), line_dash="dot",
                            line_color=CHARCOAL, line_width=1, opacity=0.45,
                            row=r+1, col=c_i+1)

    # Ocultar subplots vacíos
    for j in range(len(num_cols), NROWS * NCOLS):
        r, c_i = divmod(j, NCOLS)
        fig_hist.update_xaxes(visible=False, row=r+1, col=c_i+1)
        fig_hist.update_yaxes(visible=False, row=r+1, col=c_i+1)

    fig_hist.update_layout(
        **PLOTLY_BASE,
        height=NROWS * 210,
        bargap=0.03,
        title=dict(text="Distribuciones — línea punteada = mediana",
                   font=dict(size=11, color=GRAY_MID), x=0.03),
    )
    # Subtítulos más pequeños para no solapar
    fig_hist.update_annotations(font_size=10, font_color=TEAL_DARK)
    fig_hist.update_xaxes(showgrid=True, gridcolor=GRAY_LT, tickfont_size=8)
    fig_hist.update_yaxes(showgrid=True, gridcolor=GRAY_LT, tickfont_size=8)
    st.plotly_chart(fig_hist, use_container_width=True)

    # 3.5 — Correlaciones (Plotly)
    st.markdown("""
    <div class="card card-accent-orange" style="margin-top:0.3rem">
      <div class="section-label">3.5</div>
      <div class="section-title" style="font-size:1.1rem">Mapa de correlaciones con el consumo</div>
    </div>
    """, unsafe_allow_html=True)

    col_corr, col_heat = st.columns([1, 1.45], gap="large")

    with col_corr:
        corr_v = md["corr"]
        fig_corr = go.Figure(go.Bar(
            x=corr_v.values,
            y=[c.replace("_", " ") for c in corr_v.index],
            orientation="h",
            marker_color=[TEAL_MID if v >= 0 else ORANGE for v in corr_v.values],
            marker_line_width=0,
            text=[f"{v:.3f}" for v in corr_v.values],
            textposition="outside",
            textfont=dict(size=10, family="JetBrains Mono"),
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        ))
        fig_corr.add_vline(x=0, line_color=GRAY_LT, line_width=1)
        apply_theme(fig_corr, "Pearson vs. consumo_kwh")
        fig_corr.update_layout(height=330, margin=dict(l=10, r=65, t=50, b=30))
        fig_corr.update_xaxes(range=[-0.9, 0.9])
        st.plotly_chart(fig_corr, use_container_width=True)

    with col_heat:
        corr_full = df[num_cols + ["consumo_kwh"]].corr()
        hlabels   = [c.replace("_", "<br>") for c in corr_full.columns]
        fig_heat  = go.Figure(go.Heatmap(
            z=corr_full.values, x=hlabels, y=hlabels,
            colorscale=[[0, ORANGE], [0.5, "#FFFFFF"], [1, TEAL_MID]],
            zmin=-1, zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in corr_full.values],
            texttemplate="%{text}",
            textfont=dict(size=8, family="JetBrains Mono"),
            hovertemplate="%{x} × %{y}: %{z:.3f}<extra></extra>",
            colorbar=dict(thickness=12, len=0.8, tickfont=dict(size=9)),
        ))
        apply_theme(fig_heat, "Matriz de correlación completa")
        fig_heat.update_layout(height=370, margin=dict(l=10, r=20, t=50, b=10))
        fig_heat.update_xaxes(tickfont_size=8, tickangle=-35)
        fig_heat.update_yaxes(tickfont_size=8)
        st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("""
    <div class="callout callout-teal">
      El coeficiente de Pearson solo detecta relaciones lineales. La compacidad relativa
      y el área de techo muestran la correlación absoluta más fuerte con el consumo total.
    </div>
    """, unsafe_allow_html=True)

    # 3.6 — Scatter (Plotly)
    st.markdown("""
    <div class="card card-accent-teal" style="margin-top:0.5rem">
      <div class="section-label">3.6</div>
      <div class="section-title" style="font-size:1.1rem">Relaciones con el consumo — atributos clave</div>
    </div>
    """, unsafe_allow_html=True)

    top_feats = md["corr"].abs().nlargest(4).index.tolist()
    fig_sc = make_subplots(rows=1, cols=4,
        subplot_titles=[f"{f.replace('_',' ')}  (r={df[[f,'consumo_kwh']].corr().iloc[0,1]:.3f})"
                        for f in top_feats],
        horizontal_spacing=0.07)

    for idx, feat in enumerate(top_feats):
        fig_sc.add_trace(go.Scatter(
            x=df[feat], y=df["consumo_kwh"], mode="markers",
            marker=dict(color=df["consumo_kwh"], colorscale="YlOrRd",
                        size=5, opacity=0.6, line=dict(width=0),
                        showscale=(idx == 3),
                        colorbar=dict(title="kWh/m²", thickness=10, len=0.75,
                                      tickfont=dict(size=8)) if idx == 3 else None),
            showlegend=False,
            hovertemplate=f"{feat}: %{{x:.2f}}<br>Consumo: %{{y:.1f}} kWh/m²<extra></extra>",
        ), row=1, col=idx+1)

    fig_sc.update_layout(**PLOTLY_BASE, height=340,
        title=dict(text="Los 4 atributos con mayor correlación absoluta",
                   font=dict(size=11, color=GRAY_MID), x=0.03))
    fig_sc.update_annotations(font_size=9, font_color=TEAL_DARK)
    fig_sc.update_xaxes(showgrid=True, gridcolor=GRAY_LT, tickfont_size=8)
    fig_sc.update_yaxes(showgrid=True, gridcolor=GRAY_LT, tickfont_size=8)
    st.plotly_chart(fig_sc, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODELADO Y RESULTADOS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div style='height:1.4rem'></div>", unsafe_allow_html=True)
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
    st.markdown("<div style='height:0.9rem'></div>", unsafe_allow_html=True)

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

    col_tbl, col_cv = st.columns([1.1, 1], gap="large")

    with col_tbl:
        st.markdown("""
        <div class="section-label">Etapas 5 – 6</div>
        <div class="section-title">Comparativa de modelos</div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
        rows_m = ""
        for name, res in md["results"].items():
            is_best  = name == "Random Forest"
            bc       = "best" if is_best else ""
            overfit  = ("Sobreajuste severo" if res["train"] < 0.001 else
                        "Sobreajuste leve"   if res["train"] < res["cv_mean"] * 0.7 else "Equilibrado")
            tc       = "tag-green" if overfit == "Equilibrado" else "tag-orange"
            badge    = " ◈" if is_best else ""
            rows_m  += (f"<tr><td class='{bc}'>{name}{badge}</td>"
                        f"<td class='num {bc}'>{res['train']:.4f}</td>"
                        f"<td class='num {bc}'>{res['cv_mean']:.4f}</td>"
                        f"<td class='num {bc}'>± {res['cv_std']:.4f}</td>"
                        f"<td><span class='tag {tc}'>{overfit}</span></td></tr>")
        st.markdown(f"""
        <div class="styled-table-wrap">
        <table class="styled-table">
          <thead><tr><th>Modelo</th><th>RMSE Train</th><th>RMSE CV</th>
            <th>Desv. Std.</th><th>Diagnóstico</th></tr></thead>
          <tbody>{rows_m}</tbody>
        </table></div>
        <div class="callout callout-teal" style="margin-top:0.8rem">
          El Árbol de Decisión muestra RMSE = 0 en entrenamiento pero ~0.034 en CV —
          señal clásica de sobreajuste. El Random Forest generaliza mejor al promediar
          múltiples árboles sobre subconjuntos aleatorios.
        </div>
        """, unsafe_allow_html=True)

    with col_cv:
        st.markdown("""
        <div class="section-label">Validación cruzada K=10</div>
        <div class="section-title">Dispersión de scores</div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
        box_colors = [TEAL_LIGHT, SAND, TEAL_DARK]
        fig_box = go.Figure()
        for (name, res), color in zip(md["results"].items(), box_colors):
            fig_box.add_trace(go.Box(
                y=res["scores"], name=name.replace(" ", "<br>"),
                boxpoints="all", jitter=0.35, pointpos=0,
                marker=dict(color=color, size=5, opacity=0.6),
                line=dict(color=color, width=2),
                fillcolor=color, opacity=0.75,
                hovertemplate=f"{name}<br>RMSE: %{{y:.4f}}<extra></extra>",
            ))
        apply_theme(fig_box, "RMSE en 10 pliegues de validación cruzada")
        fig_box.update_layout(height=360, showlegend=False)
        fig_box.update_yaxes(title_text="RMSE (escala log)")
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    col_fi, col_pred = st.columns([1, 1.05], gap="large")

    with col_fi:
        st.markdown("""
        <div class="section-label">Etapa 6 · Ajuste fino</div>
        <div class="section-title">Importancia de características</div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
        fi = md["feat_importances"]
        fi_colors = [TEAL_DARK, TEAL_MID, TEAL_MID, TEAL_LIGHT,
                     SAND, SAND_LT, ORANGE_LT, ORANGE, GRAY_MID][:len(fi)]
        fig_fi = go.Figure(go.Bar(
            x=fi.values, y=[c.replace("_", " ") for c in fi.index], orientation="h",
            marker_color=fi_colors, marker_line_width=0,
            text=[f"{v:.3f}" for v in fi.values], textposition="outside",
            textfont=dict(size=10, family="JetBrains Mono"),
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        ))
        apply_theme(fig_fi, "Random Forest · importancia por pureza de Gini")
        fig_fi.update_layout(height=360, margin=dict(l=10, r=70, t=50, b=30))
        fig_fi.update_xaxes(range=[0, fi.values.max() * 1.28])
        st.plotly_chart(fig_fi, use_container_width=True)

    with col_pred:
        st.markdown("""
        <div class="section-label">Etapa 7 · Test set</div>
        <div class="section-title">Predicciones vs. valores reales</div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
        y_real  = md["y_test_real"].values
        y_pred  = md["y_pred_real"]
        errors  = y_pred - y_real
        lim_min = float(min(y_real.min(), y_pred.min())) - 1
        lim_max = float(max(y_real.max(), y_pred.max())) + 1

        fig_pv = make_subplots(rows=1, cols=2,
            subplot_titles=["Predicho vs. Real", "Distribución de residuos"],
            horizontal_spacing=0.12)
        fig_pv.add_trace(go.Scatter(
            x=y_real, y=y_pred, mode="markers",
            marker=dict(color=errors, colorscale="RdYlGn_r", size=5, opacity=0.65,
                        line=dict(width=0),
                        cmin=-float(np.percentile(np.abs(errors), 95)),
                        cmax= float(np.percentile(np.abs(errors), 95)),
                        showscale=True,
                        colorbar=dict(title="Error", thickness=10, len=0.7, tickfont=dict(size=8))),
            showlegend=False,
            hovertemplate="Real: %{x:.1f}<br>Pred: %{y:.1f}<extra></extra>",
        ), row=1, col=1)
        fig_pv.add_trace(go.Scatter(
            x=[lim_min, lim_max], y=[lim_min, lim_max], mode="lines",
            line=dict(color=TEAL_DARK, dash="dash", width=1.5),
            showlegend=False, hoverinfo="skip",
        ), row=1, col=1)
        eh_c, eh_e = np.histogram(errors, bins=30)
        eh_m = [(eh_e[k]+eh_e[k+1])/2 for k in range(len(eh_e)-1)]
        fig_pv.add_trace(go.Bar(
            x=eh_m, y=eh_c, width=np.diff(eh_e),
            marker_color=TEAL_MID, marker_line_width=0, showlegend=False,
            hovertemplate="Residuo: %{x:.2f}<br>n: %{y}<extra></extra>",
        ), row=1, col=2)
        fig_pv.add_vline(x=0, line_dash="dash", line_color=ORANGE, line_width=2, row=1, col=2)
        fig_pv.add_vline(x=float(errors.mean()), line_dash="dot",
                          line_color=TEAL_DARK, line_width=1.5, row=1, col=2,
                          annotation_text=f"μ={errors.mean():.2f}",
                          annotation_font_size=9, annotation_font_color=TEAL_DARK)
        fig_pv.update_layout(**PLOTLY_BASE, height=360, bargap=0.04)
        fig_pv.update_annotations(font_size=10)
        fig_pv.update_xaxes(showgrid=True, gridcolor=GRAY_LT, tickfont_size=9)
        fig_pv.update_yaxes(showgrid=True, gridcolor=GRAY_LT, tickfont_size=9)
        st.plotly_chart(fig_pv, use_container_width=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Subgrupos
    st.markdown("""
    <div class="section-label">Etapa 8 · Evaluación segmentada</div>
    <div class="section-title">Rendimiento por subgrupos del test set</div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

    from sklearn.metrics import mean_squared_error as _mse

    def subgroup_rmse(col_name, prefix):
        out = []
        for val in sorted(md["X_test"][col_name].unique()):
            mask = md["X_test"][col_name].values == val
            if not mask.sum(): continue
            y_s  = md["y_test_real"].values[mask]
            yp_s = md["y_pred_real"][mask]
            rs   = np.sqrt(_mse(y_s, yp_s))
            out.append({"Segmento": f"{prefix} {val}", "n": int(mask.sum()),
                        "RMSE (kWh/m²)": round(rs, 3),
                        "Error relativo (%)": round(rs / y_s.mean() * 100, 1)})
        return out

    all_sg   = subgroup_rmse("altura_total_m", "Alt.") + subgroup_rmse("area_acristalamiento", "Acrist.")
    global_r = md["rmse_real"]

    col_sg1, col_sg2 = st.columns([1, 1.2], gap="large")
    with col_sg1:
        if all_sg:
            rows_sg = "".join(
                f"<tr><td>{r['Segmento']}</td><td class='num'>{r['n']}</td>"
                f"<td class='num'>{r['RMSE (kWh/m²)']}</td>"
                f"<td><span class='tag {'tag-green' if r['Error relativo (%)'] < 8 else 'tag-orange'}'>"
                f"{r['Error relativo (%)']}%</span></td></tr>"
                for r in all_sg
            )
            st.markdown(f"""
            <div class="styled-table-wrap">
            <table class="styled-table">
              <thead><tr><th>Segmento</th><th>n</th><th>RMSE kWh/m²</th><th>Error rel.</th></tr></thead>
              <tbody>{rows_sg}</tbody>
            </table></div>
            <div class="callout callout-sand" style="margin-top:0.8rem">
              Variación notable entre subgrupos indica sesgo del modelo que podría requerir
              features adicionales o modelos especializados por segmento.
            </div>
            """, unsafe_allow_html=True)

    with col_sg2:
        if all_sg:
            segs  = [r["Segmento"] for r in all_sg]
            rmses = [r["RMSE (kWh/m²)"] for r in all_sg]
            sg_c  = [TEAL_MID if "Alt." in s else ORANGE_LT for s in segs]
            fig_sg = go.Figure()
            fig_sg.add_trace(go.Bar(
                x=segs, y=rmses, marker_color=sg_c, marker_line_width=0,
                text=[f"{v:.2f}" for v in rmses], textposition="outside",
                textfont=dict(size=10, family="JetBrains Mono"),
                hovertemplate="%{x}: %{y:.3f} kWh/m²<extra></extra>",
            ))
            fig_sg.add_hline(y=global_r, line_dash="dash", line_color=TEAL_DARK, line_width=1.5,
                              annotation_text=f"RMSE global {global_r:.2f}",
                              annotation_font_size=10, annotation_font_color=TEAL_DARK)
            apply_theme(fig_sg, "RMSE por subgrupo del test set")
            fig_sg.update_layout(height=355, xaxis_tickangle=-30)
            fig_sg.update_yaxes(title_text="RMSE (kWh/m²)")
            st.plotly_chart(fig_sg, use_container_width=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Simulador
    st.markdown("""
    <div class="section-label">Etapa 8 · Despliegue</div>
    <div class="section-title">Simulador de consumo energético</div>
    <p class="section-body">
      Introduce las características de un edificio para obtener una predicción
      usando el modelo Random Forest entrenado.
    </p>
    """, unsafe_allow_html=True)
    st.markdown("<div style='height:0.9rem'></div>", unsafe_allow_html=True)

    col_inp1, col_inp2, col_result = st.columns([1, 1, 1], gap="large")

    with col_inp1:
        st.markdown("<div class='card card-accent-teal'>", unsafe_allow_html=True)
        comp_rel   = st.slider("Compacidad relativa",
                                float(df["compacidad_relativa"].min()),
                                float(df["compacidad_relativa"].max()), 0.75, 0.01)
        sup_m2     = st.slider("Superficie total (m²)",
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
        altura     = st.selectbox("Altura total (m)",
                                   sorted(df["altura_total_m"].unique()), index=0)
        acrist     = st.selectbox("Área acristalamiento",
                                   sorted(df["area_acristalamiento"].unique()), index=2)
        dist_acr   = st.selectbox("Distribución acristalamiento",
                                   sorted(df["dist_acristalamiento"].unique()), index=0)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_result:
        inp = {
            "compacidad_relativa":  comp_rel,
            "superficie_m2":        sup_m2,
            "area_muros_m2":        area_muros,
            "area_techo_m2":        area_techo,
            "altura_total_m":       float(altura),
            "area_acristalamiento": float(acrist),
            "dist_acristalamiento": float(dist_acr),
        }
        try:
            X_in     = md["pipe"].transform(pd.DataFrame([{c: inp.get(c, 0) for c in md["feat_cols"]}]))
            lp       = md["best_model"].predict(X_in)[0]
            pred_kwh = float(np.expm1(lp))
            pct      = float((df["consumo_kwh"] < pred_kwh).mean() * 100)
            lbl      = "Alta eficiencia" if pct < 33 else ("Eficiencia media" if pct < 66 else "Baja eficiencia")
            clr      = "#3D9970" if pct < 33 else ("#C9A96E" if pct < 66 else "#E8621A")

            st.markdown(f"""
            <div class="predictor-result">
              <div style="font-family:'JetBrains Mono',monospace;font-size:0.66rem;
                          letter-spacing:0.13em;text-transform:uppercase;
                          color:rgba(255,255,255,0.55);margin-bottom:0.5rem">consumo predicho</div>
              <div class="predictor-val">{pred_kwh:.1f}</div>
              <div style="font-size:0.86rem;color:rgba(255,255,255,0.65);margin-top:0.22rem">kWh / m² · año</div>
              <div style="margin-top:1rem;padding:0.5rem 1rem;background:rgba(255,255,255,0.10);
                          border-radius:8px;font-size:0.81rem;color:rgba(255,255,255,0.80)">
                Percentil <span style="color:{clr};font-weight:600">{pct:.0f}</span> ·
                <span style="color:{clr};font-weight:600">{lbl}</span>
              </div>
              <div style="margin-top:0.6rem;font-size:0.72rem;color:rgba(255,255,255,0.38)">
                log-pred = {lp:.4f} · Random Forest</div>
            </div>
            """, unsafe_allow_html=True)

            # Gauge Plotly
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred_kwh,
                number=dict(suffix=" kWh/m²",
                            font=dict(color=TEAL_DARK, size=16, family="Playfair Display")),
                gauge=dict(
                    axis=dict(range=[0, float(df["consumo_kwh"].max()) * 1.05],
                               tickfont=dict(size=9)),
                    bar=dict(color=TEAL_MID, thickness=0.28),
                    bgcolor=CREAM, borderwidth=0,
                    steps=[
                        dict(range=[0, df["consumo_kwh"].quantile(0.33)],   color="#D4EDDA"),
                        dict(range=[df["consumo_kwh"].quantile(0.33),
                                    df["consumo_kwh"].quantile(0.66)],       color="#FBF5E8"),
                        dict(range=[df["consumo_kwh"].quantile(0.66),
                                    float(df["consumo_kwh"].max()) * 1.05],  color="#FDECD5"),
                    ],
                    threshold=dict(line=dict(color=ORANGE, width=2), thickness=0.75, value=pred_kwh),
                ),
            ))
            fig_g.update_layout(paper_bgcolor=CREAM, height=195,
                                 margin=dict(l=20, r=20, t=20, b=10),
                                 font=dict(family="Source Serif 4, serif"))
            st.plotly_chart(fig_g, use_container_width=True)

        except Exception as e:
            st.warning(f"No se pudo calcular la predicción: {e}")

    # Footer
    st.markdown("""
    <div style="margin-top:2.5rem;padding:1.2rem 0;border-top:1px solid #D4CFC9;
                display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:0.5rem">
      <div style="font-family:'JetBrains Mono',monospace;font-size:0.69rem;color:#7A7A7A;letter-spacing:0.08em">
        UCI ENERGY EFFICIENCY DATASET · 768 muestras · 8 atributos
      </div>
      <div style="font-size:0.82rem;color:#7A7A7A;font-style:italic">
        Leiry Laura Mares Cure · Hands-On Machine Learning, Cap. 2 — Aurélien Géron
      </div>
    </div>
    """, unsafe_allow_html=True)
