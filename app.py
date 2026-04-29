"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   SHARK TANK INDIA — STARTUP VALUATION PREDICTOR  v4.0                    ║
║                                                                            ║
║   Deployment  : GitHub + Streamlit Cloud (model loaded from repo path)    ║
║   Architecture: @st.fragment  +  O(1) list history  +  persistent CSV     ║
║   Theme       : Dark Navy — Shark Tank India aesthetic                     ║
║   Animation   : SVG shark swim / bite on prediction success               ║
╚══════════════════════════════════════════════════════════════════════════════╝

DEPLOYMENT NOTES
────────────────
1. Place this file at the root of your GitHub repo.
2. Put the model at:  models/valuation_model_1.pkl
3. Your repo structure should look like:
       repo/
       ├── app.py
       ├── requirements.txt
       ├── models/
       │   └── valuation_model_1.pkl
       └── data/
           ├── MLR_Friendly_Shark_Tank_India.csv
           └── Cleaned_Shark_Tank_India.xlsx
4. Prediction history is saved to  data/prediction_history.csv  (auto-created).
5. On Streamlit Cloud the CSV write will work within the session; for true
   persistence across deploys use a database or Streamlit's built-in secrets.
"""

# ── stdlib ───────────────────────────────────────────────────────────────────
import io
import os
import datetime
import math

# ── third-party ──────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import statsmodels.api as sm
import streamlit as st
import streamlit.components.v1 as components

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Shark Tank India · Valuation Predictor",
    page_icon="🦈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL & DATA PATHS  (relative to app.py — works on GitHub + Streamlit Cloud)
# ─────────────────────────────────────────────────────────────────────────────
_BASE        = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(_BASE, "models", "valuation_model_1.pkl")
HISTORY_PATH = os.path.join(_BASE, "data",   "prediction_history.csv")

# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING CONSTANTS  (extracted from exact EDA + log_valuation pipeline)
# ─────────────────────────────────────────────────────────────────────────────

# Columns that get np.log1p() THEN z-score  (EDA cell 10 → cell 26)
LOG_THEN_SCALE = {"Yearly Revenue", "SKUs", "Original Ask Amount"}

# Z-score params fitted on 423-row training set (verified to 4 decimal places)
SCALER_PARAMS: dict[str, tuple[float, float]] = {
    "Yearly Revenue":          (4.753898,  2.176600),
    "Gross Margin":            (55.500000, 15.374598),
    "Net Margin":              (19.781324, 6.761300),
    "EBITDA":                  (13.402128, 9.087214),
    "SKUs":                    (3.998586,  1.201641),
    "Original Ask Amount":     (4.192264,  0.586417),
    "Original Offered Equity": (3.130709,  2.962779),
    "Company Age":             (3.903073,  2.355531),
}

# sym_log: safely handles negative EBITDA / margins without crashing on log(negative)
# Used for display/radar normalisation only — NOT in model preprocessing
# (EDA did NOT log-transform EBITDA/margins; they are z-scored directly)
def sym_log(x: float) -> float:
    """Symmetric log: sign(x) * log1p(|x|) — handles negatives gracefully."""
    return float(np.sign(x) * np.log1p(np.abs(x)))

INDUSTRY_DUMMIES = [
    "Agriculture", "Animal/Pets", "Beauty/Fashion", "Business Services",
    "Children/Education", "Electronics", "Entertainment",
    "Fitness/Sports/Outdoors", "Food And Beverage", "Green/Cleantech",
    "Lifestyle/Home", "Liquor/Alcohol", "Manufacturing", "Medical/Health",
    "Others", "Technology/Software", "Vehicles/Electrical Vehicles",
]
ALL_INDUSTRIES = sorted(INDUSTRY_DUMMIES + ["Hardware"])

MODEL_FEATURE_ORDER = INDUSTRY_DUMMIES + [
    "Pitchers Average Age", "Yearly Revenue", "Gross Margin",
    "Net Margin", "EBITDA", "Has Patents", "Bootstrapped", "SKUs",
    "Original Ask Amount", "Original Offered Equity", "Company Age",
]

PITCHER_AGE_MAP = {"Young (18-35)": 0, "Middle (36-50)": 1, "Old (51+)": 2}

# ─────────────────────────────────────────────────────────────────────────────
# INDUSTRY BENCHMARKS  (z-scored model space — from MLR CSV)
# ─────────────────────────────────────────────────────────────────────────────
INDUSTRY_BENCHMARKS = {
    "Agriculture":                  {"Count":  5, "DealVal": 6.41, "Revenue":  0.13, "GrossMargin": -2.18, "NetMargin": -1.74, "EBITDA":  0.18},
    "Animal/Pets":                  {"Count":  5, "DealVal": 7.31, "Revenue":  0.37, "GrossMargin": -0.21, "NetMargin": -0.56, "EBITDA": -1.03},
    "Beauty/Fashion":               {"Count": 86, "DealVal": 7.21, "Revenue":  0.29, "GrossMargin":  0.35, "NetMargin": -0.46, "EBITDA":  0.02},
    "Business Services":            {"Count": 18, "DealVal": 7.46, "Revenue": -0.09, "GrossMargin": -0.16, "NetMargin": -1.04, "EBITDA": -0.39},
    "Children/Education":           {"Count": 18, "DealVal": 7.05, "Revenue":  0.05, "GrossMargin": -0.13, "NetMargin":  0.03, "EBITDA":  0.12},
    "Electronics":                  {"Count":  2, "DealVal": 6.22, "Revenue":  1.52, "GrossMargin": -2.70, "NetMargin": -0.86, "EBITDA": -0.32},
    "Entertainment":                {"Count":  3, "DealVal": 8.58, "Revenue":  0.06, "GrossMargin": -1.44, "NetMargin":  0.03, "EBITDA": -0.81},
    "Fitness/Sports/Outdoors":      {"Count": 16, "DealVal": 7.52, "Revenue":  0.43, "GrossMargin":  1.03, "NetMargin":  1.14, "EBITDA":  2.28},
    "Food And Beverage":            {"Count": 95, "DealVal": 6.95, "Revenue":  0.04, "GrossMargin": -0.23, "NetMargin":  0.12, "EBITDA": -0.81},
    "Green/Cleantech":              {"Count":  8, "DealVal": 6.49, "Revenue":  0.34, "GrossMargin": -1.50, "NetMargin": -1.45, "EBITDA":  0.92},
    "Hardware":                     {"Count":  1, "DealVal": 3.53, "Revenue":  0.23, "GrossMargin": -0.23, "NetMargin":  0.03, "EBITDA": -0.32},
    "Lifestyle/Home":               {"Count": 20, "DealVal": 7.39, "Revenue":  0.53, "GrossMargin":  0.89, "NetMargin":  0.78, "EBITDA": -0.46},
    "Liquor/Alcohol":               {"Count":  3, "DealVal": 7.24, "Revenue": -0.19, "GrossMargin": -0.57, "NetMargin": -0.26, "EBITDA": -1.47},
    "Manufacturing":                {"Count": 24, "DealVal": 6.98, "Revenue":  0.26, "GrossMargin": -0.62, "NetMargin": -0.24, "EBITDA": -0.50},
    "Medical/Health":               {"Count": 45, "DealVal": 7.58, "Revenue": -0.41, "GrossMargin":  0.96, "NetMargin": -0.01, "EBITDA": -0.38},
    "Others":                       {"Count": 17, "DealVal": 6.99, "Revenue":  0.14, "GrossMargin": -2.31, "NetMargin": -1.74, "EBITDA":  0.67},
    "Technology/Software":          {"Count": 48, "DealVal": 7.47, "Revenue": -0.62, "GrossMargin": -0.30, "NetMargin":  1.54, "EBITDA":  1.50},
    "Vehicles/Electrical Vehicles": {"Count":  9, "DealVal": 6.59, "Revenue": -1.53, "GrossMargin":  2.57, "NetMargin":  0.03, "EBITDA":  0.30},
}

HISTORY_COLS = [
    "Timestamp", "Industry", "Pitcher Age Group", "Company Age (yrs)",
    "Revenue (₹L)", "Gross Margin (%)", "Net Margin (%)", "EBITDA (₹L)",
    "SKUs", "Has Patents", "Bootstrapped", "Ask Amount (₹L)",
    "Offered Equity (%)", "Predicted Valuation (₹L)",
    "Implied Valuation (₹L)", "Delta (%)",
]

# ─────────────────────────────────────────────────────────────────────────────
# CSS — Dark Navy Shark Tank India theme with animated SVG background
# ─────────────────────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Rajdhani:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

/* ── Design tokens ── */
:root {
  --navy-900: #020B18;
  --navy-800: #041525;
  --navy-700: #061D33;
  --navy-600: #0A2540;
  --navy-500: #0D2E50;
  --navy-400: #123860;
  --teal:     #00C8BE;
  --teal-b:   #00A89E;
  --teal-dim: rgba(0,200,190,.12);
  --teal-glow:rgba(0,200,190,.25);
  --gold:     #F5C842;
  --gold-dim: rgba(245,200,66,.12);
  --red:      #E63946;
  --red-dim:  rgba(230,57,70,.12);
  --txt:      #C8E6F5;
  --muted:    #4A7A9B;
  --border:   rgba(0,200,190,.13);
  --border2:  rgba(255,255,255,.055);
  --font-head:'Bebas Neue', sans-serif;
  --font-body:'Rajdhani', sans-serif;
  --font-mono:'DM Mono', monospace;
}

/* ── Base ── */
.stApp {
  background: var(--navy-900);
  font-family: var(--font-body);
  color: var(--txt);
  letter-spacing: .02em;
}
.block-container { padding: 1rem 2.2rem 2.5rem !important; }

/* ── Animated ocean background ── */
.stApp::before {
  content: '';
  position: fixed;
  inset: 0;
  background:
    radial-gradient(ellipse 80% 40% at 20% 80%, rgba(0,60,100,.35) 0%, transparent 60%),
    radial-gradient(ellipse 60% 50% at 80% 20%, rgba(0,40,80,.3)  0%, transparent 55%),
    radial-gradient(ellipse 100% 60% at 50% 100%, rgba(0,30,60,.5) 0%, transparent 50%),
    linear-gradient(180deg, #020B18 0%, #041525 40%, #061830 70%, #020B18 100%);
  pointer-events: none;
  z-index: 0;
}

/* subtle animated scan-lines / waves */
.stApp::after {
  content: '';
  position: fixed;
  inset: 0;
  background: repeating-linear-gradient(
    0deg,
    transparent,
    transparent 3px,
    rgba(0,200,190,.012) 3px,
    rgba(0,200,190,.012) 4px
  );
  pointer-events: none;
  z-index: 0;
  animation: scanlines 8s linear infinite;
}
@keyframes scanlines { from { background-position: 0 0; } to { background-position: 0 40px; } }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #030F1E 0%, #051829 100%) !important;
  border-right: 1px solid var(--border);
}

/* ── Typography helpers ── */
.hd   { font-family: var(--font-head); }
.grad-teal {
  background: linear-gradient(100deg, var(--teal) 0%, #7FF5F0 50%, var(--gold) 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.page-h1 { font-family: var(--font-head); font-size: 3rem; line-height: 1; margin: 0; letter-spacing: .04em; }
.page-sub { color: var(--muted); font-size: .95rem; margin-top: .4rem; font-weight: 400; line-height: 1.6; }
.badge {
  display: inline-block;
  background: var(--teal-dim); border: 1px solid var(--teal); color: var(--teal);
  font-family: var(--font-head); font-size: .75rem; letter-spacing: .18em;
  padding: .2rem .75rem; border-radius: 3px; margin-bottom: .75rem;
}
.cap {
  font-family: var(--font-head); font-size: .72rem; letter-spacing: .18em;
  color: var(--teal); margin-bottom: .5rem;
}
.divl {
  border: none; height: 1px;
  background: linear-gradient(90deg, transparent, var(--teal), transparent);
  margin: 1.5rem 0;
}

/* ── Cards ── */
.card {
  background: rgba(6,25,51,.65);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 1.2rem 1.35rem;
  margin-bottom: .85rem;
  backdrop-filter: blur(14px);
  position: relative;
  overflow: hidden;
}
.card::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, transparent, var(--teal), transparent);
  opacity: .5;
}
.csm {
  background: var(--navy-600);
  border: 1px solid var(--border2);
  border-radius: 8px;
  padding: .7rem 1rem;
  margin-bottom: .6rem;
}

/* ── Result box ── */
.rbox {
  background: linear-gradient(135deg, rgba(0,200,190,.1) 0%, rgba(245,200,66,.07) 100%);
  border: 1.5px solid var(--teal);
  border-radius: 14px;
  padding: 2rem 1.5rem;
  text-align: center;
  position: relative;
  overflow: hidden;
  animation: rbox-pulse 3s ease-in-out infinite;
}
@keyframes rbox-pulse {
  0%,100% { box-shadow: 0 0 25px rgba(0,200,190,.1), inset 0 0 20px rgba(0,200,190,.03); }
  50%      { box-shadow: 0 0 55px rgba(0,200,190,.22), inset 0 0 35px rgba(0,200,190,.07); }
}
.rlabel { font-family: var(--font-head); font-size: .75rem; letter-spacing: .18em; color: var(--muted); margin-bottom: .3rem; }
.rval {
  font-family: var(--font-head); font-size: 4.2rem; line-height: 1; letter-spacing: .04em;
  background: linear-gradient(90deg, var(--teal) 0%, #A0F5F0 55%, var(--gold) 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.runit { font-size: .88rem; color: var(--gold); margin-top: .3rem; letter-spacing: .08em; }

/* ── KPI tiles ── */
.kpi { background: var(--navy-600); border: 1px solid var(--border2); border-radius: 9px; padding: .62rem .9rem; text-align: center; margin-bottom: .55rem; }
.klbl { font-family: var(--font-head); font-size: .62rem; letter-spacing: .14em; color: var(--muted); }
.kval { font-family: var(--font-head); font-size: 1.3rem; letter-spacing: .04em; color: var(--teal); }
.kdelta { font-size: .72rem; margin-top: .1rem; }

/* ── Score bars ── */
.sbw { margin-bottom: .5rem; }
.sbl { display: flex; justify-content: space-between; font-size: .75rem; color: var(--muted); margin-bottom: 3px; }
.sbt { background: rgba(255,255,255,.055); border-radius: 3px; height: 5px; }
.sbf { border-radius: 3px; height: 100%; transition: width .5s ease; }

/* ── Widget overrides ── */
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stTextInput > div > div > input {
  background: var(--navy-500) !important;
  border: 1px solid var(--border) !important;
  color: var(--txt) !important;
  border-radius: 7px !important;
  font-family: var(--font-body) !important;
}
/* ── Button ── */
.stButton > button {
  background: linear-gradient(90deg, var(--teal) 0%, var(--teal-b) 100%) !important;
  color: var(--navy-900) !important;
  font-family: var(--font-head) !important;
  font-weight: 400 !important;
  font-size: 1rem !important;
  letter-spacing: .14em !important;
  border: none !important;
  border-radius: 7px !important;
  padding: .72rem 1.8rem !important;
  width: 100% !important;
  box-shadow: 0 4px 20px rgba(0,200,190,.28) !important;
  transition: all .18s !important;
}
.stButton > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 30px rgba(0,200,190,.45) !important;
}
.stDownloadButton > button {
  background: var(--navy-500) !important;
  color: var(--teal) !important;
  border: 1px solid var(--border) !important;
  border-radius: 7px !important;
  font-family: var(--font-head) !important;
  letter-spacing: .12em !important;
  font-size: .82rem !important;
}
/* ── Metrics ── */
[data-testid="stMetric"] { background: var(--navy-600) !important; border: 1px solid var(--border2) !important; border-radius: 10px !important; padding: .85rem 1rem !important; }
[data-testid="stMetricValue"] { font-family: var(--font-head) !important; color: var(--teal) !important; letter-spacing: .04em !important; }
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: .68rem !important; text-transform: uppercase; letter-spacing: .1em; }
/* ── Expander ── */
[data-testid="stExpander"] summary { background: var(--navy-600) !important; border: 1px solid var(--border) !important; border-radius: 9px !important; font-family: var(--font-head) !important; letter-spacing: .1em !important; color: var(--teal) !important; }
/* ── Tabs ── */
[data-baseweb="tab-list"] { background: var(--navy-600) !important; border-radius: 10px !important; padding: 4px !important; }
[data-baseweb="tab"] { color: var(--muted) !important; font-family: var(--font-head) !important; letter-spacing: .1em !important; border-radius: 7px !important; }
[aria-selected="true"] { background: var(--teal) !important; color: var(--navy-900) !important; }
/* ── Misc ── */
.stAlert { border-radius: 9px !important; }
.stRadio > div { gap: .3rem !important; }
[data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: 9px !important; }

/* ── Shark swim animation ── */
@keyframes shark-swim {
  0%   { transform: translateX(-120px) scaleX(1); opacity: 0; }
  10%  { opacity: 1; }
  45%  { transform: translateX(calc(50vw - 60px)) scaleX(1); opacity: 1; }
  55%  { transform: translateX(calc(50vw - 60px)) scaleX(-1); opacity: 1; }
  90%  { transform: translateX(-120px) scaleX(-1); opacity: 1; }
  100% { transform: translateX(-120px) scaleX(-1); opacity: 0; }
}
@keyframes shark-bite {
  0%,100% { transform: scaleY(1); }
  50%      { transform: scaleY(1.08) rotate(-3deg); }
}
@keyframes bubble-rise {
  0%   { transform: translateY(0) scale(1); opacity: .7; }
  100% { transform: translateY(-80px) scale(1.5); opacity: 0; }
}
@keyframes result-reveal {
  0%   { opacity: 0; transform: translateY(20px) scale(.96); }
  100% { opacity: 1; transform: translateY(0) scale(1); }
}
@keyframes teal-flicker {
  0%,100% { opacity: 1; }
  50%      { opacity: .7; }
}
@keyframes wave-drift {
  0%   { transform: translateX(0); }
  100% { transform: translateX(-50%); }
}

.result-reveal { animation: result-reveal .6s cubic-bezier(.22,1,.36,1) both; }

/* ── Ocean wave band at bottom of sidebar ── */
.wave-band {
  position: relative;
  height: 40px;
  overflow: hidden;
  margin-top: 1rem;
}
</style>
"""

# Plotly base layout
_PL = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Rajdhani, sans-serif", color="#C8E6F5", size=12),
    xaxis=dict(gridcolor="rgba(255,255,255,.05)", linecolor="rgba(255,255,255,.07)", zeroline=False),
    yaxis=dict(gridcolor="rgba(255,255,255,.05)", linecolor="rgba(255,255,255,.07)", zeroline=False),
    margin=dict(l=10, r=10, t=40, b=10),
)

# ADD THIS after _PL = dict(...)
def _pl(**overrides):
    """Merge layout overrides into _PL without duplicate-key crashes."""
    base = dict(_PL)
    for k, v in overrides.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            base[k] = {**base[k], **v}   # deep-merge axis dicts
        else:
            base[k] = v
    return base

CHART_CFG = {"displayModeBar": False}



# ─────────────────────────────────────────────────────────────────────────────
# SHARK ANIMATION  (SVG + JS injected via st.markdown)
# ─────────────────────────────────────────────────────────────────────────────
SHARK_ANIM_HTML = """
<!DOCTYPE html>
<html>
<head>
<style>
  /* Reset iframe body */
  * { margin:0; padding:0; box-sizing:border-box; }
  html, body { background: transparent; overflow: hidden; }

  /* ── All keyframes must live here — iframe has no parent CSS ── */
  @keyframes shark-swim {
    0%   { left: -220px; opacity: 0; transform: scaleX(1); }
    8%   { opacity: 1; }
    42%  { left: calc(50vw - 90px); opacity: 1; transform: scaleX(1); }
    50%  { left: calc(50vw - 90px); opacity: 1; transform: scaleX(-1); }
    92%  { left: -220px; opacity: 1; transform: scaleX(-1); }
    100% { left: -220px; opacity: 0; transform: scaleX(-1); }
  }
  @keyframes shark-bite {
    0%,100% { transform: scaleY(1) rotate(0deg); }
    50%     { transform: scaleY(1.1) rotate(-4deg); }
  }
  @keyframes bubble-rise {
    0%   { transform: translateY(0) scale(1); opacity: 0.75; }
    100% { transform: translateY(-120px) scale(1.6); opacity: 0; }
  }
  @keyframes flash-in {
    0%   { opacity: 0; transform: translate(-50%,-50%) scale(0.6); }
    30%  { opacity: 1; transform: translate(-50%,-50%) scale(1.08); }
    70%  { opacity: 1; transform: translate(-50%,-50%) scale(1); }
    100% { opacity: 0; transform: translate(-50%,-50%) scale(0.9); }
  }
  @keyframes ripple {
    0%   { transform: translate(-50%,-50%) scale(0); opacity: 0.6; }
    100% { transform: translate(-50%,-50%) scale(4); opacity: 0; }
  }

  /* Overlay covers the iframe viewport which sits in the page */
  #overlay {
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    pointer-events: none;
    overflow: hidden;
  }

  /* Shark */
  #shark {
    position: fixed;
    bottom: 5%;
    left: -220px;
    width: 200px; height: 100px;
    filter: drop-shadow(0 0 20px rgba(0,200,190,.65));
    animation: shark-swim 3.8s cubic-bezier(.45,0,.55,1) forwards;
    z-index: 100;
  }
  #shark-mouth-group {
    animation: shark-bite 0.3s ease-in-out 1.6s 5 both;
    transform-origin: 165px 55px;
  }

  /* Deal flash */
  #deal-flash {
    display: none;
    position: fixed;
    top: 40%; left: 50%;
    transform: translate(-50%,-50%);
    font-family: 'Bebas Neue', 'Impact', sans-serif;
    font-size: 3rem;
    letter-spacing: .12em;
    color: #F5C842;
    text-shadow: 0 0 30px rgba(245,200,66,.9), 0 0 60px rgba(0,200,190,.6);
    animation: flash-in 1.8s ease forwards;
    z-index: 200;
    white-space: nowrap;
  }

  /* Ripple ring at centre */
  .ripple {
    position: fixed;
    top: 40%; left: 50%;
    width: 60px; height: 60px;
    border-radius: 50%;
    border: 3px solid #00C8BE;
    transform: translate(-50%,-50%) scale(0);
    animation: ripple 1.2s ease-out forwards;
    z-index: 150;
  }
</style>
</head>
<body>
<div id="overlay">

  <!-- Shark SVG -->
  <svg id="shark" viewBox="0 0 200 100" xmlns="http://www.w3.org/2000/svg">
    <!-- body -->
    <ellipse cx="100" cy="55" rx="75" ry="28" fill="#0A2540" stroke="#00C8BE" stroke-width="1.4"/>
    <!-- belly highlight -->
    <ellipse cx="100" cy="67" rx="54" ry="11" fill="#062035" opacity=".75"/>
    <!-- glow line top -->
    <path d="M32,36 Q100,22 170,43" stroke="#00C8BE" stroke-width=".7" fill="none" opacity=".45"/>
    <!-- dorsal fin -->
    <polygon points="92,27 118,8 131,27" fill="#0D2E50" stroke="#00C8BE" stroke-width="1.1"/>
    <!-- tail fin -->
    <polygon points="27,55 4,28 4,82" fill="#0D2E50" stroke="#00C8BE" stroke-width="1.1"/>
    <!-- tail fin lower -->
    <polygon points="27,55 4,55 12,72" fill="#0A2540" stroke="#00C8BE" stroke-width=".8"/>
    <!-- pectoral fin -->
    <polygon points="112,65 88,86 132,76" fill="#0A2540" stroke="#00C8BE" stroke-width=".9"/>
    <!-- gills -->
    <line x1="138" y1="39" x2="133" y2="69" stroke="#00C8BE" stroke-width="1" opacity=".55"/>
    <line x1="146" y1="37" x2="141" y2="67" stroke="#00C8BE" stroke-width="1" opacity=".55"/>
    <!-- eye outer glow -->
    <circle cx="155" cy="47" r="7" fill="#00C8BE" opacity=".25"/>
    <!-- eye -->
    <circle cx="155" cy="47" r="5" fill="#00C8BE" opacity=".95"/>
    <circle cx="156" cy="46" r="2.2" fill="#020B18"/>
    <circle cx="157" cy="45" r=".7" fill="#00C8BE" opacity=".6"/>
    <!-- animated mouth group -->
    <g id="shark-mouth-group">
      <path d="M164,55 Q180,46 190,55 Q180,64 164,55Z" fill="#00C8BE" opacity=".9"/>
      <polygon points="167,55 170,49 173,55" fill="white"/>
      <polygon points="173,55 176,49 179,55" fill="white"/>
      <polygon points="179,55 182,49 185,55" fill="white"/>
      <polygon points="186,55 188,50 190,55" fill="white"/>
    </g>
  </svg>

  <!-- Deal flash text -->
  <div id="deal-flash">DEAL STRUCK!</div>

</div>

<script>
(function(){
  const overlay = document.getElementById('overlay');

  // ── Bubbles ──────────────────────────────────────────────────────────────
  for (let i = 0; i < 22; i++) {
    const b  = document.createElement('div');
    const sz = 5 + Math.random() * 16;
    const delay = Math.random() * 2.5;
    const dur   = 1.4 + Math.random() * 2;
    b.style.cssText = [
      'position:fixed',
      `width:${sz}px`, `height:${sz}px`,
      'border-radius:50%',
      `border:1.5px solid rgba(0,200,190,${(0.25 + Math.random() * 0.55).toFixed(2)})`,
      `background:rgba(0,200,190,${(0.03 + Math.random() * 0.09).toFixed(2)})`,
      `left:${8 + Math.random() * 84}%`,
      `bottom:${4 + Math.random() * 45}%`,
      `animation:bubble-rise ${dur}s ease-out ${delay}s forwards`,
      'z-index:90',
    ].join(';');
    overlay.appendChild(b);
  }

  // ── Ripple rings ─────────────────────────────────────────────────────────
  [1700, 2000, 2300].forEach((t, i) => {
    setTimeout(() => {
      const r = document.createElement('div');
      r.className = 'ripple';
      r.style.animationDelay = '0s';
      overlay.appendChild(r);
    }, t);
  });

  // ── Deal flash ────────────────────────────────────────────────────────────
  setTimeout(() => {
    const f = document.getElementById('deal-flash');
    if (f) f.style.display = 'block';
  }, 1700);

  // ── Cleanup ───────────────────────────────────────────────────────────────
  setTimeout(() => {
    if (overlay) overlay.style.display = 'none';
  }, 5000);

})();
</script>
</body>
</html>
"""

# Ocean wave background strip for sidebar
WAVE_HTML = """
<div class="wave-band">
<svg viewBox="0 0 400 40" preserveAspectRatio="none" width="100%" height="40">
  <defs>
    <linearGradient id="wg" x1="0" x2="0" y1="0" y2="1">
      <stop offset="0%" stop-color="#00C8BE" stop-opacity=".18"/>
      <stop offset="100%" stop-color="#00C8BE" stop-opacity=".03"/>
    </linearGradient>
  </defs>
  <path d="M0,20 Q50,5 100,20 Q150,35 200,20 Q250,5 300,20 Q350,35 400,20 L400,40 L0,40Z"
        fill="url(#wg)" style="animation:wave-drift 4s linear infinite;"/>
  <path d="M0,25 Q50,12 100,25 Q150,38 200,25 Q250,12 300,25 Q350,38 400,25 L400,40 L0,40Z"
        fill="url(#wg)" style="animation:wave-drift 6s linear infinite reverse;" opacity=".6"/>
</svg>
</div>
"""

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
def init_state() -> None:
    defaults = {
        "page":         "engine",
        "last_inputs":  None,
        "last_pred":    None,
        "pred_done":    False,
        "show_anim":    False,
        "history_list": [],          # O(1) list — converted to DF only on render
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING  — auto-loads from repo path; sidebar upload as fallback
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model_from_path(path: str):
    """Load statsmodels model from a file path (repo-relative)."""
    try:
        return sm.load(path), None
    except Exception as e:
        return None, str(e)

@st.cache_resource
def load_model_from_bytes(raw: bytes):
    """Load statsmodels model from raw bytes (sidebar upload fallback)."""
    try:
        return sm.load(io.BytesIO(raw)), None
    except Exception as e:
        return None, str(e)

def get_model():
    """
    Model resolution order:
    1. Auto-load from  models/valuation_model_1.pkl  (present in GitHub repo)
    2. Fallback to sidebar file-uploader bytes
    Returns (model, error_str | None)
    """
    if os.path.exists(MODEL_PATH):
        return load_model_from_path(MODEL_PATH)
    raw = st.session_state.get("model_bytes")
    if raw:
        return load_model_from_bytes(raw)
    return None, "no_model"


# ─────────────────────────────────────────────────────────────────────────────
# Checking for logical deal breakers
# ─────────────────────────────────────────────────────────────────────────────

def check_dealbreakers(inputs: dict) -> list[str]:
    """Evaluates inputs against hard investor thresholds."""
    flags = []
    
    # 1. EBITDA Threshold
    if inputs["ebitda"] <= -20:
        flags.append("<strong>Severe Cash Burn:</strong> EBITDA ≤ -20L. High execution risk; sharks rarely fund deep burn unless it's a high-growth tech play.")
        
    # 2. SKU Bloat
    if inputs["skus"] > 3000:
        flags.append("<strong>Inventory Nightmare:</strong> > 3,000 SKUs. Indicates a lack of product focus (no 'hero' product) and massive working capital drain.")
        
    # 3. Gross Margin Floor
    if inputs["gross_margin"] < 15:
        flags.append("<strong>Broken Unit Economics:</strong> Gross Margin < 15%. Leaves almost no room to absorb marketing (CAC) and logistics costs at scale.")
        
    # 4. Net Margin Floor
    if inputs["net_margin"] <= -30:
        flags.append("<strong>Unsustainable Margins:</strong> Net Margin ≤ -30%. The path to profitability is incredibly steep, requiring a total operational pivot.")
        
    return flags
       
# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING  (exact replica of training pipeline)
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(inputs: dict) -> pd.DataFrame:
    """
    Transform raw user inputs → model-ready feature row.

    Pipeline (mirrors EDA.ipynb cell 10 → cell 26 → log_valuation.ipynb):
      Step 1  Industry → 17 OHE dummies (Hardware = all zeros baseline)
      Step 2  log1p(x) for: Yearly Revenue, SKUs, Original Ask Amount
      Step 3  raw (no log) for: Gross Margin%, Net Margin%, EBITDA, Equity%, Age
              NOTE: sym_log is defined for display/radar use only — NOT applied
              here. EDA never log-transformed these columns before scaling.
      Step 4  z-score all 8 continuous cols using SCALER_PARAMS
      Step 5  Ordinal pass-through: Pitchers Average Age (0/1/2)
      Step 6  Binary pass-through: Has Patents, Bootstrapped
      Step 7  Insert const=1.0 at column 0 (avoids statsmodels has_constant bug)
    """
    row: dict[str, float] = {}

    # Step 1 — industry dummies
    for d in INDUSTRY_DUMMIES:
        row[d] = 1.0 if inputs["industry"] == d else 0.0

    # Steps 2-4 — numeric features
    raw_map = {
        "Yearly Revenue":          inputs["yearly_revenue"],
        "Gross Margin":            inputs["gross_margin"],
        "Net Margin":              inputs["net_margin"],
        "EBITDA":                  inputs["ebitda"],
        "SKUs":                    inputs["skus"],
        "Original Ask Amount":     inputs["ask_amount"],
        "Original Offered Equity": inputs["offered_equity"],
        "Company Age":             inputs["company_age"],
    }
    for col, val in raw_map.items():
        v = float(val)
        if col in LOG_THEN_SCALE:
            v = np.log1p(v)      # only Revenue, SKUs, Ask Amount
        mean, std = SCALER_PARAMS[col]
        row[col] = (v - mean) / std

    # Steps 5-6 — pass-throughs
    row["Pitchers Average Age"] = float(PITCHER_AGE_MAP[inputs["pitcher_age_group"]])
    row["Has Patents"]          = float(inputs["has_patents"])
    row["Bootstrapped"]         = float(inputs["bootstrapped"])

    df_row = pd.DataFrame([row])[MODEL_FEATURE_ORDER].astype(float)
    df_row.insert(0, "const", 1.0)   # intercept — avoids has_constant=True bug
    return df_row


def run_prediction(model, inputs: dict) -> float:
    """Predict → invert log1p target → ₹ Lakhs."""
    log_pred = model.predict(preprocess(inputs))[0]
    return float(np.expm1(log_pred))

# ─────────────────────────────────────────────────────────────────────────────
# HISTORY HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _implied(ask: float, eq: float) -> float:
    return (ask / eq * 100) if eq > 0 else 0.0

def append_history(inputs: dict, pred: float) -> None:
    """O(1) list append + persistent CSV write."""
    ask   = inputs["ask_amount"]
    eq    = inputs["offered_equity"]
    impl  = _implied(ask, eq)
    delta = ((pred - impl) / impl * 100) if impl > 0 else 0.0
    ts    = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    record = {
        "Timestamp":                ts,
        "Industry":                 inputs["industry"],
        "Pitcher Age Group":        inputs["pitcher_age_group"],
        "Company Age (yrs)":        inputs["company_age"],
        "Revenue (₹L)":             inputs["yearly_revenue"],
        "Gross Margin (%)":         inputs["gross_margin"],
        "Net Margin (%)":           inputs["net_margin"],
        "EBITDA (₹L)":              inputs["ebitda"],
        "SKUs":                     inputs["skus"],
        "Has Patents":              "Yes" if inputs["has_patents"] else "No",
        "Bootstrapped":             "Yes" if inputs["bootstrapped"] else "No",
        "Ask Amount (₹L)":          ask,
        "Offered Equity (%)":       eq,
        "Predicted Valuation (₹L)": round(pred, 2),
        "Implied Valuation (₹L)":   round(impl, 2),
        "Delta (%)":                round(delta, 2),
    }

    # O(1) in-memory append
    st.session_state["history_list"].append(record)

    # Persistent CSV — creates file if missing, appends thereafter
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    pd.DataFrame([record]).to_csv(
        HISTORY_PATH,
        mode="a",
        header=not os.path.exists(HISTORY_PATH),
        index=False,
    )

def get_history_df() -> pd.DataFrame:
    """Materialise list → DataFrame only when rendering the history table."""
    if not st.session_state["history_list"]:
        return pd.DataFrame(columns=HISTORY_COLS)
    return pd.DataFrame(st.session_state["history_list"])

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar(model_available: bool) -> str:
    with st.sidebar:
        # Logo block
        st.markdown("""
        <div style='text-align:center;padding:1.2rem 0 .8rem'>
          <div style='font-size:2.8rem;filter:drop-shadow(0 0 12px rgba(0,200,190,.6))'>🦈</div>
          <div style='font-family:"Bebas Neue",sans-serif;font-size:1.4rem;
                      letter-spacing:.18em;color:#00C8BE;margin-top:.3rem'>VALUATION ENGINE</div>
          <div style='font-size:.68rem;color:#4A7A9B;letter-spacing:.12em;margin-top:.15rem'>
            SHARK TANK INDIA · MLR ANALYTICS</div>
        </div>
        <hr style='border:none;height:1px;background:rgba(0,200,190,.13);margin:.3rem 0 .9rem'>
        """, unsafe_allow_html=True)

        # Model status
        if model_available:
            st.markdown("""
            <div style='background:rgba(0,200,190,.08);border:1px solid rgba(0,200,190,.25);
                        border-radius:7px;padding:.5rem .8rem;text-align:center;margin-bottom:.8rem;
                        font-family:"Bebas Neue",sans-serif;letter-spacing:.12em;color:#00C8BE;font-size:.85rem'>
              MODEL LOADED
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown('<p class="cap">MODEL FILE</p>', unsafe_allow_html=True)
            up = st.file_uploader(
                "Upload valuation_model_1.pkl", type=["pkl"],
                help="Or place it at  models/valuation_model_1.pkl  in your repo.",
                label_visibility="collapsed",
            )
            if up is not None:
                raw = up.read()
                if raw != st.session_state.get("model_bytes"):
                    st.session_state["model_bytes"] = raw
                    st.cache_resource.clear()
                st.success("Model uploaded!")

        st.markdown("<br>", unsafe_allow_html=True)

        # Navigation
        st.markdown('<p class="cap">NAVIGATE</p>', unsafe_allow_html=True)
        for key, label in [("engine", "🦈  Valuation Engine"), ("analytics", "📊  Pitch Analytics"), ("history", "🗂️  History")]:
            if st.button(label, key=f"nav_{key}", use_container_width=True):
                st.session_state["page"] = key
                st.rerun()

        # Last prediction pill
        if st.session_state.get("last_pred") is not None:
            v   = st.session_state["last_pred"]
            ind = st.session_state["last_inputs"]["industry"]
            n   = len(st.session_state["history_list"])
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style='background:rgba(0,200,190,.07);border:1px solid rgba(0,200,190,.22);
                        border-radius:9px;padding:.8rem;text-align:center;'>
              <div style='font-family:"Bebas Neue",sans-serif;font-size:.62rem;
                          letter-spacing:.15em;color:#4A7A9B;margin-bottom:.3rem'>LAST PREDICTION</div>
              <div style='font-family:"Bebas Neue",sans-serif;font-size:1.8rem;
                          letter-spacing:.06em;color:#00C8BE'>₹ {v:,.1f} L</div>
              <div style='font-size:.68rem;color:#4A7A9B;margin-top:.18rem'>{ind}</div>
            </div>
            <div style='text-align:center;margin-top:.4rem;font-size:.65rem;color:#4A7A9B;
                        letter-spacing:.1em'>{n} PREDICTION(S) SAVED</div>
            """, unsafe_allow_html=True)

        # Wave strip
        st.markdown(WAVE_HTML, unsafe_allow_html=True)
        st.markdown("<div style='font-size:.6rem;color:#1A3A55;text-align:center;letter-spacing:.1em'>OLS · STATSMODELS · SHARK TANK S1–S3</div>", unsafe_allow_html=True)

    return st.session_state["page"]

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — VALUATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────
@st.fragment
def _input_fragment(model) -> None:
    """
    @st.fragment — only this block reruns on widget interaction.
    Dragging a slider will NOT rerun the hero section, charts, or result.
    """
    st.markdown('<p class="cap">CONFIGURE YOUR STARTUP</p>', unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns(3, gap="medium")

    # ── Column A: Identity ────────────────────────────────────────────────────
    with col_a:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="cap">IDENTITY</p>', unsafe_allow_html=True)
        industry = st.selectbox("Industry Sector", ALL_INDUSTRIES,
                                index=ALL_INDUSTRIES.index("Beauty/Fashion"),
                                help="Hardware = all-zero baseline (dropped dummy)")
        pitcher_age_group = st.selectbox("Founder Age Group", list(PITCHER_AGE_MAP.keys()),
                                         help="Ordinal encoded: Young=0, Middle=1, Old=2")
        company_age = st.number_input("Company Age (years)", 0.0, 50.0, 3.0, 0.5)
        st.markdown("</div>", unsafe_allow_html=True)

        bench = INDUSTRY_BENCHMARKS.get(industry, INDUSTRY_BENCHMARKS["Others"])
        st.markdown(f"""
        <div class="csm">
          <div style='font-family:"Bebas Neue",sans-serif;font-size:.62rem;letter-spacing:.14em;color:#4A7A9B'>
            AVG DEAL VAL · {industry.upper()}</div>
          <div style='font-family:"Bebas Neue",sans-serif;color:#00C8BE;font-size:1.1rem;letter-spacing:.04em'>
            {bench['DealVal']:.2f} <span style='font-size:.7rem;color:#4A7A9B'>LOG SCALE</span></div>
        </div>
        <div class="csm">
          <div style='font-family:"Bebas Neue",sans-serif;font-size:.62rem;letter-spacing:.14em;color:#4A7A9B'>
            DEALS IN TRAINING SET</div>
          <div style='font-family:"Bebas Neue",sans-serif;color:#F5C842;font-size:1.1rem;letter-spacing:.04em'>
            {bench['Count']}</div>
        </div>""", unsafe_allow_html=True)

    # ── Column B: Ask Details ─────────────────────────────────────────────────
    with col_b:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="cap">ASK DETAILS</p>', unsafe_allow_html=True)
        ask_amount     = st.number_input("Ask Amount (₹ Lakhs)", 1.0, 500.0, 50.0, 5.0,
                                         help="log1p-transformed in training. Capped at 500 in EDA.")
        offered_equity = st.slider("Equity Offered (%)", 0.1, 100.0, 10.0, 0.5)
        impl_live      = _implied(ask_amount, offered_equity)
        st.markdown(
            f"<div style='background:rgba(245,200,66,.07);border:1px solid rgba(245,200,66,.25);"
            f"border-radius:7px;padding:.45rem .8rem;text-align:center;margin-top:.5rem;"
            f"font-family:\"Bebas Neue\",sans-serif;letter-spacing:.08em;font-size:.82rem;color:#4A7A9B'>"
            f"IMPLIED VALUATION: <span style='color:#F5C842;font-size:1.1rem'>₹ {impl_live:,.1f} L</span></div>",
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="cap">BUSINESS PROFILE</p>', unsafe_allow_html=True)
        has_patents_raw  = st.radio("Intellectual Property", ["Has Patents", "No Patents"], index=1)
        bootstrapped_raw = st.radio("Funding Status", ["Bootstrapped", "Externally Funded"], index=0)
        has_patents  = 1 if has_patents_raw == "Has Patents" else 0
        bootstrapped = 1 if bootstrapped_raw == "Bootstrapped" else 0
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Column C: Financials ──────────────────────────────────────────────────
    with col_c:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="cap">FINANCIALS</p>', unsafe_allow_html=True)
        yearly_revenue = st.number_input("Yearly Revenue (₹ Lakhs)", 0.0, 10000.0, 120.0, 10.0,
                                         help="log1p-transformed then z-scored during training")
        gross_margin   = st.number_input("Gross Margin (%)", -200.0, 100.0, 35.0, 1.0,
                                         help="Raw % — z-scored directly, no log transform in EDA")
        net_margin     = st.number_input("Net Margin (%)", -500.0, 100.0, -5.0, 1.0,
                                         help="Raw % — z-scored directly, no log transform in EDA")
        ebitda         = st.number_input("EBITDA (₹ Lakhs)", -5000.0, 50000.0, 10.0, 5.0,
                                         help="Raw ₹L — z-scored directly. sym_log used for display only.")
        skus           = st.number_input("Number of SKUs", 0, 2000, 5, 1,
                                         help="log1p-transformed then z-scored during training")
        st.markdown("</div>", unsafe_allow_html=True)

    # ── CTA ───────────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    btn_col, _ = st.columns([1, 2])
    with btn_col:
        clicked = st.button("🦈  CALCULATE VALUATION", use_container_width=True)


    # AFTER — add st.rerun() so the outer page_engine re-runs to show animation + result:
    if clicked:
        inp = dict(
                industry=industry, pitcher_age_group=pitcher_age_group,
                company_age=company_age, ask_amount=ask_amount,
                offered_equity=offered_equity, has_patents=has_patents,
                bootstrapped=bootstrapped, yearly_revenue=yearly_revenue,
                gross_margin=gross_margin, net_margin=net_margin,
                ebitda=ebitda, skus=skus,
            )
        try:
            pred = run_prediction(model, inp)
            st.session_state.update({"last_inputs": inp, "last_pred": pred,
                                    "pred_done": True, "show_anim": True})
            append_history(inp, pred)
            st.rerun(scope="app")   # ← Forces the outer page to render the result
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.session_state["pred_done"] = False


def page_engine(model) -> None:
    # Hero
    st.markdown('<div class="badge">SHARK TANK INDIA · SEASON ANALYTICS</div>', unsafe_allow_html=True)
    h1, h2 = st.columns([2, 1], gap="large")
    with h1:
        st.markdown('<h1 class="page-h1 grad-teal">STARTUP<br>VALUATION<br>PREDICTOR</h1>', unsafe_allow_html=True)
        st.markdown('<p class="page-sub">OLS regression trained on 423 real Shark Tank India pitches. Enter your metrics — the model predicts how the Sharks value your deal.</p>', unsafe_allow_html=True)
    with h2:
        st.markdown("""
        <div style='display:flex;flex-direction:column;gap:.5rem;margin-top:.5rem'>
          <div class='csm' style='text-align:center'>
            <div style='font-family:"Bebas Neue",sans-serif;font-size:.6rem;letter-spacing:.14em;color:#4A7A9B'>TRAINING SAMPLES</div>
            <div style='font-family:"Bebas Neue",sans-serif;font-size:1.6rem;letter-spacing:.04em;color:#00C8BE'>423</div>
          </div>
          <div class='csm' style='text-align:center'>
            <div style='font-family:"Bebas Neue",sans-serif;font-size:.6rem;letter-spacing:.14em;color:#4A7A9B'>INDUSTRIES</div>
            <div style='font-family:"Bebas Neue",sans-serif;font-size:1.6rem;letter-spacing:.04em;color:#F5C842'>18</div>
          </div>
          <div class='csm' style='text-align:center'>
            <div style='font-family:"Bebas Neue",sans-serif;font-size:.6rem;letter-spacing:.14em;color:#4A7A9B'>MODEL FEATURES</div>
            <div style='font-family:"Bebas Neue",sans-serif;font-size:1.6rem;letter-spacing:.04em;color:#E63946'>28</div>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="divl">', unsafe_allow_html=True)

    if model is None:
        st.warning(
            "**No model found.** Place `valuation_model_1.pkl` inside a `models/` folder in your repo.\n\n"
            "Save from your notebook:\n```python\nmodel_1.save('models/valuation_model_1.pkl')\n```\n\n"
            "Or upload it via the sidebar uploader."
        )
        st.stop()

    # Fragment — partial rerun only
    _input_fragment(model)

    # ── Shark animation ───────────────────────────────────────────────────────
    # FIX 1: st.markdown strips <script> tags — use components.html() instead.
    # FIX 2: Do NOT set show_anim=False here; clear it AFTER rendering so the
    #         browser actually receives the HTML before the state is wiped.
    if st.session_state.get("show_anim"):
        # height=1 keeps the iframe minimal in layout; the animation uses
        # position:fixed inside the iframe which IS visible within the iframe
        # viewport. We need enough height for the shark to show.
        components.html(SHARK_ANIM_HTML, height=300, scrolling=False)
        st.session_state["show_anim"] = False   # clear AFTER injection

    # ── Result ────────────────────────────────────────────────────────────────
   # ── Result ────────────────────────────────────────────────────────────────
    if st.session_state.get("pred_done") and st.session_state.get("last_pred") is not None:
        pred = st.session_state["last_pred"]
        inp  = st.session_state["last_inputs"]
        impl = _implied(inp["ask_amount"], inp["offered_equity"])
        delta_pct = ((pred - impl) / impl * 100) if impl > 0 else 0.0

        st.markdown('<hr class="divl">', unsafe_allow_html=True)
        
        # --- DEALBREAKER EVALUATION ---
        dealbreakers = check_dealbreakers(inp)
        if dealbreakers:
            alerts_html = "".join([f"<li style='margin-bottom:.3rem; color:#C8E6F5; font-size:.85rem;'>{msg}</li>" for msg in dealbreakers])
            st.markdown(f"""
            <div style='background:rgba(230,57,70,.08); border:1px solid #E63946; border-radius:10px; padding:1.2rem; margin-bottom:1.5rem; animation: rbox-pulse 3s ease-in-out infinite;'>
              <div style='font-family:"Bebas Neue",sans-serif; font-size:1.2rem; letter-spacing:.1em; color:#E63946; margin-bottom:.5rem;'>
                🚨 DEALBREAKER WARNING — HIGH PROBABILITY OF "I'M OUT"
              </div>
              <ul style='margin:0; padding-left:1.2rem;'>
                {alerts_html}
              </ul>
            </div>
            """, unsafe_allow_html=True)
        # ------------------------------

        r1, r2 = st.columns([1.1, 0.9], gap="large")

        with r1:
            st.markdown(f"""
            <div class='rbox result-reveal'>
              <div class='rlabel'>PREDICTED DEAL VALUATION</div>
              <div class='rval'>₹ {pred:,.1f}</div>
              <div class='runit'>LAKHS · {inp['industry'].upper()}</div>
            </div>""", unsafe_allow_html=True)

        with r2:
            over = delta_pct > 0
            dc   = "#00C8BE" if over else "#E63946"
            icon = "▲" if over else "▼"
            verb = "MODEL EXCEEDS IMPLIED" if over else "MODEL BELOW IMPLIED"
            st.markdown(f"""
            <div>
              <div class='kpi'><div class='klbl'>ASK AMOUNT</div><div class='kval'>₹ {inp['ask_amount']:,.0f} L</div></div>
              <div class='kpi'><div class='klbl'>EQUITY OFFERED</div><div class='kval'>{inp['offered_equity']:.1f}%</div></div>
              <div class='kpi'><div class='klbl'>IMPLIED VALUATION</div><div class='kval'>₹ {impl:,.1f} L</div></div>
              <div class='kpi' style='border-color:{dc};background:rgba(0,0,0,.2)'>
                <div class='klbl'>{icon} MODEL VS IMPLIED</div>
                <div class='kval' style='color:{dc}'>{delta_pct:+.1f}%</div>
                <div class='kdelta' style='color:{dc}'>{verb}</div>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.info("Swim over to **Pitch Analytics** in the sidebar for benchmark charts and the health radar.")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — PITCH ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────
def page_analytics() -> None:
    st.markdown('<div class="badge">PITCH ANALYTICS & INSIGHTS</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="page-h1 grad-teal">HOW DOES YOUR<br>STARTUP STACK UP?</h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Benchmarking in the z-scored model space. Run a prediction first to unlock your personalised view.</p>', unsafe_allow_html=True)
    st.markdown('<hr class="divl">', unsafe_allow_html=True)

    inputs   = st.session_state.get("last_inputs")
    pred_val = st.session_state.get("last_pred")

    if inputs is None:
        st.warning("No prediction yet. Go to Valuation Engine first.")
        return

    industry = inputs["industry"]

    # ── User scaled values for overlays ──────────────────────────────────────
    def _sc(col: str, v: float) -> float:
        x = np.log1p(v) if col in LOG_THEN_SCALE else v
        m, s = SCALER_PARAMS[col]
        return (x - m) / s

    u = {
        "Revenue":     _sc("Yearly Revenue",  inputs["yearly_revenue"]),
        "GrossMargin": _sc("Gross Margin",    inputs["gross_margin"]),
        "NetMargin":   _sc("Net Margin",      inputs["net_margin"]),
        "EBITDA":      _sc("EBITDA",          inputs["ebitda"]),
    }

    # ── 1. Benchmark bar charts ───────────────────────────────────────────────
    st.markdown('<p class="cap">INDUSTRY BENCHMARK COMPARISON (Z-SCORED MODEL SPACE)</p>', unsafe_allow_html=True)
    t1, t2, t3, t4, t5 = st.tabs(["REVENUE", "GROSS MARGIN", "NET MARGIN", "EBITDA", "DEAL VALUATION"])

    def _bar(bkey: str, uval: float, title: str) -> go.Figure:
        inds = list(INDUSTRY_BENCHMARKS.keys())
        vals = [INDUSTRY_BENCHMARKS[i][bkey] for i in inds]
        df_b = pd.DataFrame({"Industry": inds, "Value": vals})
        df_b.loc[df_b["Industry"] == industry, "Value"] = uval
        df_b = df_b.sort_values("Value", ascending=True)
        colors = ["#F5C842" if r["Industry"] == industry else "rgba(0,200,190,.28)"
                  for _, r in df_b.iterrows()]
        fig = go.Figure(go.Bar(
            x=df_b["Value"], y=df_b["Industry"], orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{v:.2f}" for v in df_b["Value"]],
            textposition="outside", textfont=dict(size=10, color="#C8E6F5"),
        ))
        fig.add_vline(x=uval, line_dash="dot", line_color="#E63946", line_width=2,
                      annotation_text="You", annotation_position="top right",
                      annotation_font=dict(color="#E63946", size=11))
        fig.add_vline(x=float(df_b["Value"].mean()), line_dash="dash", line_color="#4A7A9B", line_width=1,
                      annotation_text="Mean", annotation_position="bottom right",
                      annotation_font=dict(color="#4A7A9B", size=10))
        fig.update_layout(title=dict(text=title, font=dict(family="Bebas Neue, sans-serif", size=14,
                                                            color="#00C8BE"), x=0),
                          height=500, **_PL)
        return fig

    with t1: st.plotly_chart(_bar("Revenue",     u["Revenue"],     "YEARLY REVENUE (Z-SCORED)"),     use_container_width=True, config=CHART_CFG)
    with t2: st.plotly_chart(_bar("GrossMargin", u["GrossMargin"], "GROSS MARGIN % (Z-SCORED)"),     use_container_width=True, config=CHART_CFG)
    with t3: st.plotly_chart(_bar("NetMargin",   u["NetMargin"],   "NET MARGIN % (Z-SCORED)"),       use_container_width=True, config=CHART_CFG)
    with t4: st.plotly_chart(_bar("EBITDA",      u["EBITDA"],      "EBITDA (Z-SCORED)"),             use_container_width=True, config=CHART_CFG)
    with t5:
        inds  = list(INDUSTRY_BENCHMARKS.keys())
        dvals = [INDUSTRY_BENCHMARKS[i]["DealVal"] for i in inds]
        df_v  = pd.DataFrame({"Industry": inds, "V": dvals}).sort_values("V", ascending=True)
        cols  = ["#F5C842" if r["Industry"] == industry else "rgba(0,200,190,.28)" for _, r in df_v.iterrows()]
        fv    = go.Figure(go.Bar(x=df_v["V"], y=df_v["Industry"], orientation="h",
                                 marker=dict(color=cols, line=dict(width=0)),
                                 text=[f"{v:.2f}" for v in df_v["V"]],
                                 textposition="outside", textfont=dict(size=10, color="#C8E6F5")))
        if pred_val:
            fv.add_vline(x=math.log1p(pred_val), line_dash="dash", line_color="#E63946", line_width=2,
                         annotation_text=f"Prediction: ₹{pred_val:,.0f}L",
                         annotation_position="top right", annotation_font=dict(color="#E63946", size=11))
        fv.update_layout(title=dict(text="AVG DEAL VALUATION (LOG SPACE) BY INDUSTRY",
                                    font=dict(family="Bebas Neue, sans-serif", size=14, color="#00C8BE"), x=0),
                         height=500, **_PL)
        st.plotly_chart(fv, use_container_width=True, config=CHART_CFG)

    st.markdown('<hr class="divl">', unsafe_allow_html=True)

    # ── 2. Implied vs Predicted ───────────────────────────────────────────────
    st.markdown('<p class="cap">IMPLIED VS MODEL PREDICTED VALUATION</p>', unsafe_allow_html=True)
    impl = _implied(inputs["ask_amount"], inputs["offered_equity"])
    fig_cmp = go.Figure(go.Bar(
        x=["Implied (Ask/Equity)", "Model Predicted"],
        y=[impl, pred_val or 0],
        marker=dict(color=["#F5C842", "#00C8BE"], line=dict(width=0)),
        text=[f"₹{impl:,.1f} L", f"₹{pred_val:,.1f} L" if pred_val else "—"],
        textposition="outside", textfont=dict(size=13, color="#C8E6F5"),
    ))

    fig_cmp.update_layout(**_pl(
        title=dict(text="IMPLIED VS PREDICTED VALUATION (₹ LAKHS)",
                font=dict(family="Bebas Neue, sans-serif", size=14, color="#00C8BE"), x=0),
        height=320,
        yaxis=dict(title="₹ Lakhs"),
    ))
    st.plotly_chart(fig_cmp, use_container_width=True, config=CHART_CFG)

    st.markdown('<hr class="divl">', unsafe_allow_html=True)

    # ── 3. Health Radar + Gauge + Bars ───────────────────────────────────────
    st.markdown('<p class="cap">STARTUP HEALTH DASHBOARD</p>', unsafe_allow_html=True)

    def _cl(v, lo, hi): return float(np.clip((v - lo) / (hi - lo) * 10, 0, 10))

    # sym_log used here for DISPLAY normalisation of metrics that can be negative
    scores = {
        "Gross Margin":    _cl(sym_log(inputs["gross_margin"]), -5,  5),
        "Net Margin":      _cl(sym_log(inputs["net_margin"]),   -6,  4),
        "Revenue":         _cl(inputs["yearly_revenue"],         0, 500),
        "EBITDA":          _cl(sym_log(inputs["ebitda"]),       -7,  7),
        "IP / Patents":    float(inputs["has_patents"]) * 10,
        "Self-Funded":     float(inputs["bootstrapped"]) * 7,
        "Product Breadth": _cl(inputs["skus"],                   0,  50),
    }
    cats   = list(scores.keys())
    vals   = list(scores.values())
    health = float(np.mean(vals))

    rl, rm, rr = st.columns([1.3, 1, 0.9], gap="medium")

    with rl:
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=cats + [cats[0]], fill="toself",
            fillcolor="rgba(0,200,190,.13)", line=dict(color="#00C8BE", width=2.2), name="Your Startup",
        ))
        fig_r.add_trace(go.Scatterpolar(
            r=[5]*(len(cats)+1), theta=cats + [cats[0]], fill="toself",
            fillcolor="rgba(245,200,66,.06)", line=dict(color="#F5C842", width=1, dash="dot"), name="Neutral",
        ))
        fig_r.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0,10], gridcolor="rgba(255,255,255,.07)",
                                color="rgba(255,255,255,.25)", tickfont=dict(size=9)),
                angularaxis=dict(gridcolor="rgba(255,255,255,.05)", color="#4A7A9B",
                                 tickfont=dict(size=10, family="Rajdhani")),
            ),
            showlegend=True, legend=dict(font=dict(color="#C8E6F5", size=10), bgcolor="rgba(0,0,0,0)", x=0),
            height=360, paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Rajdhani, sans-serif", color="#C8E6F5"),
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig_r, use_container_width=True, config=CHART_CFG)

    with rm:
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=health,
            title={"text": "HEALTH SCORE", "font": {"size": 13, "family": "Bebas Neue,sans-serif", "color": "#C8E6F5"}},
            number={"font": {"size": 44, "family": "Bebas Neue,sans-serif", "color": "#00C8BE"}, "valueformat": ".1f"},
            gauge={
                "axis": {"range": [0,10], "tickcolor": "#4A7A9B", "tickfont": {"color": "#4A7A9B", "size": 9}},
                "bar":  {"color": "#00C8BE", "thickness": .22},
                "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
                "steps": [
                    {"range": [0,   3.5], "color": "rgba(230,57,70,.18)"},
                    {"range": [3.5, 6.5], "color": "rgba(245,200,66,.13)"},
                    {"range": [6.5, 10],  "color": "rgba(0,200,190,.13)"},
                ],
                "threshold": {"line": {"color": "#F5C842", "width": 3}, "thickness": .8, "value": 5},
            },
        ))
        fig_g.update_layout(height=260, paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=20, r=20, t=55, b=10))
        st.plotly_chart(fig_g, use_container_width=True, config=CHART_CFG)

        if health >= 7:
            lbl, dc, msg = "SHARK MAGNET",  "#00C8BE", "Solid fundamentals — Sharks will compete for this deal."
        elif health >= 4.5:
            lbl, dc, msg = "CIRCLING",      "#F5C842", "Good traction. Sharpen margins to close the deal."
        else:
            lbl, dc, msg = "CHUM",          "#E63946", "High risk. A compelling story is your lifeline."
        st.markdown(f"""
        <div style='text-align:center;padding:.85rem;background:rgba(6,25,51,.7);
                    border:1px solid {dc};border-radius:10px;'>
          <div style='font-family:"Bebas Neue",sans-serif;font-size:1.4rem;letter-spacing:.12em;color:{dc}'>{lbl}</div>
          <div style='color:#4A7A9B;font-size:.8rem;margin-top:.3rem'>{msg}</div>
        </div>""", unsafe_allow_html=True)

    with rr:
        st.markdown('<p class="cap" style="margin-top:.4rem">SCORE BREAKDOWN</p>', unsafe_allow_html=True)
        for cat, score in scores.items():
            bw = int(score * 10)
            c  = "#00C8BE" if score >= 6.5 else ("#F5C842" if score >= 4 else "#E63946")
            st.markdown(f"""
            <div class='sbw'>
              <div class='sbl'><span>{cat}</span><span style='color:{c};font-weight:600'>{score:.1f}</span></div>
              <div class='sbt'><div class='sbf' style='width:{bw}%;background:{c}'></div></div>
            </div>""", unsafe_allow_html=True)

        diff = [round(v - 5, 2) for v in vals]
        df_d = pd.DataFrame({"Dimension": cats, "Edge": diff})
        fig_d = px.bar(df_d, x="Dimension", y="Edge", color="Edge",
                       color_continuous_scale=["#E63946", "#F5C842", "#00C8BE"],
                       range_color=[-5, 5], title="EDGE VS NEUTRAL BASELINE")
        fig_d.update_layout(height=210, showlegend=False, coloraxis_showscale=False,
                             title=dict(font=dict(family="Bebas Neue, sans-serif", size=11,
                                                   color="#00C8BE")), **_PL)
        fig_d.update_traces(marker_line_width=0)
        st.plotly_chart(fig_d, use_container_width=True, config=CHART_CFG)

    st.markdown('<hr class="divl">', unsafe_allow_html=True)

    with st.expander("PREPROCESSING PIPELINE & MODEL ARCHITECTURE"):
        st.markdown(f"""
#### Exact Training Pipeline

| Step | Operation | Columns |
|---|---|---|
| 1 | OHE → 18 dummies, **Hardware dropped** | `Industry` |
| 2 | `np.log1p(x)` | `Yearly Revenue`, `SKUs`, `Original Ask Amount` |
| 3 | **Raw (no log)** | `Gross Margin %`, `Net Margin %`, `EBITDA`, `Equity %`, `Company Age` |
| 4 | Z-score `(x−μ)/σ` | All 8 continuous cols (post-log where applicable) |
| 5 | Ordinal encode | `Pitchers Average Age` → Young=0, Middle=1, Old=2 |
| 6 | Binary pass-through | `Has Patents`, `Bootstrapped` |
| 7 | Intercept | `const = 1.0` prepended manually |

**Target:** `log1p(Deal Valuation)` → inverted with `np.expm1()` for display.

**sym_log usage:** `sym_log = sign(x) * log1p(|x|)` is used **for the Health Radar display only**
(normalising metrics that can go negative). It is **NOT applied in model preprocessing** —
EDA never log-transformed EBITDA or margin columns before `StandardScaler`.

#### Scaler Parameters (extracted from training pipeline)

| Feature | μ (mean) | σ (std) |
|---|---|---|
| Yearly Revenue (after log1p) | {SCALER_PARAMS['Yearly Revenue'][0]:.4f} | {SCALER_PARAMS['Yearly Revenue'][1]:.4f} |
| Gross Margin % | {SCALER_PARAMS['Gross Margin'][0]:.4f} | {SCALER_PARAMS['Gross Margin'][1]:.4f} |
| Net Margin % | {SCALER_PARAMS['Net Margin'][0]:.4f} | {SCALER_PARAMS['Net Margin'][1]:.4f} |
| EBITDA ₹L | {SCALER_PARAMS['EBITDA'][0]:.4f} | {SCALER_PARAMS['EBITDA'][1]:.4f} |
| SKUs (after log1p) | {SCALER_PARAMS['SKUs'][0]:.4f} | {SCALER_PARAMS['SKUs'][1]:.4f} |
| Ask Amount (after log1p) | {SCALER_PARAMS['Original Ask Amount'][0]:.4f} | {SCALER_PARAMS['Original Ask Amount'][1]:.4f} |
| Offered Equity % | {SCALER_PARAMS['Original Offered Equity'][0]:.4f} | {SCALER_PARAMS['Original Offered Equity'][1]:.4f} |
| Company Age yrs | {SCALER_PARAMS['Company Age'][0]:.4f} | {SCALER_PARAMS['Company Age'][1]:.4f} |
        """)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — PREDICTION HISTORY
# ─────────────────────────────────────────────────────────────────────────────
def page_history() -> None:
    st.markdown('<div class="badge">PREDICTION HISTORY</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="page-h1 grad-teal">PREDICTION LOG</h1>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="page-sub">Every prediction is saved in-session (O(1) list) '
        f'and written to <code>{HISTORY_PATH}</code> for cross-session analysis. '
        f'The CSV file is committed to your repo automatically if the <code>data/</code> folder exists.</p>',
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="divl">', unsafe_allow_html=True)

    hist = get_history_df()

    if hist.empty:
        st.info("No predictions yet. Head to the Valuation Engine to get started.")
        return

    n        = len(hist)
    avg_pred = hist["Predicted Valuation (₹L)"].mean()
    max_pred = hist["Predicted Valuation (₹L)"].max()
    min_pred = hist["Predicted Valuation (₹L)"].min()

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Predictions Made", n)
    s2.metric("Average Valuation",  f"₹ {avg_pred:,.1f} L")
    s3.metric("Highest Valuation",  f"₹ {max_pred:,.1f} L")
    s4.metric("Lowest Valuation",   f"₹ {min_pred:,.1f} L")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="cap">ALL PREDICTIONS</p>', unsafe_allow_html=True)
    st.dataframe(
        hist.style.format({
            "Predicted Valuation (₹L)": "{:,.2f}",
            "Implied Valuation (₹L)":   "{:,.2f}",
            "Delta (%)":                "{:+.2f}",
            "Revenue (₹L)":             "{:,.1f}",
            "EBITDA (₹L)":              "{:,.1f}",
        }).background_gradient(subset=["Predicted Valuation (₹L)"], cmap="Blues"),
        use_container_width=True, height=300,
    )

    if n > 1:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="cap">VALUATION TREND</p>', unsafe_allow_html=True)
        hp = hist.copy(); hp["Run #"] = range(1, n + 1)
        ft = go.Figure()
        ft.add_trace(go.Scatter(
            x=hp["Run #"], y=hp["Predicted Valuation (₹L)"],
            mode="lines+markers+text",
            line=dict(color="#00C8BE", width=2.4),
            marker=dict(size=8, color="#00C8BE", line=dict(color="#020B18", width=2)),
            text=[f"₹{v:,.0f}" for v in hp["Predicted Valuation (₹L)"]],
            textposition="top center", textfont=dict(size=9, color="#C8E6F5"), name="Model Prediction",
        ))
        ft.add_trace(go.Scatter(
            x=hp["Run #"], y=hp["Implied Valuation (₹L)"],
            mode="lines+markers", line=dict(color="#F5C842", width=1.8, dash="dot"),
            marker=dict(size=7, color="#F5C842"), name="Implied (Ask/Equity)",
        ))

        ft.update_layout(**_pl(
            xaxis=dict(title="Prediction Run #", tickmode="linear"),
            yaxis=dict(title="₹ Lakhs"),
            height=320,
            legend=dict(font=dict(color="#C8E6F5"), bgcolor="rgba(0,0,0,0)"),
        ))
        st.plotly_chart(ft, use_container_width=True, config=CHART_CFG)

        st.markdown('<p class="cap">INDUSTRIES TESTED</p>', unsafe_allow_html=True)
        ic = hist["Industry"].value_counts().reset_index(); ic.columns = ["Industry", "Count"]
        fi = px.bar(ic, x="Count", y="Industry", orientation="h", color="Count",
                    color_continuous_scale=["rgba(0,200,190,.3)", "#00C8BE"])
        fi.update_layout(**_pl(
                          height=max(180, len(ic)*38), showlegend=False,
                          coloraxis_showscale=False))
        fi.update_traces(marker_line_width=0)
        st.plotly_chart(fi, use_container_width=True, config=CHART_CFG)

    # ── Export ────────────────────────────────────────────────────────────────
    st.markdown('<hr class="divl">', unsafe_allow_html=True)
    st.markdown('<p class="cap">EXPORT DATA</p>', unsafe_allow_html=True)
    ec1, ec2, ec3, _ = st.columns([1, 1, 1, 1])

    with ec1:
        st.download_button("⬇ Session CSV", data=hist.to_csv(index=False).encode(),
                           file_name="predictions_session.csv", mime="text/csv",
                           use_container_width=True)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        hist.to_excel(w, index=False, sheet_name="Predictions")
    with ec2:
        st.download_button("⬇ Session Excel", data=buf.getvalue(),
                           file_name="predictions_session.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True)
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "rb") as f:
            with ec3:
                st.download_button("⬇ Full History CSV", data=f.read(),
                                   file_name="prediction_history.csv", mime="text/csv",
                                   use_container_width=True)
        st.caption(f"Full history file: `{HISTORY_PATH}` — persists across sessions on your local machine / server.")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("CLEAR SESSION HISTORY"):
        st.session_state["history_list"] = []
        st.session_state.update({"pred_done": False, "last_pred": None, "last_inputs": None})
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    init_state()
    st.markdown(CSS, unsafe_allow_html=True)

    model, err = get_model()
    model_ok   = model is not None

    page = render_sidebar(model_ok)

    if page == "engine":
        page_engine(model)
    elif page == "analytics":
        page_analytics()
    elif page == "history":
        page_history()


if __name__ == "__main__":
    main()
