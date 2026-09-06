import streamlit as st
import numpy as np
import plotly.graph_objects as go
import base64
import os
import json
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ── Load images ────────────────────────────────────────────────
def _load_img(filename, mime="image/jpeg"):
    candidates = [
        os.path.join(os.path.dirname(__file__), filename),
        filename,
    ]
    for p in candidates:
        if os.path.exists(p):
            with open(p, "rb") as f:
                return f"data:{mime};base64," + base64.b64encode(f.read()).decode()
    return None

PHOTO_SRC    = _load_img("shaziabyat.jpg")
COMPARE_SRC  = _load_img("comparission_chart.png", "image/png")

# ── Load ensemble model (cached so it loads once per session) ──
@st.cache_resource
def load_models():
    try:
        import joblib
        from pathlib import Path
        MODEL_DIR = Path(__file__).parent / "production_models"
        rf_m   = joblib.load(MODEL_DIR / "rf_model.pkl")
        xgb_m  = joblib.load(MODEL_DIR / "xgb_model.pkl")
        lgbm_m = joblib.load(MODEL_DIR / "lgbm_model.pkl")
        config  = json.loads((MODEL_DIR / "ensemble_config.json").read_text())
        return rf_m, xgb_m, lgbm_m, config["threshold"], True
    except Exception:
        return None, None, None, 0.5, False

_RF, _XGB, _LGBM, _THRESHOLD, _MODELS_OK = load_models()

st.set_page_config(
    page_title="Cold Chain Intelligence | Single Use Support",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Force white background regardless of OS dark mode */
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
section.stMain, .main, body { background: #ffffff !important; color: #111827 !important; }

/* Hide Streamlit chrome */
#MainMenu          { visibility: hidden; }
header             { visibility: hidden; height: 0 !important; }
footer             { visibility: hidden; }
[data-testid="stToolbar"]          { display: none !important; }
[data-testid="stDecoration"]       { display: none !important; }
[data-testid="stStatusWidget"]     { display: none !important; }

.block-container {
    padding: 0 56px 64px !important;
    max-width: 1100px !important;
    margin: 0 auto !important;
    margin-top: 0 !important;
}

/* ── TOP NAV ── */
.top-nav {
    background: #fff;
    border-bottom: 1px solid #E5E7EB;
    padding: 0 56px;
    height: 52px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: 0 -56px;
}
.nav-brand {
    font-size: 13px; font-weight: 700;
    color: #111827; letter-spacing: -0.01em;
    display: flex; align-items: center; gap: 8px;
}
.nav-brand-dot { width: 7px; height: 7px; border-radius: 50%; background: #0047BB; }
.nav-status {
    display: flex; align-items: center; gap: 6px;
    font-size: 11px; font-weight: 500; color: #6B7280;
    background: #FFFBEB; border: 1px solid #FCD34D;
    border-radius: 5px; padding: 4px 10px;
}
.nav-status-dot { width: 5px; height: 5px; border-radius: 50%; background: #F59E0B; }

/* ── PAGE WRAPPER ── */
.page {
    background: #fff;
    padding: 0 52px 52px;
    max-width: 1080px;
    margin: 0 auto;
}

/* ── HERO (full-bleed dark) ── */
.hero-full {
    background: linear-gradient(135deg, #0C1F3F 0%, #0047BB 100%);
    padding: 64px 56px 56px;
    margin: 0 -56px;
    position: relative;
    overflow: hidden;
}
.hero-full::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 420px; height: 420px;
    border-radius: 50%;
    background: rgba(147,197,253,0.07);
    pointer-events: none;
}
.hero-inner { max-width: 1080px; margin: 0 auto; }
.hero-tag {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 11px; font-weight: 600; letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #93C5FD; background: rgba(147,197,253,0.12);
    border: 1px solid rgba(147,197,253,0.3);
    border-radius: 4px; padding: 4px 10px; margin-bottom: 20px;
}
.hero-full h1 {
    font-size: 52px; font-weight: 800; color: #fff;
    letter-spacing: -0.04em; line-height: 1.08;
    margin-bottom: 18px; max-width: 800px;
}
.hero-full h1 em { font-style: normal; color: #93C5FD; }
.hero-full .hero-body {
    font-size: 17px; color: rgba(255,255,255,0.7);
    max-width: 600px; line-height: 1.7; margin-bottom: 28px;
}
.hero-notice {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(245,158,11,0.12); border: 1px solid rgba(245,158,11,0.35);
    border-radius: 6px; padding: 8px 14px;
    font-size: 12px; color: #FCD34D; font-weight: 500;
}

/* ── METRICS ROW ── */
.metrics-row {
    display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 0; background: #fff;
    border-bottom: 1px solid #E5E7EB;
    border-top: 1px solid #E5E7EB;
    margin: 0 -56px 56px;
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}
.metric-cell {
    background: #fff; padding: 28px 24px;
    border-right: 1px solid #E5E7EB;
    opacity: 0; animation: fadeUp 0.55s ease forwards;
    transition: background 0.2s ease, transform 0.2s ease, box-shadow 0.2s ease;
    cursor: default;
}
.metric-cell:last-child { border-right: none; }
.metric-cell:nth-child(1) { animation-delay: 0.05s; }
.metric-cell:nth-child(2) { animation-delay: 0.15s; }
.metric-cell:nth-child(3) { animation-delay: 0.25s; }
.metric-cell:nth-child(4) { animation-delay: 0.35s; }
.metric-cell:hover { background: #F8FAFF; transform: translateY(-3px); box-shadow: 0 8px 24px rgba(0,71,187,0.1); }
.metric-cell .val {
    font-size: 40px; font-weight: 800; letter-spacing: -0.04em;
    color: #111827; line-height: 1; margin-bottom: 8px;
}
.metric-cell .val.accent { color: #0047BB; }
.metric-cell .val.green  { color: #059669; }
.metric-cell .lbl { font-size: 13px; font-weight: 600; color: #374151; margin-bottom: 3px; }
.metric-cell .sub { font-size: 11px; color: #9CA3AF; }

/* ── SECTION ── */
.s-label {
    font-size: 11px; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: #0047BB; margin-bottom: 8px;
}
.s-title {
    font-size: 30px; font-weight: 800; color: #111827;
    letter-spacing: -0.03em; margin-bottom: 8px; line-height: 1.2;
}
.s-sub {
    font-size: 15px; color: #6B7280; line-height: 1.7;
    max-width: 640px; margin-bottom: 28px;
}

/* ── SECTION TINT ── */
.section-tint {
    background: #F5F8FF;
    margin: 0 -52px;
    padding: 40px 52px;
    margin-bottom: 48px;
}

/* ── ABOUT ── */
.about-card {
    display: flex; align-items: center; gap: 28px;
    background: #F5F8FF; border-radius: 12px;
    border: 1px solid #DBEAFE;
    padding: 28px 32px; margin-bottom: 32px;
}
.about-avatar {
    width: 72px; height: 72px; border-radius: 50%;
    background: #0047BB; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    font-size: 28px; font-weight: 800; color: #fff;
    letter-spacing: -0.02em;
}
.about-info h4 { font-size: 16px; font-weight: 700; color: #111827; margin-bottom: 3px; }
.about-info .about-title { font-size: 13px; color: #0047BB; font-weight: 600; margin-bottom: 6px; }
.about-info p { font-size: 13px; color: #6B7280; line-height: 1.6; margin: 0; }

/* ── COMPARE TABLE ── */
.compare-tbl {
    width: 100%;
    border-collapse: collapse;
    font-size: 14px;
}
.compare-tbl thead th {
    background: #F9FAFB;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #6B7280;
    padding: 10px 16px;
    text-align: left;
    border-bottom: 1px solid #E5E7EB;
    border-top: 1px solid #E5E7EB;
}
.compare-tbl tbody td {
    padding: 13px 16px;
    border-bottom: 1px solid #F3F4F6;
    color: #374151;
    vertical-align: top;
}
.compare-tbl tbody tr:last-child td { border-bottom: none; }
.compare-tbl tbody tr:hover td { background: #F9FAFB; }
.compare-tbl .row-best td {
    background: #F0FDF4;
}
.compare-tbl .td-name {
    font-weight: 600;
    color: #111827;
}
.compare-tbl .td-best {
    color: #059669;
    font-weight: 700;
}
.compare-tbl .td-badge {
    display: inline-block;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    padding: 2px 7px;
    border-radius: 4px;
    margin-left: 6px;
    vertical-align: middle;
}
.badge-rec { background: #059669; color: #fff; }
.badge-note { background: #EFF6FF; color: #0047BB; border: 1px solid #BFDBFE; }

/* ── TWO-COL LAYOUT ── */
.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
.three-col { display: grid; grid-template-columns: repeat(3,1fr); gap: 16px; }
.four-col { display: grid; grid-template-columns: repeat(4,1fr); gap: 16px; }

/* ── PANEL ── */
.panel {
    border: 1px solid #E5E7EB;
    border-radius: 8px;
    padding: 24px;
    background: #fff;
}
.panel.panel-red { border-left: 3px solid #DC2626; }
.panel.panel-green { border-left: 3px solid #059669; }
.panel h4 {
    font-size: 13px;
    font-weight: 700;
    color: #111827;
    margin-bottom: 14px;
}
.panel ul { list-style: none; padding: 0; }
.panel li {
    font-size: 13px;
    color: #4B5563;
    padding: 6px 0;
    border-bottom: 1px solid #F3F4F6;
    display: flex;
    gap: 10px;
    line-height: 1.5;
}
.panel li:last-child { border-bottom: none; }
.li-icon { flex-shrink: 0; font-size: 12px; margin-top: 2px; color: #9CA3AF; }
.li-icon.red { color: #DC2626; }
.li-icon.green { color: #059669; }

/* ── STEPS ── */
.step-item {
    padding: 24px; border-radius: 10px; background: #fff;
    border: 1px solid #E5E7EB; position: relative;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.step-item:hover { transform: translateY(-4px); box-shadow: 0 12px 32px rgba(0,71,187,0.1); }
.step-n {
    font-size: 48px; font-weight: 800; color: #EFF6FF;
    letter-spacing: -0.04em; line-height: 1; margin-bottom: 12px;
    font-variant-numeric: tabular-nums;
}
.step-label {
    font-size: 10px; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: #0047BB; margin-bottom: 8px;
}
.step-item h4 { font-size: 14px; font-weight: 700; color: #111827; margin-bottom: 6px; }
.step-item p  { font-size: 13px; color: #6B7280; line-height: 1.6; }

/* ── OUTCOMES ── */
.outcome-item {
    padding: 20px;
    border: 1px solid #E5E7EB;
    border-top: 3px solid #0047BB;
    border-radius: 8px;
    background: #fff;
}
.outcome-item h4 {
    font-size: 13px;
    font-weight: 700;
    color: #111827;
    margin-bottom: 6px;
}
.outcome-item p {
    font-size: 13px;
    color: #6B7280;
    line-height: 1.6;
}

/* ── DIFFERENTIATOR LIST ── */
.diff-list { border: 1px solid #E5E7EB; border-radius: 8px; overflow: hidden; }
.diff-row {
    display: grid;
    grid-template-columns: 200px 1fr;
    border-bottom: 1px solid #F3F4F6;
}
.diff-row:last-child { border-bottom: none; }
.diff-row:hover { background: #F9FAFB; }
.diff-key {
    padding: 14px 16px;
    font-size: 13px;
    font-weight: 600;
    color: #111827;
    border-right: 1px solid #F3F4F6;
}
.diff-val {
    padding: 14px 16px;
    font-size: 13px;
    color: #4B5563;
    line-height: 1.5;
}

/* ── HIGHLIGHT BOX ── */
.highlight-box {
    background: #0047BB;
    border-radius: 8px;
    padding: 28px;
    color: #fff;
}
.highlight-box h3 {
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 8px;
    letter-spacing: -0.01em;
}
.highlight-box p {
    font-size: 13px;
    opacity: 0.75;
    line-height: 1.65;
    margin-bottom: 0;
}

/* ── NOTICE ── */
.notice {
    background: #FFFBEB;
    border: 1px solid #FCD34D;
    border-left: 3px solid #D97706;
    border-radius: 6px;
    padding: 14px 16px;
    margin: 20px 0;
}
.notice h5 {
    font-size: 12px;
    font-weight: 700;
    color: #92400E;
    margin-bottom: 4px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.notice p {
    font-size: 13px;
    color: #78350F;
    line-height: 1.6;
    margin: 0;
}

/* ── CALCULATOR ── */
.result-panel {
    background: #0047BB;
    border-radius: 8px;
    padding: 28px;
    color: #fff;
    height: 100%;
}
.result-panel .rl { font-size: 10px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; opacity: 0.55; margin-bottom: 4px; }
.result-panel .rv { font-size: 26px; font-weight: 800; letter-spacing: -0.02em; margin-bottom: 18px; }
.result-panel .rv.big { font-size: 36px; color: #4ADE80; }
.result-panel hr { border: none; border-top: 1px solid rgba(255,255,255,0.15); margin: 4px 0 18px; }
.result-panel .rn { font-size: 11px; opacity: 0.45; margin-top: 16px; line-height: 1.6; }

/* ── CTA ── */
.cta-block {
    background: #0047BB;
    border-radius: 12px;
    padding: 56px 48px;
    text-align: center;
    margin-top: 8px;
}
.cta-block .cta-eyebrow {
    font-size: 11px; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase;
    color: rgba(255,255,255,0.55); margin-bottom: 16px;
}
.cta-block h3 {
    font-size: 36px; font-weight: 800; color: #fff;
    letter-spacing: -0.03em; line-height: 1.15; margin-bottom: 16px;
}
.cta-block h3 em { font-style: normal; color: #93C5FD; }
.cta-block .cta-sub {
    font-size: 15px; color: rgba(255,255,255,0.72); line-height: 1.7;
    max-width: 520px; margin: 0 auto 28px;
}
.cta-block .cta-punch {
    display: inline-block;
    font-size: 13px; font-weight: 700; color: #fff;
    border: 1.5px solid rgba(255,255,255,0.4);
    border-radius: 6px; padding: 10px 24px; letter-spacing: 0.02em;
}

/* ── DIVIDER ── */
.div { border: none; border-top: 1px solid #E5E7EB; margin: 40px 0; }

/* ── FOOTER ── */
.footer {
    text-align: center;
    font-size: 12px;
    color: #9CA3AF;
    padding-top: 32px;
    border-top: 1px solid #E5E7EB;
    line-height: 1.8;
}

/* ── Hide Streamlit anchor/link icons on headers ── */
[data-testid="stHeaderActionElements"] { display: none !important; }
a.anchor-link { display: none !important; }
.stMarkdown h1 a, .stMarkdown h2 a, .stMarkdown h3 a { display: none !important; }

/* ── Hide nav brand dot ── */
.nav-brand-dot { display: none !important; }

/* ── SLIDERS: replace Streamlit red with brand blue ── */
[role="slider"] {
    background-color: #0047BB !important;
    border-color: #0047BB !important;
}
[role="slider"]:focus {
    box-shadow: 0 0 0 4px rgba(0, 71, 187, 0.18) !important;
}
/* Track fill — Streamlit uses st-ah for the active (filled) portion */
.st-ah { background-color: #0047BB !important; }
[data-testid="stSliderTickBar"] { display: none !important; }
</style>
""", unsafe_allow_html=True)


# ── Temperature data ─────────────────────────────────────────────
@st.cache_data
def generate_temp_data():
    rng = np.random.default_rng(42)
    n = 145; fail_i = 110; alert_i = 94; drift_i = 80
    hours = np.linspace(0, 72, n)
    cycle = 0.6 * np.sin(hours * np.pi / 1.5)
    temp = np.zeros(n)
    for i in range(n):
        if i < drift_i:
            temp[i] = 4.5 + cycle[i] + rng.uniform(-0.25, 0.25)
        elif i < fail_i:
            p = (i - drift_i) / (fail_i - drift_i)
            temp[i] = 4.5 + cycle[i] + 14*p**2 + rng.uniform(-0.4,0.4)*(1+p)
        else:
            temp[i] = min(22 + (i-fail_i)*0.15 + rng.uniform(-0.2,0.2), 25)
    return hours, np.round(temp, 1), fail_i, alert_i, drift_i

HOURS, TEMP, FAIL_I, ALERT_I, DRIFT_I = generate_temp_data()


# ═══════════════════════════════════════════════════════════════
# NAV
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="top-nav">
    <div class="nav-brand">
        <div class="nav-brand-dot"></div>
        Cold Chain Research
    </div>
    <div class="nav-status">
        <div class="nav-status-dot"></div>
        Research Project — Awaiting Real Biopharma Data for Validation
    </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# HERO (full-bleed dark, outside .page)
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-full">
  <div class="hero-inner">
    <div class="hero-tag">Masters Research · Machine Learning · Pharma Cold Chain</div>
    <h1>Can cold chain failures be predicted<br><em>hours before</em> they happen?</h1>
    <p class="hero-body">
        A personal research project applying machine learning to pharmaceutical cold chain
        temperature data — investigating whether subtle patterns that precede equipment
        failure can be detected in advance. Five models built and tested on public research data.
        Results are promising. Real data validation is the missing piece.
    </p>
    <div class="hero-notice">
        All results are from a public research dataset — not from real biopharma production hardware.
        Performance on real pharmaceutical cold chain sensors is unknown until validated. That is exactly what this research is asking for.
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="page">', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="metrics-row">
    <div class="metric-cell">
        <div class="val accent">8+ hrs</div>
        <div class="lbl">Advance Warning</div>
        <div class="sub">on public research dataset only</div>
    </div>
    <div class="metric-cell">
        <div class="val green">4 of 4</div>
        <div class="lbl">Failures Caught in Testing</div>
        <div class="sub">zero misses on research data</div>
    </div>
    <div class="metric-cell">
        <div class="val">9 in 10</div>
        <div class="lbl">Alerts Were Genuine</div>
        <div class="sub">low false alarm rate in testing</div>
    </div>
    <div class="metric-cell">
        <div class="val accent">5</div>
        <div class="lbl">Models Built &amp; Compared</div>
        <div class="sub">from baseline to deep learning</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# ABOUT
# ═══════════════════════════════════════════════════════════════
_avatar_html = (
    f'<img src="{PHOTO_SRC}" style="width:72px;height:72px;border-radius:50%;object-fit:cover;flex-shrink:0;">'
    if PHOTO_SRC else
    '<div class="about-avatar">BJ</div>'
)
st.markdown(f"""
<div class="about-card">
    {_avatar_html}
    <div class="about-info">
        <h4>Shazia Jatoi</h4>
        <div class="about-title">Working Student · Data Engineer · Single Use Support</div>
        <p>
            Five ML models trained and validated on pharmaceutical cold chain research data —
            <strong>motivated by a genuine belief that SUS hardware can do more than monitor.
            It can predict.</strong>
        </p>
    </div>
</div>
<hr class="div">
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# THE PROBLEM
# ═══════════════════════════════════════════════════════════════
st.markdown('<p class="s-label">Research Question</p>', unsafe_allow_html=True)
st.markdown('<h2 class="s-title">Cold chain failures are expensive — could they be predicted earlier?</h2>', unsafe_allow_html=True)
st.markdown('<p class="s-sub">Every biopharma customer stores and ships product worth hundreds of thousands to millions of euros per batch. The current industry standard is a threshold alarm that fires after the damage is already done.</p>', unsafe_allow_html=True)

st.markdown("""
<div class="two-col">
    <div class="panel panel-red">
        <h4>How it works today — Threshold Alarms</h4>
        <ul>
            <li><span class="li-icon red">—</span> Alarm fires after temperature exits the safe range</li>
            <li><span class="li-icon red">—</span> By alert time, the batch is already compromised</li>
            <li><span class="li-icon red">—</span> No time left to reroute, replace cooling, or recover</li>
            <li><span class="li-icon red">—</span> Subtle pre-failure signals — gradual drift, instability — go unnoticed</li>
            <li><span class="li-icon red">—</span> Outcome: batch discarded, loss logged</li>
        </ul>
    </div>
    <div class="panel panel-green">
        <h4>With predictive intelligence</h4>
        <ul>
            <li><span class="li-icon green">+</span> System detects warning signs hours before any threshold is crossed</li>
            <li><span class="li-icon green">+</span> Alert fires while product is still within the safe zone</li>
            <li><span class="li-icon green">+</span> Hours to reroute, swap cooling units, or escalate</li>
            <li><span class="li-icon green">+</span> Catches subtle drift patterns invisible to simple threshold logic</li>
            <li><span class="li-icon green">+</span> Outcome: batch saved, failure avoided</li>
        </ul>
    </div>
</div>
<hr class="div">
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# THE DATA
# ═══════════════════════════════════════════════════════════════
st.markdown('<p class="s-label">The Research Data</p>', unsafe_allow_html=True)
st.markdown('<h2 class="s-title">Same failure pattern — two different contexts.</h2>', unsafe_allow_html=True)
st.markdown('<p class="s-sub">No real biopharma cold chain sensor data was available, so two public datasets were used. A real pharmaceutical cold chain excursion (COVID-19 vaccine, Sun et al. 2022) established what failure actually looks like. The NAB industrial temperature dataset showed the identical drift-before-failure pattern and was used to train and test all five ML models.</p>', unsafe_allow_html=True)

if COMPARE_SRC:
    st.markdown(f"""
    <div style="border:1px solid #E5E7EB;border-radius:10px;overflow:hidden;margin-bottom:8px;">
        <img src="{COMPARE_SRC}" style="width:100%;display:block;" alt="Pharmaceutical vs Industrial Temperature Excursions">
    </div>
    <p style="font-size:12px;color:#9CA3AF;margin-bottom:32px;">
        Top: Real COVID-19 vaccine ultra-cold storage excursion — temperature drifting toward the −60°C safe limit before failure (Sun et al., Nature Scientific Data, 2022).
        Bottom: NAB industrial machine temperature dataset used for ML training — same drift-before-failure pattern, different scale.
    </p>
    """, unsafe_allow_html=True)

st.markdown('<hr class="div">', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# HOW IT WORKS
# ═══════════════════════════════════════════════════════════════
st.markdown('<p class="s-label">How It Works</p>', unsafe_allow_html=True)
st.markdown('<h2 class="s-title">Simple in concept. Powerful in practice.</h2>', unsafe_allow_html=True)
st.markdown('<p class="s-sub">The system reads the same temperature data the hardware already captures — and learns to recognise the patterns that precede failure, often hours before any threshold is crossed.</p>', unsafe_allow_html=True)

st.markdown("""
<div class="three-col" style="margin-bottom:0">
    <div class="step-item">
        <div class="step-n">01</div>
        <div class="step-label">Reads the data</div>
        <h4>Sensor stream, no new hardware</h4>
        <p>Monitors temperature readings from existing cold chain sensors continuously. Works with what is already there.</p>
    </div>
    <div class="step-item">
        <div class="step-n">02</div>
        <div class="step-label">Finds the pattern</div>
        <h4>Hours before any threshold is crossed</h4>
        <p>Detects subtle drift, rising instability, time outside optimal band — signals invisible to a simple alarm.</p>
    </div>
    <div class="step-item">
        <div class="step-n">03</div>
        <div class="step-label">Fires the alert</div>
        <h4>Time to act, not time to react</h4>
        <p>Sends advance notification while product is still safe — giving the team hours to reroute, replace cooling, or escalate.</p>
    </div>
</div>
<hr class="div">
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# DEMO CHART
# ═══════════════════════════════════════════════════════════════
st.markdown('<p class="s-label">Demonstration</p>', unsafe_allow_html=True)
st.markdown('<h2 class="s-title">The difference 8 hours makes.</h2>', unsafe_allow_html=True)
st.markdown('<p class="s-sub">Same temperature trace. Same sensor data. Toggle to see what a traditional alarm catches versus what predictive intelligence catches.</p>', unsafe_allow_html=True)

mode = st.radio("", ["Traditional Threshold Alarm", "Predictive Intelligence — Early Warning"],
                horizontal=True, label_visibility="collapsed")

fig = go.Figure()
fig.add_hrect(y0=2, y1=8, fillcolor="rgba(16,185,129,0.08)",
    line=dict(color="rgba(16,185,129,0.35)", width=1),
    annotation_text="Safe zone  2–8°C",
    annotation_position="top left",
    annotation_font=dict(color="#059669", size=11))

# Normal segment — always shown
fig.add_trace(go.Scatter(x=HOURS[:DRIFT_I+1], y=TEMP[:DRIFT_I+1],
    mode='lines', name='Normal', line=dict(color='#0047BB', width=2),
    hovertemplate='%{y:.1f}°C  h%{x:.1f}<extra></extra>'))

if "Traditional" in mode:
    # Show full drift + excursion — alarm fires too late
    fig.add_trace(go.Scatter(x=HOURS[DRIFT_I:FAIL_I+1], y=TEMP[DRIFT_I:FAIL_I+1],
        mode='lines', name='Pre-failure drift',
        line=dict(color='#D97706', width=2),
        hovertemplate='%{y:.1f}°C  h%{x:.1f}<extra></extra>'))
    fig.add_trace(go.Scatter(x=HOURS[FAIL_I:], y=TEMP[FAIL_I:],
        mode='lines', name='Excursion',
        line=dict(color='#DC2626', width=2.5),
        hovertemplate='%{y:.1f}°C  h%{x:.1f}<extra></extra>'))
    cross_i = next((i for i in range(DRIFT_I, len(TEMP)) if TEMP[i] >= 8), FAIL_I)
    fig.add_vline(x=HOURS[cross_i], line=dict(color="#DC2626", width=1.5, dash="dash"),
        annotation_text="Alarm fires — batch already compromised",
        annotation_position="top right",
        annotation_font=dict(color="#DC2626", size=12))
    fig.add_hline(y=8, line=dict(color="#DC2626", width=1, dash="dot"))

else:
    # Drift shown only up to the alert point
    fig.add_trace(go.Scatter(x=HOURS[DRIFT_I:ALERT_I+1], y=TEMP[DRIFT_I:ALERT_I+1],
        mode='lines', name='Pre-failure drift',
        line=dict(color='#D97706', width=2),
        hovertemplate='%{y:.1f}°C  h%{x:.1f}<extra></extra>'))

    # What would have happened — faint dashed red (no intervention)
    fig.add_trace(go.Scatter(x=HOURS[ALERT_I:], y=TEMP[ALERT_I:],
        mode='lines', name='Without action (what-if)',
        line=dict(color='#DC2626', width=1.5, dash='dot'),
        opacity=0.28,
        hovertemplate='%{y:.1f}°C  h%{x:.1f}<extra></extra>'))

    # Recovery line — team acted on the alert, temperature returns to safe zone
    T_alert = float(TEMP[ALERT_I])
    T_target = 4.5
    n_rec = len(TEMP) - ALERT_I
    temp_recovery = np.round(
        np.array([T_target + (T_alert - T_target) * np.exp(-k / 18.0) for k in range(n_rec)]), 1)
    fig.add_trace(go.Scatter(x=HOURS[ALERT_I:], y=temp_recovery,
        mode='lines', name='With early intervention',
        line=dict(color='#059669', width=2.5),
        hovertemplate='%{y:.1f}°C  h%{x:.1f}<extra></extra>'))

    # Alert vertical line
    fig.add_vline(x=HOURS[ALERT_I], line=dict(color="#D97706", width=1.5, dash="dash"),
        annotation_text="Predictive alert — 8+ hours before failure",
        annotation_position="top right",
        annotation_font=dict(color="#D97706", size=12))

    # Action window shading
    fig.add_vrect(x0=HOURS[ALERT_I], x1=HOURS[FAIL_I],
        fillcolor="rgba(217,119,6,0.07)", line=dict(width=0))

    # "Batch secured" label on the green line end
    fig.add_annotation(
        x=HOURS[-1], y=float(temp_recovery[-1]) + 1.5,
        text="✓ Batch secured", showarrow=True, arrowhead=2,
        arrowcolor="#059669", ax=-50, ay=0,
        font=dict(color="#059669", size=12, family="Inter"),
        bgcolor="#F0FDF4", bordercolor="#059669", borderwidth=1, borderpad=7)

    fig.add_annotation(x=(HOURS[ALERT_I]+HOURS[FAIL_I])/2, y=-0.5,
        text="8+ hours to reroute · replace cooling · alert team",
        font=dict(color="#92400E", size=11), showarrow=False,
        bgcolor="#FFFBEB", bordercolor="#FCD34D", borderwidth=1, borderpad=6)

fig.update_layout(height=320, margin=dict(t=16, b=36, l=16, r=16),
    plot_bgcolor='#FAFAFA', paper_bgcolor='white',
    xaxis=dict(title="Time (hours)", showgrid=True, gridcolor='#F3F4F6',
               ticksuffix='h', range=[0,72], color='#9CA3AF'),
    yaxis=dict(title="°C", showgrid=True, gridcolor='#F3F4F6', range=[-1,27], color='#9CA3AF'),
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
                font=dict(size=11, color='#6B7280')),
    font=dict(family='Inter, sans-serif'))

st.plotly_chart(fig, use_container_width=True)

st.markdown("""
<div class="notice">
    <h5>Important — Proof of Concept</h5>
    <p>
        The 8+ hour advance warning was measured on a validated pharmaceutical cold chain research dataset —
        not on real pharmaceutical production hardware. Real-world performance will depend on the specific characteristics
        of biopharma cold chain sensors and customer conditions.
        <strong>Validation against real pharmaceutical cold chain data is the defined next step.</strong>
    </p>
</div>
<hr class="div">
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════════════
st.markdown('<p class="s-label">Research Results</p>', unsafe_allow_html=True)
st.markdown('<h2 class="s-title">What was demonstrated on research data.</h2>', unsafe_allow_html=True)
st.markdown('<p class="s-sub">Tested across five model configurations. Results shown in business terms, not model metrics.</p>', unsafe_allow_html=True)

st.markdown("""
<div style="border:1px solid #E5E7EB;border-radius:8px;overflow:hidden;margin-bottom:24px">
<table class="compare-tbl">
<thead>
<tr>
    <th>Approach</th>
    <th>Failures Detected</th>
    <th>Advance Warning</th>
    <th>False Alarm Rate</th>
</tr>
</thead>
<tbody>
<tr>
    <td class="td-name">Gradient Boosting (baseline)</td>
    <td>4 of 4</td>
    <td>~7.5 hours</td>
    <td>Low</td>
</tr>
<tr>
    <td class="td-name">Gradient Boosting + Explainability layer</td>
    <td>4 of 4</td>
    <td>~7.8 hours</td>
    <td>Low</td>
</tr>
<tr>
    <td class="td-name">Deep Learning (neural network)</td>
    <td>4 of 4</td>
    <td>~9.2 hours</td>
    <td>Higher — conservative by design</td>
</tr>
<tr class="row-best">
    <td class="td-name td-best">Combined ensemble model</td>
    <td class="td-best">4 of 4</td>
    <td class="td-best">~8.3 hours</td>
    <td class="td-best">Very low (9 in 10 alerts genuine)</td>
</tr>
</tbody>
</table>
</div>
<hr class="div">
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# DIFFERENTIATOR
# ═══════════════════════════════════════════════════════════════
st.markdown('<p class="s-label">If Validated — Potential Implications</p>', unsafe_allow_html=True)
st.markdown('<h2 class="s-title">What real-world validation could unlock.</h2>', unsafe_allow_html=True)
st.markdown('<p class="s-sub">These are not claims — they are research questions. If the model performs on real biopharma cold chain data the way it does on research data, the following become worth exploring.</p>', unsafe_allow_html=True)

col_l, col_r = st.columns([3, 2])

with col_l:
    st.markdown("""
    <div class="diff-list">
        <div class="diff-row">
            <div class="diff-key">Competitive position</div>
            <div class="diff-val">No freeze & thaw equipment provider currently offers predictive intelligence. First-mover opportunity in a premium segment.</div>
        </div>
        <div class="diff-row">
            <div class="diff-key">Pricing power</div>
            <div class="diff-val">Smart hardware commands a premium. An intelligence subscription sits alongside the hardware sale as recurring revenue.</div>
        </div>
        <div class="diff-row">
            <div class="diff-key">Customer retention</div>
            <div class="diff-val">Customers integrated into an intelligence platform have a significantly higher switching cost than hardware-only customers.</div>
        </div>
        <div class="diff-row">
            <div class="diff-key">Brand alignment</div>
            <div class="diff-val">"Protect Every Drop" becomes measurable and verifiable — not a marketing statement but a quantified commitment.</div>
        </div>
        <div class="diff-row">
            <div class="diff-key">Validation path</div>
            <div class="diff-val">System is built. Runs on standard sensor data formats. Real-world validation can begin with one shared dataset.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_r:
    st.markdown("""
    <div class="highlight-box">
        <h3>The market is ready for this.</h3>
        <p>
            Gene therapies, monoclonal antibodies, and advanced biologics
            are increasing in both value and volume. A single batch failure
            can cost hundreds of thousands to millions of euros — and damage
            the relationship with the customer far beyond the product value.
            <br><br>
            Biopharma manufacturers are actively seeking ways to de-risk
            their cold chain. This is that solution.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="div">', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# CALCULATOR
# ═══════════════════════════════════════════════════════════════
st.markdown('<p class="s-label">Illustrative Impact — If Validated</p>', unsafe_allow_html=True)
st.markdown('<h2 class="s-title">What could early warning mean commercially?</h2>', unsafe_allow_html=True)
st.markdown('<p class="s-sub">Purely illustrative. These numbers assume the research results hold on real pharmaceutical cold chain hardware — which has not been tested. This is what the research is trying to find out.</p>', unsafe_allow_html=True)

col_s, col_r2 = st.columns([3, 2])
with col_s:
    batch_cost = st.slider("Cost per failed batch (€)", 50_000, 5_000_000, 500_000, 50_000, format="€%d")
    batches_yr = st.slider("Batches managed per year",   10, 1000, 100, 10)
    fail_rate  = st.slider("Current failure rate (%)",   0.5, 10.0, 2.6, 0.1)
    avert_pct  = st.slider("Failures preventable with early warning (%)", 40, 95, 75, 5)

annual_loss = batch_cost * batches_yr * (fail_rate / 100)
savings     = annual_loss * (avert_pct / 100)
remaining   = annual_loss - savings

with col_r2:
    st.markdown(f"""
    <div class="result-panel">
        <div style="font-size:13px;font-weight:700;margin-bottom:20px;opacity:0.7">Annual impact — per customer</div>
        <div class="rl">Current annual exposure</div>
        <div class="rv">€{annual_loss:,.0f}</div>
        <hr>
        <div class="rl">Recoverable with predictive monitoring</div>
        <div class="rv big">€{savings:,.0f}</div>
        <div class="rl">Remaining exposure</div>
        <div class="rv" style="font-size:22px">€{remaining:,.0f}</div>
        <p class="rn">Illustrative estimate. Actual figures require real-world validation
        of model performance on real biopharma cold chain hardware and customer data.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="div">', unsafe_allow_html=True)




# ═══════════════════════════════════════════════════════════════
# LIVE PREDICTION DEMO
# ═══════════════════════════════════════════════════════════════
st.markdown('<p class="s-label">Live Model</p>', unsafe_allow_html=True)
st.markdown('<h2 class="s-title">Try the ensemble — in real time.</h2>', unsafe_allow_html=True)
st.markdown('<p class="s-sub">Five models were built and compared — the final ensemble combines the 3 best-performing ones (Random Forest + XGBoost + LightGBM). Adjust the sensor readings below and watch the risk score update instantly.</p>', unsafe_allow_html=True)

_FEATURE_SLIDERS = [
    ("avg_temp_last_6hours",  "Average temperature — last 6 hours (°C)",         -5.0, 30.0,  6.5, 0.1),
    ("temp_3hours_ago",       "Temperature 3 hours ago (°C)",                     -5.0, 30.0,  5.8, 0.1),
    ("rolling_std_3hour",     "Temperature variability — 3hr standard deviation",  0.0,  5.0,  1.2, 0.05),
    ("duration_outside_safe", "Hours spent outside the 2–8°C safe zone",           0.0, 12.0,  0.5, 0.1),
    ("temp_drop_last_1hour",  "Temp change last hour (°C, positive = warming)",   -5.0,  5.0,  0.8, 0.1),
    ("hour_of_day",           "Hour of day",                                         0,   23,   14,  1),
]

_col_sliders, _col_result = st.columns([3, 2])

with _col_sliders:
    _vals = {}
    for _feat, _label, _lo, _hi, _def, _step in _FEATURE_SLIDERS:
        if _feat == "hour_of_day":
            _vals[_feat] = float(st.slider(_label, int(_lo), int(_hi), int(_def), int(_step)))
        else:
            _vals[_feat] = st.slider(_label, float(_lo), float(_hi), float(_def), float(_step))
    st.markdown("""
    <p style="font-size:11px;color:#9CA3AF;margin-top:8px;line-height:1.6;">
    Features are computed from raw sensor temperature readings.
    The safe zone for pharma cold chain is typically 2–8°C.
    </p>""", unsafe_allow_html=True)

with _col_result:
    if _MODELS_OK:
        _X = [[_vals[f] for f, *_ in _FEATURE_SLIDERS]]
        _p_rf   = float(_RF.predict_proba(_X)[0, 1])
        _p_xgb  = float(_XGB.predict_proba(_X)[0, 1])
        _p_lgbm = float(_LGBM.predict_proba(_X)[0, 1])
        _risk   = round((_p_rf + _p_xgb + _p_lgbm) / 3, 3)
        _alert  = _risk >= _THRESHOLD
        _pct    = int(_risk * 100)

        _rc = "#DC2626" if _alert else "#059669"
        _status_txt  = "⚠ High Risk — Alert would fire" if _alert else "✓ Low Risk — Normal"
        _status_bg   = "#FEF2F2" if _alert else "#F0FDF4"

        def _bar(p, threshold):
            filled = int(p * 20)
            c = "#DC2626" if p >= threshold else "#059669"
            bar = '█' * filled + '░' * (20 - filled)
            return f'<span style="font-family:monospace;color:{c};font-size:11px;">{bar}</span> <span style="font-weight:700;color:{c};">{p:.0%}</span>'

        st.markdown(f"""
        <div style="background:#fff;border:1px solid #E5E7EB;border-radius:10px;padding:28px;">
            <div style="font-size:11px;font-weight:700;letter-spacing:0.08em;
                        text-transform:uppercase;color:#9CA3AF;margin-bottom:6px;">
                Ensemble Risk Score
            </div>
            <div style="font-size:72px;font-weight:800;color:{_rc};
                        letter-spacing:-0.04em;line-height:1;margin-bottom:14px;">
                {_pct}%
            </div>
            <div style="background:{_status_bg};border:1px solid {_rc};border-radius:6px;
                        padding:10px 14px;font-size:13px;font-weight:600;
                        color:{_rc};margin-bottom:22px;">
                {_status_txt}
            </div>
            <div style="font-size:11px;color:#9CA3AF;font-weight:600;
                        text-transform:uppercase;letter-spacing:0.07em;margin-bottom:12px;">
                Model breakdown
            </div>
            <div style="font-size:13px;color:#374151;">
                <div style="display:flex;flex-direction:column;gap:2px;padding:7px 0;
                            border-bottom:1px solid #F3F4F6;">
                    <span style="font-size:11px;color:#9CA3AF;">Random Forest</span>
                    {_bar(_p_rf, _THRESHOLD)}
                </div>
                <div style="display:flex;flex-direction:column;gap:2px;padding:7px 0;
                            border-bottom:1px solid #F3F4F6;">
                    <span style="font-size:11px;color:#9CA3AF;">XGBoost</span>
                    {_bar(_p_xgb, _THRESHOLD)}
                </div>
                <div style="display:flex;flex-direction:column;gap:2px;padding:7px 0;">
                    <span style="font-size:11px;color:#9CA3AF;">LightGBM</span>
                    {_bar(_p_lgbm, _THRESHOLD)}
                </div>
            </div>
            <p style="font-size:11px;color:#9CA3AF;margin-top:16px;line-height:1.6;">
                Threshold: {_THRESHOLD}. Research prototype — not validated on real biopharma hardware.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#F9FAFB;border:1px solid #E5E7EB;border-radius:10px;
                    padding:28px;text-align:center;color:#9CA3AF;">
            <div style="font-size:32px;margin-bottom:12px;">🔒</div>
            <div style="font-size:13px;">Model files not available on this deployment.</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<hr class="div">', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# CTA
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="cta-block">
    <div class="cta-eyebrow">Research Goal</div>
    <h3>One dataset away from<br>a <em>real</em> answer.</h3>
    <p class="cta-sub">
        The models are trained. The framework runs on standard sensor data formats.
        The one thing this research cannot answer without real biopharma data is the most
        important thing: does it actually work on your hardware?
    </p>
    <div class="cta-punch">One shared dataset. One honest answer.</div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="footer">
    Cold Chain Intelligence — Predictive Temperature Monitoring Research<br>
    <span style="font-size:11px">
    Proof of concept validated on pharmaceutical cold chain research data
    (NAB Machine Temperature Dataset + Sun et al. 2022, Nature Scientific Data).
    Real-world performance on pharmaceutical production hardware requires data validation.
    </span>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
