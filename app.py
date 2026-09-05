import streamlit as st
import numpy as np
import plotly.graph_objects as go

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
    padding: 0 !important;
    max-width: 100% !important;
    margin-top: 0 !important;
}

/* ── TOP NAV ── */
.top-nav {
    background: #fff;
    border-bottom: 1px solid #E5E7EB;
    padding: 0 48px;
    height: 52px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0;
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
    padding: 36px 52px 52px;
    max-width: 1080px;
    margin: 0 auto;
}

/* ── HERO ── */
.hero {
    border-bottom: 1px solid #E5E7EB;
    padding-bottom: 32px;
    margin-bottom: 32px;
}
.hero-tag {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #0047BB;
    background: #EFF6FF;
    border: 1px solid #BFDBFE;
    border-radius: 4px;
    padding: 4px 10px;
    margin-bottom: 20px;
}
.hero h1 {
    font-size: 48px;
    font-weight: 800;
    color: #111827;
    letter-spacing: -0.04em;
    line-height: 1.1;
    margin-bottom: 16px;
    max-width: 760px;
}
.hero h1 em {
    font-style: normal;
    color: #0047BB;
}
.hero p {
    font-size: 16px;
    color: #6B7280;
    max-width: 620px;
    line-height: 1.7;
    margin-bottom: 24px;
}
.hero-notice {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: #FFFBEB;
    border: 1px solid #FCD34D;
    border-radius: 6px;
    padding: 8px 14px;
    font-size: 12px;
    color: #92400E;
    font-weight: 500;
}

/* ── METRICS ROW ── */
.metrics-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1px;
    background: #E5E7EB;
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    overflow: hidden;
    margin-bottom: 48px;
}
.metric-cell {
    background: #fff;
    padding: 24px 20px;
}
.metric-cell .val {
    font-size: 32px;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: #111827;
    line-height: 1;
    margin-bottom: 6px;
}
.metric-cell .val.accent { color: #0047BB; }
.metric-cell .val.green  { color: #059669; }
.metric-cell .lbl {
    font-size: 12px;
    font-weight: 600;
    color: #374151;
    margin-bottom: 2px;
}
.metric-cell .sub {
    font-size: 11px;
    color: #9CA3AF;
}

/* ── SECTION ── */
.s-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #9CA3AF;
    margin-bottom: 6px;
}
.s-title {
    font-size: 22px;
    font-weight: 800;
    color: #111827;
    letter-spacing: -0.02em;
    margin-bottom: 6px;
    line-height: 1.3;
}
.s-sub {
    font-size: 14px;
    color: #6B7280;
    line-height: 1.7;
    max-width: 640px;
    margin-bottom: 24px;
}

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
    padding: 20px;
    border: 1px solid #E5E7EB;
    border-radius: 8px;
    background: #fff;
}
.step-n {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #0047BB;
    margin-bottom: 8px;
}
.step-item h4 {
    font-size: 13px;
    font-weight: 700;
    color: #111827;
    margin-bottom: 6px;
}
.step-item p {
    font-size: 13px;
    color: #6B7280;
    line-height: 1.6;
}

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
        Cold Chain Intelligence
    </div>
    <div class="nav-status">
        <div class="nav-status-dot"></div>
        Proof of Concept — Pending Real Data Validation
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="page">', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# HERO
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div class="hero-tag">Pharmaceutical Cold Chain · Predictive Intelligence</div>
    <h1>What if your freeze &amp; thaw platforms could predict failure <em>before</em> it happens?</h1>
    <p>
        A predictive intelligence layer designed for pharmaceutical cold chain hardware —
        detecting temperature excursion risk hours in advance, giving customers real time
        to protect their batch. Not a smarter alarm. A fundamentally different approach.
    </p>
    <div class="hero-notice">
        Proof of concept — performance numbers are based on research data and require
        validation against real SUS sensor hardware before any production commitment.
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="metrics-row">
    <div class="metric-cell">
        <div class="val accent">8+ hrs</div>
        <div class="lbl">Advance Warning</div>
        <div class="sub">on research dataset</div>
    </div>
    <div class="metric-cell">
        <div class="val green">100%</div>
        <div class="lbl">Failures Detected</div>
        <div class="sub">zero misses in testing</div>
    </div>
    <div class="metric-cell">
        <div class="val">9 in 10</div>
        <div class="lbl">Alerts Are Genuine</div>
        <div class="sub">low false alarm rate</div>
    </div>
    <div class="metric-cell">
        <div class="val accent">€M+</div>
        <div class="lbl">Batch Value at Stake</div>
        <div class="sub">per biologics customer</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# THE PROBLEM
# ═══════════════════════════════════════════════════════════════
st.markdown('<p class="s-label">The Problem</p>', unsafe_allow_html=True)
st.markdown('<h2 class="s-title">Cold chain failures are expensive — and often preventable.</h2>', unsafe_allow_html=True)
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
# HOW IT WORKS
# ═══════════════════════════════════════════════════════════════
st.markdown('<p class="s-label">How It Works</p>', unsafe_allow_html=True)
st.markdown('<h2 class="s-title">Simple in concept. Powerful in practice.</h2>', unsafe_allow_html=True)
st.markdown('<p class="s-sub">The system reads the same temperature data the hardware already captures — and learns to recognise the patterns that precede failure, often hours before any threshold is crossed.</p>', unsafe_allow_html=True)

st.markdown("""
<div class="three-col" style="margin-bottom:0">
    <div class="step-item">
        <div class="step-n">Step 01</div>
        <h4>Reads the sensor stream</h4>
        <p>Continuously monitors temperature readings from existing cold chain sensors. No new hardware required — works with what is already there.</p>
    </div>
    <div class="step-item">
        <div class="step-n">Step 02</div>
        <h4>Identifies early warning patterns</h4>
        <p>Detects subtle signals — gradual temperature drift, rising instability, time outside optimal band — that appear hours before a failure event.</p>
    </div>
    <div class="step-item">
        <div class="step-n">Step 03</div>
        <h4>Issues an advance alert</h4>
        <p>Sends notification while product is still safe — giving the team hours to act, not minutes to react after the fact.</p>
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
        not on SUS production hardware. Real-world performance will depend on the specific characteristics
        of SUS sensors and customer cold chain conditions.
        <strong>Validation against real Single Use Support data is the defined next step.</strong>
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
st.markdown('<p class="s-label">Strategic Opportunity</p>', unsafe_allow_html=True)
st.markdown('<h2 class="s-title">A hardware company becomes an intelligence company.</h2>', unsafe_allow_html=True)
st.markdown('<p class="s-sub">This is not an incremental feature. It is a positioning move — from selling equipment that manages cold chain to owning the outcome of protecting every batch.</p>', unsafe_allow_html=True)

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
            <div class="diff-val">System is built. Runs on standard sensor data formats. Real-world validation can begin with one data-sharing agreement.</div>
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
st.markdown('<p class="s-label">Business Case</p>', unsafe_allow_html=True)
st.markdown('<h2 class="s-title">What does early warning save per customer account?</h2>', unsafe_allow_html=True)
st.markdown('<p class="s-sub">Adjust these inputs to reflect a real customer situation. This is the commercial conversation behind every biopharma account.</p>', unsafe_allow_html=True)

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
        of model performance on SUS hardware and customer data.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="div">', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# CTA
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="cta-block">
    <div class="cta-eyebrow">Next Step</div>
    <h3>One dataset away from<br>a <em>real</em> answer.</h3>
    <p class="cta-sub">
        The model is built. The framework runs on standard sensor data formats already
        captured by SUS hardware. One data-sharing agreement is all that stands between
        a proof of concept and a production-ready intelligence layer.
    </p>
    <div class="cta-punch">One test run. A clear, evidence-based answer.</div>
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
    Real-world performance on SUS production hardware requires data validation.
    </span>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
