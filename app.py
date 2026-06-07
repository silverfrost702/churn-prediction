"""
ChurnSight — Customer Churn Prediction & Retention Analytics
Streamlit Dashboard — Production Design
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import plotly.graph_objects as go
import plotly.express as px

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="ChurnSight",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# DESIGN SYSTEM — Full CSS Override
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&family=Syne:wght@700;800&display=swap');

/* ── Root tokens ── */
:root {
    --bg-base:       #08090d;
    --bg-surface:    #0f1117;
    --bg-raised:     #161820;
    --bg-hover:      #1c1f2b;
    --border:        #1f2235;
    --border-light:  #262a3d;
    --accent:        #f97316;
    --accent-dim:    rgba(249,115,22,0.12);
    --accent-glow:   rgba(249,115,22,0.35);
    --cyan:          #22d3ee;
    --cyan-dim:      rgba(34,211,238,0.10);
    --green:         #34d399;
    --green-dim:     rgba(52,211,153,0.10);
    --red:           #f87171;
    --red-dim:       rgba(248,113,113,0.10);
    --yellow:        #fbbf24;
    --text-primary:  #f0f2f8;
    --text-secondary:#8b92b3;
    --text-muted:    #4a5080;
    --font-display:  'Syne', sans-serif;
    --font-body:     'DM Sans', sans-serif;
    --font-mono:     'DM Mono', monospace;
    --radius-sm:     6px;
    --radius-md:     10px;
    --radius-lg:     16px;
}

/* ── Base reset ── */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg-base) !important;
    font-family: var(--font-body) !important;
    color: var(--text-primary) !important;
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse 80% 60% at 50% -10%, rgba(249,115,22,0.06) 0%, transparent 60%),
                var(--bg-base) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div { padding: 1.5rem 1rem !important; }

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden !important; }
[data-testid="stDecoration"] { display: none !important; }

/* ── Block container ── */
.block-container {
    padding: 2rem 2.5rem 3rem !important;
    max-width: 1400px !important;
}

/* ── Typography ── */
h1, h2, h3 { font-family: var(--font-display) !important; letter-spacing: -0.02em; }

/* ── Metrics ── */
[data-testid="metric-container"] {
    background: var(--bg-raised) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    padding: 1rem 1.25rem !important;
    transition: border-color 0.2s;
}
[data-testid="metric-container"]:hover { border-color: var(--border-light) !important; }
[data-testid="metric-container"] label {
    font-size: 11px !important;
    font-weight: 500 !important;
    letter-spacing: 0.07em !important;
    text-transform: uppercase !important;
    color: var(--text-secondary) !important;
    font-family: var(--font-body) !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: var(--font-display) !important;
    font-size: 1.9rem !important;
    font-weight: 800 !important;
    color: var(--text-primary) !important;
}
[data-testid="stMetricDelta"] { font-size: 12px !important; font-family: var(--font-mono) !important; }

/* ── Plotly charts ── */
.js-plotly-plot { border-radius: var(--radius-md) !important; }

/* ── Buttons ── */
[data-testid="stButton"] > button {
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-family: var(--font-body) !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    letter-spacing: 0.02em !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s !important;
    box-shadow: 0 0 20px var(--accent-glow) !important;
}
[data-testid="stButton"] > button:hover {
    filter: brightness(1.1) !important;
    box-shadow: 0 0 30px var(--accent-glow) !important;
    transform: translateY(-1px) !important;
}

/* ── Sliders ── */
[data-testid="stSlider"] [role="slider"] {
    background: var(--accent) !important;
    box-shadow: 0 0 8px var(--accent-glow) !important;
}
[data-testid="stSlider"] > div > div > div > div {
    background: var(--accent) !important;
}

/* ── Selectboxes ── */
[data-testid="stSelectbox"] > div > div {
    background: var(--bg-raised) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-body) !important;
}
[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px var(--accent-dim) !important;
}

/* ── Radio ── */
[data-testid="stRadio"] label {
    font-family: var(--font-body) !important;
    font-size: 13px !important;
    color: var(--text-secondary) !important;
    transition: color 0.15s;
}
[data-testid="stRadio"] label:hover { color: var(--text-primary) !important; }

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
}
[data-testid="stTabs"] [role="tab"] {
    font-family: var(--font-body) !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    color: var(--text-muted) !important;
    border-bottom: 2px solid transparent !important;
    padding: 0.5rem 1.25rem !important;
    transition: all 0.15s !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    background: var(--bg-surface) !important;
}
[data-testid="stExpander"] summary {
    font-family: var(--font-body) !important;
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
}

/* ── Info / warning boxes ── */
[data-testid="stInfo"] {
    background: var(--cyan-dim) !important;
    border: 1px solid rgba(34,211,238,0.2) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--cyan) !important;
    font-family: var(--font-body) !important;
    font-size: 13px !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    overflow: hidden !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

/* ── Checkbox ── */
[data-testid="stCheckbox"] label {
    font-family: var(--font-body) !important;
    font-size: 14px !important;
    color: var(--text-secondary) !important;
}

/* ───────────────────────────────────────
   CUSTOM COMPONENTS
─────────────────────────────────────── */

/* Page header */
.cs-page-header {
    margin-bottom: 2rem;
}
.cs-page-header h1 {
    font-family: var(--font-display) !important;
    font-size: 2rem !important;
    font-weight: 800 !important;
    color: var(--text-primary) !important;
    margin: 0 0 4px !important;
    letter-spacing: -0.03em !important;
}
.cs-page-header p {
    font-size: 14px !important;
    color: var(--text-secondary) !important;
    margin: 0 !important;
}

/* Section label */
.cs-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 10px;
    font-family: var(--font-body);
}

/* Insight card */
.cs-insight {
    background: var(--bg-raised);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: var(--radius-md);
    padding: 14px 18px;
    font-size: 13.5px;
    line-height: 1.6;
    color: var(--text-secondary);
    margin: 12px 0;
}
.cs-insight b { color: var(--text-primary); }
.cs-insight .cs-icon { margin-right: 6px; }

/* Risk badge */
.cs-risk-critical { color: var(--red);    font-weight: 700; font-size: 1.1rem; }
.cs-risk-high     { color: var(--accent); font-weight: 700; font-size: 1.1rem; }
.cs-risk-medium   { color: var(--yellow); font-weight: 700; font-size: 1.1rem; }
.cs-risk-low      { color: var(--green);  font-weight: 700; font-size: 1.1rem; }

/* Stat pill */
.cs-pill {
    display: inline-block;
    background: var(--accent-dim);
    color: var(--accent);
    border: 1px solid rgba(249,115,22,0.25);
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 11px;
    font-weight: 600;
    font-family: var(--font-mono);
    letter-spacing: 0.04em;
}

/* Logo / brand */
.cs-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 6px;
}
.cs-brand-icon {
    width: 34px; height: 34px;
    background: var(--accent);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px;
    box-shadow: 0 0 16px var(--accent-glow);
}
.cs-brand-name {
    font-family: var(--font-display) !important;
    font-size: 1.25rem !important;
    font-weight: 800 !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.02em;
}
.cs-brand-sub {
    font-size: 11px;
    color: var(--text-muted);
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

/* Action item */
.cs-action {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 10px 14px;
    background: var(--bg-raised);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    margin-bottom: 7px;
    font-size: 13.5px;
    color: var(--text-secondary);
    transition: border-color 0.15s, background 0.15s;
}
.cs-action:hover {
    border-color: var(--border-light);
    background: var(--bg-hover);
    color: var(--text-primary);
}

/* Section divider with label */
.cs-section-divider {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 1.5rem 0 1rem;
}
.cs-section-divider span {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-muted);
    white-space: nowrap;
    font-family: var(--font-body);
}
.cs-section-divider::before, .cs-section-divider::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* Decision banner */
.cs-decision-flag {
    background: linear-gradient(135deg, rgba(249,115,22,0.08) 0%, rgba(249,115,22,0.04) 100%);
    border: 1px solid rgba(249,115,22,0.3);
    border-radius: var(--radius-md);
    padding: 14px 20px;
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 12px 0;
}
.cs-decision-ok {
    background: linear-gradient(135deg, rgba(52,211,153,0.06) 0%, rgba(52,211,153,0.02) 100%);
    border: 1px solid rgba(52,211,153,0.2);
    border-radius: var(--radius-md);
    padding: 14px 20px;
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 12px 0;
}

/* Sidebar nav item */
.cs-nav-header {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin: 1.2rem 0 0.5rem;
    font-family: var(--font-body);
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────
PLOT_LAYOUT = dict(
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='DM Sans, sans-serif', color='#8b92b3', size=12),
    title_font=dict(family='Syne, sans-serif', color='#f0f2f8', size=14),
    margin=dict(t=50, b=40, l=10, r=10),
    xaxis=dict(gridcolor='#1f2235', linecolor='#1f2235', tickfont=dict(size=11)),
    yaxis=dict(gridcolor='#1f2235', linecolor='#1f2235', tickfont=dict(size=11)),
    legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='#1f2235', borderwidth=1, font=dict(size=11))
)


# ─────────────────────────────────────────
# LOAD ARTIFACTS
# ─────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open('models/xgboost_model.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)

@st.cache_data
def load_batch():
    try:
        return pd.read_csv('data/predictions_with_shap.csv')
    except:
        return None

@st.cache_data
def load_shap_values():
    try:
        return np.load('models/shap_values_test.npy')
    except:
        return None

artifacts = load_artifacts()
model     = artifacts['xgb_model']
threshold = artifacts['optimal_threshold']
features  = artifacts['feature_names']
metrics   = artifacts['metrics']
explainer = load_explainer(model)
batch_df  = load_batch()
shap_vals = load_shap_values()


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="cs-brand">
        <div class="cs-brand-icon">📡</div>
        <div>
            <div class="cs-brand-name">ChurnSight</div>
            <div class="cs-brand-sub">Retention Intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="cs-nav-header">Model Performance</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.metric("ROC-AUC",   f"{metrics['roc_auc']:.3f}")
    c2.metric("F1",        f"{metrics['f1']:.3f}")
    c1.metric("Recall",    f"{metrics['recall']:.3f}")
    c2.metric("Precision", f"{metrics['precision']:.3f}")

    st.markdown(f"""
    <div style="margin-top:6px; padding: 8px 10px; background: var(--bg-raised);
         border: 1px solid var(--border); border-radius: var(--radius-sm);
         font-size: 11px; color: var(--text-muted); font-family: var(--font-mono);">
        CV AUC {metrics['cv_mean_auc']:.3f} ± {metrics['cv_std_auc']:.3f} · 5-fold
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="cs-nav-header">Threshold</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background: var(--accent-dim); border: 1px solid rgba(249,115,22,0.25);
         border-radius: var(--radius-sm); padding: 10px 12px; font-size: 12px; line-height: 1.8;">
        <span style="font-family: var(--font-display); font-size: 1.4rem; color: var(--accent);
              font-weight: 800;">{threshold:.2f}</span>
        <span style="color: var(--text-muted); margin-left: 6px; font-size: 11px;">optimal</span><br>
        <span style="color: var(--text-muted); font-size: 11px;">
            FN = $1,500 &nbsp;·&nbsp; FP = $200<br>
            Precision floor ≥ 0.50
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="cs-nav-header">Navigate</div>', unsafe_allow_html=True)
    page = st.radio("", ["Single Prediction", "Batch Analysis", "Model Insights"],
                    label_visibility="collapsed")


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def get_risk(prob):
    if prob >= 0.75:   return "Critical", "#f87171",  "🔴", "cs-risk-critical"
    elif prob >= 0.50: return "High",     "#f97316",  "🟠", "cs-risk-high"
    elif prob >= 0.30: return "Medium",   "#fbbf24",  "🟡", "cs-risk-medium"
    else:              return "Low",      "#34d399",  "🟢", "cs-risk-low"


def build_input_vector(inputs):
    row = {f: 0 for f in features}
    row['tenure']                = inputs['tenure']
    row['MonthlyCharges']        = inputs['monthly']
    row['TotalCharges']          = inputs['monthly'] * inputs['tenure']
    row['SeniorCitizen']         = int(inputs['senior'])
    row['TotalServices']         = inputs['total_services']
    row['IsNewCustomer']         = int(inputs['tenure'] <= 6)
    row['IsLongTermCustomer']    = int(inputs['tenure'] > 36)
    row['IsHighValue']           = int(inputs['monthly'] > 64.76 and inputs['tenure'] > 29)
    row['ContractRisk']          = {'Month-to-month': 3, 'One year': 2, 'Two year': 1}[inputs['contract']]
    row['ChargesPerTenureMonth'] = inputs['monthly'] / (inputs['tenure'] + 1)
    row['gender']                = 1 if inputs['gender'] == 'Male' else 0
    row['Partner']               = 1 if inputs['partner'] == 'Yes' else 0
    row['Dependents']            = 1 if inputs['dependents'] == 'Yes' else 0
    row['PhoneService']          = 1 if inputs['phone'] == 'Yes' else 0
    row['PaperlessBilling']      = 1 if inputs['paperless'] == 'Yes' else 0
    for col in ['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']:
        row[col + '_bin'] = 1 if inputs.get(col,'No') == 'Yes' else 0
    row['MultipleLines_No phone service'] = 1 if inputs['phone'] == 'No' else 0
    row['MultipleLines_Yes']              = 1 if inputs.get('multilines') == 'Yes' else 0
    row['InternetService_Fiber optic']    = 1 if inputs['internet'] == 'Fiber optic' else 0
    row['InternetService_No']             = 1 if inputs['internet'] == 'No' else 0
    row['Contract_One year']              = 1 if inputs['contract'] == 'One year' else 0
    row['Contract_Two year']              = 1 if inputs['contract'] == 'Two year' else 0
    pm = inputs['payment']
    row['PaymentMethod_Credit card (automatic)'] = 1 if pm == 'Credit card (automatic)' else 0
    row['PaymentMethod_Electronic check']        = 1 if pm == 'Electronic check' else 0
    row['PaymentMethod_Mailed check']            = 1 if pm == 'Mailed check' else 0
    no_internet = inputs['internet'] == 'No'
    for col in ['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']:
        row[col + '_No internet service'] = 1 if no_internet else 0
    return pd.DataFrame([row]).reindex(columns=features, fill_value=0)


def shap_bar(shap_row, feat_names, title=""):
    df = pd.DataFrame({'feature': feat_names, 'shap': shap_row})
    df = df.reindex(df['shap'].abs().sort_values(ascending=False).index).head(12).sort_values('shap')
    colors = ['#22d3ee' if v < 0 else '#f97316' for v in df['shap']]
    fig = go.Figure(go.Bar(
        x=df['shap'], y=df['feature'], orientation='h',
        marker_color=colors, marker_line_width=0,
        text=[f'{v:+.3f}' for v in df['shap']],
        textposition='outside',
        textfont=dict(family='DM Mono', size=10, color='#8b92b3')
    ))
    layout = {**PLOT_LAYOUT,
              'title': dict(text=title, font=dict(size=13)),
              'xaxis_title': 'SHAP value',
              'height': 420,
              'margin': dict(l=170, r=80, t=50, b=40),
              'yaxis': dict(gridcolor='#1f2235', tickfont=dict(size=11, family='DM Mono'))}
    fig.update_layout(**layout)
    return fig


RETENTION_ACTIONS = {
    "Critical": [
        ("🚨", "Assign dedicated account manager immediately"),
        ("💰", "Offer 20–30% discount on next 3 months"),
        ("📞", "Schedule proactive call within 24 hours"),
        ("🔒", "Propose annual contract with loyalty incentive"),
        ("⚡", "Escalate to senior retention team"),
    ],
    "High": [
        ("📧", "Send personalised retention offer this week"),
        ("🎁", "Offer free service upgrade (e.g. Tech Support)"),
        ("📊", "Investigate billing pain points"),
        ("🔄", "Propose switching to annual contract"),
    ],
    "Medium": [
        ("📩", "Include in next loyalty email campaign"),
        ("📦", "Suggest relevant service bundle"),
        ("⭐", "Invite to customer feedback programme"),
    ],
    "Low": [
        ("✅", "No action needed — customer appears satisfied"),
        ("📈", "Good candidate for upsell / cross-sell campaign"),
    ]
}


# ─────────────────────────────────────────
# PAGE 1 — SINGLE PREDICTION
# ─────────────────────────────────────────
if page == "Single Prediction":
    st.markdown("""
    <div class="cs-page-header">
        <h1>Single Customer Analysis</h1>
        <p>Enter customer details to score churn risk, explain the prediction, and generate retention actions.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Customer Details", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="cs-label">Account</div>', unsafe_allow_html=True)
            tenure   = st.slider("Tenure (months)", 0, 72, 12)
            monthly  = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
            contract = st.selectbox("Contract", ['Month-to-month','One year','Two year'])
            payment  = st.selectbox("Payment Method", [
                'Electronic check','Mailed check',
                'Bank transfer (automatic)','Credit card (automatic)'
            ])
        with col2:
            st.markdown('<div class="cs-label">Demographics</div>', unsafe_allow_html=True)
            gender     = st.selectbox("Gender", ['Male','Female'])
            senior     = st.checkbox("Senior Citizen")
            partner    = st.selectbox("Partner", ['No','Yes'])
            dependents = st.selectbox("Dependents", ['No','Yes'])
            paperless  = st.selectbox("Paperless Billing", ['No','Yes'])
        with col3:
            st.markdown('<div class="cs-label">Services</div>', unsafe_allow_html=True)
            internet   = st.selectbox("Internet", ['DSL','Fiber optic','No'])
            phone      = st.selectbox("Phone", ['Yes','No'])
            multilines = st.selectbox("Multiple Lines", ['No','Yes','No phone service'])
            online_sec = st.selectbox("Online Security", ['No','Yes','No internet service'])
            tech_sup   = st.selectbox("Tech Support", ['No','Yes','No internet service'])
            stream_tv  = st.selectbox("Streaming TV", ['No','Yes','No internet service'])
            stream_mov = st.selectbox("Streaming Movies", ['No','Yes','No internet service'])

    total_services = sum(1 for s in [phone,multilines,online_sec,tech_sup,stream_tv,stream_mov] if s=='Yes')

    if st.button("Run Churn Analysis", type="primary", use_container_width=True):
        inputs = {
            'tenure':tenure,'monthly':monthly,'contract':contract,'payment':payment,
            'gender':gender,'senior':senior,'partner':partner,'dependents':dependents,
            'paperless':paperless,'internet':internet,'phone':phone,'multilines':multilines,
            'OnlineSecurity':online_sec,'TechSupport':tech_sup,
            'StreamingTV':stream_tv,'StreamingMovies':stream_mov,
            'total_services':total_services
        }
        X_in  = build_input_vector(inputs)
        prob  = float(model.predict_proba(X_in)[:,1][0])
        tier, color, emoji, css = get_risk(prob)
        flag  = prob >= threshold

        st.markdown('<div class="cs-section-divider"><span>Risk Assessment</span></div>', unsafe_allow_html=True)

        k1,k2,k3,k4 = st.columns(4)
        k1.metric("Churn Probability", f"{prob:.1%}")
        k2.metric("Risk Tier", f"{emoji}  {tier}")
        k3.metric("Decision", "Flag for Retention" if flag else "Monitor Only")
        k4.metric("Threshold", f"{threshold:.2f}")

        if flag:
            st.markdown(f"""
            <div class="cs-decision-flag">
                <span style="font-size:20px">⚠️</span>
                <div>
                    <span style="font-weight:600; color: var(--accent); font-family: var(--font-display);">
                        Retention Action Required
                    </span>
                    <span style="color: var(--text-muted); font-size: 12px; margin-left: 10px;">
                        Probability {prob:.1%} exceeds threshold {threshold:.2f}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="cs-decision-ok">
                <span style="font-size:20px">✅</span>
                <div>
                    <span style="font-weight:600; color: var(--green); font-family: var(--font-display);">
                        No Immediate Action Needed
                    </span>
                    <span style="color: var(--text-muted); font-size: 12px; margin-left: 10px;">
                        Probability {prob:.1%} is below threshold {threshold:.2f}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="cs-section-divider"><span>Prediction Explanation</span></div>', unsafe_allow_html=True)

        g_col, s_col = st.columns([1, 2])
        with g_col:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={'suffix':'%','font':{'size':40,'family':'Syne','color':'#f0f2f8'}},
                gauge={
                    'axis':{'range':[0,100],'tickcolor':'#4a5080','tickfont':{'size':10}},
                    'bar':{'color':color,'thickness':0.22},
                    'bgcolor':'rgba(0,0,0,0)',
                    'borderwidth':0,
                    'steps':[
                        {'range':[0,30],  'color':'rgba(52,211,153,0.08)'},
                        {'range':[30,50], 'color':'rgba(251,191,36,0.08)'},
                        {'range':[50,75], 'color':'rgba(249,115,22,0.08)'},
                        {'range':[75,100],'color':'rgba(248,113,113,0.08)'},
                    ],
                    'threshold':{'line':{'color':'rgba(255,255,255,0.3)','width':2},'thickness':0.8,'value':threshold*100}
                },
                title={'text':f'<span style="font-size:12px;color:#4a5080;font-family:DM Sans">CHURN RISK SCORE</span>'}
            ))
            fig_gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=260,
                margin=dict(t=40,b=0,l=20,r=20)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with s_col:
            sv     = explainer.shap_values(X_in)
            sv_row = sv[0]
            st.plotly_chart(shap_bar(sv_row, features, "Feature Contributions to Churn Score"), use_container_width=True)

        top_idx  = int(np.argmax(np.abs(sv_row)))
        top_name = features[top_idx]
        top_val  = sv_row[top_idx]
        direction = "increased" if top_val > 0 else "reduced"
        st.markdown(f"""
        <div class="cs-insight">
            <span class="cs-icon">🔎</span>
            <b>Model Reasoning:</b> The primary driver for this prediction is
            <b>{top_name}</b> <span class="cs-pill">SHAP {top_val:+.3f}</span>,
            which {direction} the churn probability. The retention team should
            prioritise investigating this dimension.
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="cs-section-divider"><span>Business Impact & Actions</span></div>', unsafe_allow_html=True)

        bi_col, rec_col = st.columns([1, 1])
        with bi_col:
            st.markdown('<div class="cs-label">Business Impact</div>', unsafe_allow_html=True)
            annual = monthly * 12
            b1,b2 = st.columns(2)
            b1.metric("Annual Revenue at Risk", f"${annual:,.0f}" if flag else "—")
            b2.metric("Retention Budget Cap",   f"${int(annual*0.25):,}" if flag else "—")
            b1.metric("Replacement Cost",       "$1,500" if flag else "—")
            b2.metric("Services Subscribed",    str(total_services))

        with rec_col:
            st.markdown(f'<div class="cs-label">Recommended Actions — <span class="{css}">{emoji} {tier}</span></div>', unsafe_allow_html=True)
            for icon, text in RETENTION_ACTIONS[tier]:
                st.markdown(f'<div class="cs-action"><span>{icon}</span><span>{text}</span></div>', unsafe_allow_html=True)


# ─────────────────────────────────────────
# PAGE 2 — BATCH ANALYSIS
# ─────────────────────────────────────────
elif page == "Batch Analysis":
    st.markdown("""
    <div class="cs-page-header">
        <h1>Batch Churn Analysis</h1>
        <p>Risk segmentation and churn driver analysis across the full scored customer base.</p>
    </div>
    """, unsafe_allow_html=True)

    if batch_df is None:
        st.warning("Run notebook 03_shap_analysis.ipynb to generate data/predictions_with_shap.csv")
        st.stop()

    def tier_label(p):
        if p >= 0.75:   return 'Critical  >75%'
        elif p >= 0.50: return 'High  50–75%'
        elif p >= 0.30: return 'Medium  30–50%'
        else:           return 'Low  <30%'

    batch_df['risk_tier'] = batch_df['churn_probability'].apply(tier_label)
    flagged = batch_df[batch_df['predicted_churn'] == 1]

    # KPI row
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Total Customers",        f"{len(batch_df):,}")
    k2.metric("Flagged for Retention",  f"{len(flagged):,}")
    k3.metric("Flag Rate",              f"{len(flagged)/len(batch_df)*100:.1f}%")
    k4.metric("Est. Annual Revenue Risk",f"${len(flagged)*70*12:,.0f}")

    st.markdown('<div class="cs-section-divider"><span>Risk Distribution</span></div>', unsafe_allow_html=True)
    chart_col, dist_col = st.columns(2)

    with chart_col:
        tier_counts = batch_df['risk_tier'].value_counts()
        fig_pie = go.Figure(go.Pie(
            labels=tier_counts.index,
            values=tier_counts.values,
            hole=0.55,
            marker=dict(
                colors=['#f87171','#f97316','#fbbf24','#34d399'],
                line=dict(color='#08090d', width=2)
            ),
            textfont=dict(family='DM Sans', size=12),
            hovertemplate='<b>%{label}</b><br>%{value} customers<br>%{percent}<extra></extra>'
        ))
        fig_pie.add_annotation(
            text=f"<b>{len(batch_df)}</b><br><span style='font-size:10px'>customers</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, family='Syne', color='#f0f2f8')
        )
        layout = {**PLOT_LAYOUT, 'title': 'Customer Risk Distribution', 'height': 350,
                  'margin': dict(t=50,b=20,l=20,r=20),
                  'legend': dict(orientation='v', x=1, y=0.5, bgcolor='rgba(0,0,0,0)')}
        fig_pie.update_layout(**layout)
        st.plotly_chart(fig_pie, use_container_width=True)

    with dist_col:
        fig_hist = go.Figure(go.Histogram(
            x=batch_df['churn_probability'], nbinsx=30,
            marker_color='#f97316', marker_line_color='#08090d',
            marker_line_width=0.5, opacity=0.85
        ))
        fig_hist.add_vline(
            x=threshold, line_dash='dot', line_color='rgba(255,255,255,0.4)', line_width=2,
            annotation_text=f'  Threshold {threshold:.2f}',
            annotation_font=dict(size=11, color='#8b92b3', family='DM Mono')
        )
        layout = {**PLOT_LAYOUT, 'title': 'Churn Probability Distribution',
                  'xaxis_title': 'Churn Probability', 'yaxis_title': 'Customers', 'height': 350}
        fig_hist.update_layout(**layout)
        st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown('<div class="cs-section-divider"><span>Risk Tier Summary</span></div>', unsafe_allow_html=True)
    tier_summary = batch_df.groupby('risk_tier').agg(
        Customers=('churn_probability','count'),
        Avg_Probability=('churn_probability','mean'),
        Actual_Churn_Rate=('actual_churn','mean')
    ).round(3).reset_index().rename(columns={'risk_tier':'Risk Tier'})
    st.dataframe(tier_summary, use_container_width=True, hide_index=True)

    if 'top_driver' in batch_df.columns:
        st.markdown('<div class="cs-section-divider"><span>Top Churn Drivers</span></div>', unsafe_allow_html=True)
        drivers = flagged['top_driver'].value_counts().head(8)
        fig_d = go.Figure(go.Bar(
            x=drivers.values, y=drivers.index, orientation='h',
            marker_color='#f97316', marker_line_width=0, opacity=0.85,
            text=drivers.values, textposition='outside',
            textfont=dict(family='DM Mono', size=10, color='#8b92b3')
        ))
        layout = {**PLOT_LAYOUT,
                  'title': f'Most Common Top SHAP Driver — {len(flagged)} Flagged Customers',
                  'xaxis_title': 'Customer Count', 'height': 340,
                  'margin': dict(l=200, r=60, t=50, b=40),
                  'yaxis': dict(gridcolor='#1f2235', tickfont=dict(size=11, family='DM Mono'))}
        fig_d.update_layout(**layout)
        st.plotly_chart(fig_d, use_container_width=True)

        st.markdown("""
        <div class="cs-insight">
            <span class="cs-icon">💡</span>
            <b>Key Insight:</b> <b>ContractRisk</b> and <b>ChargesPerTenureMonth</b> are the
            top two drivers across flagged customers — both are engineered features created
            from domain knowledge, not raw data columns. This validates that deliberate feature
            engineering outperformed raw inputs in this model.
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────
# PAGE 3 — MODEL INSIGHTS
# ─────────────────────────────────────────
elif page == "Model Insights":
    st.markdown("""
    <div class="cs-page-header">
        <h1>Model Insights</h1>
        <p>Global explainability, performance evaluation, and key modeling decisions.</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Global SHAP", "Performance", "Modeling Decisions"])

    with tab1:
        st.markdown('<div class="cs-label">Mean absolute SHAP value across all test customers</div>', unsafe_allow_html=True)
        if shap_vals is not None:
            mean_abs = pd.DataFrame({
                'feature': features,
                'importance': np.abs(shap_vals).mean(axis=0)
            }).sort_values('importance', ascending=False).head(15)

            fig_imp = go.Figure(go.Bar(
                x=mean_abs['importance'][::-1],
                y=mean_abs['feature'][::-1],
                orientation='h',
                marker_color='#f97316', marker_line_width=0, opacity=0.85,
                text=[f'{v:.3f}' for v in mean_abs['importance'][::-1]],
                textposition='outside',
                textfont=dict(family='DM Mono', size=10, color='#8b92b3')
            ))
            layout = {**PLOT_LAYOUT,
                      'xaxis_title': 'Mean |SHAP Value|', 'height': 520,
                      'margin': dict(l=200, r=80, t=30, b=40),
                      'yaxis': dict(gridcolor='#1f2235', tickfont=dict(size=11, family='DM Mono'))}
            fig_imp.update_layout(**layout)
            st.plotly_chart(fig_imp, use_container_width=True)

            top3 = mean_abs.head(3)['feature'].tolist()
            st.markdown(f"""
            <div class="cs-insight">
                <span class="cs-icon">🔎</span>
                <b>Top 3 global drivers:</b>
                <span class="cs-pill">{top3[0]}</span>&nbsp;
                <span class="cs-pill">{top3[1]}</span>&nbsp;
                <span class="cs-pill">{top3[2]}</span><br><br>
                The top two drivers are <b>engineered features</b>, not raw data columns —
                confirming that domain-informed feature engineering added meaningful signal
                beyond what was available in the raw dataset.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Run notebook 03_shap_analysis.ipynb to generate SHAP values.")

    with tab2:
        perf_df = pd.DataFrame({
            'Metric': ['ROC-AUC','F1 Score','Recall','Precision','CV Mean AUC','CV Std'],
            'Score':  [f"{metrics['roc_auc']:.4f}", f"{metrics['f1']:.4f}",
                       f"{metrics['recall']:.4f}",  f"{metrics['precision']:.4f}",
                       f"{metrics['cv_mean_auc']:.4f}", f"± {metrics['cv_std_auc']:.4f}"],
            'Interpretation': [
                'Strong separation between churners and non-churners',
                'Balanced precision and recall at optimal threshold',
                '84% of all churners successfully identified',
                '1 in 2 flagged customers actually churns',
                'Consistent across all 5 CV folds',
                'Low variance — stable model, not a lucky split'
            ]
        })
        st.dataframe(perf_df, use_container_width=True, hide_index=True)

        st.markdown('<div class="cs-section-divider"><span>Threshold Economics</span></div>', unsafe_allow_html=True)
        t1,t2,t3 = st.columns(3)
        t1.metric("Default Threshold",  "0.50", delta="$166,500 cost")
        t2.metric("Optimal Threshold",  f"{threshold:.2f}", delta="-$14,900 saved", delta_color="inverse")
        t3.metric("Precision Floor",    "0.50", delta="≥1 in 2 flags correct")

        try:
            c1,c2 = st.columns(2)
            c1.image('assets/roc_pr_curves.png',    caption='ROC & PR Curves', use_column_width=True)
            c2.image('assets/threshold_tuning.png', caption='Threshold Tuning', use_column_width=True)
            st.image('assets/confusion_matrix.png', caption='Confusion Matrix at Optimal Threshold', use_column_width=True)
        except:
            st.info("Charts generated when you run notebook 02_modeling.ipynb.")

    with tab3:
        st.markdown("""
### Why XGBoost over Logistic Regression?

| Model | ROC-AUC | F1 | Recall | Status |
|---|---|---|---|---|
| Logistic Regression (baseline) | 0.845 | 0.616 | 0.789 | Baseline |
| XGBoost + SMOTE | 0.836 | 0.598 | 0.687 | ❌ Dropped |
| **XGBoost + scale_pos_weight** | **0.847** | **0.634** | **0.800** | ✅ Final |

---

### Why SMOTE was dropped

SMOTE was tested and removed. At ~26% class imbalance, synthetic oversampling
introduced noise rather than useful signal. `scale_pos_weight = 2.8` adjusts the
loss function directly — achieving better results without modifying the training
distribution.

---

### Why threshold = 0.45 instead of 0.50?

Missing a churner costs ~$1,500. A false alarm costs ~$200. The asymmetry
justifies flagging slightly earlier. A precision floor ≥ 0.50 keeps the
retention team's budget efficient.

**Savings on test set:** $14,900  
**Extrapolated to 70K-customer base:** ~$740,000 annually
        """)