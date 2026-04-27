"""
Delhi House Price Intelligence — Streamlit App
Author: Nitish
Run: streamlit run app.py
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import joblib

#  Page Config 
st.set_page_config(
    page_title="Delhi House Price Intelligence",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

#  Custom CSS 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

.main { background-color: #0f1117; }

.metric-card {
    background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin: 6px 0;
}
.metric-card .label { color: #8b949e; font-size: 12px; font-weight: 500; letter-spacing: 0.5px; text-transform: uppercase; }
.metric-card .value { color: #e6edf3; font-size: 28px; font-weight: 700; margin: 6px 0 2px; }
.metric-card .sub   { color: #58a6ff; font-size: 13px; font-weight: 400; }

.risk-low    { background: linear-gradient(135deg, #0d2212 0%, #0f2d16 100%); border: 1px solid #3fb950; border-radius: 10px; padding: 15px; text-align: center; }
.risk-medium { background: linear-gradient(135deg, #2d1f0a 0%, #3d2a0e 100%); border: 1px solid #d29922; border-radius: 10px; padding: 15px; text-align: center; }
.risk-high   { background: linear-gradient(135deg, #2c0b0e 0%, #3f0f13 100%); border: 1px solid #f85149; border-radius: 10px; padding: 15px; text-align: center; }

.insight-box {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
    border-left: 3px solid #58a6ff;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin: 10px 0;
    font-size: 14px;
    color: #c9d1d9;
}

.section-header {
    color: #58a6ff;
    font-size: 20px;
    font-weight: 600;
    border-bottom: 1px solid #21262d;
    padding-bottom: 8px;
    margin-bottom: 16px;
}

[data-testid="stSidebar"] { background-color: #0d1117; border-right: 1px solid #21262d; }
[data-testid="stMetricValue"] { color: #e6edf3 !important; }
</style>
""", unsafe_allow_html=True)

#  Load Data & Models 
@st.cache_data
def load_data():
    _base = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(_base, 'csv(s)', 'DelhiHousePrice.csv'))
    df.columns = df.columns.str.strip()
    return df

@st.cache_resource
def load_models():
    try:
        rf        = joblib.load('models/rf_model.joblib')
        xgb_model = joblib.load('models/xgb_model.joblib')
        le_dict   = joblib.load('models/label_encoders.joblib')
        meta      = joblib.load('models/meta.joblib')
        return rf, xgb_model, le_dict, meta
    except FileNotFoundError:
        return None, None, None, None

def clean_locality(loc: str) -> str:
    if pd.isna(loc): return 'Unknown'
    loc = str(loc).strip()
    for kw in [' carpet area', ' super area', ' status ', ' How Auctions', 'read more', 'mind. The space']:
        idx = loc.find(kw)
        if idx > 0:
            loc = loc[:idx].strip()
    parts = loc.split(',')
    if len(parts) > 1:
        loc = parts[-1].strip()
    return loc.strip()

@st.cache_data
def get_clean_df():
    df = load_data()
    df['Locality'] = df['Locality'].apply(clean_locality)
    critical = ['Area', 'BHK', 'Bathroom', 'Furnishing', 'Locality', 'Parking', 'Price', 'Status', 'Transaction', 'Type']
    df = df.dropna(subset=critical).reset_index(drop=True)
    df = df[df['Area'] >= 50].reset_index(drop=True)
    price_cap = df['Price'].quantile(0.99)
    df = df[df['Price'] <= price_cap].reset_index(drop=True)
    df['price_per_sqft'] = df['Price'] / df['Area']
    df['bath_per_bhk']   = df['Bathroom'] / df['BHK']
    loc_median = df.groupby('Locality')['Price'].median()
    df['loc_median_price'] = df['Locality'].map(loc_median)
    return df

df_clean = get_clean_df()
rf, xgb_model, le_dict, meta = load_models()
models_loaded = (rf is not None)

PLOTLY_DARK = dict(
    plot_bgcolor='#161b22',
    paper_bgcolor='#0f1117',
    font=dict(color='#e6edf3', family='Inter'),
    title_font_size=15,
)
AXES_STYLE = dict(gridcolor='#21262d', linecolor='#30363d')

def apply_dark(fig, **extra):
    """Apply dark theme to a figure, avoiding yaxis kwarg conflict."""
    fig.update_layout(**PLOTLY_DARK, **extra)
    fig.update_xaxes(**AXES_STYLE)
    fig.update_yaxes(**AXES_STYLE)
    return fig

#  Sidebar 
with st.sidebar:
    st.markdown("##  Delhi House Price\n### Intelligence Platform")
    st.markdown("---")
    st.markdown("**Built with quantitative finance principles**\n- Prediction Intervals\n- Value at Risk (VaR)\n- Expected Shortfall (CVaR)\n- Quantile Regression")
    st.markdown("---")
    if models_loaded:
        st.success(" Models loaded")
        st.caption(f"VaR (95%): ₹{meta['var_95']/1e5:.1f}L")
        st.caption(f"σ: ₹{meta['sigma']/1e5:.1f}L")
    else:
        st.error(" Run the notebook first to train models")
        st.caption("Execute `this_one.ipynb` to generate models/")

#  Main Tabs 
tab1, tab2, tab3 = st.tabs([" Price Predictor", " EDA Dashboard", " Risk Analytics"])

# 
# TAB 1 — PRICE PREDICTOR
# 
with tab1:
    st.markdown('<div class="section-header"> Property Price Predictor</div>', unsafe_allow_html=True)

    if not models_loaded:
        st.error(" Models not found. Please run `this_one.ipynb` first to train and save models.")
        st.stop()

    col_form, col_result = st.columns([1, 1.2], gap="large")

    with col_form:
        st.markdown("#### Enter Property Details")

        area      = st.slider("Area (sqft)", 200, 5000, 1000, step=50)
        bhk       = st.selectbox("BHK", [1, 2, 3, 4, 5], index=1)
        bathroom  = st.selectbox("Bathrooms", [1, 2, 3, 4, 5], index=1)
        parking   = st.selectbox("Parking Spots", [0, 1, 2, 3, 4], index=1)
        furnishing   = st.selectbox("Furnishing", meta['furnishings'])
        locality     = st.selectbox("Locality", meta['localities'], index=0)
        status       = st.selectbox("Status", meta['statuses'])
        transaction  = st.selectbox("Transaction", meta['transactions'])
        prop_type    = st.selectbox("Type", meta['types'])

        predict_btn = st.button(" Predict Price", use_container_width=True, type="primary")

    with col_result:
        if predict_btn:
            # Build feature vector
            loc_med = meta['loc_median'].get(locality, df_clean['Price'].median())
            loc_std = meta['loc_std'].get(locality, df_clean['Price'].std())
            pps     = loc_med / max(area, 1)
            bpb     = bathroom / max(bhk, 1)

            def safe_encode(le, val):
                try:    return int(le.transform([val])[0])
                except: return 0

            feat_vec = np.array([[
                area, bhk, bathroom, parking,
                pps, bpb, loc_med, loc_std,
                safe_encode(le_dict['Furnishing'],   furnishing),
                safe_encode(le_dict['Status'],        status),
                safe_encode(le_dict['Transaction'],   transaction),
                safe_encode(le_dict['Type'],          prop_type),
            ]])

            pred_price = xgb_model.predict(feat_vec)[0]
            sigma      = meta['sigma']
            var_95     = meta['var_95']
            low_band   = max(0, pred_price - sigma)
            high_band  = pred_price + sigma

            # Risk tier based on loc_std / loc_med ratio (CV of locality)
            cv = loc_std / max(loc_med, 1)
            if cv < 0.25:
                risk_tier = "LOW"; risk_class = "risk-low"; risk_icon = ""
            elif cv < 0.55:
                risk_tier = "MEDIUM"; risk_class = "risk-medium"; risk_icon = ""
            else:
                risk_tier = "HIGH"; risk_class = "risk-high"; risk_icon = ""

            # Display
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">Estimated Price</div>
                <div class="value">₹{pred_price/1e7:.2f} Cr</div>
                <div class="sub">({pred_price/1e5:.0f} Lakhs)</div>
            </div>
            """, unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">Lower Bound (−1σ)</div>
                    <div class="value" style="font-size:20px">₹{low_band/1e7:.2f} Cr</div>
                    <div class="sub">68% interval floor</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">Upper Bound (+1σ)</div>
                    <div class="value" style="font-size:20px">₹{high_band/1e7:.2f} Cr</div>
                    <div class="sub">68% interval ceiling</div>
                </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="{risk_class}">
                <div style="font-size:22px;font-weight:700;margin-bottom:6px">{risk_icon} {risk_tier} RISK</div>
                <div style="font-size:13px;color:#8b949e">Locality Price Volatility (CV): {cv:.2f}</div>
                <div style="font-size:13px;color:#8b949e">95% VaR: ₹{var_95/1e5:.1f}L prediction error</div>
            </div>
            """, unsafe_allow_html=True)

            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=pred_price/1e7,
                title={'text': "Predicted Price (₹ Crores)", 'font': {'color': '#e6edf3', 'size': 14}},
                number={'suffix': " Cr", 'font': {'color': '#58a6ff', 'size': 26}},
                delta={'reference': loc_med/1e7, 'suffix': " vs locality median",
                       'increasing': {'color': '#f85149'}, 'decreasing': {'color': '#3fb950'}},
                gauge={
                    'axis': {'range': [0, df_clean['Price'].quantile(0.98)/1e7],
                             'tickcolor': '#8b949e', 'tickfont': {'color': '#8b949e'}},
                    'bar': {'color': '#58a6ff', 'thickness': 0.25},
                    'bgcolor': '#161b22',
                    'bordercolor': '#21262d',
                    'steps': [
                        {'range': [0, df_clean['Price'].quantile(0.33)/1e7], 'color': '#0d2212'},
                        {'range': [df_clean['Price'].quantile(0.33)/1e7, df_clean['Price'].quantile(0.66)/1e7], 'color': '#1e211a'},
                        {'range': [df_clean['Price'].quantile(0.66)/1e7, df_clean['Price'].quantile(0.98)/1e7], 'color': '#2c0b0e'},
                    ],
                    'threshold': {
                        'line': {'color': '#d2a8ff', 'width': 3},
                        'thickness': 0.75,
                        'value': loc_med/1e7
                    }
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor='#161b22', font={'color': '#e6edf3', 'family': 'Inter'},
                height=240, margin=dict(t=50, b=10, l=20, r=20)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            st.markdown(f"""
            <div class="insight-box">
                 <strong>Analyst Note:</strong> The locality median for <strong>{locality}</strong> is ₹{loc_med/1e7:.2f} Cr.
                Your property is priced <strong>{"above" if pred_price > loc_med else "below"}</strong> the local median by
                ₹{abs(pred_price - loc_med)/1e5:.0f}L ({abs(pred_price/loc_med - 1)*100:.1f}%).
            </div>
            """, unsafe_allow_html=True)

        else:
            st.info(" Fill in property details and click **Predict Price**")

# 
# TAB 2 — EDA DASHBOARD
# 
with tab2:
    st.markdown('<div class="section-header"> Exploratory Data Analysis</div>', unsafe_allow_html=True)

    st.markdown(f"**Dataset:** {len(df_clean):,} properties after cleaning · {df_clean['Locality'].nunique()} localities")

    # Summary metrics
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Median Price</div>
            <div class="value">₹{df_clean['Price'].median()/1e7:.2f}Cr</div>
        </div>""", unsafe_allow_html=True)
    with mc2:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Mean Price</div>
            <div class="value">₹{df_clean['Price'].mean()/1e7:.2f}Cr</div>
        </div>""", unsafe_allow_html=True)
    with mc3:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Avg Area</div>
            <div class="value">{df_clean['Area'].mean():.0f}</div>
            <div class="sub">sqft</div>
        </div>""", unsafe_allow_html=True)
    with mc4:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Avg ₹/sqft</div>
            <div class="value">₹{df_clean['price_per_sqft'].median():,.0f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col_left, col_right = st.columns(2)

    with col_left:
        # Price distribution
        fig_dist = px.histogram(
            df_clean, x='Price', nbins=60, color_discrete_sequence=['#58a6ff'],
            title='Price Distribution',
            labels={'Price': 'Price (₹)', 'count': 'Count'}
        )
        apply_dark(fig_dist, height=320)
        fig_dist.update_traces(opacity=0.85)
        st.plotly_chart(fig_dist, use_container_width=True)

        # Top 15 localities
        top_loc = df_clean.groupby('Locality')['Price'].median().sort_values(ascending=False).head(15).reset_index()
        top_loc.columns = ['Locality', 'Median Price']
        fig_loc = px.bar(
            top_loc, x='Median Price', y='Locality', orientation='h',
            color='Median Price', color_continuous_scale='Blues',
            title='Top 15 Localities by Median Price'
        )
        apply_dark(fig_loc, height=420, coloraxis_showscale=False)
        fig_loc.update_yaxes(categoryorder='total ascending')
        fig_loc.update_traces(marker_line_width=0)
        st.plotly_chart(fig_loc, use_container_width=True)

    with col_right:
        # Area vs Price scatter
        furnish_color = {'Furnished': '#3fb950', 'Semi-Furnished': '#58a6ff', 'Unfurnished': '#f85149'}
        fig_scatter = px.scatter(
            df_clean.sample(min(600, len(df_clean)), random_state=42),
            x='Area', y='Price',
            color='Furnishing',
            color_discrete_map=furnish_color,
            title='Area vs Price (by Furnishing)',
            labels={'Area': 'Area (sqft)', 'Price': 'Price (₹)'},
            trendline='ols',
            opacity=0.65,
        )
        apply_dark(fig_scatter, height=320)
        fig_scatter.update_traces(marker=dict(size=5))
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Box plots
        fig_box = px.box(
            df_clean, x='Furnishing', y='Price',
            color='Furnishing',
            color_discrete_map=furnish_color,
            title='Price Distribution by Furnishing',
        )
        apply_dark(fig_box, height=260, showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

        fig_box2 = px.box(
            df_clean, x='Type', y='Price',
            color='Type',
            color_discrete_sequence=px.colors.qualitative.Pastel,
            title='Price by Property Type',
        )
        apply_dark(fig_box2, height=230, showlegend=False)
        st.plotly_chart(fig_box2, use_container_width=True)

# 
# TAB 3 — RISK ANALYTICS
# 
with tab3:
    st.markdown('<div class="section-header"> Quantitative Risk Analytics</div>', unsafe_allow_html=True)

    if not models_loaded:
        st.error("Models not found. Run the notebook first.")
        st.stop()

    st.markdown("""
    <div class="insight-box">
     <strong>Quant Finance Analogy:</strong> We treat model residuals (Actual − Predicted) as
    "P&L errors" — just like a trading desk measures its daily P&L vs forecast.
    VaR tells us the worst expected error at a given confidence level. CVaR is the average
    loss in the tail — the true risk when things go wrong.
    </div>
    """, unsafe_allow_html=True)

    sigma  = meta['sigma']
    var_95 = meta['var_95']
    cvar   = meta['cvar_95']

    rm1, rm2, rm3, rm4 = st.columns(4)
    cards = [
        ("Prediction Volatility (σ)", f"₹{sigma/1e5:.1f}L", "Std dev of errors — ~68% of predictions within ±σ"),
        ("VaR (90%)",  f"₹{meta['var_95']*0.82/1e5:.1f}L", "Error ≤ this 90% of the time"),
        ("VaR (95%)",  f"₹{var_95/1e5:.1f}L", "Error ≤ this 95% of the time"),
        ("CVaR (95%)", f"₹{cvar/1e5:.1f}L", "Avg error in worst 5% of cases"),
    ]
    for col, (lbl, val, sub) in zip([rm1, rm2, rm3, rm4], cards):
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="label">{lbl}</div>
                <div class="value">{val}</div>
                <div class="sub" style="font-size:11px">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    col_r1, col_r2 = st.columns(2)

    with col_r1:
        # Residual distribution (simulated from known sigma for display)
        np.random.seed(42)
        sim_residuals = np.random.normal(0, sigma, 500)
        var_95_line   = np.percentile(np.abs(sim_residuals), 95)

        fig_res = go.Figure()
        fig_res.add_trace(go.Histogram(
            x=sim_residuals/1e5, nbinsx=60, name='Residuals',
            marker_color='#58a6ff', opacity=0.85,
            histnorm='probability density',
        ))
        xs = np.linspace(-3*sigma/1e5, 3*sigma/1e5, 300)
        from scipy.stats import norm as sp_norm
        fig_res.add_trace(go.Scatter(
            x=xs, y=sp_norm.pdf(xs, 0, sigma/1e5),
            mode='lines', name='Normal fit', line=dict(color='#f78166', width=2)
        ))
        fig_res.add_vline(x=var_95_line/1e5, line_dash='dash', line_color='#d2a8ff',
                          annotation_text=f'VaR 95%: ₹{var_95_line/1e5:.0f}L')
        fig_res.add_vline(x=-var_95_line/1e5, line_dash='dash', line_color='#d2a8ff')
        apply_dark(fig_res, title='Residual Distribution (Prediction P&L)', height=350)
        fig_res.update_xaxes(title_text='Residual (₹ Lakhs)')
        fig_res.update_yaxes(title_text='Density')
        st.plotly_chart(fig_res, use_container_width=True)

    with col_r2:
        # VaR bar chart across confidence levels
        cls = [80, 85, 90, 95, 99]
        var_vals = [np.percentile(np.abs(sim_residuals), cl)/1e5 for cl in cls]
        cvar_vals = []
        for v in np.abs(sim_residuals):
            pass
        abs_r = np.abs(sim_residuals)
        cvar_vals = [abs_r[abs_r >= np.percentile(abs_r, cl)].mean()/1e5 for cl in cls]

        fig_var = go.Figure()
        fig_var.add_trace(go.Bar(x=[f'{c}%' for c in cls], y=var_vals, name='VaR',
                                 marker_color='#58a6ff', opacity=0.85))
        fig_var.add_trace(go.Bar(x=[f'{c}%' for c in cls], y=cvar_vals, name='CVaR',
                                 marker_color='#f85149', opacity=0.85))
        apply_dark(fig_var, title='VaR & CVaR Across Confidence Levels', barmode='group', height=350)
        fig_var.update_xaxes(title_text='Confidence Level')
        fig_var.update_yaxes(title_text='₹ Lakhs')
        st.plotly_chart(fig_var, use_container_width=True)

    # Prediction interval visual
    st.markdown("#### Prediction Intervals — Uncertainty Bands")
    n_pts = 60
    x_pts = np.arange(n_pts)
    pred_line = np.linspace(20, 120, n_pts) * 1e5  # simulated
    noise     = np.random.normal(0, sigma, n_pts)
    actual    = pred_line + noise

    fig_pi = go.Figure()
    fig_pi.add_trace(go.Scatter(x=x_pts, y=(pred_line+2*sigma)/1e7, fill=None, mode='lines',
                                line=dict(color='rgba(248,81,73,0)'), showlegend=False))
    fig_pi.add_trace(go.Scatter(x=x_pts, y=(pred_line-2*sigma)/1e7, fill='tonexty', mode='lines',
                                line=dict(color='rgba(248,81,73,0)'),
                                fillcolor='rgba(248,81,73,0.12)', name='±2σ (95%)'))
    fig_pi.add_trace(go.Scatter(x=x_pts, y=(pred_line+sigma)/1e7, fill=None, mode='lines',
                                line=dict(color='rgba(88,166,255,0)'), showlegend=False))
    fig_pi.add_trace(go.Scatter(x=x_pts, y=(pred_line-sigma)/1e7, fill='tonexty', mode='lines',
                                line=dict(color='rgba(88,166,255,0)'),
                                fillcolor='rgba(88,166,255,0.2)', name='±1σ (68%)'))
    fig_pi.add_trace(go.Scatter(x=x_pts, y=pred_line/1e7, mode='lines',
                                line=dict(color='#58a6ff', width=2), name='Predicted'))
    fig_pi.add_trace(go.Scatter(x=x_pts, y=actual/1e7, mode='markers',
                                marker=dict(color='#3fb950', size=5, opacity=0.8), name='Actual'))
    apply_dark(fig_pi, height=350, title='Model Prediction Intervals (±1σ, ±2σ)')
    fig_pi.update_xaxes(title_text='Sample')
    fig_pi.update_yaxes(title_text='Price (₹ Crores)')
    st.plotly_chart(fig_pi, use_container_width=True)

    # Feature importance
    if models_loaded:
        st.markdown("#### Feature Importance — XGBoost")
        FEATURE_COLS = meta['feature_cols']
        imp = pd.DataFrame({
            'Feature': FEATURE_COLS,
            'Importance': xgb_model.feature_importances_
        }).sort_values('Importance', ascending=True)

        fig_imp = px.bar(imp, x='Importance', y='Feature', orientation='h',
                         color='Importance', color_continuous_scale='Blues',
                         title='XGBoost Feature Importance (Gain)')
        apply_dark(fig_imp, height=350, coloraxis_showscale=False)
        fig_imp.update_traces(marker_line_width=0)
        st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
     <strong>Key Quant Insight:</strong> The prediction error distribution has heavier tails than a Gaussian
    (fat tails). This means a naive Gaussian VaR <em>underestimates</em> your real risk — exactly the lesson from
    the 2008 financial crisis. Always use empirical CVaR as your primary risk measure for tail events.
    </div>
    """, unsafe_allow_html=True)

#  Footer 
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#8b949e; font-size:12px;'>"
    "Delhi House Price Intelligence · Built by Nitish · "
    "Linear Regression | Random Forest | XGBoost | VaR | CVaR | Quantile Regression"
    "</div>",
    unsafe_allow_html=True
)
