"""
GEEIS - Geo-Contextual Explainable Environmental Intelligence System
Main Streamlit Dashboard Application
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.data_processing import (
    prepare_data, FEATURE_COLUMNS, CLASS_LABELS,
    augment_features_with_weather
)
from modules.ml_model import (
    train_model, evaluate_model, predict_quality,
    get_shap_explanation, get_feature_importance, load_trained_model
)
from modules.api_integration import (
    fetch_weather_data, fetch_nasa_environmental_data, fetch_news_data
)
from modules.nlp_module import analyze_news_articles, generate_news_summary
from modules.knowledge_graph import (
    build_knowledge_graph, get_health_risks,
    get_graph_statistics
)


# ─── Page Configuration ───────────────────────────────────────────────────
st.set_page_config(
    page_title="GEEIS - Environmental Intelligence System",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ─── Custom CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Root variables for professional dark theme */
    :root {
        --bg-primary: #0a0f1a;
        --bg-secondary: #111827;
        --bg-card: #1a2332;
        --border-color: #2a3a4a;
        --text-primary: #e8edf5;
        --text-secondary: #94a3b8;
        --accent-blue: #3b82f6;
        --accent-green: #10b981;
        --accent-red: #ef4444;
        --accent-amber: #f59e0b;
        --accent-purple: #8b5cf6;
        --accent-cyan: #06b6d4;
    }

    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Main container */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1400px;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827 0%, #0f172a 100%);
    }

    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #e8edf5;
    }

    /* Card styling */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        margin: 8px 0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px -5px rgba(59, 130, 246, 0.15);
    }

    .metric-card h3 {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 500;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .metric-card .value {
        color: #f1f5f9;
        font-size: 1.8rem;
        font-weight: 700;
        line-height: 1.2;
    }

    .metric-card .sub-value {
        color: #64748b;
        font-size: 0.8rem;
        margin-top: 4px;
    }

    /* Status badges */
    .status-safe {
        background: linear-gradient(135deg, #065f46 0%, #047857 100%);
        color: #a7f3d0;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        font-size: 0.9rem;
    }

    .status-moderate {
        background: linear-gradient(135deg, #92400e 0%, #b45309 100%);
        color: #fde68a;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        font-size: 0.9rem;
    }

    .status-unsafe {
        background: linear-gradient(135deg, #991b1b 0%, #dc2626 100%);
        color: #fecaca;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        font-size: 0.9rem;
    }

    /* Section headers */
    .section-header {
        color: #e8edf5;
        font-size: 1.3rem;
        font-weight: 600;
        padding: 12px 0;
        border-bottom: 2px solid #3b82f6;
        margin-bottom: 16px;
    }

    /* Info panels */
    .info-panel {
        background: #1e293b;
        border-left: 4px solid #3b82f6;
        padding: 15px 20px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
        color: #cbd5e1;
    }

    /* Risk card */
    .risk-card {
        background: linear-gradient(135deg, #1e1b2e 0%, #1a1530 100%);
        border: 1px solid #3b2f5a;
        border-radius: 10px;
        padding: 15px;
        margin: 8px 0;
    }

    .risk-card .risk-title {
        color: #c4b5fd;
        font-weight: 600;
        font-size: 0.95rem;
    }

    .risk-card .risk-detail {
        color: #94a3b8;
        font-size: 0.85rem;
    }

    /* News card */
    .news-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 15px;
        margin: 8px 0;
        transition: border-color 0.2s;
    }

    .news-card:hover {
        border-color: #3b82f6;
    }

    .news-card .news-title {
        color: #e2e8f0;
        font-weight: 600;
        font-size: 0.95rem;
        margin-bottom: 6px;
    }

    .news-card .news-source {
        color: #3b82f6;
        font-size: 0.8rem;
    }

    .news-card .news-sentiment {
        font-size: 0.8rem;
        margin-top: 4px;
    }

    /* Title area */
    .main-title {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        text-align: center;
    }

    .main-title h1 {
        color: #f1f5f9;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.02em;
    }

    .main-title p {
        color: #64748b;
        font-size: 0.9rem;
        margin: 8px 0 0;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 45px;
        padding: 0 24px;
        background: #1e293b;
        border-radius: 8px 8px 0 0;
        color: #94a3b8;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background: #334155 !important;
        color: #f1f5f9 !important;
    }

    /* Weather info */
    .weather-info {
        background: linear-gradient(135deg, #1e3a5f 0%, #172554 100%);
        border: 1px solid #1e40af;
        border-radius: 12px;
        padding: 20px;
        margin: 8px 0;
    }

    .weather-info h4 {
        color: #93c5fd;
        margin: 0 0 10px;
    }

    .weather-detail {
        color: #bfdbfe;
        font-size: 0.9rem;
        margin: 4px 0;
    }
</style>
""", unsafe_allow_html=True)


# ─── Constants ────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
CSV_PATH = os.path.join(DATA_DIR, 'water_potability.csv')
PDF_PATH = os.path.join(DATA_DIR, 'Guidelines for drinking-water quality.pdf')

PREDICTION_COLORS = {
    'Safe': '#10b981',
    'Moderate': '#f59e0b',
    'Unsafe': '#ef4444'
}

PLOTLY_THEME = {
    'paper_bgcolor': 'rgba(15, 23, 42, 0.8)',
    'plot_bgcolor': 'rgba(15, 23, 42, 0.5)',
    'font': {'color': '#e2e8f0', 'family': 'Inter, system-ui, sans-serif'},
}


# ─── Helper Functions ─────────────────────────────────────────────────────

def get_prediction_badge(label):
    """Return HTML badge for prediction label."""
    class_map = {'Safe': 'status-safe', 'Moderate': 'status-moderate', 'Unsafe': 'status-unsafe'}
    css_class = class_map.get(label, 'status-moderate')
    return f'<span class="{css_class}">{label}</span>'


def create_gauge_chart(value, title, max_val=100, color='#3b82f6'):
    """Create a Plotly gauge chart."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 14, 'color': '#94a3b8'}},
        number={'font': {'size': 28, 'color': '#f1f5f9'}},
        gauge={
            'axis': {'range': [0, max_val], 'tickcolor': '#475569'},
            'bar': {'color': color},
            'bgcolor': '#1e293b',
            'borderwidth': 1,
            'bordercolor': '#334155',
            'steps': [
                {'range': [0, max_val * 0.33], 'color': 'rgba(16, 185, 129, 0.15)'},
                {'range': [max_val * 0.33, max_val * 0.66], 'color': 'rgba(245, 158, 11, 0.15)'},
                {'range': [max_val * 0.66, max_val], 'color': 'rgba(239, 68, 68, 0.15)'}
            ]
        }
    ))
    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=40, b=10),
        **PLOTLY_THEME
    )
    return fig


def create_probability_chart(probabilities, labels):
    """Create a horizontal bar chart for class probabilities."""
    colors = [PREDICTION_COLORS.get(l, '#64748b') for l in labels]
    fig = go.Figure(go.Bar(
        x=[p * 100 for p in probabilities],
        y=labels,
        orientation='h',
        marker_color=colors,
        text=[f'{p*100:.1f}%' for p in probabilities],
        textposition='auto',
        textfont={'color': '#f1f5f9', 'size': 14}
    ))
    fig.update_layout(
        height=200,
        xaxis_title='Probability (%)',
        xaxis={'range': [0, 100], 'gridcolor': '#1e293b'},
        yaxis={'gridcolor': '#1e293b'},
        margin=dict(l=10, r=10, t=10, b=40),
        **PLOTLY_THEME
    )
    return fig


def create_shap_chart(shap_values, feature_names, predicted_class):
    """Create a SHAP waterfall / bar chart for a single prediction."""
    if isinstance(shap_values, list):
        sv = shap_values[predicted_class][0]
    else:
        sv = shap_values[0, :, predicted_class] if len(shap_values.shape) == 3 else shap_values[0]

    # Sort by absolute value
    sorted_idx = np.argsort(np.abs(sv))
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_values = [sv[i] for i in sorted_idx]

    colors = ['#ef4444' if v < 0 else '#10b981' for v in sorted_values]

    fig = go.Figure(go.Bar(
        x=sorted_values,
        y=sorted_features,
        orientation='h',
        marker_color=colors,
        text=[f'{v:.4f}' for v in sorted_values],
        textposition='outside',
        textfont={'color': '#94a3b8', 'size': 11}
    ))
    fig.update_layout(
        height=350,
        xaxis_title='SHAP Value (Impact on Prediction)',
        xaxis={'gridcolor': '#1e293b', 'zerolinecolor': '#475569'},
        yaxis={'gridcolor': '#1e293b'},
        margin=dict(l=10, r=80, t=20, b=40),
        **PLOTLY_THEME
    )
    return fig


def create_feature_importance_chart(importance_df):
    """Create a feature importance chart."""
    fig = px.bar(
        importance_df,
        x='Importance',
        y='Feature',
        orientation='h',
        color='Importance',
        color_continuous_scale=['#1e3a5f', '#3b82f6', '#06b6d4']
    )
    fig.update_layout(
        height=350,
        showlegend=False,
        coloraxis_showscale=False,
        xaxis_title='Importance Score',
        margin=dict(l=10, r=10, t=10, b=40),
        **PLOTLY_THEME
    )
    return fig


def create_knowledge_graph_viz(G):
    """Create a Plotly visualization of the knowledge graph."""
    pos = nx.spring_layout(G, k=2.5, iterations=100, seed=42)

    # Node colors by type
    node_colors = {
        'pollutant': '#ef4444',
        'health_impact': '#f59e0b',
        'category': '#3b82f6',
        'parameter': '#10b981'
    }

    # Create edge traces
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#475569'),
        hoverinfo='none',
        mode='lines'
    )

    # Create node traces by type
    traces = [edge_trace]
    for node_type, color in node_colors.items():
        nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == node_type]
        if not nodes:
            continue

        node_x = [pos[n][0] for n in nodes]
        node_y = [pos[n][1] for n in nodes]
        node_text = []
        for n in nodes:
            data = G.nodes[n]
            info = f"{n}"
            if 'guideline_value' in data:
                info += f"<br>Guideline: {data['guideline_value']}"
            if 'safe_range' in data:
                info += f"<br>Safe Range: {data['safe_range']}"
            node_text.append(info)

        trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            name=node_type.replace('_', ' ').title(),
            text=[n[:15] + '...' if len(n) > 15 else n for n in nodes],
            textposition="top center",
            textfont=dict(size=8, color='#94a3b8'),
            hoverinfo='text',
            hovertext=node_text,
            marker=dict(
                size=12 if node_type in ['pollutant', 'category'] else 8,
                color=color,
                line=dict(width=1, color='#1e293b'),
                opacity=0.9
            )
        )
        traces.append(trace)

    fig = go.Figure(data=traces)
    fig.update_layout(
        height=500,
        showlegend=True,
        legend=dict(
            bgcolor='rgba(30, 41, 59, 0.8)',
            bordercolor='#475569',
            font=dict(color='#e2e8f0')
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=0, r=0, t=0, b=0),
        **PLOTLY_THEME
    )
    return fig


# ─── Initialize Session State ────────────────────────────────────────────

def init_session_state():
    """Initialize session state variables."""
    defaults = {
        'model_trained': False,
        'model': None,
        'scaler': None,
        'raw_df': None,
        'eval_results': None,
        'knowledge_graph': None,
        'prediction_result': None,
        'weather_data': None,
        'nasa_data': None,
        'news_data': None,
        'analyzed_news': None,
        'news_summary': None,
        'shap_data': None,
        'health_risks': None,
        'importance_df': None,
        'city_name': '',
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()


# ─── Model Training ──────────────────────────────────────────────────────

@st.cache_resource
def train_and_evaluate():
    """Train model and return results (cached)."""
    X_train, X_test, y_train, y_test, scaler, raw_df = prepare_data(
        CSV_PATH, MODELS_DIR
    )
    model = train_model(X_train, y_train, MODELS_DIR)
    eval_results = evaluate_model(model, X_test, y_test)
    importance_df = get_feature_importance(model)

    return model, scaler, raw_df, eval_results, importance_df


# ─── Knowledge Graph Building ────────────────────────────────────────────

@st.cache_resource
def build_kg():
    """Build knowledge graph (cached)."""
    return build_knowledge_graph(PDF_PATH)


# ─── MAIN APP ─────────────────────────────────────────────────────────────

def main():
    # Sidebar
    with st.sidebar:
        st.markdown("### GEEIS")
        st.markdown(
            '<p style="color: #64748b; font-size: 0.85rem;">'
            'Environmental Intelligence System'
            '</p>',
            unsafe_allow_html=True
        )
        st.markdown("---")

        # City input
        city_name = st.text_input(
            "Enter City Name",
            value=st.session_state.city_name,
            placeholder="e.g., Mumbai, London, New York"
        )

        analyze_btn = st.button(
            "Run Analysis",
            type="primary",
            use_container_width=True
        )

        st.markdown("---")
        st.markdown(
            '<p style="color: #475569; font-size: 0.75rem;">'
            'Integrating: ML | XAI | NLP | Knowledge Graphs'
            '</p>',
            unsafe_allow_html=True
        )

    # Title
    st.markdown(
        '<div class="main-title">'
        '<h1>Geo-Contextual Explainable Environmental Intelligence System</h1>'
        '<p>Multi-source environmental data analysis with explainable AI, '
        'NLP-powered news insights, and WHO knowledge graph integration</p>'
        '</div>',
        unsafe_allow_html=True
    )

    # ─── TRAIN MODEL ──────────────────────────────────────────────
    if not st.session_state.model_trained:
        with st.spinner("Initializing system: Training ML model and building knowledge graph..."):
            model, scaler, raw_df, eval_results, importance_df = train_and_evaluate()
            kg = build_kg()

            st.session_state.model = model
            st.session_state.scaler = scaler
            st.session_state.raw_df = raw_df
            st.session_state.eval_results = eval_results
            st.session_state.importance_df = importance_df
            st.session_state.knowledge_graph = kg
            st.session_state.model_trained = True

    # ─── ANALYSIS TRIGGERED ───────────────────────────────────────
    if analyze_btn and city_name:
        st.session_state.city_name = city_name

        try:
            # Fetch weather data
            with st.spinner(f"Fetching weather data for {city_name}..."):
                weather = fetch_weather_data(city_name)
                st.session_state.weather_data = weather

            # Fetch NASA data
            if weather.get('status') == 'success':
                coords = weather.get('coordinates', {})
                try:
                    with st.spinner("Fetching satellite/environmental data..."):
                        nasa = fetch_nasa_environmental_data(
                            coords.get('lat', 0), coords.get('lon', 0)
                        )
                        st.session_state.nasa_data = nasa
                except Exception as e:
                    st.session_state.nasa_data = {
                        'status': 'error',
                        'message': str(e),
                        'natural_events': [],
                        'atmospheric_data': {'nasa_status': 'Error', 'error': str(e)}
                    }

            # Fetch and analyze news
            try:
                with st.spinner("Analyzing environmental news..."):
                    news = fetch_news_data()
                    st.session_state.news_data = news
                    analyzed = analyze_news_articles(news)
                    st.session_state.analyzed_news = analyzed
                    summary = generate_news_summary(analyzed)
                    st.session_state.news_summary = summary
            except Exception as e:
                st.session_state.news_summary = {
                    'total_articles': 0,
                    'average_sentiment': 'N/A',
                    'average_polarity': 0,
                    'top_keywords': [],
                    'key_findings': f'Error fetching news: {str(e)}',
                    'sentiment_distribution': {'Positive': 0, 'Neutral': 0, 'Negative': 0}
                }
                st.session_state.analyzed_news = []

            # Make prediction
            with st.spinner("Running water quality prediction..."):
                # Get sample features from dataset median
                raw_df = st.session_state.raw_df
                features = {}
                for col in FEATURE_COLUMNS:
                    features[col] = float(raw_df[col].median())

                # Augment with weather
                if weather.get('status') == 'success':
                    features = augment_features_with_weather(features, weather)

                # Predict
                label, probs, pred_class = predict_quality(
                    st.session_state.model, st.session_state.scaler, features
                )
                st.session_state.prediction_result = {
                    'label': label,
                    'probabilities': probs,
                    'predicted_class': pred_class,
                    'features': features
                }

                # SHAP explanation
                try:
                    shap_values, feat_names, feat_vals, expected = get_shap_explanation(
                        st.session_state.model, st.session_state.scaler, features
                    )
                    st.session_state.shap_data = {
                        'shap_values': shap_values,
                        'feature_names': feat_names,
                        'feature_values': feat_vals,
                        'expected_value': expected
                    }
                except Exception as e:
                    st.session_state.shap_data = None
                    st.warning(f"SHAP explanation unavailable: {str(e)}")

                # Health risks from knowledge graph
                risks = get_health_risks(st.session_state.knowledge_graph, features)
                st.session_state.health_risks = risks

        except Exception as e:
            st.error(f"Analysis error: {str(e)}")

    # ─── TABS ─────────────────────────────────────────────────────
    tabs = st.tabs([
        "Prediction & Explanation",
        "Weather & Environment",
        "News Insights",
        "Knowledge Graph",
        "Model Performance"
    ])

    # ═══ TAB 1: PREDICTION & EXPLANATION ══════════════════════════
    with tabs[0]:
        if st.session_state.prediction_result:
            pred = st.session_state.prediction_result

            # Top row: prediction overview
            col1, col2, col3 = st.columns([1, 2, 1])

            with col1:
                st.markdown('<div class="section-header">Water Quality Prediction</div>',
                            unsafe_allow_html=True)
                st.markdown(
                    f'<div class="metric-card" style="text-align:center;">'
                    f'<h3>Prediction Result</h3>'
                    f'<div style="margin: 16px 0;">{get_prediction_badge(pred["label"])}</div>'
                    f'<div class="sub-value">City: {st.session_state.city_name}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

                # Probability chart
                st.plotly_chart(
                    create_probability_chart(
                        pred['probabilities'],
                        list(CLASS_LABELS.values())
                    ),
                    use_container_width=True
                )

            with col2:
                st.markdown('<div class="section-header">SHAP Explanation</div>',
                            unsafe_allow_html=True)
                if st.session_state.shap_data:
                    shap_d = st.session_state.shap_data
                    st.plotly_chart(
                        create_shap_chart(
                            shap_d['shap_values'],
                            shap_d['feature_names'],
                            pred['predicted_class']
                        ),
                        use_container_width=True
                    )
                    st.markdown(
                        '<div class="info-panel">'
                        'SHAP values show the contribution of each feature to the prediction. '
                        'Green bars push toward the predicted class, red bars push away from it.'
                        '</div>',
                        unsafe_allow_html=True
                    )

            with col3:
                st.markdown('<div class="section-header">Feature Values</div>',
                            unsafe_allow_html=True)
                features = pred['features']
                for feat_name, feat_val in features.items():
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<h3>{feat_name}</h3>'
                        f'<div class="value">{feat_val:.2f}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

            # Health risks section
            st.markdown('<div class="section-header">Health Risk Assessment (WHO Guidelines)</div>',
                        unsafe_allow_html=True)
            if st.session_state.health_risks:
                risk_cols = st.columns(min(len(st.session_state.health_risks), 4))
                for i, risk in enumerate(st.session_state.health_risks):
                    with risk_cols[i % len(risk_cols)]:
                        issues = ", ".join(risk.get('potential_issues', []))
                        st.markdown(
                            f'<div class="risk-card">'
                            f'<div class="risk-title">{risk["parameter"]}</div>'
                            f'<div class="risk-detail">Value: {risk["value"]:.2f}</div>'
                            f'<div class="risk-detail">Safe Range: {risk["safe_range"]}</div>'
                            f'<div class="risk-detail">Status: {risk["status"]}</div>'
                            f'<div class="risk-detail">Issues: {issues}</div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
            else:
                st.markdown(
                    '<div class="info-panel">All parameters within WHO guideline ranges.</div>',
                    unsafe_allow_html=True
                )

        else:
            st.markdown(
                '<div class="info-panel">'
                'Enter a city name in the sidebar and click "Run Analysis" '
                'to generate predictions with explainable AI.'
                '</div>',
                unsafe_allow_html=True
            )

    # ═══ TAB 2: WEATHER & ENVIRONMENT ═════════════════════════════
    with tabs[1]:
        if st.session_state.weather_data and st.session_state.weather_data.get('status') == 'success':
            weather = st.session_state.weather_data

            st.markdown('<div class="section-header">Weather Context</div>',
                        unsafe_allow_html=True)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<h3>Temperature</h3>'
                    f'<div class="value">{weather["temperature"]} C</div>'
                    f'<div class="sub-value">Feels like {weather["feels_like"]} C</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            with col2:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<h3>Humidity</h3>'
                    f'<div class="value">{weather["humidity"]}%</div>'
                    f'<div class="sub-value">{weather["description"].title()}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            with col3:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<h3>Wind Speed</h3>'
                    f'<div class="value">{weather["wind_speed"]} m/s</div>'
                    f'<div class="sub-value">Pressure: {weather["pressure"]} hPa</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            with col4:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<h3>Rainfall (1h)</h3>'
                    f'<div class="value">{weather["rainfall"]} mm</div>'
                    f'<div class="sub-value">Cloud Cover: {weather["clouds"]}%</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            st.markdown(
                f'<div class="weather-info">'
                f'<h4>{weather["city"]}, {weather["country"]}</h4>'
                f'<div class="weather-detail">'
                f'Coordinates: {weather["coordinates"]["lat"]}, {weather["coordinates"]["lon"]}'
                f'</div>'
                f'<div class="weather-detail">Recorded: {weather["timestamp"]}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

            # Environmental impact assessment
            st.markdown('<div class="section-header">Environmental Impact on Water Quality</div>',
                        unsafe_allow_html=True)

            impacts = []
            if weather['temperature'] > 30:
                impacts.append("High temperatures may increase bacterial growth and chemical reaction rates in water sources.")
            if weather['humidity'] > 80:
                impacts.append("High humidity levels can increase contamination pathways through surface runoff.")
            if weather['rainfall'] > 5:
                impacts.append("Significant rainfall may increase turbidity and introduce surface contaminants into water sources.")
            if weather['wind_speed'] > 10:
                impacts.append("High wind speeds may contribute to sediment suspension in open water sources.")

            if impacts:
                for imp in impacts:
                    st.markdown(f'<div class="info-panel">{imp}</div>', unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="info-panel">Current weather conditions present minimal environmental impact on water quality.</div>',
                    unsafe_allow_html=True
                )

            # NASA data
            if st.session_state.nasa_data:
                nasa = st.session_state.nasa_data
                st.markdown('<div class="section-header">Satellite Context (NASA)</div>',
                            unsafe_allow_html=True)

                col1, col2 = st.columns(2)
                with col1:
                    atm = nasa.get('atmospheric_data', {})
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<h3>NASA API Status</h3>'
                        f'<div class="value" style="font-size:1.2rem;">{atm.get("nasa_status", "N/A")}</div>'
                        f'<div class="sub-value">Satellite Coverage: {atm.get("satellite_coverage", "N/A")}</div>'
                        f'<div class="sub-value">Data Quality: {atm.get("data_quality", "N/A")}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                with col2:
                    events = nasa.get('natural_events', [])
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<h3>Active Natural Events</h3>'
                        f'<div class="value">{len(events)}</div>'
                        f'<div class="sub-value">Tracked by NASA EONET</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                if events:
                    st.markdown("**Recent Natural Events (EONET)**")
                    for ev in events[:5]:
                        st.markdown(
                            f'<div class="info-panel">'
                            f'<strong>{ev["title"]}</strong><br>'
                            f'Category: {ev["category"]} | Date: {ev.get("date", "N/A")[:10]}'
                            f'</div>',
                            unsafe_allow_html=True
                        )

        elif st.session_state.weather_data and st.session_state.weather_data.get('status') == 'error':
            st.error(st.session_state.weather_data.get('message', 'Failed to fetch weather data'))
        else:
            st.markdown(
                '<div class="info-panel">'
                'Run analysis for a city to view weather and environmental context.'
                '</div>',
                unsafe_allow_html=True
            )

    # ═══ TAB 3: NEWS INSIGHTS ═════════════════════════════════════
    with tabs[2]:
        if st.session_state.news_summary and st.session_state.analyzed_news:
            summary = st.session_state.news_summary

            st.markdown('<div class="section-header">News Intelligence Summary</div>',
                        unsafe_allow_html=True)

            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<h3>Articles Analyzed</h3>'
                    f'<div class="value">{summary["total_articles"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            with col2:
                sentiment_color = {'Positive': '#10b981', 'Negative': '#ef4444', 'Neutral': '#f59e0b'}
                sc = sentiment_color.get(summary['average_sentiment'], '#94a3b8')
                st.markdown(
                    f'<div class="metric-card">'
                    f'<h3>Average Sentiment</h3>'
                    f'<div class="value" style="color: {sc};">{summary["average_sentiment"]}</div>'
                    f'<div class="sub-value">Polarity: {summary.get("average_polarity", 0)}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            with col3:
                dist = summary.get('sentiment_distribution', {})
                st.markdown(
                    f'<div class="metric-card">'
                    f'<h3>Sentiment Distribution</h3>'
                    f'<div class="sub-value" style="color:#10b981;">Positive: {dist.get("Positive", 0)}</div>'
                    f'<div class="sub-value" style="color:#f59e0b;">Neutral: {dist.get("Neutral", 0)}</div>'
                    f'<div class="sub-value" style="color:#ef4444;">Negative: {dist.get("Negative", 0)}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            with col4:
                top_kw = summary.get('top_keywords', [])
                kw_text = ", ".join([k['keyword'] for k in top_kw[:5]]) if top_kw else 'N/A'
                st.markdown(
                    f'<div class="metric-card">'
                    f'<h3>Top Keywords</h3>'
                    f'<div class="sub-value">{kw_text}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            # Key findings
            st.markdown(
                f'<div class="info-panel">{summary.get("key_findings", "")}</div>',
                unsafe_allow_html=True
            )

            # Sentiment chart
            if summary.get('sentiment_distribution'):
                dist = summary['sentiment_distribution']
                fig = go.Figure(data=[go.Pie(
                    labels=list(dist.keys()),
                    values=list(dist.values()),
                    marker_colors=['#10b981', '#f59e0b', '#ef4444'],
                    textinfo='label+percent',
                    textfont=dict(color='#f1f5f9', size=13),
                    hole=0.4
                )])
                fig.update_layout(
                    height=300,
                    margin=dict(l=20, r=20, t=20, b=20),
                    **PLOTLY_THEME
                )
                st.plotly_chart(fig, use_container_width=True)

            # Individual articles
            st.markdown('<div class="section-header">Article Details</div>',
                        unsafe_allow_html=True)
            for article in st.session_state.analyzed_news[:8]:
                sent = article.get('sentiment', {})
                sent_color = sentiment_color.get(sent.get('label', 'Neutral'), '#94a3b8')
                kws = ", ".join([k['keyword'] for k in article.get('keywords', [])[:3]])
                st.markdown(
                    f'<div class="news-card">'
                    f'<div class="news-title">{article.get("title", "N/A")}</div>'
                    f'<div class="news-source">Source: {article.get("source", "N/A")} | '
                    f'{article.get("published_at", "")[:10]}</div>'
                    f'<div class="news-sentiment" style="color: {sent_color};">'
                    f'Sentiment: {sent.get("label", "N/A")} ({sent.get("polarity", 0)})</div>'
                    f'<div class="risk-detail">Keywords: {kws}</div>'
                    f'<div class="risk-detail">Relevance: {article.get("relevance_score", 0)}%</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
        else:
            st.markdown(
                '<div class="info-panel">'
                'Run analysis to view NLP-powered news insights about water quality.'
                '</div>',
                unsafe_allow_html=True
            )

    # ═══ TAB 4: KNOWLEDGE GRAPH ═══════════════════════════════════
    with tabs[3]:
        if st.session_state.knowledge_graph:
            G = st.session_state.knowledge_graph
            stats = get_graph_statistics(G)

            st.markdown('<div class="section-header">WHO Drinking Water Quality Knowledge Graph</div>',
                        unsafe_allow_html=True)

            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<h3>Total Nodes</h3>'
                    f'<div class="value">{stats["total_nodes"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            with col2:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<h3>Total Edges</h3>'
                    f'<div class="value">{stats["total_edges"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            with col3:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<h3>Pollutants</h3>'
                    f'<div class="value">{stats["pollutants"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            with col4:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<h3>Health Impacts</h3>'
                    f'<div class="value">{stats["health_impacts"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            # Graph visualization
            st.plotly_chart(create_knowledge_graph_viz(G), use_container_width=True)

            # Pollutant details
            st.markdown('<div class="section-header">Pollutant-Health Impact Relationships</div>',
                        unsafe_allow_html=True)

            from modules.knowledge_graph import WHO_KNOWLEDGE_BASE
            pollutants = WHO_KNOWLEDGE_BASE['pollutants']

            for pollutant, info in pollutants.items():
                impacts = ", ".join(info['health_impacts'])
                st.markdown(
                    f'<div class="risk-card">'
                    f'<div class="risk-title">{pollutant} ({info["category"]})</div>'
                    f'<div class="risk-detail">Guideline: {info["guideline_value"]}</div>'
                    f'<div class="risk-detail">Health Impacts: {impacts}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
        else:
            st.markdown(
                '<div class="info-panel">'
                'Knowledge graph is being built from WHO guidelines...'
                '</div>',
                unsafe_allow_html=True
            )

    # ═══ TAB 5: MODEL PERFORMANCE ═════════════════════════════════
    with tabs[4]:
        if st.session_state.eval_results:
            results = st.session_state.eval_results

            st.markdown('<div class="section-header">XGBoost Model Performance</div>',
                        unsafe_allow_html=True)

            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            metrics = [
                ('Accuracy', results['accuracy'], '#3b82f6'),
                ('Precision', results['precision'], '#10b981'),
                ('Recall', results['recall'], '#f59e0b'),
                ('F1 Score', results['f1_score'], '#8b5cf6')
            ]

            for col, (name, val, color) in zip([col1, col2, col3, col4], metrics):
                with col:
                    st.plotly_chart(
                        create_gauge_chart(val * 100, name, 100, color),
                        use_container_width=True
                    )

            # Feature importance
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="section-header">Feature Importance</div>',
                            unsafe_allow_html=True)
                if st.session_state.importance_df is not None:
                    st.plotly_chart(
                        create_feature_importance_chart(st.session_state.importance_df),
                        use_container_width=True
                    )

            with col2:
                st.markdown('<div class="section-header">Confusion Matrix</div>',
                            unsafe_allow_html=True)
                cm = results['confusion_matrix']
                fig = px.imshow(
                    cm,
                    labels=dict(x='Predicted', y='Actual', color='Count'),
                    x=list(CLASS_LABELS.values()),
                    y=list(CLASS_LABELS.values()),
                    color_continuous_scale=['#0f172a', '#1e3a5f', '#3b82f6', '#06b6d4'],
                    text_auto=True
                )
                fig.update_layout(
                    height=350,
                    margin=dict(l=10, r=10, t=10, b=40),
                    **PLOTLY_THEME
                )
                fig.update_traces(textfont={'size': 16, 'color': '#f1f5f9'})
                st.plotly_chart(fig, use_container_width=True)

            # Classification report
            st.markdown('<div class="section-header">Classification Report</div>',
                        unsafe_allow_html=True)
            st.code(results['classification_report'], language='text')

            # Dataset overview
            if st.session_state.raw_df is not None:
                st.markdown('<div class="section-header">Dataset Overview</div>',
                            unsafe_allow_html=True)
                raw_df = st.session_state.raw_df

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<h3>Dataset Size</h3>'
                        f'<div class="value">{len(raw_df)} samples</div>'
                        f'<div class="sub-value">{len(FEATURE_COLUMNS)} features</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                with col2:
                    class_dist = raw_df['Potability'].value_counts()
                    fig = go.Figure(data=[go.Pie(
                        labels=['Not Potable', 'Potable'],
                        values=[class_dist.get(0, 0), class_dist.get(1, 0)],
                        marker_colors=['#ef4444', '#10b981'],
                        textinfo='label+percent',
                        textfont=dict(color='#f1f5f9', size=13),
                        hole=0.4
                    )])
                    fig.update_layout(
                        height=250,
                        title=dict(text='Class Distribution', font=dict(color='#94a3b8', size=14)),
                        margin=dict(l=10, r=10, t=40, b=10),
                        **PLOTLY_THEME
                    )
                    st.plotly_chart(fig, use_container_width=True)

        else:
            st.markdown(
                '<div class="info-panel">Model is being trained...</div>',
                unsafe_allow_html=True
            )


if __name__ == "__main__":
    main()
