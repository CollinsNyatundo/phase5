import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np

# Flexible import to support running from repo root or from within app/
try:
    from app.utils import (
        load_training_samples,
        load_csv_sample,
        load_models,
        load_metrics,
        load_css,
        inject_css_block,
        load_lottie_url,
    )
except Exception:
    from utils import (  # type: ignore
        load_training_samples,
        load_csv_sample,
        load_models,
        load_metrics,
        load_css,
        inject_css_block,
        load_lottie_url,
    )

st.set_page_config(
    page_title="🌾 Cropland Mapping ML Suite",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Base CSS
css = load_css()
if css:
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# Additional advanced CSS blocks
glassmorphism_css = """
.glass-container { background: rgba(255,255,255,0.1); backdrop-filter: blur(10px); border-radius: 20px; border: 1px solid rgba(255,255,255,0.2); padding: 30px; margin: 20px 0; box-shadow: 0 8px 32px rgba(31,38,135,.37); }
.glass-card { background: rgba(255,255,255,0.08); backdrop-filter: blur(8px); border-radius: 15px; border: 1px solid rgba(255,255,255,0.15); padding: 20px; margin: 10px; box-shadow: 0 4px 16px rgba(31, 38, 135, 0.2); transition: all .3s ease; }
.glass-card:hover { transform: translateY(-5px); box-shadow: 0 8px 32px rgba(31,38,135,.4); }
"""
animated_css = """
@keyframes float { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-10px)} }
@keyframes pulse { 0%,100%{transform:scale(1)} 50%{transform:scale(1.05)} }
.floating-card { background:#fff; border-radius:20px; padding:2rem; box-shadow:0 15px 35px rgba(0,0,0,.1); animation: float 3s ease-in-out infinite; border:1px solid #e0e0e0; transition: all .3s ease; }
.floating-card:hover { animation: pulse .5s ease-in-out; box-shadow:0 20px 40px rgba(0,0,0,.15) }
"""
button_css = """
.custom-button{background:linear-gradient(45deg,#FE6B8B 30%,#FF8E53 90%);border:none;border-radius:25px;color:#fff;padding:12px 30px;font-size:16px;font-weight:600;cursor:pointer;transition:all .3s ease;box-shadow:0 3px 15px rgba(0,0,0,.2);margin:10px 5px}
.custom-button:hover{transform:translateY(-2px);box-shadow:0 6px 25px rgba(0,0,0,.3)}
"""
inject_css_block(glassmorphism_css + animated_css + button_css)

carousel_css = """
.hs-scroll { overflow-x: hidden; position: relative; padding-bottom: 0.25rem; }
.hs-track { display: flex; gap: 16px; align-items: stretch; will-change: transform; }
.hs-card { min-width: 260px; flex: 0 0 auto; }
/* Autoscroll (pause on hover) */
.auto-scroll { animation: scrollx var(--scroll-duration, 28s) linear infinite; }
.hs-scroll:hover .auto-scroll { animation-play-state: paused; }
@keyframes scrollx { from { transform: translateX(0); } to { transform: translateX(-50%); } }
/* Subtle scrollbar styling if ever shown */
.hs-scroll::-webkit-scrollbar { height: 8px; }
.hs-scroll::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); border-radius: 8px; }
"""
inject_css_block(carousel_css)

st.markdown("""
<div class="main-header">
  <h1>🌾 Cropland Mapping ML Suite</h1>
  <p>Advanced Machine Learning for Agricultural Land Classification</p>
</div>
""", unsafe_allow_html=True)

# Quick stats and shortcuts
train_df, train_gdf = load_training_samples()
s1_sample = load_csv_sample("Sentinel1.csv", nrows=50_000)
s2_sample = load_csv_sample("Sentinel2.csv", nrows=50_000)
models = load_models()
metrics = load_metrics()

st.markdown('<div class="glass-container">', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="glass-card">
        <h4>📊 Training Samples</h4>
        <h2>{len(train_df):,}</h2>
        <p style="margin:0; opacity:0.8; font-size:0.9rem;">Ready for analysis</p>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="glass-card">
        <h4>🛰️ Sentinel-1 Rows</h4>
        <h2>{len(s1_sample):,}</h2>
        <p style="margin:0; opacity:0.8; font-size:0.9rem;">SAR data points</p>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="glass-card">
        <h4>🌍 Sentinel-2 Rows</h4>
        <h2>{len(s2_sample):,}</h2>
        <p style="margin:0; opacity:0.8; font-size:0.9rem;">Optical imagery</p>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class="glass-card">
        <h4>🤖 ML Models</h4>
        <h2>{len(models)}</h2>
        <p style="margin:0; opacity:0.8; font-size:0.9rem;">Trained & ready</p>
    </div>
    """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Optional Lottie section
try:
    from streamlit_lottie import st_lottie
    colA, colB, colC = st.columns(3)
    with colA:
        lottie = load_lottie_url("https://assets5.lottiefiles.com/packages/lf20_1a8dx7zj.json")
        if lottie: st_lottie(lottie, height=140, key="farm")
    with colB:
        st.markdown(
            '<div class="metric-card" style="height:140px; display:flex; flex-direction:column; align-items:center; justify-content:center; padding:0.5rem; margin:0; text-align:center;">\
            <div style="font-size:0.9rem; margin:0 0 2px 0; opacity:0.9;">Best Accuracy</div>\
            <div style="font-size:1.05rem; font-weight:600; margin:0 0 2px 0;">Random Forest</div>\
            <div style="font-size:1.6rem; font-weight:800; margin:0;">74.1%</div>\
            </div>',
            unsafe_allow_html=True,
        )
    with colC:
        lottie2 = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_qp1q7mct.json")
        if lottie2: st_lottie(lottie2, height=140, key="data")
except Exception:
    pass

st.markdown("""
<div class="glass-container">
<h3 class="gradient-text" style="text-align: center; margin-bottom: 1rem;">🚀 Explore the Suite</h3>
<div class="hs-scroll">
  <div class="hs-track auto-scroll" style="--scroll-duration: 26s;">
    <!-- Sequence A -->
    <div class="glass-card hs-card">
      <h4>📊 Data Explorer</h4>
      <p>Dive deep into your agricultural datasets with interactive visualizations and statistical insights.</p>
    </div>
    <div class="glass-card hs-card">
      <h4>🤖 Model Performance</h4>
      <p>Compare ML models with comprehensive metrics, charts, and performance analytics.</p>
    </div>
    <div class="glass-card hs-card">
      <h4>🗺️ Interactive Maps</h4>
      <p>Visualize cropland data on interactive maps with clustering and geospatial analysis.</p>
    </div>
    <div class="glass-card hs-card">
      <h4>🔮 Predictions</h4>
      <p>Make real-time predictions using trained models with confidence scoring.</p>
    </div>
    <div class="glass-card hs-card">
      <h4>🧠 Explainability</h4>
      <p>Understand model decisions with SHAP, PDP/ICE, and feature attributions.</p>
    </div>
    <!-- Sequence B (duplicate for seamless loop) -->
    <div class="glass-card hs-card">
      <h4>📊 Data Explorer</h4>
      <p>Dive deep into your agricultural datasets with interactive visualizations and statistical insights.</p>
    </div>
    <div class="glass-card hs-card">
      <h4>🤖 Model Performance</h4>
      <p>Compare ML models with comprehensive metrics, charts, and performance analytics.</p>
    </div>
    <div class="glass-card hs-card">
      <h4>🗺️ Interactive Maps</h4>
      <p>Visualize cropland data on interactive maps with clustering and geospatial analysis.</p>
    </div>
    <div class="glass-card hs-card">
      <h4>🔮 Predictions</h4>
      <p>Make real-time predictions using trained models with confidence scoring.</p>
    </div>
    <div class="glass-card hs-card">
      <h4>🧠 Explainability</h4>
      <p>Understand model decisions with SHAP, PDP/ICE, and feature attributions.</p>
    </div>
  </div>
</div>
<p style="text-align: center; margin-top: 1rem; opacity: 0.8;">Use the sidebar navigation to explore each section →</p>
</div>
""", unsafe_allow_html=True)

st.info("Under development")
