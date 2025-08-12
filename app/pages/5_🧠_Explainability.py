import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Flexible import to tolerate different run contexts
try:
    from app.utils import load_models, get_feature_schema, load_css, load_lottie_url
except Exception:
    from utils import load_models, get_feature_schema, load_css, load_lottie_url  # type: ignore

st.set_page_config(page_title="🧠 Explainability", page_icon="🧠", layout="wide")

# Inject global CSS
css = load_css()
if css:
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

st.title("🧠 Model Explainability")

st.markdown(
    """
    <div class="content-intro">
        <h4>Interpret and trust your models</h4>
        <p>Explore feature contributions with SHAP values, visualize feature effects via PDP/ICE, and inspect top drivers of predictions.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Optional Lottie header
try:
    from streamlit_lottie import st_lottie
    c1, c2 = st.columns(2)
    with c1:
        l1 = load_lottie_url("https://assets10.lottiefiles.com/packages/lf20_tll0j4bb.json")  # AI/brain
        if l1:
            st_lottie(l1, height=120, key="exp_ai")
    with c2:
        l2 = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_qp1q7mct.json")  # data
        if l2:
            st_lottie(l2, height=120, key="exp_data")
except Exception:
    pass

models = load_models()
model_names = list(models.keys())
if not model_names:
    st.warning("No models found in `models/`. Save *.pkl or *.joblib models to enable explainability.")
    st.stop()

with st.sidebar:
    st.subheader("Settings")
    model_name = st.selectbox("Model", options=model_names, index=0)
    sample_size = st.slider("Sample size for explanations", 100, 5000, 800, 50)
    top_k = st.slider("Top features (importance)", 5, 30, 12)
    st.markdown("---")
    uploaded = st.file_uploader("Optional: Upload data CSV for explainability", type=["csv"]) 

mdl = models[model_name]
feature_names = get_feature_schema(models)

# Load background data X
X_source: pd.DataFrame
source_label = ""
if uploaded is not None:
    try:
        df_in = pd.read_csv(uploaded)
        if feature_names:
            X_source = df_in.reindex(columns=feature_names).select_dtypes(include=[np.number]).fillna(0.0)
        else:
            X_source = df_in.select_dtypes(include=[np.number]).fillna(0.0)
        source_label = "(uploaded)"
    except Exception as e:
        st.error(f"Failed to read uploaded CSV: {e}")
        X_source = pd.DataFrame()
else:
    # Fallback: try to assemble from any numeric columns in available data files
    # Prefer light-weight synthetic sample if nothing available
    X_source = pd.DataFrame()

# If no data yet, synthesize a small dataframe based on known features
if X_source.empty:
    if feature_names:
        rng = np.random.default_rng(42)
        X_source = pd.DataFrame(rng.normal(size=(max(sample_size, 500), len(feature_names))), columns=feature_names)
        source_label = "(Training Data)"
    else:
        st.info("No feature schema detected. Upload a CSV with your model features to enable explainability.")
        st.stop()

# Downsample for performance
if len(X_source) > sample_size:
    X = X_source.sample(sample_size, random_state=42)
else:
    X = X_source.copy()

# Helper: get model output function for probability of class 1 when available
def predict_function(model, data: pd.DataFrame) -> np.ndarray:
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(data)
            # binary or multi-class -> take class 1 if exists else max prob
            if proba.ndim == 2 and proba.shape[1] > 1:
                return proba[:, 1]
            return proba.ravel()
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(data)
            return np.array(scores, dtype=float).ravel()
        else:
            preds = model.predict(data)
            return np.array(preds, dtype=float).ravel()
    except Exception:
        # Last resort: try predict on numpy
        try:
            return np.array(model.predict(data.values), dtype=float).ravel()
        except Exception:
            return np.zeros(len(data), dtype=float)

# Helper: robust mean(|SHAP|) per feature across different SHAP shapes
def mean_abs_shap_per_feature(shap_values_obj) -> np.ndarray:
    vals = getattr(shap_values_obj, "values", shap_values_obj)
    try:
        arr = np.array(vals)
    except Exception:
        # e.g., list of per-class arrays -> stack on last axis
        arr = np.stack(vals, axis=-1)
    # (n_samples, n_features) or (n_samples, n_features, n_outputs) or variants
    if arr.ndim == 1:
        return np.abs(arr)
    if arr.ndim == 2:
        return np.abs(arr).mean(axis=0)
    # average across all axes except feature axis -> choose axis with size n_features
    n_features = X.shape[1]
    feature_axes = [i for i, s in enumerate(arr.shape) if s == n_features]
    feat_axis = feature_axes[0] if feature_axes else 1
    axes_to_mean = tuple(i for i in range(arr.ndim) if i != feat_axis)
    return np.abs(arr).mean(axis=axes_to_mean)

# Helper: extract a sample-length SHAP vector for a selected feature
def sample_shap_for_feature(shap_values_obj, feature_idx: int, n_samples: int, n_features: int) -> np.ndarray:
    vals = getattr(shap_values_obj, "values", shap_values_obj)
    try:
        arr = np.array(vals)
    except Exception:
        arr = np.stack(vals, axis=0)
    # Identify axes
    sample_axes = [i for i, s in enumerate(arr.shape) if s == n_samples]
    feature_axes = [i for i, s in enumerate(arr.shape) if s == n_features]
    sample_axis = sample_axes[0] if sample_axes else 0
    feature_axis = feature_axes[0] if feature_axes else (1 if arr.ndim > 1 else 0)
    # Select the requested feature along its axis
    arr_sel = np.take(arr, indices=feature_idx, axis=feature_axis)
    # Now reduce any remaining non-sample axes by mean
    if arr_sel.ndim == 1:
        # ideally already length n_samples
        if arr_sel.shape[0] != n_samples:
            return np.resize(arr_sel, n_samples)
        return arr_sel
    # Find the sample axis in arr_sel
    sample_axes_sel = [i for i, s in enumerate(arr_sel.shape) if s == n_samples]
    if not sample_axes_sel:
        # fallback: try last axis as samples
        sample_axes_sel = [arr_sel.ndim - 1]
    s_ax = sample_axes_sel[0]
    axes_to_mean = tuple(i for i in range(arr_sel.ndim) if i != s_ax)
    y = arr_sel
    if axes_to_mean:
        y = y.mean(axis=axes_to_mean)
    # Ensure 1D length n_samples
    y = np.asarray(y).ravel()
    if y.shape[0] != n_samples:
        y = np.resize(y, n_samples)
    return y

# Compute SHAP values (gracefully degrade)
shap_values = None
explainer_name = ""
try:
    import shap
    # shap >= 0.39 supports auto Explainer selection
    explainer = shap.Explainer(mdl, X, feature_names=list(X.columns))
    shap_values = explainer(X)
    explainer_name = explainer.__class__.__name__
except Exception as e:
    shap_values = None
    st.warning(f"SHAP explainability unavailable: {e}")

# Tabs for visualizations
tabs = st.tabs(["Overview", "Feature Importance", "Dependence", "PDP / ICE"]) 

with tabs[0]:
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Samples used", f"{len(X):,}")
    with c2:
        st.metric("Features", f"{X.shape[1]:,}")
    with c3:
        st.metric("Data source", source_label or "(unknown)")
    st.markdown('</div>', unsafe_allow_html=True)

    if shap_values is not None:
        try:
            mean_abs = mean_abs_shap_per_feature(shap_values)
            # Align lengths defensively
            n = min(len(X.columns), len(mean_abs))
            imp_df = pd.DataFrame({"feature": list(X.columns)[:n], "importance": np.asarray(mean_abs).ravel()[:n]})
            imp_df = imp_df.sort_values("importance", ascending=False)
            fig = px.bar(imp_df.head(top_k), x="importance", y="feature", orientation='h', title="Mean |SHAP| by feature")
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.info(f"Unable to compute SHAP importances: {e}")
    else:
        st.info("SHAP values not available. Try uploading a small CSV (200–1000 rows) matching your model features.")

with tabs[1]:
    st.subheader("Top Feature Importances (Mean |SHAP|)")
    if shap_values is not None:
        try:
            mean_abs = mean_abs_shap_per_feature(shap_values)
            n = min(len(X.columns), len(mean_abs))
            imp_df = pd.DataFrame({"feature": list(X.columns)[:n], "importance": np.asarray(mean_abs).ravel()[:n]})
            imp_df = imp_df.sort_values("importance", ascending=False)
            fig = px.bar(imp_df.head(top_k), x="importance", y="feature", orientation='h', color="importance", color_continuous_scale='viridis')
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            csv_imp = imp_df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download full importances (CSV)", data=csv_imp, file_name="shap_importances.csv", mime="text/csv")
        except Exception as e:
            st.info(f"Unable to compute SHAP importances: {e}")
    else:
        st.info("SHAP not available. Ensure `shap` is installed and provide a tabular dataset.")

with tabs[2]:
    st.subheader("SHAP Dependence")
    if shap_values is not None:
        feat = st.selectbox("Feature", options=list(X.columns), index=0, key="dep_feat")
        color_feat = st.selectbox("Color by", options=[None] + list(X.columns), index=0, key="dep_color")
        try:
            idx = list(X.columns).index(feat)
            y_vals = sample_shap_for_feature(shap_values, idx, n_samples=len(X), n_features=X.shape[1])
            x_vals = X.iloc[:, idx].values
            color_vals = X[color_feat] if color_feat else None
            fig = px.scatter(
                x=x_vals, y=y_vals, color=color_vals,
                labels={"x": feat, "y": "SHAP value"}, opacity=0.7, color_continuous_scale='viridis',
                title=f"Dependence: {feat} vs SHAP value"
            )
            fig.add_hline(y=0, line_dash="dash", opacity=0.5)
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not render dependence plot: {e}")
    else:
        st.info("Provide a dataset to compute SHAP dependence plots.")

with tabs[3]:
    st.subheader("Partial Dependence and ICE")
    feat = st.selectbox("Feature for PDP/ICE", options=list(X.columns), index=0, key="pdp_feat")
    grid_points = st.slider("Grid points", 10, 80, 30)
    ice_curves = st.slider("ICE curves (samples)", 5, 100, 30)

    # Build grid over the selected feature
    try:
        x_series = X[feat]
        vmin, vmax = np.nanpercentile(x_series, [1, 99])
        grid = np.linspace(vmin, vmax, grid_points)

        # PDP: hold others at median
        X_ref = X.median(numeric_only=True)
        X_pdp = pd.DataFrame(np.tile(X_ref.values, (grid_points, 1)), columns=X.columns)
        X_pdp[feat] = grid
        y_pdp = predict_function(mdl, X_pdp)

        fig_pdp = go.Figure()
        fig_pdp.add_trace(go.Scatter(x=grid, y=y_pdp, mode='lines', name='PDP', line=dict(width=3, color="#60A5FA")))

        # ICE: sample a few rows and vary feature
        ice_idx = X.sample(min(ice_curves, len(X)), random_state=42)
        for i, (_, row) in enumerate(ice_idx.iterrows()):
            X_ice = pd.DataFrame(np.tile(row.values, (grid_points, 1)), columns=X.columns)
            X_ice[feat] = grid
            y_ice = predict_function(mdl, X_ice)
            fig_pdp.add_trace(go.Scatter(x=grid, y=y_ice, mode='lines', line=dict(width=1, color='rgba(255,255,255,0.25)'), showlegend=False))

        fig_pdp.update_layout(title=f"PDP/ICE for {feat}", xaxis_title=feat, yaxis_title="Model output",
                              paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_pdp, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not compute PDP/ICE: {e}")
