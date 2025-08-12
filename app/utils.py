import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import geopandas as gpd

try:
    import streamlit as st
except Exception:  # Allow non-streamlit contexts
    class _Dummy:
        def cache_data(self, func=None, **kwargs):
            def wrapper(f):
                return f
            return wrapper
        def cache_resource(self, func=None, **kwargs):
            def wrapper(f):
                return f
            return wrapper
        def session_state(self):
            return {}
        def markdown(self, *args, **kwargs):
            pass
    st = _Dummy()  # type: ignore


ROOT_DIR = Path.cwd()
TRAIN_DIR = ROOT_DIR / "Train"
CACHE_DIR = ROOT_DIR / "cache"
MODELS_DIR = ROOT_DIR / "models"
ASSETS_DIR = ROOT_DIR / "app" / "assets"

# Remote dataset configuration (Hugging Face Hub)
# You can override via environment variables if needed
USE_HF_DATA = os.getenv("USE_HF_DATA", "1") != "0"
HF_DATA_REPO_ID = os.getenv("HF_DATA_REPO", "cnyagaka/satellite-data")
HF_DATA_SUBDIR = os.getenv("HF_DATA_SUBDIR", "data")  # files live under this prefix in the repo

CACHE_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
ASSETS_DIR.mkdir(exist_ok=True, parents=True)


def _try_hf_download(relative_path: str) -> Optional[Path]:
    """Download a file from the configured HF dataset repo and return the cached local path.

    relative_path is relative to the dataset's data subdirectory (e.g., 'Sentinel1.csv' or 'Train/xxx.shp').
    """
    if not USE_HF_DATA:
        return None
    try:
        # Import locally to avoid hard dependency if unused
        from huggingface_hub import hf_hub_download  # type: ignore

        path_in_repo = f"{HF_DATA_SUBDIR}/{relative_path}".replace("\\", "/")
        downloaded = hf_hub_download(
            repo_id=HF_DATA_REPO_ID,
            repo_type="dataset",
            filename=path_in_repo,
        )
        return Path(downloaded)
    except Exception:
        return None


def _find_file(filename: str) -> Optional[Path]:
    """Resolve a file by checking local paths first, then Hugging Face dataset if enabled.

    The filename can include subdirectories (e.g., 'Test/Test.csv').
    """
    # Local candidates
    candidates = [
        ROOT_DIR / filename,
        ROOT_DIR / "data" / filename,  # optional local 'data' dir if present
        TRAIN_DIR / filename,
    ]
    for p in candidates:
        if p.exists():
            return p

    # Remote (HF Hub) fallback
    return _try_hf_download(filename)


@st.cache_data(show_spinner=False)
def load_training_samples() -> Tuple[pd.DataFrame, Optional[gpd.GeoDataFrame]]:
    """Load and combine Fergana and Orenburg training samples. Returns (df, gdf).

    If local files are not present, fetch shapefile components from the configured
    Hugging Face dataset repo under 'data/Train/'.
    """
    def resolve_shapefile(base_name: str) -> Optional[Path]:
        # 1) Local path
        local_shp = TRAIN_DIR / f"{base_name}.shp"
        if local_shp.exists():
            return local_shp
        # 2) Remote via HF Hub
        shp_path = None
        if USE_HF_DATA:
            # Ensure supporting files are cached too
            for ext in [".shp", ".dbf", ".shx", ".prj"]:
                _ = _try_hf_download(f"Train/{base_name}{ext}")
                if ext == ".shp" and _ is not None:
                    shp_path = _
        return shp_path

    gdfs: List[gpd.GeoDataFrame] = []
    for base in ["Fergana_training_samples", "Orenburg_training_samples"]:
        shp = resolve_shapefile(base)
        if shp is not None and shp.exists():
            try:
                gdfs.append(gpd.read_file(shp))
            except Exception:
                continue
    if not gdfs:
        return pd.DataFrame(), None

    # Label regions if available
    for gdf, region in zip(gdfs, ["Fergana", "Orenburg"]):
        if "region" not in gdf.columns:
            gdf["region"] = region

    gdf_all = pd.concat(gdfs, ignore_index=True)
    df = pd.DataFrame(gdf_all.drop(columns=["geometry"]))

    # Ensure lon/lat
    if "translated_lon" not in df.columns and hasattr(gdf_all, "geometry"):
        df["translated_lon"] = gdf_all.geometry.x
    if "translated_lat" not in df.columns and hasattr(gdf_all, "geometry"):
        df["translated_lat"] = gdf_all.geometry.y

    # Keep essential cols if present
    for col in ["longitude", "latitude"]:
        if col not in df.columns:
            if col == "longitude" and "translated_lon" in df.columns:
                df[col] = df["translated_lon"]
            if col == "latitude" and "translated_lat" in df.columns:
                df[col] = df["translated_lat"]

    return df, gdf_all


@st.cache_data(show_spinner=False)
def load_csv_sample(name: str, usecols: Optional[List[str]] = None, nrows: Optional[int] = 100_000) -> pd.DataFrame:
    """Load a CSV by name from local paths or the HF dataset. Optionally sample first nrows for performance."""
    path = _find_file(name)
    if path is None:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, usecols=usecols, nrows=nrows)
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_parquet_if_exists(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception:
            pass
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def list_model_files() -> List[Path]:
    patterns = ["*.pkl", "*.joblib"]
    out: List[Path] = []
    for pat in patterns:
        out.extend(list(MODELS_DIR.glob(pat)))
    return sorted(out)


@st.cache_resource(show_spinner=False)
def load_models() -> Dict[str, object]:
    import joblib
    models: Dict[str, object] = {}
    for p in list_model_files():
        try:
            models[p.stem] = joblib.load(p)
        except Exception:
            continue
    return models


@st.cache_data(show_spinner=False)
def load_metrics() -> Dict:
    metrics_path = CACHE_DIR / "metrics.json"
    if metrics_path.exists():
        try:
            return pd.read_json(metrics_path).to_dict(orient="list")  # type: ignore
        except Exception:
            try:
                import json
                return json.loads(metrics_path.read_text(encoding="utf-8"))
            except Exception:
                return {}
    return {}


def load_css() -> str:
    css_path = ASSETS_DIR / "styles.css"
    if css_path.exists():
        try:
            return css_path.read_text(encoding="utf-8")
        except Exception:
            return ""
    return ""


def inject_css_block(css: str) -> None:
    if css:
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def get_feature_schema(models: Dict[str, object]) -> List[str]:
    """Attempt to infer feature names from a saved schema or from a model."""
    schema_json = CACHE_DIR / "final_features.json"
    if schema_json.exists():
        try:
            import json
            return json.loads(schema_json.read_text(encoding="utf-8"))
        except Exception:
            pass
    # Try scikit-learn feature_names_in_
    for mdl in models.values():
        feats = getattr(mdl, "feature_names_in_", None)
        if feats is not None:
            return list(map(str, feats))
    return []


def format_number(x: float, digits: int = 3) -> str:
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


# Lottie helpers
@st.cache_data(show_spinner=False)
def load_lottie_url(url: str) -> Optional[Dict[str, Any]]:
    try:
        import requests
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None
