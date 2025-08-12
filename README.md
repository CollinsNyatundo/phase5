# GEO-AI Cropland Mapping
The notebook and dataset outline a complete workflow for the GEO-AI Cropland Mapping Challenge. The task focuses on identifying cropland areas using multi-temporal satellite imagery. The project walks through each stage of the process, from preparing and analyzing the data to training and evaluating models, with practical insights at every step.
## Table of Contents
- Business Understanding
- Environment Setup
- Data Description
- Preprocessing Pipeline
- Feature Engineering
- Exploratory Data Analysis (EDA)
- Model Training
- Recommendations
## Business Understanding
- **Goal**: Classify pixels/points as crop vs non-crop in target regions, producing per-location cropland probabilities or labels.
- **Objectives**: Maximize competition score, produce interpretable regional reports, and have a reproducible pipeline.
- **Constraints**: Limited labeled samples, regional variability, risk of spatial leakage (use spatial CV).
### Objectives:
- Maximize official competition score.
- Generate interpretable regional performance reports.
- Provide reproducible and modular pipeline code.
### Constraints & Risks:
- Limited labeled samples with class imbalance (~30% crop).
- Regional variability (Fergana and Orenburg).
- Risk of spatial leakage—must use spatially aware cross-validation.
## Environment Setup
- Implemented in Python, this pipeline requires the following packages:
- Data processing: pandas, numpy, geopandas
- Geospatial handling: rasterio, pyproj, folium
- Machine learning: scikit-learn, xgboost, lightgbm, catboost
- Imbalanced data tools: imblearn (e.g., SMOTE)
- Optimization and deep learning (optional): optuna, torch
- Visualization: matplotlib, seaborn
## Data Description
 ### Training samples:
- 1000 labeled points combined from two regions:
- Fergana (500 samples)
- Orenburg (500 samples)
- Class imbalance: 703 non-crop (0), 297 crop (1)
### Satellite data:
- Sentinel-1 radar (~1.75 million rows) with VH and VV bands
- Sentinel-2 optical multispectral (~5.6 million rows) including bands B2–B12, cloud coverage, solar azimuth/zenith
### Test data:
- 600 unlabeled points for final inference and submission
## Preprocessing Pipeline
- Aggregate Sentinel-1 and Sentinel-2 data spatially by unique (lat, lon) coordinates using mean statistics, reducing data size.
- Use BallTree nearest neighbor search with adaptive angular thresholds for matching labeled points to satellite pixels.
- Fallback approach: merge by rounded coordinate keys when coverage is low.
- Cache intermediate aggregation results for long-term reproducibility and speed.
- Handle missing data by imputation and fallback merges.
- Compute and attach vegetation indices and derived radar features.
## Feature Engineering
### Base Satellite Features
Sentinel-1: VH and VV radar backscatter.
Sentinel-2: Optical bands B2 (Blue), B3 (Green), B4 (Red), B5–B7 (Red Edge), B8 and B8A (NIR), B11 and B12 (SWIR), cloud_pct, solar_azimuth, solar_zenith.
### Vegetation Indices Computed
NDVI = (B8 - B4) / (B8 + B4)
NDWI = (B3 - B8) / (B3 + B8)
EVI = 2.5 × ((B8 - B4) / (B8 + 6×B4 - 7.5×B2 + 1))
SAVI = (1 + L) × (B8 - B4) / (B8 + B4 + L), where L=0.5
MSAVI = (2×B8 + 1 - sqrt((2×B8 + 1)² - 8×(B8 - B4))) / 2
RECI = (B8 / B5) - 1
GCI = (B8 / B3) - 1
### Radar Feature Ratio
VH_VV_ratio = VH / VV
#### Key Notes
- Features are matched spatially with train points, ensuring high spatial fidelity (~81.6% coverage).
- Missing matches are addressed with fallback key-based merges.
- All numeric features are downcast to float32 for memory efficiency.
- Temporal feature engineering is recommended for improvement but not implemented explicitly here (e.g., monthly/seasonal aggregates, cloud masking).
## Exploratory Data Analysis (EDA)
- Analyzed spatial distribution and class balance.
- Visualized spectral index histograms and multi-temporal patterns.
- Confirmed Sentinel-1 and Sentinel-2 data coverage (~81.7% of training points).
- Assessed missing data and quality flags.
- Recommended spatial CV for train-validation splitting to prevent spatial leakage.
## Model Training
- Use spatial cross-validation split strategies (block or region-based) to avoid data leakage.
- Employ classical ML models (XGBoost, LightGBM, CatBoost) or deep learning frameworks.
- Address class imbalance with SMOTE or related oversampling techniques.
- Tune hyperparameters using Optuna.
- Output cropland probability scores formatted for official competition submission.
## Recommendations
- Spatially aware train-test splits are critical.
- Use cloud and quality masks to filter noisy satellite observations.
- Feature engineering (vegetation indices) greatly improves class separability.
- Consider temporal aggregation to capture phenological signals.
- Perform data augmentation or synthetic sample generation if feasible.




