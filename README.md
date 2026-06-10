# 🌡️ Spatio-Temporal Urban Heat Island (UHI) Prediction System

**AI-powered system for predicting and simulating Urban Heat Island effects in Delhi, India**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-ff4b4b.svg)](https://streamlit.io)

---

## 📋 Overview

This system combines **satellite image analysis**, **weather sensor data analytics**, **spatial data mining**, and **deep learning** to:

1. **Classify** land cover from satellite imagery using ResNet-50 CNN (Achieving **97.8%** validation accuracy).
2. **Cluster** heat hotspot zones using DBSCAN spatial mining.
3. **Discover** UHI patterns via FP-Growth association rules (Extracting 40+ actionable rules).
4. **Predict** future temperatures using LSTM + XGBoost models (Both achieving **R² > 0.97**).
5. **Simulate** how construction plans affect local temperature via an interactive dashboard.

> *"If I build a concrete mall in Sector 15, how much will that raise the local temperature in 3 years?"*

---

## 🏗️ Project Structure

```
uhi-prediction-delhi/
├── config.py                  # Central configuration (paths, hyperparams)
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── .gitignore
│
├── data/
│   ├── raw/                   # Downloaded satellite images & CSV sensor files
│   ├── processed/             # Cleaned DataFrames (.csv)
│   ├── labels/                # Training labels for CNN
│   └── models/                # Saved model weights (.pth, .joblib)
│
├── src/
│   ├── __init__.py
│   ├── utils.py               # Shared utility functions
│   ├── preprocessing.py       # Data cleaning, NDVI/NDBI, feature engineering
│   ├── cnn_model.py           # ResNet-50 transfer learning (land cover)
│   ├── clustering.py          # DBSCAN spatial clustering
│   ├── mining.py              # FP-Growth association rules
│   └── predictor.py           # LSTM + XGBoost heat forecasting
│
├── dashboard/
│   ├── app.py                 # Streamlit main app
│   └── pages/
│       ├── 1_Current_Heat_Map.py    # Interactive Delhi heat map
│       ├── 2_Historical_Trends.py   # 10-year temperature analysis
│       ├── 3_Simulation_Tool.py     # Construction impact simulator
│       └── 4_Mined_Rules.py         # Association rules explorer
│
└── outputs/
    ├── maps/                  # Generated Folium maps
    ├── plots/                 # Generated charts
    └── rules/                 # Exported association rules
```

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Preprocessing Pipeline

```bash
python -m src.preprocessing
```

This loads NASA POWER weather data, computes thermal features, and generates spatial features.

### 3. Run Mining Pipeline

```bash
python -m src.mining
```

Discovers association rules from the preprocessed data.

### 4. Train CNN Model

```bash
# With synthetic data (demo mode)
python -m src.cnn_model --synthetic --epochs 10

# With EuroSAT dataset
python -m src.cnn_model --eurosat-dir path/to/EuroSAT
```

### 5. Train Prediction Models

```bash
python -m src.predictor
```

Trains both XGBoost and LSTM models and compares performance.

### 6. Launch Dashboard

```bash
streamlit run dashboard/app.py
```

---

## 📊 Technology Stack

| Category | Primary Technologies | Description / Purpose |
| :--- | :--- | :--- |
| **Data Processing** | `Python`, `Pandas`, `NumPy` | Core data manipulation, time-series handling, and matrix operations. |
| **Satellite & Geospatial** | `Rasterio`, `GDAL`, `GeoPandas` | Extracting thermal/optical bands and clipping to Delhi NCT shapefiles. |
| **Deep Learning (Vision)** | `PyTorch`, `Torchvision` | Transfer learning using ResNet-50 for land cover classification. |
| **Machine Learning** | `Scikit-learn`, `XGBoost` | DBSCAN for spatial heat clustering and Gradient Boosting for temperature forecasting. |
| **Data Mining** | `mlxtend` | FP-Growth algorithm for extracting meteorological association rules. |
| **Deep Learning (Time-Series)**| `PyTorch (LSTM)` | Sequential modeling for long-term climate trend predictions. |
| **Frontend & Simulation** | `Streamlit` | Interactive dashboard for real-time visualization and "What-If" scenario building. |
| **Visualization & Mapping** | `Plotly`, `Seaborn`, `Folium` | Dynamic interactive charts, correlation heatmaps, and spatial coordinate mapping. |

---

## 📡 Data Sources

| Dataset | Source | Format |
|---|---|---|
| Daily weather (2015–2025) | [NASA POWER](https://power.larc.nasa.gov) | CSV |
| Satellite imagery | [USGS EarthExplorer](https://earthexplorer.usgs.gov) | GeoTIFF |
| Land cover labels | [EuroSAT (Kaggle)](https://www.kaggle.com/datasets/apollo2506/eurosat-dataset) | JPG |
| Delhi boundary | [GADM](https://gadm.org) | Shapefile |

---

## 🔬 Model Performance Details

### CNN — Land Cover Classification
- **Architecture:** ResNet-50 (pre-trained on ImageNet)
- **Classes:** Dense Concrete, Asphalt, Dense Vegetation, Sparse Vegetation, Water
- **Training:** Progressive unfreezing, Adam optimizer, CrossEntropyLoss
- **Performance:** **97.8%** Validation Accuracy

### DBSCAN Clustering
- **Features:** Lat, Lon, NDVI, NDBI, concrete %, vegetation %
- **Output:** Identifies localized Heat Hotspot zones vs. Cool Islands

### Association Rule Mining
- **Algorithm:** FP-Growth
- **Parameters:** min_support=0.08, min_confidence=0.6, min_lift=1.2
- **Result:** Extracted 40+ deterministic rules mapping land cover and weather to extreme heat.

### Predictive Models
- **XGBoost:** Gradient boosted trees (**R² = 0.975**)
- **LSTM:** 2-layer LSTM (64→32 units) with early stopping (**R² = 0.978**)

---

## 📄 Research Paper

The project includes an extensive **9+ page IEEE format research paper** (`paper/uhi_paper.tex`) complete with detailed mathematical formulations, literature review, and comprehensive case studies derived from the Simulation tool. Analytical figures (feature importance, historical trends, correlation matrices) are generated programmatically and included in the LaTeX compilation.

---

## 👤 Author

Madhav Gupta
B.Tech Computer Science and Engineering
Bennett University
Email: guptamadhav7718@gmail.com

---

## 📜 License

This project is for academic/research purposes.
