# 🌾 Farmer Income Predictor

A full-stack ML application that predicts Indian farmers' total annual income using demographic, agricultural, geographic, and socio-economic features.

## 📁 Project Structure

```
farmer-income-data.zip/
├── lte_train.csv                        # Training data (47,970 rows)
├── lte_test.csv                         # Test data
├── lte_dictionary.csv                   # Column descriptions
└── farmer_income_predictor/
    ├── app.py                           # Main Streamlit app (entry point)
    ├── requirements.txt                 # Python dependencies
    ├── backend/
    │   ├── train_model.py               # ML model training pipeline
    │   └── predictor.py                 # Prediction helper
    ├── models/                          # Saved model + metadata (auto-created)
    │   ├── income_model.joblib
    │   └── model_meta.joblib
    └── pages/
        ├── home.py                      # Home & Overview page
        ├── explorer.py                  # Data Explorer page
        ├── predict.py                   # Predict Income page
        └── insights.py                  # Model Insights page
```

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the app

```bash
cd farmer_income_predictor
streamlit run app.py
```

### 3. Train the model (first time)

- Navigate to **🔮 Predict Income** in the sidebar
- Click **"🚀 Train Model Now"** — this trains on `lte_train.csv` (~2 minutes)
- All subsequent runs load the saved model instantly

---

## 🧠 Model Details

| Property | Value |
|---|---|
| Algorithm | Gradient Boosting Regressor |
| Features | 34 numeric + 10 categorical |
| Target | `log1p(Total Income)` |
| Train/Test split | 80% / 20% |
| Imputation | Median (numeric), Mode (categorical) |
| Encoding | OrdinalEncoder |
| Scaler | StandardScaler |

## 📊 App Pages

| Page | Description |
|---|---|
| 🏠 Home & Overview | KPI cards, income distribution, region-wise charts |
| 📊 Data Explorer | Filter data, scatter/box plots, correlation heatmap |
| 🔮 Predict Income | Interactive form → real-time income prediction |
| 🧠 Model Insights | Feature importance, metrics, R² gauge |

## 🔬 Key Features Used

- Non-Agriculture Income, Total Land for Agriculture
- Active Loans, Avg Disbursement Amount
- State, Region, Village category (Agri & Socio-Econ)
- Proximity to Mandi and Railway
- Seasonal rainfall (Kharif / Rabi seasons 2020–2022)
- Groundwater thickness & replenishment
- Infrastructure indicators (electricity, sanitation, road density)
- Night light index, land holding index

## 📦 Tech Stack

- **Frontend / UI**: Streamlit + Plotly
- **Backend / ML**: scikit-learn, pandas, numpy, joblib
- **Model persistence**: joblib

---
*Built with Python + Streamlit | IBM Farmer Income Dataset*
