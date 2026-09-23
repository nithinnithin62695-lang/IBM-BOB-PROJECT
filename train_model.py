"""
Farmer Income Predictor – Model Training Script
Trains a GradientBoosting model on lte_train.csv and saves:
  - models/income_model.joblib   (trained pipeline)
  - models/model_meta.joblib     (feature names, metrics, importance)
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_CSV = os.path.join(BASE_DIR, "..", "lte_train.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

TARGET = "Target_Variable/Total Income"

# ── Feature groups ─────────────────────────────────────────────────────────────
NUMERIC_FEATURES = [
    "Non_Agriculture_Income",
    "Total_Land_For_Agriculture",
    "No_of_Active_Loan_In_Bureau",
    "Avg_Disbursement_Amount_Bureau",
    "K022-Proximity to nearest mandi (Km)",
    "K022-Proximity to nearest railway (Km)",
    "KO22-Village score based on socio-economic parameters (0 to 100)",
    "K022-Seasonal Average Rainfall (mm)",
    "R022-Seasonal Average Rainfall (mm)",
    "K021-Seasonal Average Rainfall (mm)",
    "R021-Seasonal Average Rainfall (mm)",
    "R020-Seasonal Average Rainfall (mm)",
    "Perc_of_house_with_6plus_room",
    "perc_of_pop_living_in_hh_electricity",
    "perc_Households_with_Pucca_House_That_Has_More_Than_3_Rooms",
    "mat_roof_Metal_GI_Asbestos_sheets",
    "perc_of_Wall_material_with_Burnt_brick",
    "Households_with_improved_Sanitation_Facility",
    "perc_Households_do_not_have_KCC_With_The_Credit_Limit_Of_50k",
    "K022-Total Geographical Area (in Hectares)-",
    "K022-Net Agri area (in Ha)-",
    "K022-Net Agri area (% of total geog area)-",
    "Kharif Seasons  Irrigated area in 2022",
    "Kharif Seasons  Cropping density in 2022",
    "Kharif Seasons  Agricultural Score in 2022",
    "Kharif Seasons  Seasonal average groundwater thickness (cm) in 2022",
    "Kharif Seasons  Seasonal average groundwater replenishment rate (cm) in 2022",
    "Rabi Seasons  Season Irrigated area in 2022",
    "Rabi Seasons Cropping density in 2022",
    "Rabi Seasons Agricultural Score in 2022",
    " Night light index",
    " Village score based on socio-economic parameters (0 to 100)",
    " Land Holding Index source (Total Agri Area/ no of people)",
    " Road density (Km/ SqKm)",
]

CATEGORICAL_FEATURES = [
    "State",
    "REGION",
    "SEX",
    "MARITAL_STATUS",
    "K022-Village category based on Agri parameters (Good, Average, Poor)",
    "K022-Village category based on socio-economic parameters (Good, Average, Poor)",
    "R022-Village category based on Agri parameters (Good, Average, Poor)",
    "Kharif Seasons  Agricultural performance in 2022",
    "Rabi Seasons Agricultural performance in 2022",
    " Village category based on socio-economic parameters (Good, Average, Poor)",
]


def load_data():
    print(f"Loading training data from: {TRAIN_CSV}")
    df = pd.read_csv(TRAIN_CSV, low_memory=False)
    print(f"  Shape: {df.shape}")
    return df


def prepare_features(df: pd.DataFrame):
    # Keep only columns we need
    all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    available_num = [c for c in NUMERIC_FEATURES if c in df.columns]
    available_cat = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    all_available = available_num + available_cat

    X = df[all_available].copy()
    y = df[TARGET].copy()

    # Drop rows where target is missing
    mask = y.notna()
    X, y = X[mask], y[mask]

    # Log-transform target to reduce skewness
    y = np.log1p(y)

    print(f"  Features used – numeric: {len(available_num)}, categorical: {len(available_cat)}")
    print(f"  Samples after cleaning: {len(X)}")
    return X, y, available_num, available_cat


def build_pipeline(num_features, cat_features):
    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    preprocessor = ColumnTransformer([
        ("num", num_transformer, num_features),
        ("cat", cat_transformer, cat_features),
    ])
    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.08,
        max_depth=5,
        subsample=0.8,
        min_samples_leaf=10,
        random_state=42,
        verbose=0,
    )
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])
    return pipeline


def evaluate(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    # Back to original scale for reporting
    y_test_orig = np.expm1(y_test)
    y_pred_orig = np.expm1(y_pred)

    mae = mean_absolute_error(y_test_orig, y_pred_orig)
    rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
    r2 = r2_score(y_test_orig, y_pred_orig)
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_test_orig - y_pred_orig) / (y_test_orig + 1e-9))) * 100

    print(f"\n  -- Evaluation Metrics --")
    print(f"  MAE  : Rs {mae:,.0f}")
    print(f"  RMSE : Rs {rmse:,.0f}")
    print(f"  R2   : {r2:.4f}")
    print(f"  MAPE : {mape:.2f}%")
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}


def get_feature_importance(pipeline, num_features, cat_features):
    model = pipeline.named_steps["model"]
    all_features = num_features + cat_features
    importances = model.feature_importances_
    fi = pd.Series(importances, index=all_features).sort_values(ascending=False)
    return fi.to_dict()


def train():
    df = load_data()
    X, y, num_features, cat_features = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTraining on {len(X_train)} samples, validating on {len(X_test)} …")

    pipeline = build_pipeline(num_features, cat_features)
    pipeline.fit(X_train, y_train)
    print("  Training complete.")

    metrics = evaluate(pipeline, X_test, y_test)
    feature_importance = get_feature_importance(pipeline, num_features, cat_features)

    # Save pipeline
    model_path = os.path.join(MODELS_DIR, "income_model.joblib")
    joblib.dump(pipeline, model_path)

    # Save metadata
    meta = {
        "num_features": num_features,
        "cat_features": cat_features,
        "all_features": num_features + cat_features,
        "metrics": metrics,
        "feature_importance": feature_importance,
        "target_transform": "log1p",
    }
    meta_path = os.path.join(MODELS_DIR, "model_meta.joblib")
    joblib.dump(meta, meta_path)

    print(f"\n  Model saved: {model_path}")
    print(f"  Metadata saved: {meta_path}")
    return pipeline, meta


if __name__ == "__main__":
    train()
