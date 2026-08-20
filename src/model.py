"""Reusable preprocessing and modelling utilities for crop-yield prediction."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
TARGET = "yield_tonnes_ha"
CATEGORICAL_FEATURES = ["crop", "state", "season", "irrigated"]
NUMERIC_FEATURES = [
    "rainfall_mm", "avg_temperature_c", "soil_ph", "soil_organic_matter_pct",
    "nitrogen_kg_ha", "farm_size_ha",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def load_data(path: str | Path) -> pd.DataFrame:
    """Load and validate the project dataset."""
    frame = pd.read_csv(path)
    required = set(FEATURES + [TARGET])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("Dataset is empty.")
    if frame[TARGET].isna().any():
        raise ValueError(f"Target column '{TARGET}' contains missing values.")
    return frame


def build_preprocessor() -> ColumnTransformer:
    """Create the preprocessing transformer used during training and prediction."""
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])


def build_model() -> Pipeline:
    """Build the selected Gradient Boosting regression pipeline."""
    estimator = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=2,
        loss="huber",
        random_state=42,
    )
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", estimator),
    ])


def evaluate(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Return standard regression metrics."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    prediction = model.predict(X)
    return {
        "mae": float(mean_absolute_error(y, prediction)),
        "rmse": float(mean_squared_error(y, prediction) ** 0.5),
        "r2": float(r2_score(y, prediction)),
    }


def save_model(model: Pipeline, path: str | Path) -> None:
    """Persist a fitted model pipeline."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str | Path) -> Pipeline:
    """Load a persisted model pipeline."""
    return joblib.load(path)


def compute_permutation_importance(
    model: Pipeline, X: pd.DataFrame, y: pd.Series, random_state: int = 42
) -> pd.DataFrame:
    """Calculate feature importance on the original input features."""
    result = permutation_importance(
        model, X, y, scoring="neg_mean_absolute_error",
        n_repeats=10, random_state=random_state, n_jobs=-1
    )
    return (
        pd.DataFrame({
            "feature": X.columns,
            "mae_increase": result.importances_mean,
            "mae_increase_std": result.importances_std,
        })
        .sort_values("mae_increase", ascending=False)
        .reset_index(drop=True)
    )
