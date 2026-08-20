"""Train, compare, validate, and persist the crop-yield model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from model import (
    FEATURES, TARGET, build_preprocessor, build_model, compute_permutation_importance,
    evaluate, load_data, save_model
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "crop_yield_prediction.csv"
MODEL_PATH = ROOT / "models" / "crop_yield_model.joblib"
METRICS_PATH = ROOT / "reports" / "evaluation.json"
COMPARISON_PATH = ROOT / "reports" / "model_comparison.csv"
IMPORTANCE_PATH = ROOT / "reports" / "feature_importance.csv"


def candidate_models() -> dict[str, Pipeline]:
    """Return candidate regressors using the same preprocessing."""
    specs = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=2,
            loss="huber", random_state=42
        ),
    }
    return {
        name: Pipeline([("preprocessor", build_preprocessor()), ("model", estimator)])
        for name, estimator in specs.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the crop-yield prediction model.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()

    frame = load_data(args.data)
    X, y = frame[FEATURES], frame[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    comparison = []
    fitted = {}
    for name, pipeline in candidate_models().items():
        pipeline.fit(X_train, y_train)
        metrics = evaluate(pipeline, X_test, y_test)
        comparison.append({"model": name, **metrics})
        fitted[name] = pipeline

    comparison_df = pd.DataFrame(comparison).sort_values(["rmse", "mae"])
    best_name = str(comparison_df.iloc[0]["model"])
    best_model = fitted[best_name]

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = -cross_val_score(
        best_model, X, y, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1
    )

    test_metrics = evaluate(best_model, X_test, y_test)
    importance = compute_permutation_importance(best_model, X_test, y_test)

    save_model(best_model, MODEL_PATH)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        "dataset": args.data.name,
        "rows": int(len(frame)),
        "features": int(len(FEATURES)),
        "test_size": 0.20,
        "random_state": 42,
        "selected_model": best_name,
        "test_metrics": test_metrics,
        "five_fold_cv_rmse_mean": float(cv_scores.mean()),
        "five_fold_cv_rmse_std": float(cv_scores.std()),
        "candidate_models": comparison_df.to_dict(orient="records"),
    }
    METRICS_PATH.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    comparison_df.to_csv(COMPARISON_PATH, index=False)
    importance.to_csv(IMPORTANCE_PATH, index=False)

    print(f"Dataset: {args.data}")
    print(f"Rows: {len(frame):,}")
    print(comparison_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Selected model: {best_name}")
    print(f"5-fold CV RMSE: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"Saved model: {MODEL_PATH}")


if __name__ == "__main__":
    main()
