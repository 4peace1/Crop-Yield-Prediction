"""Command-line prediction utility for crop yield."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model import CATEGORICAL_FEATURES, NUMERIC_FEATURES, load_model

MODEL_PATH = ROOT / "models" / "crop_yield_model.joblib"


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict yield in tonnes per hectare.")
    for feature in CATEGORICAL_FEATURES:
        parser.add_argument(f"--{feature.replace('_', '-')}", required=True)
    for feature in NUMERIC_FEATURES:
        parser.add_argument(f"--{feature.replace('_', '-')}", required=True, type=float)
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model missing. Run 'python src/train.py' first.")

    row = {feature: getattr(args, feature) for feature in CATEGORICAL_FEATURES + NUMERIC_FEATURES}
    model = load_model(MODEL_PATH)
    prediction = max(0.0, float(model.predict([row])[0]))
    print(f"Predicted yield: {prediction:.2f} tonnes/hectare")


if __name__ == "__main__":
    main()
