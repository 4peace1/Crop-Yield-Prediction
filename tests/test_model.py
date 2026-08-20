"""Unit tests for the crop-yield model pipeline."""
from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model import (  # noqa: E402
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    build_model,
    load_data,
)


DATA_PATH = ROOT / "data" / "crop_yield_prediction.csv"


def test_dataset_exists():
    assert DATA_PATH.exists(), "Expected dataset is missing."


def test_dataset_schema():
    df = load_data(DATA_PATH)
    expected = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET])
    assert expected.issubset(df.columns)
    assert len(df) > 0


def test_target_is_numeric():
    df = load_data(DATA_PATH)
    assert pd.api.types.is_numeric_dtype(df[TARGET])


def test_model_can_fit_and_predict():
    df = load_data(DATA_PATH)
    model = build_model()
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].head(20)
    y = df[TARGET].head(20)

    model.fit(X, y)
    predictions = model.predict(X)

    assert len(predictions) == len(y)
    assert all(pd.notna(predictions))
