"""Repository-level smoke tests."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_required_files_exist():
    required = [
        ROOT / "README.md",
        ROOT / "requirements.txt",
        ROOT / "app.py",
        ROOT / "data" / "crop_yield_prediction.csv",
        ROOT / "models" / "crop_yield_model.joblib",
        ROOT / "reports" / "evaluation.json",
        ROOT / "reports" / "model_comparison.csv",
        ROOT / "reports" / "feature_importance.csv",
    ]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
    assert not missing, f"Missing required project files: {missing}"
