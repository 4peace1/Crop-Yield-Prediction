# Sprint 4 — Code Quality & Testing

## Automated tests

Run from the repository root:

```bash
python -m pytest -q
```

The test suite checks:

- required project files
- dataset existence
- expected feature schema
- numeric target type
- model fitting
- prediction output length and validity

## Python syntax check

Run:

```bash
python -m compileall app.py src
```

## Reproducibility check

Create a clean virtual environment and run:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python src/train.py --data data/crop_yield_prediction.csv
python -m streamlit run app.py
```

## Manual application checks

- Generate a prediction.
- Verify the result is displayed in tonnes/hectare.
- Verify estimated production responds to farm size.
- Verify model comparison charts load.
- Verify feature importance loads.
- Verify prediction history can be downloaded.

## Repository hygiene

Do not commit:

- `.venv/`
- `venv/`
- `__pycache__/`
- `*.pyc`
- `.ipynb_checkpoints/`
- local secrets
- personal absolute file paths
