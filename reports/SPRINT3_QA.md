# Sprint 3 QA Checklist

## Streamlit
- [ ] Run `python -m streamlit run app.py`
- [ ] Verify all input widgets load.
- [ ] Generate a prediction.
- [ ] Verify estimated production equals predicted yield × farm size.
- [ ] Verify model comparison chart loads.
- [ ] Verify feature importance chart loads.
- [ ] Verify prediction history appears after a prediction.
- [ ] Verify CSV download works.

## Repository
- [ ] Confirm `data/crop_yield_prediction.csv` exists.
- [ ] Confirm `models/crop_yield_model.joblib` exists.
- [ ] Confirm `reports/evaluation.json` is free of personal absolute paths.
- [ ] Confirm no `__pycache__` or `.pyc` files are committed.
