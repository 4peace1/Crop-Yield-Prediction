# Final Capstone Submission Checklist

## Repository
- [x] README documents the current project architecture and metrics.
- [x] Relative paths are used for project data/model/report artifacts.
- [x] Generated Python cache files are excluded from the final package.
- [x] MIT License is included.
- [x] Requirements include runtime and test dependencies.

## Machine Learning
- [x] Dataset schema is aligned with the training pipeline.
- [x] Multiple regression models are compared.
- [x] Holdout metrics are reported.
- [x] Five-fold cross-validation is reported.
- [x] Feature importance is documented.
- [x] Model is persisted for application use.
- [ ] External independent validation remains a future improvement.

## Application
- [x] Streamlit application uses the same feature schema as training.
- [x] Prediction result is displayed in tonnes/hectare.
- [x] Estimated total production is calculated from farm area.
- [x] Model performance is visible.
- [x] Feature importance is visible.
- [x] Prediction history can be exported.

## Testing
- [x] Pytest suite included.
- [x] Syntax compilation check completed.
- [x] Repository smoke tests completed.

## Before GitHub push
1. Run `python -m pytest -q`.
2. Run `python -m compileall -q app.py src tests`.
3. Run `python src/train.py --data data/crop_yield_prediction.csv` if you want to regenerate model/report artifacts.
4. Run `python -m streamlit run app.py` and capture real screenshots.
5. Review the Git diff before committing.
