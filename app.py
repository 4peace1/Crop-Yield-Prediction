"""Interactive Streamlit dashboard for the Crop Yield Prediction capstone."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from model import CATEGORICAL_FEATURES, NUMERIC_FEATURES, load_data, load_model  # noqa: E402

MODEL_PATH = ROOT / "models" / "crop_yield_model.joblib"
DATA_PATH = ROOT / "data" / "crop_yield_prediction.csv"
METRICS_PATH = ROOT / "reports" / "evaluation.json"
COMPARISON_PATH = ROOT / "reports" / "model_comparison.csv"
IMPORTANCE_PATH = ROOT / "reports" / "feature_importance.csv"

st.set_page_config(
    page_title="Crop Yield Predictor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title {font-size: 2.6rem; font-weight: 800; margin-bottom: 0.1rem;}
    .subtitle {font-size: 1.05rem; color: #5f6368; margin-bottom: 1.5rem;}
    .result-card {
        padding: 1.25rem 1.5rem; border-radius: 14px;
        border: 1px solid rgba(49, 51, 63, .15);
        margin: .5rem 0 1rem 0;
    }
    .small-note {font-size: .88rem; color: #6b7280;}
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data
def get_data() -> pd.DataFrame:
    return load_data(DATA_PATH)

@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)

@st.cache_data
def get_metrics() -> dict:
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))

@st.cache_data
def get_comparison() -> pd.DataFrame:
    return pd.read_csv(COMPARISON_PATH)

@st.cache_data
def get_importance() -> pd.DataFrame:
    return pd.read_csv(IMPORTANCE_PATH)

if not MODEL_PATH.exists():
    st.error("The trained model is missing. Run `python src/train.py --data data/crop_yield_prediction.csv`.")
    st.stop()

data = get_data()
model = get_model()
metrics = get_metrics()

st.markdown('<div class="main-title">🌾 Crop Yield Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Interactive machine-learning dashboard for estimating crop yield in tonnes per hectare.</div>',
    unsafe_allow_html=True,
)

# Top-level KPIs
k1, k2, k3, k4 = st.columns(4)
k1.metric("Dataset records", f"{len(data):,}")
k2.metric("Test R²", f"{metrics['test_metrics']['r2']:.3f}")
k3.metric("Test RMSE", f"{metrics['test_metrics']['rmse']:.3f} t/ha")
k4.metric("Test MAE", f"{metrics['test_metrics']['mae']:.3f} t/ha")

with st.sidebar:
    st.header("🌱 Prediction Inputs")
    st.caption("Adjust the farm and environmental conditions, then generate an estimate.")

    crop = st.selectbox("Crop", sorted(data["crop"].dropna().unique()))
    state = st.selectbox("State", sorted(data["state"].dropna().unique()))
    season = st.selectbox("Season", sorted(data["season"].dropna().unique()))
    irrigated = st.selectbox(
        "Irrigation",
        [0, 1],
        format_func=lambda x: "Yes" if x else "No",
    )
    farm_size = st.number_input(
        "Farm size (ha)",
        min_value=0.1,
        max_value=100.0,
        value=float(data["farm_size_ha"].median()),
        step=0.1,
    )
    rainfall = st.number_input(
        "Rainfall (mm)",
        min_value=0.0,
        max_value=5000.0,
        value=float(data["rainfall_mm"].median()),
        step=1.0,
    )
    temperature = st.number_input(
        "Average temperature (°C)",
        min_value=-10.0,
        max_value=60.0,
        value=float(data["avg_temperature_c"].median()),
        step=0.1,
    )
    soil_ph = st.number_input(
        "Soil pH",
        min_value=0.0,
        max_value=14.0,
        value=float(data["soil_ph"].median()),
        step=0.1,
    )
    organic = st.number_input(
        "Soil organic matter (%)",
        min_value=0.0,
        max_value=20.0,
        value=float(data["soil_organic_matter_pct"].median()),
        step=0.1,
    )
    nitrogen = st.number_input(
        "Nitrogen (kg/ha)",
        min_value=0.0,
        max_value=500.0,
        value=float(data["nitrogen_kg_ha"].median()),
        step=1.0,
    )

    predict_clicked = st.button("🌱 Predict Yield", type="primary", use_container_width=True)

row = pd.DataFrame([{
    "crop": crop,
    "state": state,
    "season": season,
    "irrigated": irrigated,
    "rainfall_mm": rainfall,
    "avg_temperature_c": temperature,
    "soil_ph": soil_ph,
    "soil_organic_matter_pct": organic,
    "nitrogen_kg_ha": nitrogen,
    "farm_size_ha": farm_size,
}])

if "history" not in st.session_state:
    st.session_state.history = []

if predict_clicked:
    prediction = max(0.0, float(model.predict(row)[0]))
    st.session_state.history.append({
        "Crop": crop,
        "State": state,
        "Season": season,
        "Irrigated": "Yes" if irrigated else "No",
        "Rainfall (mm)": rainfall,
        "Nitrogen (kg/ha)": nitrogen,
        "Predicted Yield (t/ha)": round(prediction, 3),
    })

    st.markdown("### 🎯 Prediction Result")
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    r1, r2 = st.columns([2, 1])
    with r1:
        st.metric("Estimated crop yield", f"{prediction:.2f} tonnes/hectare")
    with r2:
        st.metric("Estimated production", f"{prediction * farm_size:.2f} tonnes")
    st.markdown(
        '<div class="small-note">The second figure multiplies predicted yield by the entered farm area and is an estimate, not a guarantee.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.success("Prediction generated successfully.")
else:
    st.info("Use the sidebar to enter farm conditions and click **Predict Yield**.")

tab1, tab2, tab3 = st.tabs(["📊 Model Performance", "🔎 Feature Importance", "📋 Dataset"])

with tab1:
    st.subheader("Model comparison")
    comparison = get_comparison().copy()
    display = comparison.rename(
        columns={"model": "Model", "mae": "MAE", "rmse": "RMSE", "r2": "R²"}
    )
    st.dataframe(
        display.style.format({"MAE": "{:.3f}", "RMSE": "{:.3f}", "R²": "{:.3f}"}),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        f"Selected model: {metrics['selected_model']}. "
        f"Five-fold CV RMSE: {metrics['five_fold_cv_rmse_mean']:.3f} ± "
        f"{metrics['five_fold_cv_rmse_std']:.3f} t/ha."
    )

    chart = comparison.set_index("model")[["mae", "rmse"]].rename(
        columns={"mae": "MAE", "rmse": "RMSE"}
    )
    st.bar_chart(chart)

with tab2:
    st.subheader("Permutation feature importance")
    importance = get_importance().copy().sort_values("mae_increase", ascending=True)
    importance_chart = importance.set_index("feature")[["mae_increase"]].rename(
        columns={"mae_increase": "Increase in MAE"}
    )
    st.bar_chart(importance_chart)
    st.dataframe(
        get_importance().rename(
            columns={
                "feature": "Feature",
                "mae_increase": "MAE increase",
                "mae_increase_std": "Std. deviation",
            }
        ).style.format({"MAE increase": "{:.4f}", "Std. deviation": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
    )

with tab3:
    st.subheader("Dataset profile")
    d1, d2, d3 = st.columns(3)
    d1.metric("Rows", f"{len(data):,}")
    d2.metric("Input features", f"{len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)}")
    d3.metric("Target", "yield_tonnes_ha")
    st.dataframe(data.head(20), use_container_width=True, hide_index=True)

if st.session_state.history:
    st.divider()
    st.subheader("🧾 Prediction History")
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df, use_container_width=True, hide_index=True)
    csv = history_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Prediction History",
        data=csv,
        file_name="crop_yield_predictions.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "Educational capstone demonstration. Model outputs should not be treated as "
    "independent agronomic advice or validated field-yield guarantees."
)
