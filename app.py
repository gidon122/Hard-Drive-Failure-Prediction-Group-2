"""
Hard Drive Failure Prediction - Streamlit app.

Loads the pipeline trained by train.py (median imputer + RandomForest) and predicts
the probability that a drive fails at its next daily record, from SMART raw values.

Run locally:  streamlit run app.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "drive_failure_prediction_model.pkl"

st.set_page_config(page_title="Drive Failure Predictor", page_icon="💽", layout="wide")

# Friendly names for common SMART attributes (unknown ones fall back to "SMART <id>")
SMART_LABELS = {
    1: "Read error rate",
    3: "Spin-up time",
    4: "Start/stop count",
    5: "Reallocated sectors count",
    7: "Seek error rate",
    9: "Power-on hours",
    10: "Spin retry count",
    12: "Power cycle count",
    187: "Reported uncorrectable errors",
    188: "Command timeout",
    189: "High fly writes",
    190: "Airflow temperature",
    192: "Power-off retract count",
    193: "Load cycle count",
    194: "Temperature (°C)",
    197: "Current pending sector count",
    198: "Offline uncorrectable sectors",
    199: "UDMA CRC error count",
}
# Attributes most associated with failure - shown up front in the single-drive form
KEY_ATTRIBUTES = [5, 187, 188, 197, 198, 9, 194]


def smart_id(feature: str) -> int:
    return int(feature.split("_")[1])


def label_for(feature: str) -> str:
    sid = smart_id(feature)
    return f"{SMART_LABELS.get(sid, 'SMART ' + str(sid))} (SMART {sid})"


# ----------------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return joblib.load(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error(f"Model file not found: `{MODEL_PATH.name}`. Put it in the same folder as app.py.")
    st.stop()

model = load_model()

if not hasattr(model, "feature_names_in_"):
    st.error("The model has no stored feature names. Re-train it with a pandas DataFrame (train.py does this).")
    st.stop()

FEATURES = list(model.feature_names_in_)
POSITIVE_IDX = list(model.classes_).index(1)


def predict_failure_proba(df: pd.DataFrame) -> np.ndarray:
    """Align columns to what the model was trained on; missing values are imputed by the pipeline."""
    X = df.reindex(columns=FEATURES).apply(pd.to_numeric, errors="coerce")
    return model.predict_proba(X)[:, POSITIVE_IDX]


def risk_band(p: float, threshold: float) -> str:
    if p >= threshold:
        return "High"
    if p >= threshold / 2:
        return "Elevated"
    return "Low"


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.title("💽 Hard Drive Failure Predictor")
st.write(
    "Estimates the probability that a drive **fails at its next daily record** "
    "from its SMART raw values, using a Random Forest trained on historical drive telemetry."
)

with st.sidebar:
    st.header("Settings")
    threshold = st.slider(
        "Alert threshold",
        min_value=0.01, max_value=0.99, value=0.50, step=0.01,
        help="Drives with a failure probability at or above this value are flagged High risk. "
             "Failures are rare, so lowering it catches more failures at the cost of more false alarms.",
    )
    st.caption(f"Model uses {len(FEATURES)} SMART raw features.")
    st.info("Decision-support tool only. It is not a guarantee that a drive will or won't fail.")

tab_single, tab_batch, tab_model = st.tabs(["Single drive", "Batch (CSV)", "Model info"])

# ----------------------------------------------------------------------------
# Tab 1: single drive
# ----------------------------------------------------------------------------
with tab_single:
    st.subheader("Enter SMART values for one drive")
    st.caption("Leave a field empty if the value is unknown - it is filled with the training median.")

    key_feats = [f for f in FEATURES if smart_id(f) in KEY_ATTRIBUTES]
    other_feats = [f for f in FEATURES if f not in key_feats]

    with st.form("single_form"):
        values = {}

        cols = st.columns(3)
        for i, feat in enumerate(key_feats):
            with cols[i % 3]:
                values[feat] = st.number_input(
                    label_for(feat), value=None, min_value=0.0, step=1.0, format="%g", key=feat
                )

        if other_feats:
            with st.expander(f"All other SMART attributes ({len(other_feats)})"):
                cols = st.columns(4)
                for i, feat in enumerate(other_feats):
                    with cols[i % 4]:
                        values[feat] = st.number_input(
                            label_for(feat), value=None, min_value=0.0, step=1.0, format="%g", key=feat
                        )

        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        row = pd.DataFrame([{k: (np.nan if v is None else v) for k, v in values.items()}])
        n_given = int(row.notna().sum(axis=1).iloc[0])
        proba = float(predict_failure_proba(row)[0])
        band = risk_band(proba, threshold)

        c1, c2, c3 = st.columns(3)
        c1.metric("Failure probability", f"{proba:.1%}")
        c2.metric("Risk level", band)
        c3.metric("Values provided", f"{n_given} / {len(FEATURES)}")
        st.progress(min(proba, 1.0))

        if band == "High":
            st.error("High risk - consider backing up data and scheduling replacement.")
        elif band == "Elevated":
            st.warning("Elevated risk - monitor this drive closely.")
        else:
            st.success("Low risk based on the values provided.")

        if n_given < len(key_feats):
            st.caption("Few values were provided, so most inputs were imputed. Treat this estimate with caution.")

# ----------------------------------------------------------------------------
# Tab 2: batch prediction
# ----------------------------------------------------------------------------
with tab_batch:
    st.subheader("Score many drives at once")
    st.write(
        "Upload a CSV with one row per drive observation. Columns must be named like the training "
        "features (e.g. `smart_197_raw`). Extra columns such as `serial_number`, `date` or `model` are kept."
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        try:
            raw = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Could not read the file: {exc}")
            st.stop()

        present = [f for f in FEATURES if f in raw.columns]
        missing = [f for f in FEATURES if f not in raw.columns]

        if not present:
            st.error("None of the model's SMART columns were found in this file.")
            st.stop()
        if missing:
            st.warning(
                f"{len(missing)} of {len(FEATURES)} model features are missing from the file "
                "and will be imputed with training medians."
            )
            with st.expander("Show missing columns"):
                st.write(missing)

        result = raw.copy()
        result["failure_probability"] = predict_failure_proba(raw)
        result["risk_level"] = result["failure_probability"].apply(lambda p: risk_band(p, threshold))
        result = result.sort_values("failure_probability", ascending=False).reset_index(drop=True)

        n_high = int((result["risk_level"] == "High").sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows scored", f"{len(result):,}")
        c2.metric("High risk", f"{n_high:,}")
        c3.metric("Share flagged", f"{n_high / len(result):.2%}")

        st.dataframe(result.head(1000), use_container_width=True)
        if len(result) > 1000:
            st.caption("Showing the 1,000 highest-risk rows. The download contains everything.")

        st.download_button(
            "Download results (CSV)",
            data=result.to_csv(index=False).encode("utf-8"),
            file_name="drive_failure_predictions.csv",
            mime="text/csv",
        )

# ----------------------------------------------------------------------------
# Tab 3: model info
# ----------------------------------------------------------------------------
with tab_model:
    st.subheader("What drives the predictions")
    try:
        clf = model.named_steps["classifier"]
        importances = (
            pd.Series(clf.feature_importances_, index=FEATURES)
            .sort_values(ascending=False)
            .head(15)
        )
        importances.index = [label_for(f) for f in importances.index]
        st.bar_chart(importances)
        st.caption("Top 15 features by Random Forest importance.")

        params = clf.get_params()
        st.write(
            f"**Trees:** {params['n_estimators']}  |  **Max depth:** {params['max_depth']}  |  "
            f"**Class weight:** {params['class_weight']}"
        )
    except Exception:
        st.info("Feature importances are not available for this model.")

    st.markdown(
        "**Target:** whether the same drive fails at its next daily record.  \n"
        "**Validation:** chronological split (earlier 80% of dates for training, later 20% for testing)."
    )
