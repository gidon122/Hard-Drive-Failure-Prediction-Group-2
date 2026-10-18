"""
Hard Drive Failure Prediction - single-file Streamlit app (converted from group2_model.ipynb).

What it does
  * Loads the dataset from  clean.parquet  (same relative path the notebook uses; it must sit
    next to this file)
  * Trains the notebook's final model (predict whether a drive fails at its NEXT daily record):
    median imputer + RandomForest, tuned with GridSearchCV, chronological 80/20 split
  * Saves it as drive_failure_prediction_model.pkl and reuses it on later runs
  * Serves predictions for a single drive or a CSV of drives

Run:  streamlit run app.py
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import sklearn
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# ----------------------------------------------------------------------------
# Config (values taken from the notebook)
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "clean.parquet"                          # notebook: pd.read_parquet("clean.parquet")
MODEL_PATH = BASE_DIR / "drive_failure_prediction_model.pkl"    # notebook: joblib.dump(...)

TRAIN_FRACTION = 0.80
RANDOM_STATE = 42
PARAM_GRID = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [10, 20],
}
CV_FOLDS = 3
SCORING = "recall"   # as in the notebook; "average_precision" is a safer choice for rare failures

# Friendly names for common SMART attributes (unknown ones fall back to "SMART <id>")
SMART_LABELS = {
    1: "Read error rate", 3: "Spin-up time", 4: "Start/stop count",
    5: "Reallocated sectors count", 7: "Seek error rate", 9: "Power-on hours",
    10: "Spin retry count", 12: "Power cycle count",
    187: "Reported uncorrectable errors", 188: "Command timeout", 189: "High fly writes",
    190: "Airflow temperature", 192: "Power-off retract count", 193: "Load cycle count",
    194: "Temperature (°C)", 197: "Current pending sector count",
    198: "Offline uncorrectable sectors", 199: "UDMA CRC error count",
}
KEY_ATTRIBUTES = [5, 187, 188, 197, 198, 9, 194]   # shown up front in the single-drive form

st.set_page_config(page_title="Drive Failure Predictor", page_icon="💽", layout="wide")


def smart_id(feature: str) -> int:
    return int(feature.split("_")[1])


def label_for(feature: str) -> str:
    sid = smart_id(feature)
    return f"{SMART_LABELS.get(sid, 'SMART ' + str(sid))} (SMART {sid})"


# ----------------------------------------------------------------------------
# Data loading (reads only the columns it needs, to keep memory low)
# ----------------------------------------------------------------------------
def smart_columns(all_columns) -> list:
    return [c for c in all_columns if c.startswith("smart_") and c.endswith("_raw")]


@st.cache_resource(show_spinner="Reading clean.parquet ...")   # cache_resource: no per-run copy of a big frame
def load_dataset(path: str, extra_cols: tuple = ()) -> pd.DataFrame:
    """Load clean.parquet with only: date, serial_number, failure, SMART raw columns (+ extras)."""
    names = pq.ParquetFile(path).schema.names
    wanted = [c for c in ["date", "serial_number", "failure", *extra_cols] if c in names]
    wanted += smart_columns(names)
    df = pd.read_parquet(path, columns=list(dict.fromkeys(wanted)))
    df["date"] = pd.to_datetime(df["date"])
    return df


def build_next_failure_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Label each row with whether the same drive fails at its next record (notebook: next_failure)."""
    data = df.sort_values(["serial_number", "date"]).copy()
    data["next_failure"] = data.groupby("serial_number")["failure"].shift(-1)
    data = data.dropna(subset=["next_failure"]).copy()   # each drive's last row has no "next"
    data["next_failure"] = data["next_failure"].astype(int)
    return data


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def build_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
        )),
    ])


def train_model(df: pd.DataFrame):
    """Chronological split -> GridSearchCV -> evaluate on later data. Returns (model, metrics)."""
    data = build_next_failure_dataset(df)
    features = smart_columns(data.columns)

    split_date = data["date"].quantile(TRAIN_FRACTION)
    train_df = data[data["date"] < split_date]
    test_df = data[data["date"] >= split_date]
    X_train, y_train = train_df[features], train_df["next_failure"]
    X_test, y_test = test_df[features], test_df["next_failure"]

    if y_train.nunique() < 2:
        raise ValueError("The training period contains no failure examples, so a model cannot be trained.")

    grid = GridSearchCV(build_pipeline(), PARAM_GRID, cv=CV_FOLDS, scoring=SCORING, n_jobs=-1)
    grid.fit(X_train, y_train)
    model = grid.best_estimator_

    y_pred = model.predict(X_test)
    proba_idx = list(model.classes_).index(1)
    y_proba = model.predict_proba(X_test)[:, proba_idx]
    metrics = {
        "best_params": grid.best_params_,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "split_date": str(split_date.date()),
        "positives_train": int(y_train.sum()),
        "positives_test": int(y_test.sum()),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1]),
        "report": pd.DataFrame(
            classification_report(y_test, y_pred, zero_division=0, output_dict=True)
        ).T,
        "pr_auc": float(average_precision_score(y_test, y_proba)) if y_test.sum() > 0 else None,
    }
    return model, metrics


def train_and_save():
    df = load_dataset(str(DATA_PATH))
    model, metrics = train_model(df)
    joblib.dump(model, MODEL_PATH, compress=3)
    load_model.clear()
    st.session_state["train_metrics"] = metrics
    st.session_state["train_sklearn"] = sklearn.__version__


def show_training_metrics(metrics: dict):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Train rows", f"{metrics['n_train']:,}")
    c2.metric("Test rows", f"{metrics['n_test']:,}")
    c3.metric("Failures in test", f"{metrics['positives_test']:,}")
    c4.metric("PR-AUC", "n/a" if metrics["pr_auc"] is None else f"{metrics['pr_auc']:.4f}")
    st.caption(f"Train = dates before {metrics['split_date']}; test = later dates. "
               f"Best parameters: {metrics['best_params']}")
    cm = pd.DataFrame(metrics["confusion_matrix"],
                      index=["Actual: no failure", "Actual: failure"],
                      columns=["Predicted: no failure", "Predicted: failure"])
    left, right = st.columns(2)
    left.write("**Confusion matrix**")
    left.dataframe(cm)
    right.write("**Classification report**")
    right.dataframe(metrics["report"].round(3))


# ----------------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model ...")
def load_model():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


st.title("💽 Hard Drive Failure Predictor")
st.write(
    "Estimates the probability that a drive **fails at its next daily record** from its "
    "SMART raw values, using a Random Forest trained on historical drive telemetry."
)

try:
    model = load_model()
except Exception as exc:
    st.error(f"Could not load `{MODEL_PATH.name}`: {exc}\n\n"
             "This usually means scikit-learn on the server differs from the version used to train. "
             "Pin the training version in requirements.txt.")
    st.stop()

# --- No model yet: offer to train it from clean.parquet and stop -------------
if model is None:
    st.warning(f"No trained model found (`{MODEL_PATH.name}`).")
    if not DATA_PATH.exists():
        st.error(f"Dataset not found at `{DATA_PATH}`. Place `clean.parquet` next to `app.py`, "
                 f"or add a pre-trained `{MODEL_PATH.name}`.")
        st.stop()
    st.write(f"Dataset found: `{DATA_PATH.name}`. Training runs the full grid search "
             "(4 settings x 3 folds) and can take a while on a large file.")
    if st.button("Train model from clean.parquet", type="primary"):
        with st.spinner("Training ... please wait"):
            try:
                train_and_save()
            except Exception as exc:
                st.error(f"Training failed: {exc}")
                st.stop()
        st.rerun()
    st.stop()



if not hasattr(model, "feature_names_in_"):
    st.error("The saved model has no stored feature names. Retrain it from the Model tab.")
    st.stop()

FEATURES = list(model.feature_names_in_)
POSITIVE_IDX = list(model.classes_).index(1)


def predict_failure_proba(df: pd.DataFrame) -> np.ndarray:
    """Align columns to training features; missing values are imputed by the pipeline."""
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
with st.sidebar:
    st.header("Settings")
    threshold = st.slider(
        "Alert threshold", min_value=0.01, max_value=0.99, value=0.50, step=0.01,
        help="Drives with failure probability at or above this value are flagged High risk. "
             "Failures are rare, so lowering it catches more failures at the cost of more false alarms.",
    )
    st.caption(f"Model uses {len(FEATURES)} SMART raw features.")
    st.info("Decision-support tool only. It is not a guarantee that a drive will or won't fail.")

tab_single, tab_batch, tab_data, tab_model = st.tabs(
    ["Single drive", "Batch (CSV)", "Dataset", "Model"]
)

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
                    label_for(feat), value=None, min_value=0.0, step=1.0, format="%g", key=feat)

        if other_feats:
            with st.expander(f"All other SMART attributes ({len(other_feats)})"):
                cols = st.columns(4)
                for i, feat in enumerate(other_feats):
                    with cols[i % 4]:
                        values[feat] = st.number_input(
                            label_for(feat), value=None, min_value=0.0, step=1.0, format="%g", key=feat)

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
    st.write("Upload a CSV with one row per drive observation. Columns must be named like the "
             "training features (e.g. `smart_197_raw`). Extra columns such as `serial_number` are kept.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        try:
            raw = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Could not read the file: {exc}")
            raw = None

        if raw is not None:
            missing = [f for f in FEATURES if f not in raw.columns]
            if len(missing) == len(FEATURES):
                st.error("None of the model's SMART columns were found in this file.")
            else:
                if missing:
                    st.warning(f"{len(missing)} of {len(FEATURES)} model features are missing from the file "
                               "and will be imputed with training medians.")
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
                    file_name="drive_failure_predictions.csv", mime="text/csv",
                )

# ----------------------------------------------------------------------------
# Tab 3: dataset overview (the notebook's EDA, from clean.parquet)
# ----------------------------------------------------------------------------
with tab_data:
    st.subheader("Dataset overview")
    if not DATA_PATH.exists():
        st.info(f"`{DATA_PATH.name}` is not available in this deployment, so the dataset overview is hidden. "
                "Predictions still work from the saved model.")
    elif not st.toggle("Load dataset overview", value=False):
        st.caption("Loading the full dataset is memory-heavy, so it only runs when you switch this on.")
    else:
        df = load_dataset(str(DATA_PATH), extra_cols=("model",))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Records", f"{len(df):,}")
        c2.metric("Unique drives", f"{df['serial_number'].nunique():,}")
        c3.metric("Drive models", f"{df['model'].nunique():,}" if "model" in df.columns else "n/a")
        c4.metric("Failure records", f"{int(df['failure'].sum()):,}")
        st.caption(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}  |  "
                   f"Failure rate: {df['failure'].mean() * 100:.4f}%")

        left, right = st.columns(2)
        with left:
            st.write("**Class distribution (0 = no failure, 1 = failure)**")
            st.bar_chart(df["failure"].value_counts().sort_index())
        with right:
            st.write("**Pending sector count (SMART 197) by failure status**")
            if "smart_197_raw" in df.columns:
                # all failures + a sample of healthy rows so the plot stays fast
                healthy = df[df["failure"] == 0]
                sample = pd.concat([
                    df[df["failure"] == 1],
                    healthy.sample(min(len(healthy), 100_000), random_state=RANDOM_STATE),
                ])
                fig, ax = plt.subplots(figsize=(5, 3.5))
                groups = [sample.loc[sample["failure"] == k, "smart_197_raw"].dropna() for k in (0, 1)]
                ax.boxplot(groups, tick_labels=["No failure", "Failure"])
                ax.set_yscale("symlog")
                ax.set_ylabel("SMART 197 raw (symlog)")
                st.pyplot(fig)
                plt.close(fig)
                st.caption("Healthy rows are sampled (up to 100k) for speed.")
            else:
                st.info("smart_197_raw not found in the dataset.")

        st.write("**Missing values per SMART feature**")
        nulls = df[smart_columns(df.columns)].isnull().sum()
        st.bar_chart(nulls[nulls > 0].sort_values(ascending=False).head(25))

# ----------------------------------------------------------------------------
# Tab 4: model info + retraining
# ----------------------------------------------------------------------------
with tab_model:
    st.subheader("What drives the predictions")
    try:
        clf = model.named_steps["classifier"]
        importances = pd.Series(clf.feature_importances_, index=FEATURES).sort_values(ascending=False).head(15)
        importances.index = [label_for(f) for f in importances.index]
        st.bar_chart(importances)
        st.caption("Top 15 features by Random Forest importance.")
        p = clf.get_params()
        st.write(f"**Trees:** {p['n_estimators']}  |  **Max depth:** {p['max_depth']}  |  "
                 f"**Class weight:** {p['class_weight']}")
    except Exception:
        st.info("Feature importances are not available for this model.")

    if "train_metrics" in st.session_state:
        st.subheader("Latest training run")
        show_training_metrics(st.session_state["train_metrics"])
        st.info(f"Trained with scikit-learn {st.session_state['train_sklearn']}. "
                "Pin this version in requirements.txt before deploying.")

    st.subheader("Retrain")
    if DATA_PATH.exists():
        st.caption("Retrains from clean.parquet and overwrites the saved model. "
                   "Note: on Streamlit Cloud the disk is temporary, so train locally and commit the .pkl.")
        if st.button("Retrain model"):
            with st.spinner("Training ... please wait"):
                try:
                    train_and_save()
                except Exception as exc:
                    st.error(f"Training failed: {exc}")
                else:
                    st.rerun()
    else:
        st.caption("Retraining needs clean.parquet next to app.py.")
