from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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

import streamlit as st

st.set_page_config(page_title="Hard Drive Failure Prediction", layout="wide")

st.header("Hard Drive Failure Prediction")

st.write(
    "A Streamlit web app that explores hard drive health data and predicts "
    "drive failures using a Random Forest classifier."
)
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "clean.parquet"
MODEL_PATH = BASE_DIR / "drive_failure_prediction_model.pkl"


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_parquet(DATA_PATH)


def make_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])


def show_report(y_true, y_pred):
    st.write("**Confusion Matrix**")
    cm = confusion_matrix(y_true, y_pred)
    st.dataframe(
        pd.DataFrame(
            cm,
            index=["Actual 0", "Actual 1"],
            columns=["Predicted 0", "Predicted 1"],
        )
    )
    st.write("**Classification Report**")
    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    st.dataframe(pd.DataFrame(report).T.round(2))


@st.cache_resource(show_spinner="Training baseline model...")
def train_baseline(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    smart = [c for c in df.columns if c.startswith("smart_") and c.endswith("_raw")]

    split_date = df.loc[int(len(df) * 0.8), "date"]
    train_df = df[df["date"] < split_date]
    test_df = df[df["date"] >= split_date]

    model = make_pipeline()
    model.fit(train_df[smart], train_df["failure"])

    y_test = test_df["failure"]
    y_pred = model.predict(test_df[smart])
    y_proba = model.predict_proba(test_df[smart])[:, 1]
    return y_test, y_pred, average_precision_score(y_test, y_proba)


@st.cache_resource(show_spinner="Preparing final model...")
def get_final_model(X_train, y_train):
    # Use the saved model if it is in the folder, otherwise train it.
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH), None
    grid = GridSearchCV(
        make_pipeline(),
        {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [10, 20],
        },
        cv=3,
        scoring="recall",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_


# ----------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------
st.title("Hard Drive Failure Prediction")

if not DATA_PATH.exists():
    st.error("clean.parquet was not found next to app.py.")
    st.stop()

df = load_data()
smart_features = [c for c in df.columns if c.startswith("smart_") and c.endswith("_raw")]

# 1. Data overview ------------------------------------------------------
st.header("1. Data Overview")
st.dataframe(df.head())

st.write("**Missing values per column**")
st.dataframe(df.isnull().sum().rename("missing"))

st.write("**Date range and drive counts**")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Earliest date", str(df["date"].min().date()))
c2.metric("Latest date", str(df["date"].max().date()))
c3.metric("Unique drives", f"{df['serial_number'].nunique():,}")
c4.metric("Unique models", df["model"].nunique())

# 2. Failure distribution ----------------------------------------------
st.header("2. Failure Distribution")
failure_count = df["failure"].value_counts().sort_index()
failure_pct = (df["failure"].value_counts(normalize=True).sort_index() * 100).round(3)
st.dataframe(pd.DataFrame({"count": failure_count, "percentage": failure_pct}))

col_a, col_b = st.columns(2)
with col_a:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df, x="failure", ax=ax)
    ax.set_title("Distribution of Hard Drive Failure Records")
    ax.set_xlabel("Failure (0 = No Failure, 1 = Failure)")
    ax.set_ylabel("Number of Records")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with col_b:
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=df, x="failure", y="smart_197_raw", ax=ax)
    ax.set_title("Pending Sector Count By Recorded Failure Status")
    ax.set_ylabel("SMART 197 Raw Value")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# 3. Failed drives ------------------------------------------------------
st.header("3. Failed Drives")
failed_rows = df[df["failure"] == 1]
st.write(f"Number of unique failed drives: **{failed_rows['serial_number'].nunique()}**")
st.dataframe(failed_rows[["date", "serial_number", "model", "failure"]].head(10))

df_sorted = df.sort_values(["serial_number", "date"])
failed_serials = df_sorted.loc[df_sorted["failure"] == 1, "serial_number"].unique()
sample_serials = failed_serials[:5]

st.write("**First and last observation for a sample of failed drives**")
st.dataframe(
    df_sorted[df_sorted["serial_number"].isin(sample_serials)]
    .groupby("serial_number")
    .agg(
        first_obs=("date", "min"),
        last_obs=("date", "max"),
        number_of_observations=("date", "size"),
        failure_records=("failure", "sum"),
    )
)

last_obs = df.sort_values("date").groupby("serial_number").tail(1)
st.write("**Final observation label for every drive**")
st.dataframe(last_obs["failure"].value_counts().rename("drives"))

# 4. SMART features -----------------------------------------------------
st.header("4. SMART Features Used")
st.write(smart_features)

# 5. Baseline model -----------------------------------------------------
st.header("5. Baseline Model (chronological 80/20 split)")
y_test_b, y_pred_b, pr_auc = train_baseline(df)
show_report(y_test_b, y_pred_b)
st.metric("Average precision (PR-AUC)", round(pr_auc, 4))

# 6. Next-observation model --------------------------------------------
st.header("6. Next-Observation Failure Model")
data = df.copy()
data["date"] = pd.to_datetime(data["date"])
data = data.sort_values(["serial_number", "date"])
data["next_failure"] = data.groupby("serial_number")["failure"].shift(-1)
data = data.dropna(subset=["next_failure"]).copy()
data["next_failure"] = data["next_failure"].astype(int)

nf_counts = data["next_failure"].value_counts()
nf_pct = data["next_failure"].value_counts(normalize=True).round(6)
st.dataframe(pd.DataFrame({"count": nf_counts, "proportion": nf_pct}))

split_date = data["date"].quantile(0.80)
train_df = data[data["date"] < split_date]
test_df = data[data["date"] >= split_date]
X_train, y_train = train_df[smart_features], train_df["next_failure"]
X_test, y_test = test_df[smart_features], test_df["next_failure"]

model, best_params = get_final_model(X_train, y_train)
if best_params:
    st.write("**Best parameters:**", best_params)
else:
    st.write("**Best parameters:**", model.get_params().get("classifier__n_estimators"),
             "trees, max depth", model.get_params().get("classifier__max_depth"))

y_pred = model.predict(X_test)
show_report(y_test, y_pred)