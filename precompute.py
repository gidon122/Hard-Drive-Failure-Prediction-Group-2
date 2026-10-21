"""Run this ONCE on your own computer (not on Streamlit Cloud).

It runs the full analysis from the notebook on clean.parquet and saves the
results (tables, charts, metrics) into the 'results' folder. The Streamlit
app then only displays those files, so it stays light enough to deploy.

Usage:  python precompute.py
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "clean.parquet"
MODEL_PATH = BASE_DIR / "drive_failure_prediction_model.pkl"
OUT = BASE_DIR / "results"
OUT.mkdir(exist_ok=True)


def save_table(df, name):
    df.to_csv(OUT / f"{name}.csv")


def make_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=100, class_weight="balanced",
            random_state=42, n_jobs=-1)),
    ])


def save_report(y_true, y_pred, prefix):
    cm = confusion_matrix(y_true, y_pred)
    save_table(pd.DataFrame(cm, index=["Actual 0", "Actual 1"],
                            columns=["Predicted 0", "Predicted 1"]), f"{prefix}_cm")
    rep = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    save_table(pd.DataFrame(rep).T.round(2), f"{prefix}_report")


print("Loading data...")
df = pd.read_parquet(DATA_PATH)
summary = {}

# 1. Overview
print("Overview...")
save_table(df.head().astype(str), "head")
save_table(df.isnull().sum().rename("missing").to_frame(), "missing")
summary["earliest"] = str(pd.to_datetime(df["date"].min()).date())
summary["latest"] = str(pd.to_datetime(df["date"].max()).date())
summary["unique_drives"] = int(df["serial_number"].nunique())
summary["unique_models"] = int(df["model"].nunique())

# 2. Failure distribution
print("Failure distribution...")
fc = df["failure"].value_counts().sort_index()
fp = (df["failure"].value_counts(normalize=True).sort_index() * 100).round(3)
save_table(pd.DataFrame({"count": fc, "percentage": fp}), "failure_dist")

fig, ax = plt.subplots(figsize=(8, 5))
sns.countplot(data=df, x="failure", ax=ax)
ax.set_title("Distribution of Hard Drive Failure Records")
ax.set_xlabel("Failure (0 = No Failure, 1 = Failure)")
ax.set_ylabel("Number of Records")
fig.tight_layout()
fig.savefig(OUT / "failure_countplot.png", dpi=120)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 5))
sns.boxplot(data=df, x="failure", y="smart_197_raw", ax=ax)
ax.set_title("Pending Sector Count By Recorded Failure Status")
ax.set_ylabel("SMART 197 Raw Value")
fig.tight_layout()
fig.savefig(OUT / "smart197_boxplot.png", dpi=120)
plt.close(fig)

# 3. Failed drives
print("Failed drives...")
failed_rows = df[df["failure"] == 1]
summary["n_failed_drives"] = int(failed_rows["serial_number"].nunique())
save_table(failed_rows[["date", "serial_number", "model", "failure"]]
           .head(10).astype(str).reset_index(drop=True), "failed_head")

df_sorted = df.sort_values(["serial_number", "date"])
failed_serials = df_sorted.loc[df_sorted["failure"] == 1, "serial_number"].unique()
sample = df_sorted[df_sorted["serial_number"].isin(failed_serials[:5])]
save_table(sample.groupby("serial_number").agg(
    first_obs=("date", "min"), last_obs=("date", "max"),
    number_of_observations=("date", "size"),
    failure_records=("failure", "sum")).astype(str), "failed_sample")

last_obs = df.sort_values("date").groupby("serial_number").tail(1)
save_table(last_obs["failure"].value_counts().rename("drives").to_frame(), "last_obs")
del df_sorted, sample, last_obs

# 4. Features
smart_features = [c for c in df.columns if c.startswith("smart_") and c.endswith("_raw")]
summary["smart_features"] = smart_features

# 5. Baseline model
print("Baseline model (this can take a while)...")
base = df.copy()
base["date"] = pd.to_datetime(base["date"])
base = base.sort_values("date").reset_index(drop=True)
split_date = base.loc[int(len(base) * 0.8), "date"]
tr, te = base[base["date"] < split_date], base[base["date"] >= split_date]
m = make_pipeline()
m.fit(tr[smart_features], tr["failure"])
yp = m.predict(te[smart_features])
ypr = m.predict_proba(te[smart_features])[:, 1]
save_report(te["failure"], yp, "baseline")
summary["baseline_pr_auc"] = round(float(average_precision_score(te["failure"], ypr)), 4)
del base, tr, te, m

# 6. Next-observation model
print("Next-observation model...")
data = df.copy()
data["date"] = pd.to_datetime(data["date"])
data = data.sort_values(["serial_number", "date"])
data["next_failure"] = data.groupby("serial_number")["failure"].shift(-1)
data = data.dropna(subset=["next_failure"]).copy()
data["next_failure"] = data["next_failure"].astype(int)
save_table(pd.DataFrame({
    "count": data["next_failure"].value_counts(),
    "proportion": data["next_failure"].value_counts(normalize=True).round(6),
}), "next_failure_dist")

split_date = data["date"].quantile(0.80)
train_df, test_df = data[data["date"] < split_date], data[data["date"] >= split_date]
X_train, y_train = train_df[smart_features], train_df["next_failure"]
X_test, y_test = test_df[smart_features], test_df["next_failure"]

if MODEL_PATH.exists():
    print("Loading saved model...")
    model = joblib.load(MODEL_PATH)
    summary["best_params"] = "Loaded from drive_failure_prediction_model.pkl"
else:
    print("Running grid search (slow)...")
    grid = GridSearchCV(
        make_pipeline(),
        {"classifier__n_estimators": [100, 200], "classifier__max_depth": [10, 20]},
        cv=3, scoring="recall", n_jobs=-1)
    grid.fit(X_train, y_train)
    model = grid.best_estimator_
    summary["best_params"] = {k: int(v) for k, v in grid.best_params_.items()}

save_report(y_test, model.predict(X_test), "final")

with open(OUT / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("Done. Results saved in:", OUT)
