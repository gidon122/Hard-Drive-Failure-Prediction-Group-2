"""
Train the hard-drive failure model (converted from group2_model.ipynb).

Target : next_failure  -> does the SAME drive fail at its next daily record?
Model  : median imputer + RandomForest, tuned with GridSearchCV
Output : drive_failure_prediction_model.pkl  (loaded by app.py)

Run locally (NOT on Streamlit Cloud - it needs the big parquet file):
    python train.py
    python train.py --data clean.parquet --out drive_failure_prediction_model.pkl
"""
import argparse

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# --- settings (same values as the notebook) ---------------------------------
TRAIN_FRACTION = 0.80          # earlier 80% of dates -> train, later 20% -> test
RANDOM_STATE = 42
PARAM_GRID = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [10, 20],
}
CV_FOLDS = 3
SCORING = "recall"             # as in the notebook; "average_precision" is a safer choice for rare failures


def build_next_failure_dataset(path: str) -> pd.DataFrame:
    """Load data and label each row with whether the same drive fails at its next record."""
    data = pd.read_parquet(path)
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values(["serial_number", "date"])

    data["next_failure"] = data.groupby("serial_number")["failure"].shift(-1)
    # Each drive's last row has no "next" record -> drop it
    data = data.dropna(subset=["next_failure"]).copy()
    data["next_failure"] = data["next_failure"].astype(int)
    return data


def get_smart_features(data: pd.DataFrame) -> list:
    return [
        col for col in data.columns
        if col.startswith("smart_") and col.endswith("_raw")
    ]


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])


def main(data_path: str, model_path: str) -> None:
    data = build_next_failure_dataset(data_path)
    smart_features = get_smart_features(data)

    print("Next-failure class counts:")
    print(data["next_failure"].value_counts())
    print(data["next_failure"].value_counts(normalize=True).round(6))
    print(f"\n{len(smart_features)} SMART features:", smart_features)

    # Chronological split: earlier dates train, later dates test
    split_date = data["date"].quantile(TRAIN_FRACTION)
    train_df = data[data["date"] < split_date].copy()
    test_df = data[data["date"] >= split_date].copy()

    X_train, y_train = train_df[smart_features], train_df["next_failure"]
    X_test, y_test = test_df[smart_features], test_df["next_failure"]
    print(f"\nTrain rows: {len(train_df):,} | Test rows: {len(test_df):,} | split date: {split_date}")

    # Hyperparameter search
    grid_search = GridSearchCV(
        build_pipeline(),
        PARAM_GRID,
        cv=CV_FOLDS,
        scoring=SCORING,
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)
    print("\nBest parameters:", grid_search.best_params_)
    model = grid_search.best_estimator_

    # Evaluation on the held-out, later-in-time data
    y_pred = model.predict(X_test)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    y_proba = model.predict_proba(X_test)[:, 1]
    print("Average precision (PR-AUC):", round(average_precision_score(y_test, y_proba), 4))

    # Save (compressed to keep the file small enough for GitHub / Streamlit Cloud)
    joblib.dump(model, model_path, compress=3)
    print(f"\nSaved model to {model_path}")
    print(f"scikit-learn version used: {sklearn.__version__}  <-- pin this in requirements.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="clean.parquet")
    parser.add_argument("--out", default="drive_failure_prediction_model.pkl")
    args = parser.parse_args()
    main(args.data, args.out)
