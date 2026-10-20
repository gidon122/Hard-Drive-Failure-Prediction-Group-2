# Hard Drive Failure Prediction

A Streamlit web app that explores hard drive health data and predicts drive failures using a Random Forest classifier. It is the deployable version of the `group2_model.ipynb` notebook (SIWES Group 2 project).

## Overview

Hard drives report SMART health attributes (Self-Monitoring, Analysis and Reporting Technology) every day. This project uses those daily readings to answer two questions:

1. Can the SMART readings on a given day tell us whether a drive is failing?
2. Can they tell us whether a drive will fail on its **next** recorded day?

The app walks through the full analysis: data exploration, failure distribution, a baseline model, and a tuned next-observation model.

## Dataset

The app reads `clean.parquet`, which must sit in the same folder as `app.py`. The app never asks for an upload.

| Detail | Value |
|---|---|
| Rows | 1,766,030 |
| Date range | 2024-01-01 to 2024-03-31 |
| Unique drives | 19,511 |
| Drive models | 1 (ST12000NM0008) |
| Failure records | 144 (about 0.008% of rows) |

**Columns**

| Column | Description |
|---|---|
| `date` | Date of the daily reading |
| `serial_number` | Unique drive identifier |
| `model` | Drive model |
| `failure` | 1 if the drive failed on that day, otherwise 0 |
| `smart_5_raw` | Reallocated sector count |
| `smart_9_raw` | Power-on hours |
| `smart_187_raw` | Reported uncorrectable errors |
| `smart_188_raw` | Command timeout count |
| `smart_194_raw` | Temperature |
| `smart_197_raw` | Current pending sector count |
| `smart_198_raw` | Offline uncorrectable sector count |

The dataset has no missing values. Failures are extremely rare, so this is a highly imbalanced classification problem.

## What the App Shows

1. **Data Overview**: data preview, missing values per column, date range, and drive and model counts.
2. **Failure Distribution**: failure counts and percentages, a count plot of failure records, and a box plot of SMART 197 (pending sectors) by failure status.
3. **Failed Drives**: the number of unique failed drives, a sample of failure records, and the first and last observation for a sample of failed drives.
4. **SMART Features Used**: the list of feature columns fed to the model.
5. **Baseline Model**: a Random Forest trained on earlier dates and tested on later dates, with a confusion matrix, classification report, and PR-AUC.
6. **Next-Observation Model**: a tuned Random Forest that predicts whether a drive fails on its next recorded day, with a confusion matrix and classification report.

## Methodology

**Chronological split.** Data is split by time, not randomly. The earliest 80% of dates are used for training and the latest 20% for testing. This mirrors real use, where a model predicts the future from the past.

**Model pipeline.**
1. `SimpleImputer` (median strategy) fills any missing values.
2. `RandomForestClassifier` with `class_weight="balanced"` to handle the rare failure class.

**Baseline.** Predicts the `failure` label of the same row (100 trees).

**Next-observation target.** For each drive, readings are sorted by date and a `next_failure` label is created by shifting `failure` back one row. Each row's features are then used to predict whether the drive fails on its following record. The last row of each drive has no next record and is dropped.

**Hyperparameter tuning.** `GridSearchCV` with 3-fold cross-validation, scored on recall:

| Parameter | Values tried |
|---|---|
| `n_estimators` | 100, 200 |
| `max_depth` | 10, 20 |

Best parameters found: `max_depth=10`, `n_estimators=200`.

## Results from the Notebook

**Baseline model** (test set: 368,042 rows, 25 failures)

| Metric | Value |
|---|---|
| Failure class precision | 0.10 |
| Failure class recall | 0.08 |
| PR-AUC | 0.0155 |

**Next-observation model, tuned** (test set: 368,034 rows, 25 failures)

| Metric | Value |
|---|---|
| Failure class recall | 0.48 (12 of 25 caught) |
| Failure class precision | about 0.00 (3,366 false alarms) |

### How to read these results

Accuracy is about 99% in both cases, but that number is misleading because almost every row is a non-failure. The tuned model catches far more real failures than the baseline (recall rose from 0.08 to 0.48), but it does so by raising many false alarms. These results show the problem is hard with SMART readings alone. They are a starting point, not a production-ready predictor.

## Project Structure

```
.
├── app.py                                 # Streamlit app
├── requirements.txt                       # Python dependencies
├── clean.parquet                          # Dataset (must be in the same folder)
├── drive_failure_prediction_model.pkl     # Saved model (optional, speeds up startup)
└── README.md
```

## Run Locally

1. Clone the repo and move into the folder.

   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>
   ```

2. (Recommended) Create a virtual environment.

   ```bash
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   ```

3. Install dependencies.

   ```bash
   pip install -r requirements.txt
   ```

4. Run the app.

   ```bash
   streamlit run app.py
   ```

The app opens at `http://localhost:8501`.

## Deploy on Streamlit Community Cloud

1. Push `app.py`, `requirements.txt`, `clean.parquet`, and (optionally) `drive_failure_prediction_model.pkl` to a GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **Create app**, choose your repo and branch, and set the main file to `app.py`.
4. Click **Deploy**.

### Deployment notes

- **File size:** GitHub blocks files over 100 MB. If `clean.parquet` is larger, use [Git LFS](https://git-lfs.com).
- **Memory:** The dataset has about 1.77 million rows and the baseline model trains on most of them. Streamlit Cloud's free tier has limited memory, so the app can crash during loading or training. If that happens, use a smaller sample of the data or precompute the results.
- **Startup time:** The first load is slow because the models train once. Results are cached with `st.cache_data` and `st.cache_resource`, so later interactions are fast.
- **Saved model:** If `drive_failure_prediction_model.pkl` is in the repo, the app loads it instead of re-running the grid search. Make sure it was saved with the same scikit-learn version listed in your deployment. Pinning versions in `requirements.txt` avoids load errors.

## Tech Stack

- Python
- Streamlit
- pandas, NumPy, PyArrow
- scikit-learn
- matplotlib, seaborn
- joblib

## Limitations and Possible Improvements

- Only 144 failure records exist, and only 25 are in the test set, so the metrics are very noisy.
- The data covers a single drive model over three months.
- Precision for the failure class is very low. Ideas to improve it:
  - Add rolling or trend features (for example, the change in SMART 5 or 197 over the last 7 days).
  - Predict failure within the next N days instead of the next day only.
  - Tune the decision threshold using the precision-recall curve.
  - Try other approaches for imbalance, such as resampling or gradient boosting.

## Acknowledgements

Built as part of the SIWES Group 2 project. The hard drive data follows the SMART format used in public drive-reliability datasets such as Backblaze's.
