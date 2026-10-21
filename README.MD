# Hard Drive Failure Prediction

A Streamlit web app that explores hard drive health data and predicts drive failures with a Random Forest classifier. It is the deployable version of the `group2_model.ipynb` notebook (SIWES Group 2 project).

## Overview

Hard drives report SMART health attributes (Self-Monitoring, Analysis and Reporting Technology) every day. This project uses those daily readings to answer two questions:

1. Can the SMART readings on a given day tell us whether a drive is failing?
2. Can they tell us whether a drive will fail on its **next** recorded day?

The app walks through the full analysis: data exploration, failure distribution, a baseline model, and a tuned next-observation model.

## How the Project Is Organized

Training a Random Forest on about 1.4 million rows is too heavy for Streamlit Cloud's free tier, so the work is split in two:

| Step | File | Where it runs | What it does |
|---|---|---|---|
| 1 | `precompute.py` | Your own computer, once | Runs the full analysis on `clean.parquet` and saves tables, charts, and metrics into the `results/` folder |
| 2 | `app.py` | Streamlit Cloud | Only displays the saved files in `results/`, so it is fast and light |

The deployed app does not need the dataset or the saved model.

## Dataset

`precompute.py` reads `clean.parquet`, which must sit in the same folder as the script.

| Detail | Value |
|---|---|
| Rows | 1,766,030 |
| Date range | 2024-01-01 to 2024-03-31 (91 days) |
| Unique drives | 19,511 |
| Drive models | 1 (ST12000NM0008) |
| Failure records | 144 (about 0.008% of rows) |

Each row is **one drive on one day**.

| Column | Description |
|---|---|
| `date` | Date of the daily reading |
| `serial_number` | Unique drive identifier |
| `model` | Drive model |
| `failure` | 1 on the day the drive failed, otherwise 0 |
| `smart_5_raw` | Reallocated sector count |
| `smart_9_raw` | Power-on hours |
| `smart_187_raw` | Reported uncorrectable errors |
| `smart_188_raw` | Command timeout count |
| `smart_194_raw` | Temperature (°C) |
| `smart_197_raw` | Current pending sector count |
| `smart_198_raw` | Offline uncorrectable sector count |

### Things worth knowing about the data

- **Extreme class imbalance.** Only 144 of 1.77 million rows are failures, each from a different drive. A model that always predicts "no failure" would be 99.99% accurate, so accuracy is not a useful metric here. Recall and precision matter instead.
- **Failing drives look different.** For example, about 71% of failure rows have a non-zero reallocated sector count, compared with about 4.6% of healthy rows. Still, about 12% of failed drives show no error readings at all, so these columns cannot catch every failure.
- **`smart_197_raw` and `smart_198_raw` are identical** in every row, so the second adds no new information.
- **`smart_188_raw` has very large values** (up to about 10^12) because the drive packs several counters into one number. Treat it as a signal, not a literal count.
- **Drives stop reporting after failing.** Only 3 rows come after a drive's failure row.
- **No missing values** in the dataset.

## What the App Shows

1. **Data Overview**: data preview, missing values, date range, and drive and model counts.
2. **Failure Distribution**: failure counts and percentages, a count plot, and a box plot of SMART 197 by failure status.
3. **Failed Drives**: number of unique failed drives, sample failure records, and first and last observations for a sample of failed drives.
4. **SMART Features Used**: the feature columns given to the model.
5. **Baseline Model**: a Random Forest trained on earlier dates and tested on later dates, with a confusion matrix, classification report, and PR-AUC.
6. **Next-Observation Model**: a tuned Random Forest that predicts whether a drive fails on its next recorded day, with a confusion matrix and classification report.

## Methodology

**Chronological split.** Data is split by time, not randomly. The earliest 80% of dates train the model and the latest 20% test it. This mirrors real use, where a model predicts the future from the past.

**Model pipeline.**
1. `SimpleImputer` (median strategy) fills any missing values.
2. `RandomForestClassifier` with `class_weight="balanced"` to handle the rare failure class.

**Baseline.** Predicts the `failure` label of the same row (100 trees).

**Next-observation target.** For each drive, readings are sorted by date and a `next_failure` label is created by shifting `failure` back one row. Each row's features are used to predict whether the drive fails on its following record. The last row of each drive has no next record and is dropped.

**Hyperparameter tuning.** `GridSearchCV` with 3-fold cross-validation, scored on recall:

| Parameter | Values tried |
|---|---|
| `n_estimators` | 100, 200 |
| `max_depth` | 10, 20 |

Best parameters found: `max_depth=10`, `n_estimators=200`.

## Results

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

Accuracy is about 99% in both cases, but that number is misleading because almost every row is a non-failure. The tuned model catches far more real failures than the baseline (recall rose from 0.08 to 0.48), but it does so by raising many false alarms. The test set contains only 25 failures, so these metrics are noisy and could change with a different split. The results are a starting point, not a production-ready predictor.

## Project Structure

```
.
├── app.py              # Streamlit app (displays saved results only)
├── precompute.py       # Runs the analysis locally and writes results/
├── requirements.txt    # Dependencies for the deployed app
├── results/            # Saved tables, charts, and metrics (committed to GitHub)
│   ├── summary.json
│   ├── failure_countplot.png
│   ├── smart197_boxplot.png
│   └── *.csv
└── README.md
```

`clean.parquet` and `drive_failure_prediction_model.pkl` are needed only to run `precompute.py` on your computer. They do not need to be in the GitHub repo.

## Run It Yourself

### 1. Generate the results (once, on your computer)

Put `clean.parquet` (and, optionally, `drive_failure_prediction_model.pkl`) in the same folder as `precompute.py`.

```bash
pip install pandas pyarrow scikit-learn seaborn matplotlib joblib streamlit
python precompute.py
```

This takes several minutes. When it prints `Done. Results saved in: ...results`, a `results/` folder appears. If the `.pkl` file is present, the script loads it instead of running the slow grid search.

### 2. Test the app locally

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`. Press `Ctrl + C` to stop it.

## Deploy on Streamlit Community Cloud

1. Push `app.py`, `requirements.txt`, `precompute.py`, `README.md`, and the whole `results/` folder to a GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **Create app**, choose your repo and branch, and set the main file to `app.py`.
4. Click **Deploy**.

### Troubleshooting

- **"The 'results' folder is missing":** run `precompute.py` locally, then commit the `results/` folder (all 14 files) to GitHub.
- **`ModuleNotFoundError` on Streamlit Cloud:** make sure `requirements.txt` is in the repo, in the same folder as `app.py`, and lists `streamlit` and `pandas`.
- **App crashes while loading:** the old version trained models inside the app and ran out of memory. Use the current `app.py`, which only reads from `results/`.
- **App doesn't update after a push:** open **Manage app** and choose **Reboot app**.
- **Run `precompute.py` again** only if your data or model changes, then commit the new `results/` folder.

## Tech Stack

- Python
- Streamlit
- pandas
- scikit-learn, matplotlib, seaborn, joblib, and PyArrow (used by `precompute.py`)

## Limitations and Possible Improvements

- Only 144 failure records exist, and only 25 are in the test set, so the metrics are very noisy.
- The data covers a single drive model over three months.
- Precision for the failure class is very low. Ideas to improve it:
  - Add rolling or trend features (for example, the change in SMART 5 or 197 over the last 7 days).
  - Predict failure within the next N days instead of the next day only.
  - Tune the decision threshold using the precision-recall curve.
  - Try other approaches for imbalance, such as resampling or gradient boosting.
- Remove the duplicate `smart_198_raw` feature, since it is identical to `smart_197_raw`.

## Acknowledgements

Built as part of the SIWES Group 2 project. The data follows the SMART format used in public drive-reliability datasets such as Backblaze's.