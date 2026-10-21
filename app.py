import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Hard Drive Failure Prediction", layout="wide")

RESULTS = Path(__file__).parent / "results"


@st.cache_data
def table(name, index=True):
    return pd.read_csv(RESULTS / f"{name}.csv", index_col=0 if index else None)


st.title("Hard Drive Failure Prediction")

if not (RESULTS / "summary.json").exists():
    st.error("The 'results' folder is missing. Run precompute.py locally and "
             "commit the 'results' folder next to app.py.")
    st.stop()

summary = json.loads((RESULTS / "summary.json").read_text())

# 1. Data overview
st.header("1. Data Overview")
st.dataframe(table("head", index=True))
st.write("**Missing values per column**")
st.dataframe(table("missing"))
st.write("**Date range and drive counts**")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Earliest date", summary["earliest"])
c2.metric("Latest date", summary["latest"])
c3.metric("Unique drives", f"{summary['unique_drives']:,}")
c4.metric("Unique models", summary["unique_models"])

# 2. Failure distribution
st.header("2. Failure Distribution")
st.dataframe(table("failure_dist"))
col_a, col_b = st.columns(2)
col_a.image(str(RESULTS / "failure_countplot.png"))
col_b.image(str(RESULTS / "smart197_boxplot.png"))

# 3. Failed drives
st.header("3. Failed Drives")
st.write(f"Number of unique failed drives: **{summary['n_failed_drives']}**")
st.dataframe(table("failed_head", index=False))
st.write("**First and last observation for a sample of failed drives**")
st.dataframe(table("failed_sample"))
st.write("**Final observation label for every drive**")
st.dataframe(table("last_obs"))

# 4. SMART features
st.header("4. SMART Features Used")
st.write(summary["smart_features"])

# 5. Baseline model
st.header("5. Baseline Model (chronological 80/20 split)")
st.write("**Confusion Matrix**")
st.dataframe(table("baseline_cm"))
st.write("**Classification Report**")
st.dataframe(table("baseline_report"))
st.metric("Average precision (PR-AUC)", summary["baseline_pr_auc"])

# 6. Next-observation model
st.header("6. Next-Observation Failure Model")
st.dataframe(table("next_failure_dist"))
st.write("**Best parameters:**", summary["best_params"])
st.write("**Confusion Matrix**")
st.dataframe(table("final_cm"))
st.write("**Classification Report**")
st.dataframe(table("final_report"))