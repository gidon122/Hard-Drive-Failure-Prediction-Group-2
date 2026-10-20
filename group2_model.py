#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


# In[4]:


df = pd.read_parquet("clean.parquet")
df


# In[5]:


df.info()


# In[6]:


df.head()


# In[7]:


df.isnull().sum()


# In[8]:


failure_count = df["failure"].value_counts().sort_index()
failure_percentage = df["failure"].value_counts(normalize=True).sort_index() * 100

print(f"Failure counts: {failure_count}")
print(f"Failure percentage: {failure_percentage}.round(3)")


# In[9]:


plt.figure(figsize=(8,5))
sns.countplot(data=df, x="failure")
plt.title("Distribution of Hard Drive Failure Records")
plt.xlabel("Failure (0 = No Failure, 1 = Failure)")
plt.ylabel("Number of Records")
plt.tight_layout()
plt.show()


# In[10]:


plt.figure(figsize=(9,5))
sns.boxplot(
    data = df,
    x= "failure",
    y= "smart_197_raw"
)
plt.title("Pending Sector Count By Recorded Failure Status")
plt.ylabel("SMART 197 Raw Value")
plt.tight_layout()
plt.show()


# In[11]:


print("Earliest date:", df["date"].min())
print("Latest date:", df["date"].max())

print("Unique drives:", df["serial_number"].nunique())
print("Unique models:", df["model"].nunique())


# In[12]:


failed_rows = df[df["failure"] == 1] 
failed_rows


# In[13]:


failed_rows[[
    "date", "serial_number", "model", "failure"
 ]].head(10)


# In[14]:


#count failure per drive
failures_per_drive = (
    df[df["failure"] == 1]
    .groupby("serial_number")
    .size()
    .sort_values(ascending=False)
)

print("Number of unique failed drives:", len(failures_per_drive))
print("Failure records per drive:", failures_per_drive.value_counts().sort_index())
print("Drives with the most failure records:", failures_per_drive.head(10))


# In[15]:


#sort records in order of time
df = df.sort_values(["serial_number", "date"]).copy()

#show the first and last observation for a sample of failed drives
failed_serials = df.loc[df["failure"] == 1, "serial_number"].unique()

sample_serials = failed_serials[:5]

display(
    df[df["serial_number"].isin(sample_serials)]
.groupby("serial_number")
.agg(
    first_obs=("date", "min"),
    last_obs=("date", "max"),
    number_of_observations= ("date", "size"),
    failure_records= ("failure", "sum")
)
)


# In[16]:


#get the final observation for every drive
last_obs = (
    df.sort_values("date")
    .groupby("serial_number")
    .tail(1)
)

print("Total unique drives:", df["serial_number"].nunique())
print("Final observation labels:", last_obs["failure"].value_counts())
print("Final observation labelled as failures:")
display(
    last_obs.loc[
        last_obs["failure"] == 1,
        ["date","serial_number","model","failure"]
    ].head(10)
)


# In[17]:


#prepare features for training
smart_features = [
    col for col in df.columns
    if col.startswith("smart_") and col.endswith("_raw")
]

smart_features


# In[18]:


X = df[smart_features]
y = df["failure"]


# In[19]:


#Ensure date is stored as datetime
df["date"] = pd.to_datetime(df["date"])

#sort observations chronologically
df = df.sort_values("date").reset_index(drop=True)

#Define the chronological split point
split_index = int(len(df)*0.8)
split_date = df.loc[split_index, "date"]

#Separate earlier and later observations
train_df = df[df["date"] < split_date].copy()
test_df = df[df["date"] >= split_date].copy()


# In[20]:


X_train = train_df[smart_features]
y_train = train_df["failure"]

X_test = test_df[smart_features]
y_test = test_df["failure"]

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

model.fit(X_train,y_train)


# In[21]:


y_pred = model.predict(X_test)


# In[22]:


#confusion matrix
print("Confusion Matrix:", confusion_matrix(y_test,y_pred))


# In[23]:


print("Classification report:", classification_report(y_test,y_pred,zero_division=0))


# In[24]:


from sklearn.metrics import average_precision_score


# In[25]:


df_future = pd.read_parquet("clean.parquet")

print(type(df_future))


# In[26]:


#get the model's probability estimates for class 1
y_proba = model.predict_proba(X_test)[:, 1]

#Evaluate ranking performance for the rare failure class
pr_auc = average_precision_score(y_test, y_proba)

print("Average precision (PR-AUC):", round(pr_auc, 4))


# In[27]:


df_future = df.copy()

#sorting each drive's observations chronologically
df_future = df_future.sort_values(
    ["serial_number", "date"]
).reset_index(drop=True)
df_future


# In[28]:


#create next observation failure target
data = pd.read_parquet("clean.parquet")
data["date"] = pd.to_datetime(data["date"])
data = data.sort_values(["serial_number", "date"])

data["next_failure"] = (
    data.groupby("serial_number")["failure"].shift(-1)
)

data = data.dropna(subset=["next_failure"]).copy()
data["next_failure"] = data["next_failure"].astype(int)

print(data["next_failure"].value_counts())


# In[29]:


#features for chronological training
smart_features = [
    col for col in data.columns
    if col.startswith("smart_") and col.endswith("_raw")
]
print("SMART featrures:", smart_features)


# In[30]:


print(data["next_failure"].value_counts())
print(data["next_failure"].value_counts(normalize=True).round(6))


# In[31]:


#Earlier dates will train and later dates will test
split_date = data["date"].quantile(0.80)

train_df = data[data["date"] < split_date].copy()
test_df = data[data["date"] >= split_date].copy()

X_train = train_df[smart_features]
y_train = train_df["next_failure"]

X_test = test_df[smart_features]
y_test = test_df["next_failure"]



# In[ ]:


param_grid = {
    "classifier__n_estimators": [100,200],
    "classifier__max_depth": [10,20]
}

grid_search = GridSearchCV(
    model,
    param_grid,
    cv=3,
    scoring="recall",
    n_jobs=-1
)

grid_search.fit(X_train,y_train)
print("Best Parameters:", grid_search.best_params_)
model = grid_search.best_estimator_


# In[ ]:


y_pred = model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test,y_pred))

print("Classification Report:")
print(classification_report(y_test,y_pred, zero_division=0))


# In[ ]:


print("Best parameters:", grid_search.best_params_)


# In[ ]:


import joblib

joblib.dump(model, "drive_failure_prediction_model.pkl")


# In[ ]:




