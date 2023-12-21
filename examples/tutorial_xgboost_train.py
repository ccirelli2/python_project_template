"""
Utilize xgb train method (non sklearn approach).

"""
import os
from collections import Counter


import pandas as pd
from decouple import config as d_config

# XGboost
import xgboost as xgb
from xgboost import DMatrix

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Project Libraries
from src import utils

# Settings
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)

# Config Files
CONFIG_MASTER = utils.load_config()
FEATURE_SET = CONFIG_MASTER["ML_MODEL_FEATURE_SET"]

# Directories
DIR_DATA_RAW = d_config("DIR_DATA_RAW")
DIR_DATA_ENHANCED = d_config("DIR_DATA_ENHANCED")
DIR_MODEL = d_config("DIR_DATA_MODELS_XGB")
DIR_MLFLOW = d_config("DIR_DATA_EXAMPLES_MLFLOW")

# Globals
USE_SAMPLE = False
NUM_ROWS = 100_000
TEST_SIZE = 0.33
TARGET_COLUMN = "TARGET"
INPUT_FILE_NAME = CONFIG_MASTER["FILE_NAMES"]["MODEL"]

# Load Dataset
print(f"Loading data w/ file name => {INPUT_FILE_NAME}")
ml_df = utils.load_dataframe(
    directory=DIR_DATA_ENHANCED,
    filename=INPUT_FILE_NAME,
    extension="csv",
    sample=USE_SAMPLE,
    nrows=NUM_ROWS,
)

# Inspect Data
print(ml_df.head())
print(f"Dataframe Shape => {ml_df.shape}")

# Get Class Target
print(
    f"Class Distribution => {Counter(ml_df.TARGET)}, Num Classes => {len(Counter(ml_df.TARGET))}"
)
print(f"Percentage Class 1 => {Counter(ml_df.TARGET)[1] / len(ml_df)}")
print(f"Percentage Class 0 => {Counter(ml_df.TARGET)[0] / len(ml_df)}")

# Split Data
X = ml_df[[x for x in FEATURE_SET if x not in (TARGET_COLUMN)]]
y = ml_df[TARGET_COLUMN]
assert TARGET_COLUMN not in X.columns, "Target column found in X Feature Set"

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=7
)
print(f"Training dataset shape => {X_train.shape} Test Dataset shape => {X_test.shape}")

# Create XGBoost Dataset
"""
data -> feature data
label -> label trying to predict.
missing -> default value for missing data.  if None defaults to np.nan.
"""
print("Creating XGBoost DMatrix Datasets")
dtrain = DMatrix(
    data=X_train, label=y_train, missing=None, feature_names=X_train.columns.tolist()
)
dtest = DMatrix(
    data=X_test, label=y_test, missing=None, feature_names=X_train.columns.tolist()
)
watchlist = [(dtest, "eval"), (dtrain, "train")]

# Instantiate Model
print("Creating Parameter Space")

parameters = {
    "objective": "binary:logistic",
    "booster": "gbtree",  # 'gblinear', 'dart'
    "device": "cpu",  # 'gpu', cuda
    "verbosity": 1,  # 0 (silent), 1 (warning), 2 (info), 3 (debug)
    "n_estimators=": 2,
    "learning_rate=": 0.01,  # default 0.3
    "gamma": 0,  # default 0
    "max_depth": 0,  # default 6
    "min_child_weight": 1,  # default 1
    "max_delta_step": 0,  # default 0
    "class_weight": "balanced",  # default None. xgboost will automatically balance class weights.
    "subsample": 0.5,  # default 1
    "sampling_method": "uniform",  # default uniform, gradient based.
    "reg_alpha": 1,  # default 0
    "reg_lambda": 1,  # default 1
    "tree_method": "hist",  # default auto, same as hist method.
    "max_leaves": 0,  # default 0
    "max_bin": 256,  # default 256, only used if tree_method is set to hist.
    "max_features": "auto",  # default None
    # "eval_metric": "logloss",
    # "eval_metric": "error",
}

# Train Model using fit method.
print("Fitting Classifier")
clf = xgb.train(
    params=parameters,
    dtrain=dtrain,
    num_boost_round=100,
    evals=watchlist,
    early_stopping_rounds=10,
)
print("Fitting completed")

# Generate Predictions (Note: returns labels vs probs).
yhat_test_probs = clf.predict(dtest)
yhat_test_binary = utils.apply_threshold(yhat_test_probs)

# Evaluation
clf_report_test = classification_report(y_test, yhat_test_binary, labels=[0, 1])
print(f"Classification Report - Test \n{clf_report_test}")
