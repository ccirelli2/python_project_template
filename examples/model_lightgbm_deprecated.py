"""
Example script to train a LightGBM models.

======================
LightGBM
======================
- Tutorial: https://www.geeksforgeeks.org/binary-classification-using-lightgbm/
- Tutorial: https://www.analyticsvidhya.com/blog/2021/08/complete-guide-on-how-to-use-lightgbm-in-python/
- Tutorial: https://www.kaggle.com/code/prashant111/lightgbm-classifier-in-python
- Tutorial: https://forecastegy.com/posts/lightgbm-binary-classification-python/
- Lightgbm Dataset: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.Dataset.html
- Monotonic Constraints: https://lightgbm.readthedocs.io/en/latest/Parameters.html
- lightgbm train: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.train.html
- Warning:
    - msg: "Met categorical feature which contains sparse values.
        "Consider renumbering to consecutive integers started from zero"
    - ref: https://github.com/Microsoft/LightGBM/blob/92e95e62c4ae330e1ede01570dd3498ecbc58579/src/io/bin.cpp#L322-L358
- is_unbalance: https://stackoverflow.com/questions/68738225/use-of-is-unbalance-parameter-in-lightgbm
- is_unbalance: https://lightgbm.readthedocs.io/en/latest/Parameters.html

======================
Confusion Matrix
======================
- precision: tp / (tp + fp)
    - where tp is true positive.  tp is where we predict 1 and the actual label is 1. false positive is where we predict
        1 and the actual label is 0.
    - precision measures the ability of the classifier to not label a negative sample as positive.
- recall: tp / (tp + fn)
    - where tp is true positive, fn is false negative (predict 0 but actual is 1).
    - recall measures how well our classifier identifies tp positives.  ex: out of all the patients that had a disease,
        how many were correctly classified.
- f1-score: harmonic mean of precision and recall.


references:
- Tutorial: https://www.kdnuggets.com/2022/11/confusion-matrix-precision-recall-explained.html#:~:text=Precision%20is%20a%20metric%20that,of%20them%20actually%20have%20it%3F&text=In%20this%20case%2C%20we%20only,of%20true%20positives%20is%200.
- Tutorial: https://www.analyticsvidhya.com/blog/2020/04/confusion-matrix-machine-learning/#:~:text=The%20F1%20score%20is%20a,What%20is%20a%20confusion%20matrix%3F
- sklearn ConfusionMatrixDisplay: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html
- sklearn precision_recall_fscore_support: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html#sklearn.metrics.precision_recall_fscore_support
- roc_auc_score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html


TODO: Add MLFlow Tracking
TODO: Add Monotonic Constraint
"""
# Generic Libraries
import os
import sys
import uuid
import pickle
import copy
import csv
import logging
import functools
import pandas as pd
from itertools import chain
from decouple import config as d_config
from datetime import datetime
from collections import Counter
import graphviz
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Project Libraries
from src import utils
from src.modeling import LgbmModelingPrepare

# Scikit Learn Libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    precision_recall_fscore_support,
    RocCurveDisplay,
)

# Light GBM
import lightgbm as lgb
from lightgbm import LGBMClassifier
from lightgbm import LGBMRegressor

# Hyper Parameter Optimization
import gc
from hyperopt import hp, tpe
from hyperopt.fmin import fmin
from hyperopt import Trials
from hyperopt import hp, tpe, Trials, STATUS_OK
from hyperopt.fmin import fmin
from hyperopt.pyll.stochastic import sample

# Tracking
import mlflow

# Settings
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)
logger = logging.getLogger(__name__)
logger.setLevel(level="INFO")

# Config Files
CONFIG_MASTER = utils.load_config()
FEATURE_SET = CONFIG_MASTER["ML_MODEL_FEATURE_SET"]
FEATURE_DATA_TYPES = CONFIG_MASTER["DATA_TYPE_CATEGORIES"]
FEATURE_TYPE_CONFIG = {
    "categorical": FEATURE_DATA_TYPES["NOMINAL_BINARY"] + FEATURE_DATA_TYPES["NOMINAL"],
    "continuous": [],
    "ordinal": FEATURE_DATA_TYPES["NUMERIC_ORDINAL"],
}
NON_FEATURES_CONFIG = CONFIG_MASTER["COLUMNS_NOT_IN_FINAL_FEATURE_SET"]
NON_FEATURES = list(
    chain.from_iterable([value for key, value in NON_FEATURES_CONFIG.items()])
)

# Directories
DIR_DATA_RAW = d_config("DIR_DATA_RAW")
DIR_DATA_ENHANCED = d_config("DIR_DATA_ENHANCED")
DIR_MODEL = d_config("DIR_DATA_MODELS")

# Filenames
INPUT_FILE_NAME = CONFIG_MASTER["FILE_NAMES"]["MODEL"]

# Globals
UUID = str(uuid.uuid4())
SAVE_RESULTS = False
USE_SAMPLE = True
FILL_NULLS = False
INCLUDE_INDEX = False
NUM_ROWS = 100_000
PKEY_COLUMN = "CASEID"
TARGET_COLUMN = "TARGET"
CLASS_THRESHOLD = 0.5
NUM_ROUNDS = 10


if __name__ == "__main__":
    # Create An Experiment (ML FLOW)
    experiment_name = f"lightgbm_experiment_{UUID}"
    DIR_EXPERIMENT = os.path.join(DIR_MODEL, experiment_name)
    mlflow.set_tracking_uri(uri=DIR_EXPERIMENT)
    mlflow.set_experiment(experiment_name)
    start_time = datetime.now()

    # Log Params
    mlflow.log_param("UUID", UUID)
    mlflow.log_param("Start Time", str(start_time))
    mlflow.log_param("Sample", USE_SAMPLE)
    mlflow.log_param("Number of Rows", NUM_ROWS)
    mlflow.log_param("Fill Nulls", FILL_NULLS)
    mlflow.log_param("Feature Set", FEATURE_SET)
    mlflow.log_param("Excluded Features", NON_FEATURES_CONFIG)
    mlflow.log_param("Target Column", TARGET_COLUMN)
    mlflow.log_param("Class Threshold", CLASS_THRESHOLD)
    mlflow.log_param("Number of Rounds", NUM_ROUNDS)

    # Load Dataset
    ml_df = utils.load_dataframe(
        directory=DIR_DATA_ENHANCED,
        filename=INPUT_FILE_NAME,
        extension="csv",
        sample=USE_SAMPLE,
        nrows=NUM_ROWS,
    )

    # Restrict Dataset to Features
    ml_df = ml_df[FEATURE_SET]

    # Inspect Target Distribution
    target_df = ml_df.groupby(TARGET_COLUMN)[TARGET_COLUMN].count()
    print(f"Target Distribution (Pct. Positive):\n{target_df[1] / target_df.sum()}")
    plt.bar(target_df.index, target_df.values)
    plt.title("Target Distribution")
    filename = f"target_distribution.png"
    path = os.path.join(DIR_EXPERIMENT, filename)
    plt.savefig(path)

    # Validate Feature Set
    invalid_features = [f for f in ml_df.columns if f in NON_FEATURES]
    assert not invalid_features, f"Invalid features found: {invalid_features}"

    # Split Data into X & y
    x_feature_set = [
        f for f in ml_df.columns if f != TARGET_COLUMN and f != PKEY_COLUMN
    ]
    y = ml_df[TARGET_COLUMN]
    X = ml_df[x_feature_set]
    print(f"X Shape: {X.shape}, y Shape: {y.shape}")

    # Fill Null Values (Not required by Lightgbm but adding so that we can cast all features to int)
    if FILL_NULLS:
        # Fill Nulls
        X = X.fillna(-9)

        # Cast Data Types to Integer
        X = X.astype(int)
        y = y.astype(int)

    # Split Data into Train & Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=123, test_size=0.33
    )
    print(f"X_train Shape: {X_train.shape}, y_train Shape: {y_train.shape}")

    # Create LightGbm Dataset
    """
    data: is the training samples
    label: is the target variable
    feature_name: list of feature names and or columns of our dataset.
    categorical_feature: list of categorical features.  For this study all features are categorical.
    """
    lgb_train = lgb.Dataset(
        data=X_train.values,
        label=y_train.values,
        feature_name=x_feature_set,
        categorical_feature=x_feature_set,
        free_raw_data=False,
    )
    lgb_test = lgb.Dataset(
        data=X_test.values,
        label=y_test.values,
        feature_name=x_feature_set,
        categorical_feature=x_feature_set,
        free_raw_data=False,
    )

    # Parameter Space
    """
    Binary Loss Functions: binary, cross_entropy
    binary; binary optimizes the log loss
    is_unbalance: is used to handle unbalanced classes.
    """
    objective = "binary"
    metric = "auc"
    boosting_type = "gbdt"
    verbosity = 3
    is_unbalance = True

    params = {
        "objective": objective,
        "metric": metric,
        "boosting_type": boosting_type,
        "verbosity": verbosity,
        "is_unbalance": is_unbalance,
        # 'num_leaves': 31,
        # 'learning_rate': 0.05,
        # 'feature_fraction': 0.9,
    }

    mlflow.log_param("Objective", objective)
    mlflow.log_param("Metric", metric)
    mlflow.log_param("Boosting Type", boosting_type)
    mlflow.log_param("Verbosity", verbosity)
    mlflow.log_param("Is Unbalanced", is_unbalance)

    # Train Model
    """
    train method: params define the object, in particular the objective, which determines the type of models we are
        training.
        returns booster models.
    """
    models = lgb.train(
        params=params,
        train_set=lgb_train,
        num_boost_round=NUM_ROUNDS,
        valid_sets=[lgb_test],
    )

    # Make Predictions
    yhat_train_probs = models.predict(X_train.values)
    yhat_test_probs = models.predict(X_test.values)
    yhat_all_probs = models.predict(X.values)

    # Apply Classification Threshold
    def apply_threshold(prediction: list, threshold: float = 0.5) -> list:
        """
        Apply threshold to predictions.
        """
        return [1 if y >= threshold else 0 for y in prediction]

    # Convert Predictions to Target Variable (1, 0)
    yhat_train_binary = apply_threshold(yhat_train_probs, threshold=CLASS_THRESHOLD)
    yhat_test_binary = apply_threshold(yhat_test_probs, threshold=CLASS_THRESHOLD)
    yhat_all_binary = apply_threshold(yhat_all_probs, threshold=CLASS_THRESHOLD)

    # Evaluate Model
    accuracy_train = accuracy_score(y_train, yhat_train_binary)
    accuracy_test = accuracy_score(y_test, yhat_test_binary)
    accuracy_all = accuracy_score(y, yhat_all_binary)
    print(f"Accuracy Train => {accuracy_train}")
    print(f"Accuracy Test => {accuracy_test}")
    print(f"Accuracy All => {accuracy_all}")
    mlflow.log_metric("Accuracy Train", accuracy_train)
    mlflow.log_metric("Accuracy Test", accuracy_test)
    mlflow.log_metric("Accuracy All", accuracy_all)

    # Classification Report
    clf_report_train = classification_report(y_train, yhat_train_binary, labels=[0, 1])
    print("Classification Report - Train", "\n", clf_report_train)
    mlflow.log_text(
        text=clf_report_train, artifact_file="Classification Report - Train"
    )

    # Precision Recall F1-Score
    clf_scores_train = precision_recall_fscore_support(y_train, yhat_train_binary)
    clf_scores_test = precision_recall_fscore_support(y_test, yhat_test_binary)
    print(
        f"Training Precision: 0: {clf_scores_train[0][0]}, 0: {clf_scores_train[0][1]}"
    )
    print(f"Training Recall: 0: {clf_scores_train[1][0]}, 0: {clf_scores_train[1][1]}")
    print(
        f"Training F1-Score: 0: {clf_scores_train[2][0]}, 0: {clf_scores_train[2][1]}"
    )

    # Training Metrics
    mlflow.log_metric("Training Precision 0", clf_scores_train[0][0])
    mlflow.log_metric("Training Precision 1", clf_scores_train[0][1])
    mlflow.log_metric("Training Recall 0", clf_scores_train[1][0])
    mlflow.log_metric("Training Recall 1", clf_scores_train[1][1])
    mlflow.log_metric("Training F1-Score 0", clf_scores_train[2][0])
    mlflow.log_metric("Training F1-Score 1", clf_scores_train[2][1])

    # Test Metrics
    mlflow.log_metric("Test Precision 0", clf_scores_test[0][0])
    mlflow.log_metric("Test Precision 1", clf_scores_test[0][1])
    mlflow.log_metric("Test Recall 0", clf_scores_test[1][0])
    mlflow.log_metric("Test Recall 1", clf_scores_test[1][1])
    mlflow.log_metric("Test F1-Score 0", clf_scores_test[2][0])
    mlflow.log_metric("Test F1-Score 1", clf_scores_test[2][1])

    # Confusion Matrix
    cm_train = confusion_matrix(y_train, yhat_train_binary)
    cm_test = confusion_matrix(y_test, yhat_test_binary)
    cm_all = confusion_matrix(y, yhat_all_binary)

    mlflow.log_text(text=str(cm_train), artifact_file="Confusion Matrix - Train")
    mlflow.log_text(text=str(cm_test), artifact_file="Confusion Matrix - Test")
    mlflow.log_text(text=str(cm_all), artifact_file="Confusion Matrix - All")

    cm_train_disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_train, display_labels=[0, 1]
    ).plot()
    # cm_train_disp.plot()
    # filename = f'confusion_matrix_train.png'
    # path = os.path.join(DIR_EXPERIMENT, filename)
    # plt.savefig(path)
    mlflow.log_figure(cm_train_disp.figure_, "confusion_matrix_train.png")

    cm_test_disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_test, display_labels=[0, 1]
    ).plot()
    # cm_train_disp.plot()
    # filename = f'confusion_matrix_train.png'
    # path = os.path.join(DIR_EXPERIMENT, filename)
    # plt.savefig(path)
    mlflow.log_figure(cm_test_disp.figure_, "confusion_matrix_test.png")

    # Receiver Operator Curve
    roc_auc_train = roc_auc_score(y_train, yhat_train_probs)
    roc_auc_test = roc_auc_score(y_test, yhat_test_probs)
    mlflow.log_metric("ROC AUC Train", roc_auc_train)
    mlflow.log_metric("ROC AUC Test", roc_auc_test)

    # print('ROC AUC Test Score: ', roc_auc_test)

    roc_plot_train = RocCurveDisplay.from_predictions(
        y_true=y_train, y_pred=yhat_train_probs, plot_chance_level=True
    )
    mlflow.log_figure(roc_plot_train.figure_, "roc_curve_train.png")

    roc_plot_test = RocCurveDisplay.from_predictions(
        y_true=y_test, y_pred=yhat_test_probs, plot_chance_level=True
    )
    mlflow.log_figure(roc_plot_test.figure_, "roc_curve_test.png")

    # filename = f"roc_curve_test.png"
    # path = os.path.join(DIR_EXPERIMENT, filename)
    # plt.grid(which='both')
    # plt.title(f"ROC Curve - Test (AUC: {roc_auc})")
    # plt.savefig(path)
    # plt.close()

    # End Experiment
    end_time = datetime.now()
    mlflow.log_param("End Time", str(end_time))
    mlflow.end_run()
