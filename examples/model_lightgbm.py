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
- path smoothing: https://github.com/microsoft/LightGBM/issues/2790

TODO: Add MLFlow Tracking
TODO: Add Monotonic Constraint
"""
# Generic Libraries
import os
import uuid
import logging
import pandas as pd
import numpy as np
from itertools import chain
from decouple import config as d_config
from datetime import datetime
import matplotlib.pyplot as plt

# Project Libraries
from src import utils
from src.modeling import quick_hyperopt

# Scikit Learn Libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    precision_recall_fscore_support,
    RocCurveDisplay,
)

# Light GBM
import lightgbm as lgb

# Hyper Parameter Optimization
from hyperopt import hp

# Settings
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)

# Config Files
CONFIG_MASTER = utils.load_config()
FEATURE_SET = CONFIG_MASTER["ML_MODEL_FEATURE_SET"]
FEATURE_DATA_TYPES = CONFIG_MASTER["DATA_TYPE_CATEGORIES"]
FEATURE_TYPE_CONFIG = {
    "categorical": FEATURE_DATA_TYPES["NOMINAL_BINARY"] + FEATURE_DATA_TYPES["NOMINAL"],
    "continuous": [],
    "ordinal": FEATURE_DATA_TYPES["NUMERIC_ORDINAL"],
}
EXCLUDED_FEATURES_CONFIG = CONFIG_MASTER["COLUMNS_NOT_IN_FINAL_FEATURE_SET"]
EXCLUDED_FEATURES = list(
    chain.from_iterable([value for key, value in EXCLUDED_FEATURES_CONFIG.items()])
)

# Directories
DIR_DATA_RAW = d_config("DIR_DATA_RAW")
DIR_DATA_ENHANCED = d_config("DIR_DATA_ENHANCED")
DIR_MODEL = d_config("DIR_DATA_MODELS_LGB")

# Run Mode
DEBUG_MODE = False

# Globals
MODEL_DATE = datetime.today().strftime("%Y-%m-%d")
MODEL_UUID = str(uuid.uuid4())[:6] + "_" + MODEL_DATE
MODEL_NAME = "lightgbm"
MODEL_EXPERIMENT = (
    f"experiment-{MODEL_NAME}-{MODEL_UUID}-{MODEL_DATE}"
    if not DEBUG_MODE
    else "experiment-debug"
)
DIR_EXPERIMENT = os.path.join(DIR_MODEL, MODEL_EXPERIMENT)

# Parameter Elections
# SAVE_RESULTS = False if DEBUG_MODE else True
SAVE_RESULTS = True
USE_SAMPLE = True if DEBUG_MODE else False
FILL_NULLS = False
INCLUDE_INDEX = False
LIMIT_STATE = True
IN_OUT_SPLIT_DATA = True

# Parameter Values
NUM_ROWS = 50_000
PKEY_COLUMN = "CASEID"
TARGET_COLUMN = "TARGET"
CLASS_THRESHOLD = 0.5
FILL_NA_CONSTANT = -9
TEST_SIZE = 0.33
STATE_ELECTION_LIST = [4, 12, 24]
IN_OUT_COL = "IN_OUT"
IN_OUT_VAL = ["Inpatient", "Outpatient"]  # ["1", "2"]
INPUT_FILE_NAME = (
    CONFIG_MASTER["FILE_NAMES"]["MODEL"]
    if not IN_OUT_SPLIT_DATA
    else CONFIG_MASTER["FILE_NAMES"]["ENHANCED_TARGET_COL"]
)

# Parameter Config
PARAMS = {
    "SAVE_RESULTS": SAVE_RESULTS,
    "USE_SAMPLE": USE_SAMPLE,
    "FILL_NULLS": FILL_NULLS,
    "INCLUDE_INDEX": INCLUDE_INDEX,
    "LIMIT_STATE": LIMIT_STATE,
    "NUM_ROWS": NUM_ROWS,
    "PKEY_COLUMN": PKEY_COLUMN,
    "TARGET_COLUMN": TARGET_COLUMN,
    "CLASS_THRESHOLD": CLASS_THRESHOLD,
    "FILL_NA_CONSTANT": FILL_NA_CONSTANT,
    "TEST_SIZE": TEST_SIZE,
    "MODEL_NAME": MODEL_NAME,
    "MODEL_EXPERIMENT": MODEL_EXPERIMENT,
    "DIR_EXPERIMENT": DIR_EXPERIMENT,
    "MODEL_DATE": MODEL_DATE,
    "STATE_ELECTION_LIST": str(STATE_ELECTION_LIST),
    "IN_OUT_SPLIT_DATA": IN_OUT_SPLIT_DATA,
    "IN_OUT_COL": IN_OUT_COL,
    "IN_OUT_VAL": str(IN_OUT_VAL),
}
METRICS = {}

if __name__ == "__main__":
    ############################################################################
    # SETUP
    ############################################################################

    # Create Experiment Directory
    if not os.path.exists(DIR_EXPERIMENT):
        os.makedirs(DIR_EXPERIMENT)

    # Logger
    logger = utils.Logger(directory=DIR_EXPERIMENT, filename="run-logs").get_logger()

    # Log start time
    start_time = datetime.now()
    logger.info(f"Starting Experiment {MODEL_EXPERIMENT} at {start_time}")

    # Log Parameters
    logger.info(f"parameters => {PARAMS}")

    # Load Dataset
    logger.info(f"Loading data w/ file name => {INPUT_FILE_NAME}")
    ml_df = utils.load_dataframe(
        directory=DIR_DATA_ENHANCED,
        filename=INPUT_FILE_NAME,
        extension="csv",
        sample=USE_SAMPLE,
        nrows=NUM_ROWS,
    )

    # Limit States (STFIPS) to Arizona, Florida, Maryland
    if LIMIT_STATE:
        logger.info(f"Limiting Dataset to STFIPS: {STATE_ELECTION_LIST}")
        ml_df = ml_df[ml_df["STFIPS"].isin(STATE_ELECTION_LIST)]
        logger.info(f"\tDataset Shape: {ml_df.shape}")

    if IN_OUT_SPLIT_DATA:
        logger.info(f"Limiting Dataset By In-Out Column: {IN_OUT_VAL}")
        logger.info(f"Starting Dimensions => {ml_df.shape}")
        ml_df = ml_df[ml_df[IN_OUT_COL].isin(IN_OUT_VAL)]
        logger.info(f"\tFinal dimensions: {ml_df.shape}")

    ############################################################################
    # EDA
    ############################################################################

    # Inspect Target Distribution
    target_df = ml_df.groupby(TARGET_COLUMN)[TARGET_COLUMN].count()
    logger.info(
        f"Target Distribution (Pct. Positive):\n{target_df[1] / target_df.sum()}"
    )
    plt.bar(target_df.index, target_df.values)
    plt.title("Target Distribution")
    filename = f"target_distribution.png"
    path = os.path.join(DIR_EXPERIMENT, filename)
    plt.savefig(path)
    plt.close()
    METRICS["target_dist_pos"] = target_df[1] / target_df.sum()
    METRICS["target_dist_neg"] = target_df[0] / target_df.sum()

    ############################################################################
    # BUILD TRAINING DATASET
    ############################################################################

    # Restrict Dataset to Features
    ml_df = ml_df[FEATURE_SET]

    # Validate Feature Set
    excluded_features = [f for f in ml_df.columns if f in EXCLUDED_FEATURES]
    assert not excluded_features, f"Invalid features found: {excluded_features}"

    # Split Data into X & y
    x_feature_set = [
        f for f in ml_df.columns if f != TARGET_COLUMN and f != PKEY_COLUMN
    ]
    y = ml_df[TARGET_COLUMN]
    X = ml_df[x_feature_set]
    logger.info(f"X Shape: {X.shape}, y Shape: {y.shape}")

    # Fill Null Values (Not required by Lightgbm but adding so that we can cast all features to int)
    if FILL_NULLS:
        logger.info("Filling Null Values w/ constant.")
        # Fill Nulls
        X = X.fillna(FILL_NA_CONSTANT)

        # Cast Data Types to Integer
        X = X.astype(int)
        y = y.astype(int)

    # Split Data into Train & Test
    logger.info("Splitting Data into Train & Test")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=123, test_size=0.33
    )
    logger.info(f"X_train Shape: {X_train.shape}, y_train Shape: {y_train.shape}")

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

    ############################################################################
    # HYPER PARAMETER OPTIMIZATION
    ############################################################################
    """
    Parameter Docs: https://lightgbm.readthedocs.io/en/latest/Parameters.html
    """

    # Define Parameter Space
    N_EVALS = 50 if not DEBUG_MODE else 5
    N_FOLDS = 5
    N_ROUNDS = 100 if not DEBUG_MODE else 10

    # boosting_list = [{"boosting": "gbdt", "boosting": "dart"}]
    boosting_list = [{"boosting": "dart"}]
    objective_list = ["binary"]
    metric_list = ["binary_logloss", "auc"]
    objective_config = {
        "boosting_list": boosting_list,
        "metric_list": metric_list,
        "objective_list": objective_list,
    }
    param_config = {
        "boosting": hp.choice("boosting", boosting_list),
        "metric": hp.choice("metric", metric_list),
        "objective": hp.choice("objective", objective_list),
        "is_unbalance": hp.choice(
            "is_unbalance", [True]
        ),  # absent resampling, we can use weights.
        "num_leaves": hp.uniform("num_leaves", 2, 1000),  # max was 100
        "max_depth": hp.uniform("max_depth", 5, 100),  # max was 12
        "max_bin": hp.uniform("max_bin", 32, 250),
        "min_data_in_leaf": hp.uniform("min_data_in_leaf", 1, 750),
        "min_data_in_bin": hp.uniform("min_data_in_bin", 1, 750),
        # "min_gain_to_split": hp.uniform("min_gain_to_split", 0.01, 5),  # may want to lower rel to learning rate.
        "min_gain_to_split": hp.uniform("min_gain_to_split", 0.005, 1),
        "lambda_l1": hp.uniform("lambda_l1", 0, 30),
        "lambda_l2": hp.uniform("lambda_l2", 0, 30),
        "learning_rate": hp.uniform(
            "learning_rate", 0.001, 0.1
        ),  # may slow down training considerably.
        "feature_fraction": hp.uniform("feature_fraction", 0.5, 1),
        "bagging_fraction": hp.uniform("bagging_fraction", 0.5, 1),
        "path_smooth": hp.uniform("path_smooth", 0, 1),
    }

    # Log Params
    PARAMS["N_EVALS"] = N_EVALS
    PARAMS["N_FOLDS"] = N_FOLDS
    PARAMS["N_ROUNDS"] = N_ROUNDS
    for param in objective_config:
        PARAMS[f"objective_config-{param}"] = objective_config[param]

    # Run Hyper Parameter Optimization
    hpo_results = quick_hyperopt(
        train=lgb_train,
        space_config=param_config,
        objective_config=objective_config,
        num_evals=N_EVALS,
        num_cv_folds=N_FOLDS,
        early_stopping_rounds=N_ROUNDS,
        stratified=True,  # change this to True.
        diagnostic=True,
        integer_params=[
            "max_depth",
            "num_leaves",
            "max_bin",
            "min_data_in_leaf",
            "min_data_in_bin",
        ],
    )

    # Get Trial Results
    trials = hpo_results[1]
    best_run_loss = min([x["result"]["loss"] for x in trials])
    logging.info(f"Best Overall Trial Log Loss => {best_run_loss}")
    METRICS["best_run_loss"] = best_run_loss

    # Plot Trial Loss Ratio
    trials_loss = [x["result"]["loss"] for x in trials]
    plt.plot([str(x) for x in range(len(trials_loss))], trials_loss)
    plt.grid(which="both")
    plt.title("Hyperopt - Loss vs Trials")
    plt.xlabel("Loss")
    plt.ylabel("Trial Number")
    plt.savefig(os.path.join(DIR_EXPERIMENT, "hyperopt_loss.png"))
    plt.close()

    # Get Best Params
    best_params = hpo_results[0]
    best_params["is_unbalance"] = (
        True if best_params["is_unbalance"] else False
    )  # hpo returns 1|0 instead of True|False
    logger.info(f"Best Parameters =>\n{best_params}")
    for param in best_params:
        METRICS[f"best-param-{param}"] = best_params[param]

    # Use Cross Validation to Get Maximum Number of Rounds
    # Lightgbm early stopping: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.early_stopping.html#lightgbm.early_stopping
    logger.info(
        "Starting Cross Validation with Best Parameters - Get Max Number of Rounds"
    )
    cv = lgb.cv(
        best_params,
        lgb_train,
        num_boost_round=100 if not DEBUG_MODE else 10,
    )

    # Plot Cross Validation
    cv_loss = cv["valid binary_logloss-mean"]
    plt.plot(cv_loss)
    plt.grid(which="both")
    plt.title("Cross Validation - Binary Log Loss")
    plt.xlabel("Loss")
    plt.ylabel("Iteration")
    plt.savefig(os.path.join(DIR_EXPERIMENT, "cross_validation_loss.png"))
    plt.close()

    # Get Num Rounds
    num_rounds = np.array(cv["valid binary_logloss-mean"]).argmin() + 1
    logging.info(f"Max number of rounds => {num_rounds}")
    METRICS["num_boosting_rounds"] = num_rounds

    ############################################################################
    # TRAIN MODEL
    ############################################################################

    # Train Model
    """
    train method: params define the object, in particular the objective, which determines the type of models we are
        training.
        returns booster models.
    """
    model = lgb.train(
        params=best_params,
        train_set=lgb_train,
        num_boost_round=num_rounds,
        valid_sets=[lgb_test],
    )

    # Make Predictions
    yhat_train_probs = model.predict(X_train.values)
    yhat_test_probs = model.predict(X_test.values)
    yhat_all_probs = model.predict(X.values)

    ############################################################################
    # EVALUATE MODEL
    ############################################################################

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
    logger.info(f"Accuracy Train => {accuracy_train}")
    logger.info(f"Accuracy Test => {accuracy_test}")
    logger.info(f"Accuracy All => {accuracy_all}")
    METRICS["accuracy_train"] = accuracy_train
    METRICS["accuracy_test"] = accuracy_test
    METRICS["accuracy_all"] = accuracy_all

    # Classification Report
    clf_report_train = classification_report(y_train, yhat_train_binary, labels=[0, 1])
    logger.info(f"Classification Report - Train \n{clf_report_train}")

    clf_report_test = classification_report(y_test, yhat_test_binary, labels=[0, 1])
    logger.info(f"Classification Report - Test \n{clf_report_test}")

    # Precision Recall F1-Score
    clf_scores_train = precision_recall_fscore_support(y_train, yhat_train_binary)
    clf_scores_test = precision_recall_fscore_support(y_test, yhat_test_binary)
    logger.info(
        f"Training Precision: 0: {clf_scores_train[0][0]}, 1: {clf_scores_train[0][1]}"
    )
    logger.info(
        f"Training Recall: 0: {clf_scores_train[1][0]}, 1: {clf_scores_train[1][1]}"
    )
    logger.info(
        f"Training F1-Score: 0: {clf_scores_train[2][0]}, 1: {clf_scores_train[2][1]}\n"
    )
    logger.info(
        f"Test Precision: 0: {clf_scores_test[0][0]}, 1: {clf_scores_test[0][1]}"
    )
    logger.info(f"Test Recall: 0: {clf_scores_test[1][0]}, 1: {clf_scores_test[1][1]}")
    logger.info(
        f"Test F1-Score: 0: {clf_scores_test[2][0]}, 1: {clf_scores_test[2][1]}"
    )
    METRICS["precision_0_train"] = clf_scores_train[0][0]
    METRICS["precision_1_train"] = clf_scores_train[0][1]
    METRICS["recall_0_train"] = clf_scores_train[1][0]
    METRICS["recall_1_train"] = clf_scores_train[1][1]
    METRICS["f1_0_train"] = clf_scores_train[2][0]
    METRICS["f1_1_train"] = clf_scores_train[2][1]

    # Confusion Matrix
    cm_train = confusion_matrix(y_train, yhat_train_binary)
    cm_test = confusion_matrix(y_test, yhat_test_binary)
    cm_all = confusion_matrix(y, yhat_all_binary)

    cm_train_disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_train, display_labels=[0, 1]
    ).plot()
    cm_train_disp.plot()
    filename = f"confusion_matrix_train.png"
    path = os.path.join(DIR_EXPERIMENT, filename)
    plt.savefig(path)
    plt.close()

    cm_test_disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_test, display_labels=[0, 1]
    ).plot()
    cm_test_disp.plot()
    filename = f"confusion_matrix_test.png"
    path = os.path.join(DIR_EXPERIMENT, filename)
    plt.savefig(path)
    plt.close()

    # Receiver Operator Curve
    roc_auc_train = roc_auc_score(y_train, yhat_train_probs)
    roc_auc_test = roc_auc_score(y_test, yhat_test_probs)
    logger.info(f"ROC AUC Train Score \n{roc_auc_train}")
    logger.info(f"ROC AUC Test Score \n{roc_auc_test}")
    METRICS["roc_auc_train"] = roc_auc_train
    METRICS["roc_auc_test"] = roc_auc_test

    roc_plot_train = RocCurveDisplay.from_predictions(
        y_true=y_train, y_pred=yhat_train_probs, plot_chance_level=True
    )
    filename = f"roc_curve_train.png"
    path = os.path.join(DIR_EXPERIMENT, filename)
    plt.grid(which="both")
    plt.title(f"ROC Curve - Test (AUC: {roc_auc_train})")
    plt.savefig(path)
    plt.close()

    roc_plot_test = RocCurveDisplay.from_predictions(
        y_true=y_test, y_pred=yhat_test_probs, plot_chance_level=True
    )
    filename = f"roc_curve_test.png"
    path = os.path.join(DIR_EXPERIMENT, filename)
    plt.grid(which="both")
    plt.title(f"ROC Curve - Test (AUC: {roc_auc_test})")
    plt.savefig(path)
    plt.close()

    ############################################################################
    # INTERPRET MODEL - FEATURE IMPORTANCE
    ############################################################################

    # Global Feature Importance
    lgb.plot_importance(
        model,
        importance_type="gain",
        figsize=(7, 6),
        title="LightGBM Feature Importance (Gain)",
    )
    filename = "feature_importance_gain.png"
    path = os.path.join(DIR_EXPERIMENT, filename)
    plt.savefig(path)
    plt.close()

    lgb.plot_importance(
        model,
        importance_type="split",
        figsize=(7, 6),
        title="LightGBM Feature Importance (Split)",
    )
    filename = "feature_importance_split.png"
    path = os.path.join(DIR_EXPERIMENT, filename)
    plt.savefig(path)
    plt.close()

    ####################################################################################################################
    # SAVE MODEL RESULTS
    ####################################################################################################################

    # Feature Importance DataFrame
    if SAVE_RESULTS:
        logger.info("Saving Experiment Results")

        # Save Feature Set
        feature_set_df = pd.DataFrame({"FEATURE": x_feature_set})
        feature_set_df.to_csv(
            os.path.join(DIR_EXPERIMENT, "feature_set.csv"), index=False
        )

        # Save Excluded Features
        excluded_features_df = pd.DataFrame({"FEATURE": EXCLUDED_FEATURES})
        excluded_features_df.to_csv(
            os.path.join(DIR_EXPERIMENT, "excluded_features.csv"), index=False
        )

        # Save Parameters
        param_df = pd.DataFrame.from_dict(PARAMS, orient="index", columns=["VALUE"])
        param_df.to_csv(os.path.join(DIR_EXPERIMENT, "parameters.csv"), index=True)

        # Save Metrics
        metrics_df = pd.DataFrame.from_dict(METRICS, orient="index", columns=["VALUE"])
        metrics_df.to_csv(os.path.join(DIR_EXPERIMENT, "metrics.csv"), index=True)

        # Save Feature Importance
        imp_df = pd.DataFrame(
            {
                "FEATURE": model.feature_name(),
                "IMPORTANCE-GAIN": model.feature_importance(importance_type="gain"),
                "IMPORTANCE-SPLIT": model.feature_importance(importance_type="split"),
            }
        )
        imp_df.to_csv(
            os.path.join(DIR_EXPERIMENT, "model_feature_importance.csv"), index=False
        )

        # Trial Loss
        trail_loss_df = pd.DataFrame(
            {
                "LOSS": [x["result"]["loss"] for x in trials],
                "TRIAL": [x for x in range(len(trials))],
            }
        )
        trail_loss_df.to_csv(
            os.path.join(DIR_EXPERIMENT, "hpo_trial_loss.csv"), index=False
        )

        # Cross Validation Loss
        cv_loss_df = pd.DataFrame(
            {"LOSS": cv_loss, "ITERATION": [x for x in range(len(cv_loss))]}
        )
        cv_loss_df.to_csv(os.path.join(DIR_EXPERIMENT, "cv_loss.csv"), index=False)

        # Classification Reports
        path = os.path.join(DIR_EXPERIMENT, "classification_report_train.txt")
        with open(path, "w") as f:
            f.write(clf_report_train)
            f.close()
        path = os.path.join(DIR_EXPERIMENT, "classification_report_test.txt")
        with open(path, "w") as f:
            f.write(clf_report_test)
            f.close()

        # Save Model
        model.save_model(os.path.join(DIR_EXPERIMENT, "model.txt"))

        # Save Datasets
        X_train.to_csv(os.path.join(DIR_EXPERIMENT, "X_train.csv"), index=True)
        X_test.to_csv(os.path.join(DIR_EXPERIMENT, "X_test.csv"), index=True)
        y_train.to_csv(os.path.join(DIR_EXPERIMENT, "y_train.csv"), index=True)
        y_test.to_csv(os.path.join(DIR_EXPERIMENT, "y_test.csv"), index=True)

        # End Experiment
        duration = datetime.now() - start_time
        logger.info(
            f"Experiment {MODEL_EXPERIMENT} completed successfully in {duration}"
        )

    logger.info(f"Experiment {MODEL_EXPERIMENT} completed successfully in {duration}")
