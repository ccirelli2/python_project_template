"""

TODOS
- Add Train + Validation + Test Splits
- Early stopping call back for hyperopt and xgboost.
- Add standard deviation to hyperopt xgb.cv output.
- Hyperopt - split train and val loss. Currently only returning train loss.
- Implement MLFlow for tracking.
- Look into different random searches for hyperopt.
    See https://www.databricks.com/blog/2021/04/15/how-not-to-tune-your-model-with-hyperopt.html

Class Weights
- When class_weight is set to "balanced", XGBoost automatically adjusts the weights based on the number of samples in
    each class. For example, if one class has fewer samples than the others, its weight will be increased to make it
    more important during training.

References
- https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
- https://medium.com/@thedatabeast/handling-imbalanced-data-with-xgboost-class-weight-parameter-c67b7257515b
- https://medium.com/district-data-labs/parameter-tuning-with-hyperopt-faa86acdfdce
- xgboost callbacks: https://xgboost.readthedocs.io/en/stable/python/examples/cross_validation.html
- xgboost.cv: https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.cv
- hyperopt hmin docs: https://github.com/hyperopt/hyperopt/wiki/FMin


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
from collections import Counter

# Project Libraries
from src import utils
from src import modeling
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

# XGboost
import xgboost as xgb
from xgboost import DMatrix, plot_importance

# Hyper Parameter Optimization
import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll.stochastic import sample

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
DIR_MODEL = d_config("DIR_DATA_MODELS_XGB")

# Run Mode
DEBUG_MODE = False

# Globals
MODEL_DATE = datetime.today().strftime("%Y-%m-%d")
MODEL_UUID = str(uuid.uuid4())[:6]
MODEL_NAME = "xgboost"
MODEL_EXPERIMENT = (
    f"experiment-{MODEL_NAME}-{MODEL_UUID}-{MODEL_DATE}"
    if not DEBUG_MODE
    else "experiment-debug"
)
DIR_EXPERIMENT = os.path.join(DIR_MODEL, MODEL_EXPERIMENT)

# Parameter Elections
SAVE_RESULTS = True
USE_SAMPLE = True if DEBUG_MODE else False
FILL_NULLS = False
INCLUDE_INDEX = False
LIMIT_STATE = True
IN_OUT_SPLIT_DATA = False

# Parameter Values
NUM_ROWS = 50_000 if USE_SAMPLE else None
PKEY_COLUMN = "CASEID"
TARGET_COLUMN = "TARGET"
CLASS_THRESHOLD = 0.5
FILL_NA_CONSTANT = -9 if FILL_NULLS else None
TEST_SIZE = 0.33
VAL_SIZE = 0.1
STATE_ELECTION_LIST = [4, 12, 24]
IN_OUT_COL = "IN_OUT"
IN_OUT_VAL = ["1", "2"]  # ["Inpatient", "Outpatient"]
INPUT_FILE_NAME = (
    CONFIG_MASTER["FILE_NAMES"]["MODEL"]
    if not IN_OUT_SPLIT_DATA
    else CONFIG_MASTER["FILE_NAMES"]["ENHANCED_TARGET_COL"]
)

# Parameter Config
PARAM_LOGS = {
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
    "VAL_SIZE": VAL_SIZE,
    "MODEL_NAME": MODEL_NAME,
    "MODEL_EXPERIMENT": MODEL_EXPERIMENT,
    "DIR_EXPERIMENT": DIR_EXPERIMENT,
    "MODEL_DATE": MODEL_DATE,
    "STATE_ELECTION_LIST": str(STATE_ELECTION_LIST),
    "IN_OUT_SPLIT_DATA": IN_OUT_SPLIT_DATA,
    "IN_OUT_COL": IN_OUT_COL,
    "IN_OUT_VAL": str(IN_OUT_VAL),
}
METRICS_LOG = {}
HPO_LOGS = {}


if __name__ == "__main__":
    ############################################################################
    # SETUP
    ############################################################################

    # Create Experiment Directory
    if not os.path.exists(DIR_EXPERIMENT):
        os.makedirs(DIR_EXPERIMENT)

    # Logger
    logger = utils.Logger(directory=DIR_EXPERIMENT, filename="run").get_logger()

    # Log start time
    start_time = datetime.now()
    logger.info(f"Starting Experiment {MODEL_EXPERIMENT} at {start_time}")

    # Log Parameters
    logger.info(f"parameters => {PARAM_LOGS}")
    if DEBUG_MODE:
        logger.warning("Running in DEBUG Mode")

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
    METRICS_LOG["target_dist_pos"] = target_df[1] / target_df.sum()
    METRICS_LOG["target_dist_neg"] = target_df[0] / target_df.sum()

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
        assert X.isna().sum().sum() == 0, "Null values found in dataset."

        # Cast Data Types to Integer (source dataset already label encoded)
        X = X.astype(int)
        y = y.astype(int)

    # Create Train & Test Splits
    logger.info("Splitting Data into Train & Test")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=123, test_size=TEST_SIZE, stratify=y
    )
    logger.info(f"X_train Shape: {X_train.shape}, y_train Shape: {y_train.shape}")

    # Create XGBoost Dataset
    """
    ::data: is the training samples
    ::label: is the target variable
    ::missing: value to be considered as missing.  If None, defaults to np.nan.
    ::feature_name: list of feature names and or columns of our dataset.
    ::categorical_feature: list of categorical features.  For this study all features are categorical.
    """
    DTRAIN = DMatrix(
        data=X_train,
        label=y_train,
        missing=FILL_NA_CONSTANT if FILL_NULLS else None,
        feature_names=X_train.columns.tolist(),
    )
    DTEST = DMatrix(
        data=X_test,
        label=y_test,
        missing=FILL_NA_CONSTANT if FILL_NULLS else None,
        feature_names=X_train.columns.tolist(),
    )
    DALL = DMatrix(
        data=X,
        label=y,
        missing=FILL_NA_CONSTANT,
        feature_names=X_train.columns.tolist(),
    )

    # Calculate Scale Pos Weights for Imbalanced Dataset (can't use sampling techniques in this study).
    """
    - By default, the scale_pos_weight hyperparameter is set to the value of 1.0 and has the effect of weighing the
        balance of positive examples, relative to negative examples when boosting decision trees.
    - For an imbalanced binary classification dataset, the negative class refers to the majority class (class 0) and
        the positive class refers to the minority class (class 1).
    - XGBoost is trained to minimize a loss function and the “gradient” in gradient boosting refers to the steepness
        of this loss function, e.g. the amount of error
    - The scale_pos_weight value is used to scale the gradient for the positive class.
    - This has the effect of scaling errors made by the model during training on the positive class and encourages
        the model to over-correct them.
    - A sensible default value to set for the scale_pos_weight hyperparameter is the inverse of the class distribution.
    - scale_pos_weight = total_negative_examples / total_positive_examples
    References
    - https://machinelearningmastery.com/xgboost-for-imbalanced-classification/
    """
    counter_train = Counter(y_train)
    SCALE_POS_WEIGHT_BASE = counter_train[0] / counter_train[1]
    logger.info(f"Scale Pos Weight Base => {SCALE_POS_WEIGHT_BASE}")
    METRICS_LOG["scale_pos_weight"] = SCALE_POS_WEIGHT_BASE

    ############################################################################
    # HYPER PARAMETER OPTIMIZATION
    ############################################################################
    # HPO Parameters (Constants)
    NUM_HPO_EVALS = 150 if not DEBUG_MODE else 25  # first 20 iters are random.
    NUM_HPO_EVALS_EARLY_STOP = 3
    HPO_SEED = 123456
    HYPEROPT_ALGO = tpe.suggest  # tpe.suggest OR hyperopt.rand.suggest
    XGB_STRATIFIED = True
    HPO_CV_FOLDS = 5
    SCALE_POS_WEIGHT_LOWER = int(0.1 * SCALE_POS_WEIGHT_BASE)
    SCALE_POS_WEIGHT_UPPER = int(5.0 * SCALE_POS_WEIGHT_BASE)
    LOSS_METRIC_LABEL_TRAIN = (
        "train-logloss-mean"  # subject to objective function and metrics parameter.
    )
    LOSS_METRIC_LABEL_TEST = "test-logloss-mean"
    BOOSTER_OPTIONS = ["dart"]  # ["gbtree", "gblinear", "dart"]
    TREE_METHOD_OPTIONS = ["auto"]  # , "hist", "exact", "approx"]
    SAMPLING_METHOD_OPTIONS = ["uniform"]
    INTEGER_PARAMETERS = [
        "max_depth",
        "num_leaves",
        "max_bin",
        "max_leaves",
        "n_estimators",
    ]  # "min_data_in_leaf", "min_data_in_bin",

    # Define Parameter Space
    """
    class_weight: set weights to be applied to labels.  parameter can either be set to balanced or provided a dict of
        key=class, value = weights.  Passing balanced will ask xgboost to do the weighting itself and try to balance
        the positive and negative weights.
    """
    PARAM_SPACE = {
        "objective": "binary:logistic",
        "device": "cpu",
        "verbosity": 1,
        "class_weight": "balanced",  # auto balances based on pos/neg distribution.
        "scale_pos_weight": hp.uniform(
            "scale_pos_weight", SCALE_POS_WEIGHT_LOWER, SCALE_POS_WEIGHT_UPPER
        ),
        "max_features": "auto",
        "metric": "binary_logloss",
        "n_estimators": hp.uniform("n_estimators", 30, 100),
        "num_leaves": hp.uniform("num_leaves", 2, 1000),
        "booster": hp.choice("booster", BOOSTER_OPTIONS),
        "learning_rate": hp.uniform("learning_rate", 0.001, 0.1),
        "gamma": hp.uniform("gamma", 0, 100),
        "max_depth": hp.uniform("max_depth", 0, 1000),
        "min_child_weight": hp.uniform("min_child_weight", 0, 1000),
        "max_delta_step": hp.uniform("max_delta_step", 0, 10),
        "sub_sample": hp.uniform("sub_sample", 0.5, 1),
        "sampling_method": hp.choice("sampling_method", SAMPLING_METHOD_OPTIONS),
        "reg_alpha": hp.uniform("reg_alpha", 0, 30),
        "reg_lambda": hp.uniform("reg_lambda", 0, 30),
        "tree_method": hp.choice("tree_method", TREE_METHOD_OPTIONS),
        "max_leaves": hp.uniform("max_leaves", 10, 1000),
        "max_bin": hp.uniform("max_bin", 10, 250),
    }

    HPO_LOGS["NUM_HPO_EVALS"] = NUM_HPO_EVALS
    HPO_LOGS["NUM_HPO_EVALS_EARLY_STOP"] = NUM_HPO_EVALS_EARLY_STOP
    HPO_LOGS["HPO_SEED"] = HPO_SEED
    HPO_LOGS["HYPEROPT_ALGO"] = HYPEROPT_ALGO
    HPO_LOGS["XGB_STRATIFIED"] = XGB_STRATIFIED
    HPO_LOGS["HPO_CV_FOLDS"] = HPO_CV_FOLDS
    HPO_LOGS["SCALE_POS_WEIGHT_LOWER"] = SCALE_POS_WEIGHT_LOWER
    HPO_LOGS["SCALE_POS_WEIGHT_UPPER"] = SCALE_POS_WEIGHT_UPPER
    HPO_LOGS["LOSS_METRIC_LABEL_TRAIN"] = LOSS_METRIC_LABEL_TRAIN
    HPO_LOGS["INTEGER_PARAMETERS"] = INTEGER_PARAMETERS
    HPO_LOGS["BOOSTER_OPTIONS"] = BOOSTER_OPTIONS
    HPO_LOGS["TREE_METHOD_OPTIONS"] = TREE_METHOD_OPTIONS
    for p in PARAM_SPACE:
        HPO_LOGS[p] = PARAM_SPACE[p]

    # HPO Globals
    obj_call_count = 0
    cur_best_loss = 99999

    def objective(param_space):
        # Define Parameters as globals
        global obj_call_count, cur_best_loss, DTRAIN

        # Coerce Parameters from Float to Int
        for p in [p for p in INTEGER_PARAMETERS if param_space.get(p)]:
            param_space[p] = int(param_space[p])

        # Ensure Target Balancing Parameters in Space
        if "class_weight" not in param_space.keys():
            logger.warning("Class weight parameter not present in best parameters")
            param_space["class_weight"] = "balanced"
        if "scale_pos_weight" not in param_space.keys():
            logger.warning("Scale-pos-weight not present in best parameters")
            param_space["scale_pos_weight"] = np.choose(
                range(SCALE_POS_WEIGHT_LOWER, SCALE_POS_WEIGHT_UPPER), 1
            )
        # Log Best
        obj_call_count += 1
        logger.info(f"Objective call # {obj_call_count}, cur_best_loss {cur_best_loss}")
        logger.info(f"Parameters => {param_space}")

        # Get Num Rounds
        num_rounds = int(param_space["n_estimators"])

        # Build Model
        """
        metrics:
            metric to watch during training.
            when it is not specified, the evaluation metric is chosen according to objective function.
        ::error: binary classification error rate
        ::logloss: negative log likelihood function
        """
        cv_results = xgb.cv(
            params=param_space,
            dtrain=DTRAIN,
            num_boost_round=num_rounds,
            nfold=HPO_CV_FOLDS,
            stratified=XGB_STRATIFIED,
            # evals=watchlist,
            verbose_eval=True,
            # metrics=xgb_params['metric'],
            seed=HPO_SEED,
            callbacks=[
                xgb.callback.EvaluationMonitor(show_stdv=True),
                xgb.callback.EarlyStopping(NUM_HPO_EVALS_EARLY_STOP),
            ],
        )

        # Train Results
        losses_train = cv_results[LOSS_METRIC_LABEL_TRAIN]
        loss_best_train = min(losses_train)
        loss_var_train = np.var(losses_train, ddof=1)

        # Test Results
        losses_test = cv_results[LOSS_METRIC_LABEL_TEST]
        loss_best_test = min(losses_test)
        loss_var_test = np.var(losses_test, ddof=1)

        # Log Results
        if loss_best_train < cur_best_loss:
            logger.info(f"New Best Loss Train: {loss_best_train}")
            cur_best_loss = loss_best_train

        logger.info(
            f"\t\t CV Loss-Train: Best => {loss_best_train}, Var => {loss_var_train}"
        )
        logger.info(
            f"\t\t CV Loss-Test: Best => {loss_best_test}, Var => {loss_var_test}"
        )

        # Return Results
        """
        It can return either a scalar-valued loss, or a dictionary.  A returned dictionary must
        contain a 'status' key with a value from `STATUS_STRINGS`, must
        contain a 'loss' key if the status is `STATUS_OK`.
        https://github.com/hyperopt/hyperopt/wiki/FMin
        """
        return {
            "loss": loss_best_train,
            "loss_variance": loss_var_train,
            "status": STATUS_OK,
        }

    # Instantiate Trials Object (Track Trials)
    trials = Trials()

    # Execute fmin hyperopt
    logger.info("---------- START HPO --------------------")
    hpo_results = fmin(
        fn=objective,
        space=PARAM_SPACE,
        algo=HYPEROPT_ALGO,
        max_evals=NUM_HPO_EVALS,
        trials=trials,
        verbose=1,
    )
    logger.info("---------- END HPO --------------------")

    # Get Trial Results
    best_params = hpo_results
    best_run_loss = min([x["result"]["loss"] for x in trials])
    logging.info(f"Best Overall Trial Log Loss => {best_run_loss}")
    logging.info(f"Best Parameters => {best_params}")
    METRICS_LOG["best_run_loss"] = best_run_loss

    # Plot Trial Loss Ratio
    trial_losses = [x["result"]["loss"] for x in trials]
    trial_variances = [x["result"]["loss_variance"] for x in trials]
    assert len(trial_losses) == len(
        trial_variances
    ), "Trial Loss & Variance Lengths Do Not Match"

    # Plot HPO Performance
    logger.info("Plotting HPO Results")
    fig, ax1 = plt.subplots(figsize=(12, 5))
    xlabels = [x for x in range(len(trial_losses))]
    ax1.set_title("Trial Loss")
    ax1.plot(xlabels, trial_losses, color="blue")

    # ax2 = ax1.twinx()
    # ax2.plot(xlabels, trial_variances, color="red")
    # ax2.set_ylabel("Trial Loss Variance", color="red")
    plt.tight_layout()
    plt.grid(which="both")
    plt.savefig(os.path.join(DIR_EXPERIMENT, "hpo_loss_var_trial_results.png"))
    plt.close()

    ####################################################################################################################
    # Best Parameters - Transformations
    ####################################################################################################################

    # Convert Best Param Index (int) to Label (str pos in list)
    best_params["booster"] = BOOSTER_OPTIONS[
        best_params["booster"]
    ]  # best params returning index val as opposed to name.
    best_params["tree_method"] = TREE_METHOD_OPTIONS[best_params["tree_method"]]
    best_params["sampling_method"] = SAMPLING_METHOD_OPTIONS[
        best_params["sampling_method"]
    ]

    # Convert Integer Params from Float to Integer
    for param in [p for p in INTEGER_PARAMETERS if best_params.get(p)]:
        best_params[param] = int(best_params[param])

    # Add Back Static Params
    static_params = {
        "objective": "binary:logistic",
        "device": "cpu",
        "verbosity": 1,
        # "class_weight": "balanced",
        "max_features": "auto",
        "metric": "binary_logloss",
        # "n_estimators": 100,
    }
    # Add Static Params to dictionary
    for p in [p for p in static_params if p not in best_params]:
        logger.info(f"Adding static param => {p} with value => {static_params[p]}")
        best_params[p] = static_params[p]

    # Validate if Balancing Parameters are in Best Params
    if "class_weight" not in best_params.keys():
        best_params["class_weight"] = "balanced"
        logger.warning("Class Weight not found in best params")
    if "scale_pos_weight" not in best_params.keys():
        best_params["scale_pos_weight"] = SCALE_POS_WEIGHT_BASE
        logger.warning("Scale-Pos-Weight not found in best params")

    # Logging
    logger.info("\n\n\n------------------- Best Parameters -------------------")
    for param in best_params:
        logger.info(f"Param => {param}, Value => {best_params[param]}")
        METRICS_LOG[f"best-param-{param}"] = best_params[param]

    ############################################################################
    # CROSS VALIDATION - EARLY STOPPING
    ############################################################################

    # Use Cross Validation to Get Maximum Number of Rounds
    logger.info(
        "Starting Cross Validation with Best Parameters - Get Max Number of Rounds Early Stoppage"
    )

    # Should redefine Num Estimators as we want to see elbow point with parameters.
    NUM_BOOST_ROUNDS_CV = 100
    NUM_ROUNDS_EARLY_STOP_CV = 5
    CV_LOSS_PRECISION = 3
    PARAM_LOGS["NUM_BOOST_ROUNDS_CV"] = NUM_BOOST_ROUNDS_CV
    PARAM_LOGS["NUM_ROUNDS_EARLY_STOP_CV"] = NUM_ROUNDS_EARLY_STOP_CV
    PARAM_LOGS["CV_LOSS_PRECISION"] = CV_LOSS_PRECISION
    logger.info("---------- START CV --------------------")
    cv = xgb.cv(
        best_params,
        DTRAIN,
        num_boost_round=NUM_BOOST_ROUNDS_CV,
        callbacks=[
            xgb.callback.EvaluationMonitor(show_stdv=True),
            xgb.callback.EarlyStopping(NUM_ROUNDS_EARLY_STOP_CV),
        ],
    )
    logger.info("---------- END CV --------------------")
    # Expose & Transform Results
    cv_loss_train = cv[LOSS_METRIC_LABEL_TRAIN].round(CV_LOSS_PRECISION).values.tolist()
    cv_loss_train_change = [1] + [
        y - x for x, y in zip(cv_loss_train, cv_loss_train[1:])
    ]
    cv_loss_train_is_improving = list(
        map(lambda x: 1 if x < 0 else 0, cv_loss_train_change)
    )
    assert len(cv_loss_train) == len(
        cv_loss_train_change
    ), "CV loss delta train length does not match cv loss length"

    cv_loss_test = cv[LOSS_METRIC_LABEL_TEST].round(CV_LOSS_PRECISION).values.tolist()
    cv_loss_test_change = [1] + [y - x for x, y in zip(cv_loss_test, cv_loss_test[1:])]
    cv_loss_test_is_improving = list(
        map(lambda x: 1 if x < 0 else 0, cv_loss_test_change)
    )
    assert len(cv_loss_test) == len(
        cv_loss_test_change
    ), "CV loss delta test length does not match cv loss length"

    # Build Results DataFrame
    assert len(cv_loss_train) == len(
        cv_loss_test
    ), "CV Train & Test Loss Lengths Do Not Match"

    cv_results_df = pd.DataFrame(
        {
            "loss_train": cv_loss_train,
            "loss_train_change": cv_loss_train_change,
            "loss_train_is_improving": cv_loss_train_is_improving,
            "loss_test": cv_loss_test,
            "loss_test_change": cv_loss_test_change,
            "loss_test_is_improving": cv_loss_test_is_improving,
        }
    )
    cv_results_df.to_csv(os.path.join(DIR_EXPERIMENT, "cv_results.csv"), index=False)

    # Plot Results
    logger.info("Plotting Cross Validation Results")
    fig, ax1 = plt.subplots(figsize=(12, 5))
    xlabels = [x for x in range(len(cv_loss_train))]
    ax1.set_title("CV Loss Performance")
    ax1.set_ylabel("Train Loss", color="red")
    ax1.set_xlabel("Number of Boosting Rounds", color="red")
    ax1.plot(xlabels, cv_loss_train, color="blue")

    ax2 = ax1.twinx()
    ax2.plot(xlabels, cv_loss_test, color="red")
    ax2.set_ylabel("Test Loss", color="red")
    plt.tight_layout()
    plt.grid(which="both")
    plt.savefig(os.path.join(DIR_EXPERIMENT, "cv_train_test_results.png"))
    plt.close()

    # Get Num Rounds
    cv_opt_num_rounds = cv_results_df[
        cv_results_df["loss_train_is_improving"] == 1
    ].shape[0]
    # cv_opt_num_rounds = (
    #     np.array(cv[LOSS_METRIC_LABEL_TRAIN]).argmin() + 1
    # )  # gets index position of min value.
    logging.info(f"Max number of rounds => {cv_opt_num_rounds}")
    METRICS_LOG["max_num_boost_rounds_cv"] = cv_opt_num_rounds

    ############################################################################
    # TRAIN MODEL
    ############################################################################

    # Train Model
    """
    train method: params define the object, in particular the objective, which determines the type of models we are
        training.
        returns booster models.
    """
    # Prepare Best Parameters
    if "class_weight" not in best_params.keys():
        best_params["class_weight"] = "balanced"
        logger.warning("Class Weight not found in best params")
    if "scale_pos_weight" not in best_params.keys():
        best_params["scale_pos_weight"] = SCALE_POS_WEIGHT_BASE
        logger.warning("Scale-Pos-Weight not found in best params")

    best_params["num_boost_round"] = cv_opt_num_rounds
    logger.info(f"Training final model with best params => {best_params}")
    logger.info("---------- STARTING MODEL TRAIN --------------------")
    model = xgb.train(
        params=best_params, dtrain=DTRAIN, num_boost_round=cv_opt_num_rounds
    )
    logger.info("---------- END MODEL TRAIN --------------------")
    # Make Predictions
    yhat_train_probs = model.predict(DTRAIN)
    yhat_test_probs = model.predict(DTEST)
    yhat_all_probs = model.predict(DALL)

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
    METRICS_LOG["accuracy_train"] = accuracy_train
    METRICS_LOG["accuracy_test"] = accuracy_test
    METRICS_LOG["accuracy_all"] = accuracy_all

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
    METRICS_LOG["precision_0_train"] = clf_scores_train[0][0]
    METRICS_LOG["precision_1_train"] = clf_scores_train[0][1]
    METRICS_LOG["recall_0_train"] = clf_scores_train[1][0]
    METRICS_LOG["recall_1_train"] = clf_scores_train[1][1]
    METRICS_LOG["f1_0_train"] = clf_scores_train[2][0]
    METRICS_LOG["f1_1_train"] = clf_scores_train[2][1]

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
    METRICS_LOG["roc_auc_train"] = roc_auc_train
    METRICS_LOG["roc_auc_test"] = roc_auc_test

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
    try:
        plot_importance(model)
        plt.title("Feature Importance (Weight)")
        filename = "feature_importance_weight.png"
        path = os.path.join(DIR_EXPERIMENT, filename)
        plt.savefig(path)
        plt.close()
    except ValueError as err:
        logger.error(err)

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
        param_df = pd.DataFrame.from_dict(PARAM_LOGS, orient="index", columns=["VALUE"])
        param_df.to_csv(os.path.join(DIR_EXPERIMENT, "parameters.csv"), index=True)

        # Save Metrics
        metrics_df = pd.DataFrame.from_dict(
            METRICS_LOG, orient="index", columns=["VALUE"]
        )
        metrics_df.to_csv(os.path.join(DIR_EXPERIMENT, "metrics.csv"), index=True)

        # Save Feature Importance
        # imp_df = pd.DataFrame(
        #     {
        #         "FEATURE": model.feature_names,
        #         "IMPORTANCE": model.get_booster().get_score(importance_type="gain")
        #     }
        # )
        # imp_df.to_csv(
        #     os.path.join(DIR_EXPERIMENT, "model_feature_importance.csv"), index=False
        # )

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
        # cv_loss_df = pd.DataFrame(
        #     {"LOSS": cv_loss, "ITERATION": [x for x in range(len(cv_loss))]}
        # )
        # cv_loss_df.to_csv(os.path.join(DIR_EXPERIMENT, "cv_loss.csv"), index=False)

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

        logger.info(
            f"Experiment {MODEL_EXPERIMENT} completed successfully in {duration}"
        )
