"""
Utilize xgb train method (non sklearn approach).

TODO: investigate what xgb.cv returns (what does return look like.)
TODO: HPO returning
    Parameters: { "class_weight", "max_features", "metric", "n_estimators", "num_leaves", "sub_sample" } are not used.


References:
- https://medium.com/analytics-vidhya/hyperparameter-tuning-hyperopt-bayesian-optimization-for-xgboost-and-neural-network-8aedf278a1c9
- https://www.kaggle.com/code/yassinealouini/hyperopt-the-xgboost-model/script
- example cv: https://xgboost.readthedocs.io/en/stable/python/examples/cross_validation.html
- xgb.cv docs: https://xgboost.readthedocs.io/en/stable/python/python_api.html


fmin(fn, space, algo=None, max_evals=None, timeout=None, loss_threshold=None, trials=None, rstate=None, allow_trials_fmin=True, pass_expr_memo_ctrl=None, catch_eval_exceptions=False, verbose=True, return_argmin=True, points_to_evaluate=None, max_queue_len=1, show_progressbar=True, early_stop_fn=None, trials_save_file='')
    Minimize a function over a hyperparameter space.

    More realistically: *explore* a function over a hyperparameter space
    according to a given algorithm, allowing up to a certain number of
    function evaluations.  As points are explored, they are accumulated in
    `trials`


    Parameters
    ----------

    fn : callable (trial point -> loss)
        This function will be called with a value generated from `space`
        as the first and possibly only argument.  It can return either
        a scalar-valued loss, or a dictionary.  A returned dictionary must
        contain a 'status' key with a value from `STATUS_STRINGS`, must
        contain a 'loss' key if the status is `STATUS_OK`. Particular
        optimization algorithms may look for other keys as well.  An
        optional sub-dictionary associated with an 'attachments' key will
        be removed by fmin its contents will be available via
        `trials.trial_attachments`. The rest (usually all) of the returned
        dictionary will be stored and available later as some 'result'
        sub-dictionary within `trials.trials`.

    space : hyperopt.pyll.Apply node or "annotated"
        The set of possible arguments to `fn` is the set of objects
        that could be created with non-zero probability by drawing randomly
        from this stochastic program involving involving hp_<xxx> nodes
        (see `hyperopt.hp` and `hyperopt.pyll_utils`).
        If set to "annotated", will read space using type hint in fn. Ex:
        (`def fn(x: hp.uniform("x", -1, 1)): return x`)

    algo : search algorithm
        This object, such as `hyperopt.rand.suggest` and
        `hyperopt.tpe.suggest` provides logic for sequential search of the
        hyperparameter space.

    max_evals : int
        Allow up to this many function evaluations before returning.

    timeout : None or int, default None
        Limits search time by parametrized number of seconds.
        If None, then the search process has no time constraint.

    loss_threshold : None or double, default None
        Limits search time when minimal loss reduced to certain amount.
        If None, then the search process has no constraint on the loss,
        and will stop based on other parameters, e.g. `max_evals`, `timeout`

    trials : None or base.Trials (or subclass)
        Storage for completed, ongoing, and scheduled evaluation points.  If
        None, then a temporary `base.Trials` instance will be created.  If
        a trials object, then that trials object will be affected by
        side-effect of this call.

    rstate : numpy.random.Generator, default numpy.random or `$HYPEROPT_FMIN_SEED`
        Each call to `algo` requires a seed value, which should be different
        on each call. This object is used to draw these seeds via `randint`.
        The default rstate is
        `numpy.random.default_rng(int(env['HYPEROPT_FMIN_SEED']))`
        if the `HYPEROPT_FMIN_SEED` environment variable is set to a non-empty
        string, otherwise np.random is used in whatever state it is in.

    verbose : bool
        Print out some information to stdout during search. If False, disable
            progress bar irrespectively of show_progressbar argument

    allow_trials_fmin : bool, default True
        If the `trials` argument

    pass_expr_memo_ctrl : bool, default False
        If set to True, `fn` will be called in a different more low-level
        way: it will receive raw hyperparameters, a partially-populated
        `memo`, and a Ctrl object for communication with this Trials
        object.

    return_argmin : bool, default True
        If set to False, this function returns nothing, which can be useful
        for example if it is expected that `len(trials)` may be zero after
        fmin, and therefore `trials.argmin` would be undefined.

    points_to_evaluate : list, default None
        Only works if trials=None. If points_to_evaluate equals None then the
        trials are evaluated normally. If list of dicts is passed then
        given points are evaluated before optimisation starts, so the overall
        number of optimisation steps is len(points_to_evaluate) + max_evals.
        Elements of this list must be in a form of a dictionary with variable
        names as keys and variable values as dict values. Example
        points_to_evaluate value is [{'x': 0.0, 'y': 0.0}, {'x': 1.0, 'y': 2.0}]

    max_queue_len : integer, default 1
        Sets the queue length generated in the dictionary or trials. Increasing this
        value helps to slightly speed up parallel simulatulations which sometimes lag
        on suggesting a new trial.

    show_progressbar : bool or context manager, default True (or False is verbose is False).
        Show a progressbar. See `hyperopt.progress` for customizing progress reporting.

    early_stop_fn: callable ((result, *args) -> (Boolean, *args)).
        Called after every run with the result of the run and the values returned by the function previously.
        Stop the search if the function return true.
        Default None.

    trials_save_file: str, default ""
        Optional file name to save the trials object to every iteration.
        If specified and the file already exists, will load from this file when
        trials=None instead of creating a new base.Trials object

    Returns
    -------

    argmin : dictionary
        If return_argmin is True returns `trials.argmin` which is a dictionary.  Otherwise
        this function  returns the result of `hyperopt.space_eval(space, trails.argmin)` if there
        were successfull trails. This object shares the same structure as the space passed.
        If there were no successfull trails, it returns None.

"""
########################################################################################################################
# Setup
########################################################################################################################

import os
from collections import Counter
import pandas as pd
import numpy as np
from decouple import config as d_config

# XGboost
import xgboost as xgb
from xgboost import DMatrix

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Hyoperopt
import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll.stochastic import sample

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
DEBUG_MODE = True
USE_SAMPLE = False if not DEBUG_MODE else True
NUM_ROWS = 10_000
TEST_SIZE = 0.33
TARGET_COLUMN = "TARGET"
INPUT_FILE_NAME = CONFIG_MASTER["FILE_NAMES"]["MODEL"]

########################################################################################################################
# Load Dataset
########################################################################################################################
print(f"Loading data w/ file name => {INPUT_FILE_NAME}")
ml_df = utils.load_dataframe(
    directory=DIR_DATA_ENHANCED,
    filename=INPUT_FILE_NAME,
    extension="csv",
    sample=USE_SAMPLE,
    nrows=NUM_ROWS,
)

########################################################################################################################
# Inspect Data
########################################################################################################################

print(ml_df.head())
print(f"Dataframe Shape => {ml_df.shape}")

# Get Class Target
print(
    f"Class Distribution => {Counter(ml_df.TARGET)}, Num Classes => {len(Counter(ml_df.TARGET))}"
)
print(f"Percentage Class 1 => {Counter(ml_df.TARGET)[1] / len(ml_df)}")
print(f"Percentage Class 0 => {Counter(ml_df.TARGET)[0] / len(ml_df)}")

########################################################################################################################
# Build ML Dataset
########################################################################################################################

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

########################################################################################################################
# Hyperparameter Optimization
########################################################################################################################

# HPO Parameters
N_HPO_EVALS = 50 if not DEBUG_MODE else 5
N_HPO_EVALS_EARLY_STOP = N_HPO_EVALS
HPO_SEED = 123456
HYPEROPT_ALGO = tpe.suggest  # tpe.suggest OR hyperopt.rand.suggest
HPO_STRATIFIED = True
HPO_CV_FOLDS = 5
INTEGER_PARAMETERS = [
    "max_depth",
    "num_leaves",
    "max_bin",
    "max_leaves"
    # "min_data_in_leaf",
    # "min_data_in_bin",
]

# Globals
obj_call_count = 0
cur_best_loss = 0


def objective(params):
    # Define Parameters as globals
    global obj_call_count, cur_best_loss, dtrain, dtest

    # Log Best
    obj_call_count += 1
    print(f"Objective call # {obj_call_count}, cur_best_loss {cur_best_loss}")

    # Sample Parameter Space
    xgb_params = sample(space)
    print(f"Training with parameters => {xgb_params}")

    # Coerce Parameters from Float to Int
    for p in INTEGER_PARAMETERS:
        if xgb_params.get(p):
            xgb_params[p] = int(xgb_params[p])

    # Get Num Rounds
    num_rounds = int(params["n_estimators"])
    del params["n_estimators"]

    # Build Model
    """
    metrics:
        metric to watch during training.
        when it is not specified, the evaluation metric is chosen according to objective function.
    ::error: binary classification error rate
    ::logloss: negative log likelihood function
    """
    cv_results = xgb.cv(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=num_rounds,
        nfold=HPO_CV_FOLDS,
        stratified=HPO_STRATIFIED,
        # evals=watchlist,
        verbose_eval=True,
        # metrics=xgb_params['metric'],
        seed=HPO_SEED,
        # early_stopping_rounds=N_HPO_EVALS_EARLY_STOP,
        callbacks=[
            xgb.callback.EvaluationMonitor(show_stdv=True),
            # xgb.callback.EarlyStopping(10),
        ],
    )

    # Return Best Loss (cv_results returns a dataframe)
    "Dataframe columns contain loss metric defined"
    # loss_metric = f"train-{params['metric']}-mean"
    loss_metric = f"train-logloss-mean"  # note needs to be dynamic if we have more than one metric.
    scores = cv_results[loss_metric]
    best_loss = scores[np.array(scores).argmin()]

    if best_loss > cur_best_loss:
        cur_best_loss == best_loss

    print(f"\t\t Best Loss => {best_loss}\n\n\n")
    # Return Results
    """
    It can return either a scalar-valued loss, or a dictionary.  A returned dictionary must
    contain a 'status' key with a value from `STATUS_STRINGS`, must
    contain a 'loss' key if the status is `STATUS_OK`.
    """
    return {"loss": best_loss, "status": STATUS_OK}


# Define Parameter Space
space = {
    "objective": "binary:logistic",
    "device": "cpu",
    "verbosity": 1,
    "class_weight": "balanced",
    "max_features": "auto",
    "metric": "binary_logloss",
    "n_estimators": 100,
    "num_leaves": hp.uniform("num_leaves", 2, 1000),
    "booster": hp.choice("booster", ["gbtree", "gblinear", "dart"]),
    "learning_rate": hp.uniform("learning_rate", 0.01, 0.1),
    "gamma": hp.uniform("gamma", 0, 100),
    "max_depth": hp.uniform("max_depth", 0, 1000),
    "min_child_weight": hp.uniform("min_child_weight", 0, 1000),
    "max_delta_step": hp.uniform("max_delta_step", 0, 10),
    "sub_sample": hp.uniform("sub_sample", 0.5, 1),
    "sampling_method": hp.choice("sampling_method", ["uniform"]),
    "reg_alpha": hp.uniform("reg_alpha", 0, 30),
    "reg_lambda": hp.uniform("reg_lambda", 0, 30),
    "tree_method": hp.choice("tree_method", ["auto", "hist", "exact", "approx"]),
    "max_leaves": hp.uniform("max_leaves", 10, 1000),
    "max_bin": hp.uniform("max_bin", 10, 250),
}

########################################################################################################################
# Run Hyperopt
########################################################################################################################

# Track Trials
trials = Trials()

# Execute fmin
hpo_results = fmin(
    fn=objective,
    space=space,
    algo=HYPEROPT_ALGO,
    max_evals=N_HPO_EVALS,
    trials=trials,
    verbose=1,
)

# Get Trial Results
best_params = hpo_results
best_run_loss = min([x["result"]["loss"] for x in trials])
print(f"Best Overall Trial Log Loss => {best_run_loss}")
print(f"Best Parameters => {best_params}")
