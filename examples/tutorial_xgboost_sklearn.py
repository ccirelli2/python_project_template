"""


########################################################################################################################
Reference:
########################################################################################################################
- Documentation: https://xgboost.readthedocs.io/en/stable/
- Tutorials: https://xgboost.readthedocs.io/en/stable/get_started.html
- Gradient Boosting https://en.wikipedia.org/wiki/Gradient_boosting
- Guide to Gradient Boosting: https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/
- XGBoost Parameters: https://xgboost.readthedocs.io/en/stable/parameter.html
- Turning Parameters: https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
- Learning Rate: https://machinelearningmastery.com/tune-learning-rate-for-gradient-boosting-with-xgboost-in-python/
- Class Weights: https://medium.com/@rithpansanga/xgboost-and-imbalanced-datasets-strategies-for-handling-class-imbalance-cdd810b3905c#:~:text=Using%20class%20weights%3A%20XGBoost%20allows,influence%20on%20the%20model's%20predictions.
- Imbalanced Data: https://medium.com/@thedatabeast/handling-imbalanced-data-with-xgboost-class-weight-parameter-c67b7257515b
- Ensemble Learning: https://courses.analyticsvidhya.com/courses/ensemble-learning-and-ensemble-learning-techniques?utm_source=blog&utm_medium=complete-guide-parameter-tuning-gradient-boosting-gbm-python
- Gradient Boosting: https://www.analyticsvidhya.com/blog/2021/09/gradient-boosting-algorithm-a-complete-guide-for-beginners/


########################################################################################################################
Parameters:
########################################################################################################################
- Booster: gbtree (default), dart, gblinear.
- Device: CPU (default), GPU.
- nthreads:
    - Number of parallel threads used to run XGBoost.  Equal to the number of physical cores by default.
- disable_default_eval_metric:
    - Disable default metric.  Set to 1 to disable. ?????
- learning_rate:
    - Boosting learning rate (xgb's "eta"). Stepsize shrinkage used in update to prevent overfitting.
    - After each boosting step we can directly get the weights of each of new features, and eta shrinks the feature weights.
    - controls the step size at which the algorithm makes updates to the model weights
    - Setting values less than 1.0 has the effect of making less corrections for each tree added to the model.
- gamma: Minimum loss reduction required to make a further partition on a leaf node of the tree.
    - The larger gamma is, the more conservative the algorithm will be.
- max_depth:
    - Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit.
    - 0 indicates no limit on depth.
- n_estimators:
    - number of tress to create in model.
- min_child_weight:
    - Minimum sum of instance weight (hessian) needed in a child.  In linear regression, this corresponds
        to the minimum number of instances needed to be in each node.
    - The larger the value the more conservative the algorithm will be.
    - range is [0,∞]
- max_delta_step:
    - Maximum delta step we allow each leaf output to be. If the value is set to 0,
        it means there is no constraint.
    - If it is set to a positive value, it can help making the update step more conservative.
    - Usually this parameter is not needed, but it might help in logistic regression when class is extremely imbalanced.
    - Set it to value of 1-10 might help control the update.
- class_weights:
    - In XGBoost, the class_weight the parameter is used to adjust the weights of different classes in the
        training data to handle imbalanced class distribution.
    - The class_weight parameter can be set to either "balanced" or a dictionary of weights for each class.
    - Sklearn has a package to calculate class weights: from sklearn.utils.class_weight import compute_class_weight
- Class Imbalance & Class Weights:
    - https://www.analyticsvidhya.com/blog/2020/10/improve-class-imbalance-class-weights/#:~:text=Class%20weights%20are%20a%20technique,bias%20towards%20the%20majority%20class.
- Class Imbalance w/ Examples:
    - https://medium.com/@ravi.abhinav4/improving-class-imbalance-with-class-weights-in-machine-learning-af072fdd4aa4
    - Logistic Regression w/ Modified Loss Function:
        - L(y, p) = -(w_0 * y * log(p) + w_1 * (1 — y) * log(1 — p))
        - When y = 0(true negative class), the loss is -w_1 * log(p), where p is the predicted probability for class 1.
        - The model will be penalized more for misclassifying the positive class when w_1 is higher,
            which helps to focus more on the minority class.
- subsample:
    - Subsample ratio of the training instance.  Setting it to 0.5 means that XGBoost would randomly
        sample half of the training data prior to growing trees and this will prevent over-fitting.
- lambda:
    - L2 regularization term on weights (analogous to Ridge regression).
    - range is [0,∞]
- alpha
    - L1 regularization term on weights (analogous to Lasso regression).
    - range is [0,∞]
- tree_method:
    - The tree construction algorithm used in XGBoost.
    - auto: Same as the hist tree method.
    - exact: Exact greedy algorithm. Enumerates all split candidates.
    - approx: Approximate greedy algorithm using quantile sketch and gradient histogram.
    - hist: Faster histogram optimized approximate greedy algorithm
- scale_pos_weight:
    - Control the balance of positive and negative weights, useful for unbalanced classes.
    - A typical value to consider: sum(negative instances) / sum(positive instances)
    - Difference w/ class_weights: https://datascience.stackexchange.com/questions/54043/differences-between-class-weight-and-scale-pos-weight-in-lightgbm
- grow_policy:
    - Controls a way new nodes are added to the tree.
    - depthwise: split at nodes closest to the root.
    - lossguide: split at nodes with highest loss change.
        - finding the node where a split would result in the greatest reduction in the loss function,
            and splitting this node, regardless of how deep it will make the tree.
        - This may result in a tree with one or two very deep branches, while the other branches may not have grown very far.
        - ref: https://subscription.packtpub.com/book/data/9781800564480/6/ch06lvl1sec34/another-way-of-growing-trees:-xgboost's-grow-policy
- max_leaves
    - Maximum number of nodes to be added.  Only relevant when grow_policy is set to lossguide.
- max_bin
    - Maximum number of discrete bins to bucket continuous features.  Only used if tree_method is set to hist.
- max_cat_to_onehot
    - Maximum number of categories for one-hot encoding.  Only used if tree_method is set to gpu_hist.
- max_features
    - Maximum number of features used in each tree.  Only used if tree_method is set to gpu_hist.
    - auto: sqrt(num_features)
- init
    - This affects initialization of the output.
    - This can be used if we have made another model whose outcome is to be used as the initial estimates for GBM.
    - **Note**: Look into this further.  Is this an initial prediction for each observation?
- Objective
    - specifies the learning task.
    - default: reg:squarederror
    - reg:logistic: logistic regression, output probability
    - binary:logistic: logistic regression for binary classification, output probability
    - binary:logitraw: logistic regression for binary classification, output score before logistic transformation
    - binary: hinge: hinge loss for binary classification. This makes predictions of 0 or 1, rather than producing probabilities.
- Eval metric:
    - eval_metric: Evaluation metrics for validation data, a default metric will be assigned according to objective
    - logloss: Negative log-likelihood
    - auc: Area under the curve


########################################################################################################################
# XGBoost Dataset (DMatrix):
########################################################################################################################
- xgb.DMatrix
    - Data matrix used in XGBoost.
    - pandas dataframe:
        - data = pandas.DataFrame(np.arange(12).reshape((4,3)), columns=['a', 'b', 'c'])
        - label = pandas.DataFrame(np.random.randint(2, size=4))
        - dtrain = xgb.DMatrix(data, label=label)
- missing:
    - parameter used to identify the default value imputed for missing values.
- feature_names:
    - set names of features for X data.
- feature_types:
    - when 'enable_categoricacl' is set to 'True', string "c" represents categorical data type while "q" represents
        numerical feature type.
    - For categorical, the input is assumed to be preprocessed and encoded by the user.
    - Dictionary object to map column name to data type.
        - example: xgb.DMatrix(data, feature_types={'A': 'c', 'B': 'q'})
References:
- DMatrix: https://xgboost.readthedocs.io/en/stable/dev/group__DMatrix.html?highlight=Dmatrix
- Dmatrix Class Object: https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.core
- Dmatrix weight:
    - offset parameter?
    - https://stackoverflow.com/questions/35983565/how-is-the-parameter-weight-dmatrix-used-in-the-gradient-boosting-procedure

########################################################################################################################
# Model Fit (xgboost.sklearn)
########################################################################################################################
XGBoostClassifier Fit Method
- Fit gradient boosting classifier
- X: independent features. feature matrix.
- y: labels.
- eval_set: list of (X, y) tuple pairs to use as validation sets for which metrics will be computed.
- eval_metric: str, list of strs.
- early stopping rounds ;int


########################################################################################################################
# Hyperparameter Tuning:
########################################################################################################################

References:
- https://www.kaggle.com/code/yassinealouini/hyperopt-the-xgboost-model/script
- https://medium.com/analytics-vidhya/hyperparameter-tuning-hyperopt-bayesian-optimization-for-xgboost-and-neural-network-8aedf278a1c9

"""
import os
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import pandas as pd
from decouple import config as d_config

# XGboost
import xgboost as xgb
from xgboost import XGBClassifier, DMatrix

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
USE_SAMPLE = True
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
X = ml_df[FEATURE_SET]
y = ml_df[TARGET_COLUMN]

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
xg_train = DMatrix(
    data=X_train, label=y_train, missing=None, feature_names=X_train.columns.tolist()
)

xg_test = DMatrix(
    data=X_test, label=y_test, missing=None, feature_names=X_train.columns.tolist()
)

# Instantiate Model
print("Instantiating XGBClassifier")

clf = XGBClassifier(
    objective="binary:logistic",
    booster="gbtree",  # 'gblinear', 'dart'
    device="cpu",  # 'gpu', cuda
    verbosity=1,  # 0 (silent), 1 (warning), 2 (info), 3 (debug)
    n_estimators=2,
    learning_rate=0.01,  # default 0.3
    gamma=0,  # default 0
    max_depth=0,  # default 6
    min_child_weight=1,  # default 1
    max_delta_step=0,  # default 0
    class_weight="balanced",  # default None. xgboost will automatically balance class weights.
    subsample=0.5,  # default 1
    sampling_method="uniform",  # default uniform, gradient based.
    reg_alpha=1,  # default 0
    reg_lambda=1,  # default 1
    tree_method="hist",  # default auto, same as hist method.
    max_leaves=0,  # default 0
    max_bin=256,  # default 256, only used if tree_method is set to hist.
    max_features="auto",  # default None
)

# Train Model using fit method.
print("Fitting Classifier")
clf.fit(
    X=X_train,
    y=y_train,
    # early_stopping_rounds=10
)
print("Fitting completed")

# Generate Predictions (Note: returns labels vs probs).
yhat_binary = clf.predict(X_test)
yhat_probs = clf.predict_proba(X_test)
