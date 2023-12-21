"""
###################################################
Pipeline Transformations                          #
###################################################
1.  Transf - Map constant -9 to python None     X #
2.  Transf - Drop cols w/ pct null >= thresh    X #
3.  Desc - Describe clean dataframe             X #
4.  Transf - Create in-out column               X #
5.  Transf - Create Target column               X #
6.  Transf - Create Final ML Dataset            X #
7.  Desc - Describe Final ML Dataset            X #
8.  Feature Selection - Correlation Matrix      0 #
9.  Feature Selection - Chi2 Test               X #
10. Feature Selection - Models                  X #
11. Feature Selection - Majority Vote           0 #
###################################################
###################################################
"""
# TODO: Need to capture the accuracy of each models.  There is no point in reporting feature importance for
#   in accurate models.
import logging
import pandas as pd
from decouple import config as d_config
from src import utils
from clean import (
    describe_clean,
    transf_map_val_to_null,
)
from enhance import (
    transf_drop_null_cols,
    transf_create_in_out_col,
    transf_create_target_col,
    transf_create_final_ml_dataset,
    describe_final_dataset,
)
from eda import (
    feature_importance_chi2,
    feature_importance_models,
)

# Settings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
pd.set_option("display.max_columns", None)

# Directories
DIR_DATA = d_config("DIR_DATA")
DIR_DATA_RAW = d_config("DIR_DATA_RAW")

# Config Files
CONFIG_MASTER = utils.load_config()
FILE_NAME_RAW = CONFIG_MASTER["FILE_NAMES"]["RAW"][2019]
SEP = CONFIG_MASTER["FILE_SEP"]["ted_sd_raw_2019"]

# Globals
SAMPLE = False
NUM_ROWS = 10_000

# Add Pipeline here
if __name__ == "__main__":
    # Load Raw Data
    transf_df = utils.load_dataframe(
        directory=DIR_DATA_RAW, filename=FILE_NAME_RAW, sample=SAMPLE, nrows=NUM_ROWS
    )

    # Clean: Map -9 to Null
    transf_df = transf_map_val_to_null.transform(pd_df=transf_df)

    # Describe: Clean Data
    describe_df = describe_clean.transform(pd_df=transf_df)

    # Enhance: Drop Null Columns
    transf_df = transf_drop_null_cols.transform(pd_df=transf_df)

    # Enhance: Create In-Out Column
    transf_df = transf_create_in_out_col.transform(pd_df=transf_df)

    # Enhance: Create Target Column
    transf_df = transf_create_target_col.transform(pd_df=transf_df)

    # Enhance: Create Final ML Dataset
    transf_df = transf_create_final_ml_dataset.transform(pd_df=transf_df)

    # Describe: Enhanced Data (Final Dataset before feature selection & modeling)
    describe_df = describe_final_dataset.transform(pd_df=transf_df)

    # Feature Selection: Correlation Matrix
    # TODO: Add Correlation Matrix Transformation

    # Feature Selection: Chi2 Test
    chi2_feature_imp_df = feature_importance_chi2.transform(pd_df=transf_df)

    # Feature Selection: - Models
    # TODO: Log accuracy of each models
    models_feature_imp_df = feature_importance_models.transform(pd_df=transf_df)

    # Model: LightGBM
    # TODO: add lightgbm models
