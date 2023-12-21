"""
"""
import logging
import pandas as pd
from decouple import config as d_config
from src import utils, feature_importance

# Settings
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)
logger = logging.getLogger(__name__)
logger.setLevel(level="INFO")

# Config Files
CONFIG_MASTER = utils.load_config()
FEATURE_IMPORTANCE_FEATURES = CONFIG_MASTER["FEATURE_IMPORTANCE"]
CHI2_FEATURES = FEATURE_IMPORTANCE_FEATURES["CHI2"]

# Directories
DIR_DATA_ENHANCED = d_config("DIR_DATA_ENHANCED")
DIR_DATA_EDA_CHI2 = d_config("DIR_DATA_EDA_CHI2")

# Globals
SAVE_RESULTS = True
SAMPLE = True
NUM_ROWS = 100_000


if __name__ == "__main__":
    # Load Data
    source_df = utils.load_dataframe(
        directory=DIR_DATA_ENHANCED,
        filename="TD_SD_ENH_TARGET_COL",
        sample=SAMPLE,
        nrows=NUM_ROWS,
    )

    # Instantiate Feature Importance Class
    fi = feature_importance.FeatureImportanceChi2(
        data=source_df,
        output_dir=DIR_DATA_EDA_CHI2,
        feature_column_names=CHI2_FEATURES,
        save_results=SAVE_RESULTS,
    )

    # Execute Individual Methods
    fi.get_importance()

    print(fi.feature_importance_dataframe.head())
