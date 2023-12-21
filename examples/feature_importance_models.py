"""
#TODO: Should have one hot encoded nominal features.
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
FEATURE_IMPORTANCE = CONFIG_MASTER["FEATURE_IMPORTANCE"]
MODEL_FEATURES = FEATURE_IMPORTANCE["MODELS"]

# Directories
DIR_DATA_ENHANCED = d_config("DIR_DATA_ENHANCED")
DIR_DATA_EDA_MODELS = d_config("DIR_DATA_EDA_MODELS")

# Globals
SAVE_RESULTS = True
SAMPLE = True
NUM_ROWS = 5_000


if __name__ == "__main__":
    # Load Data
    source_df = utils.load_dataframe(
        directory=DIR_DATA_ENHANCED,
        filename="TD_SD_ENH_TARGET_COL",
        sample=SAMPLE,
        nrows=NUM_ROWS,
    )

    fi = feature_importance.FeatureImportanceModels(
        data=source_df,
        output_dir=DIR_DATA_EDA_MODELS,
        sample=False,
        nan_fill=True,
        feature_column_names=MODEL_FEATURES,
        save_results=False,
        include_index=False,
    )

    fi.get_importance()
    print(fi.feature_importance_dataframe.head())
