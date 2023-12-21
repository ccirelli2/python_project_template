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


def transform(pd_df: pd.DataFrame):
    fi = feature_importance.FeatureImportanceModels(
        data=pd_df,
        output_dir=DIR_DATA_EDA_MODELS,
        sample=False,
        nan_fill=True,
        feature_column_names=MODEL_FEATURES,
        include_index=False,
        save_results=True,
    )
    transf_df = fi.get_importance().feature_importance_dataframe
    return transf_df
