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


def transform(pd_df: pd.DataFrame):
    logger.info(f"Transform {__name__}")
    fi = feature_importance.FeatureImportanceChi2(
        data=pd_df,
        output_dir=DIR_DATA_EDA_CHI2,
        feature_column_names=CHI2_FEATURES,
        sample=False,
        save_results=True,
        include_index=False,
    )
    transf_df = fi.get_importance().feature_importance_dataframe
    return transf_df
