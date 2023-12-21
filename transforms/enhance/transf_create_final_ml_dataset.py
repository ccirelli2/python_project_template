"""
Final dataset prior to modeling.
This script takes the dataset from the final transformation and apply the final feature set to it.
"""
import os
import logging
import pandas as pd
from decouple import config as d_config
from src import utils

# Settings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
pd.set_option("display.max_columns", None)

# Config Files
CONFIG_MASTER = utils.load_config()
CONFIG_FILE_NAMES = CONFIG_MASTER["FILE_NAMES"]

# Globals
DIR_DATA_ENHANCED = d_config("DIR_DATA_ENHANCED")
FEATURE_SET = CONFIG_MASTER["ML_MODEL_DATASET"]

# Filenames
OUTPUT_FILENAME = CONFIG_FILE_NAMES["MODEL"]


def transform(pd_df: pd.DataFrame) -> pd.DataFrame:
    """ """
    assert isinstance(pd_df, pd.DataFrame), f"Expected pd.DataFrame, got {type(pd_df)}"
    logger.info(f"Transform {__name__}")

    assert not [
        c for c in FEATURE_SET if c not in pd_df.columns
    ], "Missing columns in feature set."
    transf_df = pd_df[FEATURE_SET]

    utils.write_dataframe(
        pd_df=transf_df, directory=DIR_DATA_ENHANCED, filename=OUTPUT_FILENAME
    )
    return transf_df
