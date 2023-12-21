"""
Cleaning Tasks
- Create In/Out column.
"""
import os
import logging
import pandas as pd
from decouple import config as d_config
from src import utils
from src.enhance import NewColMapper

# Settings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Load Config Files
CONFIG_MASTER = utils.load_config()
CONFIG_IN_OUT_COL_NAMES = CONFIG_MASTER["IN_OUT_COL_NAMES"]
CONFIG_IN_OUT_COL_MAPPING = CONFIG_MASTER["IN_OUT_COL_MAPPING"]

# Directories
DIR_DATA_ENHANCED = d_config("DIR_DATA_ENHANCED")

# Globals
INPUT_COL = CONFIG_IN_OUT_COL_NAMES["IN"]
OUTPUT_COL = CONFIG_IN_OUT_COL_NAMES["OUT"]


def transform(pd_df: pd.DataFrame) -> pd.DataFrame:
    """ """
    assert isinstance(pd_df, pd.DataFrame), f"Expected pd.DataFrame, got {type(pd_df)}"
    logger.info(f"Transform {__name__}")
    transformer = NewColMapper(
        dataframe=pd_df,
        mapper=CONFIG_IN_OUT_COL_MAPPING,
        input_col=INPUT_COL,
        output_col=OUTPUT_COL,
    )
    transf_df = transformer.transform().dataframe
    utils.write_dataframe(
        pd_df=transf_df, directory=DIR_DATA_ENHANCED, filename="TD_SD_ENH_IN_OUT_COL"
    )
    return transf_df
