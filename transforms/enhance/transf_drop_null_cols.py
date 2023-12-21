"""
Drop columns whose percentage of null values equals or exceeds threshold.
"""
import os
import logging
import pandas as pd
from src import utils
from decouple import config as d_config
from src.cleaning import DropNullColumns

# Settings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Load Config Files
CONFIG_MASTER = utils.load_config()
DIR_DATA_CLEAN = d_config("DIR_DATA_CLEAN")
COL_PCT_NULL_THRESH = CONFIG_MASTER["DROP_COLUMNS"]["NULL_PCT_THRESHOLD"]


def transform(pd_df: pd.DataFrame) -> pd.DataFrame:
    """ """
    assert isinstance(pd_df, pd.DataFrame), f"Expected pd.DataFrame, got {type(pd_df)}"
    logger.info(f"Transform {__name__}")
    transf_df = DropNullColumns(dataframe=pd_df, threshold=COL_PCT_NULL_THRESH).clean()
    utils.write_dataframe(
        pd_df=transf_df, directory=DIR_DATA_CLEAN, filename="TS_SD_CLEAN_DROP_NULL_COLS"
    )
    return transf_df
