"""
Structure
- Multiple transformations
- All wrapped in final clean function.

Cleaning Tasks
- Map -9 to Null
"""
import os
import logging
import pandas as pd
from decouple import config as d_config
from src.cleaning import map_val_to_null
from src import utils

# Settings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Globals
CONFIG_MASTER = utils.load_config()
NULL_VAL = CONFIG_MASTER["NULL_MAPPER"]["FROM"]
DIR_DATA_CLEAN = d_config("DIR_DATA_CLEAN")


def transform(pd_df: pd.DataFrame):
    assert isinstance(pd_df, pd.DataFrame), f"Expected pd.DataFrame, got {type(pd_df)}"
    logger.info(f"Transform {__name__}")
    transf_df = map_val_to_null(
        dataframe=pd_df, column=pd_df.columns.tolist(), val=NULL_VAL
    )
    utils.write_dataframe(
        pd_df=transf_df, directory=DIR_DATA_CLEAN, filename="TS_SD_CLEAN_VAL_TO_NULL"
    )
    return transf_df
