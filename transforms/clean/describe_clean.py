"""

"""
import os
import logging
import pandas as pd
from decouple import config as d_config
from src import utils, describer


# Settings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
pd.set_option("display.max_columns", None)

# Globals
DIR_DATA = d_config("DIR_DATA")
DIR_DATA_RAW = os.path.join(DIR_DATA, "raw")
DIR_DATA_CLEAN = d_config("DIR_DATA_CLEAN")
CONFIG_MASTER = utils.load_config()
FILE_NAME = CONFIG_MASTER["FILE_NAMES"]["CLEAN"]


# Instantiate Data Describer
def transform(pd_df: pd.DataFrame):
    assert isinstance(pd_df, pd.DataFrame), f"Expected pd.DataFrame, got {type(pd_df)}"
    logger.info(f"Transform {__name__}")
    describe_df = describer.DataDescriber(pd_df=pd_df).describe()
    utils.write_dataframe(
        pd_df=describe_df, directory=DIR_DATA_CLEAN, filename="TS_SD_CLEAN_DESC"
    )
    return describe_df
