import os
import logging
import pandas as pd
from decouple import config as d_config
from src import utils, cleaning

# Settings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
pd.set_option("display.max_columns", None)

# Globals
DIR_INPUT = d_config("DIR_DATA_CLEAN")
DIR_OUTPUT = d_config("DIR_DATA_STRUCTURE")

CONFIG_MASTER = utils.load_config()
SEP = CONFIG_MASTER["FILE_SEP"]["TED_SD_CLEAN"]
FILE_NAME_CLEAN = CONFIG_MASTER["FILE_NAMES"]["CLEAN"]
FILE_NAME_OUTPUT = "TED_SD_PCT_NULL_BY_COL"

# Load Dataframe
pd_df = utils.load_dataframe(
    directory=DIR_INPUT, filename=FILE_NAME_CLEAN, extension="csv"
)

# Get null percentage by column
null_df = pd_df.isna().sum().reset_index()
null_df["Pct_Null"] = null_df[0] / len(pd_df)
utils.write_dataframe(
    pd_df=null_df, directory=DIR_OUTPUT, filename=FILE_NAME_OUTPUT, extension="csv"
)
