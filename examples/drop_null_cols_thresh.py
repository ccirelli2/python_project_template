"""
Example of a function to drop columns where the null percentage exceeds a certain threhold.
"""
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
pd.set_option("display.max_rows", 100)

# Directories
DIR_DATA = d_config("DIR_DATA")
DIR_DATA_CLEAN = d_config("DIR_DATA_CLEAN")

# Config Files
CONFIG_MASTER = utils.load_config()
FILE_NAME_CLEAN = CONFIG_MASTER["FILE_NAMES"]["CLEAN"]
SEP = CONFIG_MASTER["FILE_SEP"]["ted_sd_raw_2019"]
NULL_THRESH = CONFIG_MASTER["DROP_COLUMNS"]["NULL_PCT_THRESHOLD"]


if __name__ == "__main__":
    # Load Data
    pd_df = utils.load_dataframe(
        directory=DIR_DATA_CLEAN, filename=FILE_NAME_CLEAN, sample=True, nrows=10_000
    )

    # Instantiate Transformer
    transformer = cleaning.DropNullColumns(dataframe=pd_df, threshold=NULL_THRESH)

    # Run Clean Pipeline
    transformer.clean()
