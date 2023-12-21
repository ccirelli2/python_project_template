"""
According to the data dictionary -9 represents a missing value.
We need to replace -9 w/ null.
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

# Globals
DIR_DATA = d_config("DIR_DATA")
DIR_DATA_RAW = os.path.join(DIR_DATA, "raw")
DIR_DATA_ENHANCED = d_config("DIR_DATA_ENHANCED")
CONFIG_MASTER = utils.load_config()
FILE_NAME_RAW = CONFIG_MASTER["FILE_NAMES"]["RAW"][2019]
FILE_NAME_OUTPUT = CONFIG_MASTER["FILE_NAMES"]["ENHANCED"]
IN_OUT_COL_MAPPING = CONFIG_MASTER["IN_OUT_COL_MAPPING"]
SEP = CONFIG_MASTER["FILE_SEP"]["ted_sd_raw_2019"]


if __name__ == "__main__":
    # Load Data
    pd_df = pd.read_csv(
        os.path.join(DIR_DATA_RAW, FILE_NAME_RAW), sep=SEP, nrows=10_000
    )

    columns = pd_df.columns.tolist()
    pd_df = cleaning.map_val_to_null(dataframe=pd_df, column=columns, val=-9)
    print(pd_df.head())
