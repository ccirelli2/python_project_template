"""
Script to show how to enhance dataste w/ additional columns
"""
import os
import logging
import pandas as pd
import numpy as np
import math
from decouple import config as d_config
from src import utils
from src.enhance import NewColMapper

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

# Project Modules
from src.enhance import NewColMapper

if __name__ == "__main__":
    # Load Data
    pd_df = utils.load_dataframe(
        directory=DIR_DATA_RAW, filename=FILE_NAME_RAW, sample=True, nrows=100_000
    )
    pd_df = pd_df[["SERVICES_D"]]

    # Insert Row w/ Null value (test mapping)
    pd_df.loc[0, "SERVICES_D"] = np.nan

    # Instantiate Class
    transformer = NewColMapper(
        dataframe=pd_df,
        mapper=IN_OUT_COL_MAPPING,
        input_col="SERVICES_D",
        output_col="IN_OUT",
    )

    transf_df = transformer.transform().dataframe
    print(transf_df.head())
