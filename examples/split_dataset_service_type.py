"""
Note: The Average LOS by In-Out Column is substantially different for services 1 & 2.
    Therefore, we will split the dataset between [In, Out] & [1, 2] and then run the model.
In-Out: Outpatient
    Mean LOS => 18.8
In-Out: 2
    Mean LOS => 6.3
In-out: Inpatient
    Mean LOS => 23.9
In-Out: 1
    Mean LOS => 5.7
"""

# Generic Libraries
import os
import uuid
import logging
import pandas as pd
from itertools import chain
from decouple import config as d_config

# Project Libraries
from src import utils

# Settings
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)
logger = logging.getLogger(__name__)
logger.setLevel(level="INFO")

# Config Files
CONFIG_MASTER = utils.load_config()
FEATURE_SET = CONFIG_MASTER["ML_MODEL_FEATURE_SET"]

# Directories
DIR_DATA_RAW = d_config("DIR_DATA_RAW")
DIR_DATA_ENHANCED = d_config("DIR_DATA_ENHANCED")
DIR_MODEL = d_config("DIR_DATA_MODELS")

# Filenames
INPUT_FILE_NAME = CONFIG_MASTER["FILE_NAMES"]["ENHANCED_TARGET_COL"]

# Run Mode
DEBUG_MODE = True

# Parameters
SPLIT_DATA_IN_OUT = True
IN_OUT_COL = "IN_OUT"
IN_OUT_VAL = ["1", "2"]


if __name__ == "__main__":
    # Load Dataset
    ml_df = utils.load_dataframe(
        directory=DIR_DATA_ENHANCED,
        filename=INPUT_FILE_NAME,
        extension="csv",
        sample=True,
        nrows=100_000,
    )

    if SPLIT_DATA_IN_OUT:
        logger.info(f"Limiting Dataset By In-Out Column: {IN_OUT_VAL}")
        logger.info(f"Starting Dimensions => {ml_df.shape}")
        ml_df = ml_df[ml_df[IN_OUT_COL].isin(IN_OUT_VAL)]
        print(f"Final dimensions: {ml_df.shape}")
