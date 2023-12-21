"""
Example script that takes the dataset from the final transformation and apply the final feature set to it.
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
DIR_DATA_MODEL = d_config("DIR_DATA_MODELS")
FEATURE_SET = CONFIG_MASTER["ML_MODEL_FEATURE_SET"]

# Filenames
INPUT_FILENAME = "TD_SD_ENH_TARGET_COL"
OUTPUT_FILENAME = CONFIG_FILE_NAMES["MODEL"]

# Load Datasets
enh_df = utils.load_dataframe(
    directory=DIR_DATA_ENHANCED,
    filename=INPUT_FILENAME,
    extension="csv",
    sample=True,
    nrows=1000,
)

# Apply Feature Set
assert not [
    c for c in FEATURE_SET if c not in enh_df.columns
], "Missing columns in feature set."

ml_df = enh_df[FEATURE_SET]

# Save Dataset
"Not saving for example script"
