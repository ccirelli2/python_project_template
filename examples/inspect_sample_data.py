"""
Inspect sample (top n rows) of each dataset provided as raw files.
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

# Globals
DIR_DATA = d_config("DIR_DATA")
DIR_DATA_RAW = os.path.join(DIR_DATA, "raw")
CONFIG_MASTER = utils.load_config()
FILE_NAME = CONFIG_MASTER["FILE_NAMES"]["RAW"][2019]

# Load Data
pd_df = utils.load_dataframe(
    directory=DIR_DATA_RAW, filename=FILE_NAME, sample=True, nrows=10_000
)
logger.info(f"pd_df.shape: {pd_df.shape}")
