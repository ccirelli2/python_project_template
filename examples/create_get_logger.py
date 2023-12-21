# Generic Libraries
import os
import logging
import pandas as pd
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
DIR_EXAMPLES = d_config("DIR_EXAMPLES")

if __name__ == "__main__":
    logger = utils.Logger(directory=os.getcwd(), filename="test").get_logger()
    logger.info("Test Info")
