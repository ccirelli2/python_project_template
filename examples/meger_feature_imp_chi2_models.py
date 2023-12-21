"""
"""
import logging
import pandas as pd
from decouple import config as d_config
from src import utils
from src.feature_importance import MergeImportance

# Settings
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)
logger = logging.getLogger(__name__)
logger.setLevel(level="INFO")

# Config Files
CONFIG_MASTER = utils.load_config()

# Directories
DIR_DATA_EDA_CHI2 = d_config("DIR_DATA_EDA_CHI2")
DIR_DATA_EDA_MODELS = d_config("DIR_DATA_EDA_MODELS")
DIR_DATA_EDA_ALL = d_config("DIR_DATA_EDA_ALL")


if __name__ == "__main__":
    chi2_df = utils.load_dataframe(
        directory=DIR_DATA_EDA_CHI2, filename="feature_importance", extension="csv"
    )

    model_df = utils.load_dataframe(
        directory=DIR_DATA_EDA_MODELS, filename="feature_importance", extension="csv"
    )

    mi = MergeImportance(
        chi2_df=chi2_df,
        model_df=model_df,
        output_dir=DIR_DATA_EDA_ALL,
        merge_column_name="FEATURE",
    )

    mi.transform()
