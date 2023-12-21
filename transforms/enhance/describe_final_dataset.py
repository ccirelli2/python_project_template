"""
Generate data description of dataset prior to building models.
"""
import logging
import pandas as pd
from decouple import config as d_config
from src import utils, describer

# Settings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Directories
DIR_DATA_ENHANCED = d_config("DIR_DATA_ENHANCED")


def transform(pd_df: pd.DataFrame) -> pd.DataFrame:
    """ """
    assert isinstance(pd_df, pd.DataFrame), f"Expected pd.DataFrame, got {type(pd_df)}"
    logger.info(f"Transform {__name__}")
    describe_df = describer.DataDescriber(pd_df=pd_df).describe()
    utils.write_dataframe(
        pd_df=describe_df, directory=DIR_DATA_ENHANCED, filename="TD_SD_FINAL_DESC"
    )
    return describe_df
