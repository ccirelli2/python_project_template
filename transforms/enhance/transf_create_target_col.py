"""
Cleaning Tasks
- Create Target Column.
"""
import logging
import pandas as pd
from decouple import config as d_config
from src.enhance import CreateTargetColumn
from src import utils

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
    transformer = CreateTargetColumn(
        dataframe=pd_df, service_col="IN_OUT", los_col="LOS", target_col="TARGET"
    )
    transf_df = transformer.transform().dataframe
    utils.write_dataframe(
        pd_df=transf_df, directory=DIR_DATA_ENHANCED, filename="TD_SD_ENH_TARGET_COL"
    )
    return transf_df
