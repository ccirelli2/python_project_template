"""
Create LOS / Target Column.
"""
import os
import logging
import pandas as pd
from decouple import config as d_config
from src import utils
from src.enhance import CreateTargetColumn

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
SEP = CONFIG_MASTER["FILE_SEP"]["ted_sd_raw_2019"]


if __name__ == "__main__":
    # Load Data
    pd_df = utils.load_dataframe(
        directory=DIR_DATA_ENHANCED,
        filename="TD_SD_ENH_IN_OUT_COL",
        sample=True,
        nrows=10_000,
    )

    transformer = CreateTargetColumn(
        dataframe=pd_df,
        service_col="IN_OUT",
        los_col="LOS",
    )
    transf_df = transformer.transform().dataframe

    print(transf_df[["IN_OUT", "LOS", "TARGET"]].head(20))
