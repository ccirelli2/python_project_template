"""
Module containing all fixtures for all tests.
Auto-loaded when the user calls pytest within this or a chile dir of test/.

TODO: Need to test column-mapper function to map nan values to nan.  should not map to Unknown.
"""
import os
import sys
import pytest
import pandas as pd
import numpy as np
from decouple import config as d_config

# Directories
DIR_SRC_ROOT = d_config("DIR_ROOT")
DIR_TESTS = d_config("DIR_TESTS")
DIR_TESTS_DATA = os.path.join(DIR_TESTS, "data")

# Import SRC Packages
sys.path.append(DIR_SRC_ROOT)
from src.utils import load_config
from tests.src.test_cleaning import DropNullColumns

######################################################
# UTILITY FIXTURES
######################################################
DATA_FILES = {
    "LOAD_DATAFRAME": {"FILE_NAME": "test_data", "DIRECTORY": DIR_TESTS_DATA},
    "LOAD_DATAFRAME_TEST_W_EXT": {
        "FILE_NAME": "test_data.csv",
        "DIRECTORY": DIR_TESTS_DATA,
    },
}
CONFIG_FILES = {
    "NO_DIRECTORY": {"FILE_NAME": "config.yaml", "DIRECTORY": None},
    "WITH_DIRECTORY": {"FILE_NAME": "config.yaml", "DIRECTORY": DIR_SRC_ROOT},
    "NO_DIRECTORY_NO_FILENAME": {"FILE_NAME": None, "DIRECTORY": None},
}


@pytest.fixture(params=list(DATA_FILES.keys()), ids=list(DATA_FILES.keys()))
def data_files(request):
    """Fixture to return file name"""
    return DATA_FILES[request.param]


@pytest.fixture(params=list(CONFIG_FILES.keys()), ids=list(CONFIG_FILES.keys()))
def config_files(request):
    """Fixture to return config keys"""
    return CONFIG_FILES[request.param]


@pytest.fixture
def master_config():
    return load_config(directory=DIR_SRC_ROOT)


################################################################
# DATA FIXTURES
################################################################


@pytest.fixture()
def data_map_val_to_null():
    data = pd.DataFrame(
        {
            "col1": [-9, 2, 3, 4, 5],
            "col2": [6, 7, 8, -9, 10],
        }
    )
    return data


@pytest.fixture()
def data_drop_null_column():
    data = pd.DataFrame(
        {
            "col1": [None, 2, 3, 4, 5],
            "col2": [6, 7, 8, None, 10],
            "col3": [None, None, None, None, 1],
        }
    )
    return data


######################################################
# CLEANING FIXTURES
######################################################
COLS_MAP_VAL_TO_NULL = ["col1", "col2", ["col1", "col2"]]


@pytest.fixture()
def val_map_val_to_null():
    """Constant value to map to null."""
    return -9


@pytest.fixture(params=COLS_MAP_VAL_TO_NULL, ids=["col1", "col2", "col1&2"])
def col_map_val_to_null(request):
    """
    fixture to return column names to test function that maps a fixed
    value to None."""
    return request.param


@pytest.fixture()
def drop_null_columns_class_fixture(data_drop_null_column):
    return DropNullColumns(dataframe=data_drop_null_column)


@pytest.fixture()
def drop_null_col_class_get_column_null_pct(drop_null_columns_class_fixture):
    """
    Fixture that has calculated the null percentage by dataframe.
    Args:
        drop_null_columns_class_fixture:

    Returns:

    """
    return drop_null_columns_class_fixture._get_column_null_pct()


@pytest.fixture()
def drop_null_col_class_get_cols_to_drop(drop_null_col_class_get_column_null_pct):
    """
    Fixture that has identified columns ot drop.
    Returns:

    """
    return drop_null_col_class_get_column_null_pct._get_columns_to_drop()


######################################################
# ENHANCEMENT FIXTURES - New Column Mapper
######################################################
"""
Note: pandas represents None values as np.nan.
Therefore, our mapping and validation fixtures need to use np.nan versus Python None.
"""


@pytest.fixture(params=["col1"], ids=["col1"])
def new_col_mapper_col_fixture(request):
    """
    fixture to return column names to test function that maps a fixed value to None.
    """
    return request.param


@pytest.fixture()
def new_col_mapper_map_fixture():
    mapping = {
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
        10: "ten",
        None: "nan",
    }
    return mapping


@pytest.fixture()
def new_col_mapper_data_fixture():
    data = pd.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
        }
    )
    return data


@pytest.fixture()
def new_col_mapper_correct_vals_fixture(new_col_mapper_col_fixture):
    correct_values = {
        "col1": ["one", "two", "three", "four", "five"],
    }
    return correct_values[new_col_mapper_col_fixture]


######################################################
# ENHANCEMENT FIXTURES - Create Target Column
######################################################


@pytest.fixture()
def create_target_col_data_fixture():
    """
    Includes expected target values.
    """
    data = pd.DataFrame(
        {
            "IN_OUT": [
                "Inpatient",
                "Inpatient",
                "Outpatient",
                "Outpatient",
                "1",
                "1",
                "2",
                "2",
            ],
            "LOS": [31, 27, 34, 32, 31, 29, 31, 29],
            "TARGET_EXP": [1, 0, 1, 0, 1, 0, 1, 0],
        }
    )
    return data
