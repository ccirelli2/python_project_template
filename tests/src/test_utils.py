"""
Test utility functions
"""
import os
import sys
from decouple import config as d_config

# Directories
DIR_ROOT = d_config("DIR_ROOT")
DIR_TESTS = d_config("DIR_TESTS")
sys.path.append(DIR_ROOT)
sys.path.append(DIR_TESTS)

# Project Modules
from src.utils import load_config, load_dataframe
from tests.conftest import *


# Tests
def test_load_dataframe(data_files):
    """
    Test to validate loading of dataframe from CSV file.
    """
    try:
        load_dataframe(
            directory=data_files["DIRECTORY"], filename=data_files["FILE_NAME"]
        )
    except Exception as e:
        pytest.fail(f"Exception raised: {e}")
    return None


def test_load_config_files(config_files):
    try:
        load_config(
            directory=config_files["DIRECTORY"], filename=config_files["FILE_NAME"]
        )
    except Exception as e:
        pytest.fail(f"Exception raised: {e}")


def test_load_config_return_dict(config_files):
    """
    Test ability to load config files with and without file name.
    Args:
        config_files:

    Returns:

    """
    config = load_config(
        directory=config_files["DIRECTORY"], filename=config_files["FILE_NAME"]
    )
    assert isinstance(config, dict), f"Config is not a dictionary: {config}"
    return None
