"""
Test src.cleaning.py functions.
Note: must be run from terminal `poetry run pytest -v`.
"""
import sys
from decouple import config as d_config

# Directories
DIR_ROOT = d_config("DIR_ROOT")
DIR_TESTS = d_config("DIR_TESTS")
sys.path.append(DIR_ROOT)
sys.path.append(DIR_TESTS)

# Project Modules
from src.cleaning import map_val_to_null, DropNullColumns
from tests.conftest import *


def test_map_val_to_null(
    data_map_val_to_null, val_map_val_to_null, col_map_val_to_null
):
    """
    Test to validate mapping of value to null.
    """
    data = map_val_to_null(
        dataframe=data_map_val_to_null,
        column=col_map_val_to_null,
        val=val_map_val_to_null,
    )
    if isinstance(col_map_val_to_null, list):
        assert (
            data.isnull().sum().sum() == 2
        ), f"Sum null values => {data.isnull().sum()}"
    else:
        assert (
            data[col_map_val_to_null].isnull().sum().sum() == 1
        ), f"Sum null values => {data['col1'].isnull().sum()}"


def test_instantiate_drop_null_columns_class(data_drop_null_column):
    """
    Args:
        data_drop_null_column:
        master_config:

    Returns:

    """
    assert DropNullColumns(dataframe=data_drop_null_column)


def test_get_column_null_pct(drop_null_columns_class_fixture):
    """
    Given a predefined datafrmae, test if function returns the correct null pct.
    Args:
        drop_col_class_instantiated:

    Returns:

    """
    null_df = drop_null_columns_class_fixture._get_column_null_pct().null_df
    assert null_df.to_dict() == {"col1": 0.2, "col2": 0.2, "col3": 0.8}


def test_get_columns_to_drop(drop_null_col_class_get_column_null_pct):
    """
    Given a predefined dataframe, test if function returns the correct columns.
    Args:
        drop_null_col_class_cols_null_pct: DropNullColumns class object instantiated and get col nulls pct already
        called.

    Returns:

    """
    cols_to_drop = (
        drop_null_col_class_get_column_null_pct._get_columns_to_drop().cols_to_drop
    )
    assert cols_to_drop == ["col3"]


def test_drop_columns(drop_null_col_class_get_cols_to_drop):
    """
    Test if function correctly drops columns
    Args:
        drop_null_col_class_get_cols_to_drop:

    Returns:

    """
    clean_df = drop_null_col_class_get_cols_to_drop._drop_columns().clean_df
    assert clean_df.columns.tolist() == ["col1", "col2"]


def test_drop_null_col_clean(drop_null_columns_class_fixture):
    """
    Test DropNullColumn class clean method.
    Returns:

    """
    clean_df = drop_null_columns_class_fixture.clean()
    assert clean_df.columns.to_list() == ["col1", "col2"]
