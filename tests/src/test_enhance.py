"""

"""
import sys
import math
from decouple import config as d_config

# Directories
DIR_ROOT = d_config("DIR_ROOT")
DIR_TESTS = d_config("DIR_TESTS")
sys.path.append(DIR_ROOT)
sys.path.append(DIR_TESTS)

# Project Modules
from src.enhance import NewColMapper, CreateTargetColumn
from tests.conftest import *


def test_instantiate_new_col_mapper(
    new_col_mapper_data_fixture, new_col_mapper_map_fixture, new_col_mapper_col_fixture
):
    """
    Test if function instantiates NewColMapper.
    """
    transformer = NewColMapper(
        dataframe=new_col_mapper_data_fixture,
        mapper=new_col_mapper_map_fixture,
        input_col=new_col_mapper_col_fixture,
        output_col="col3",
    )
    assert isinstance(transformer, NewColMapper)


def test_new_col_mapper_create_col(
    new_col_mapper_data_fixture, new_col_mapper_col_fixture, new_col_mapper_map_fixture
):
    """
    Test if function creates a third column.
    """
    transformer = NewColMapper(
        dataframe=new_col_mapper_data_fixture,
        mapper=new_col_mapper_map_fixture,
        input_col=new_col_mapper_col_fixture,
        output_col="col3",
    )
    transformer._create_new_column()
    assert (
        transformer.dataframe.shape[1] == 2
    ), f"Expected 3 columns, got {transformer.dataframe.shape[1]}."


def test_new_col_mapper_new_col_values(
    new_col_mapper_data_fixture,
    new_col_mapper_col_fixture,
    new_col_mapper_map_fixture,
    new_col_mapper_correct_vals_fixture,
):
    """
    Test if function creates a third column.
    """
    transformer = NewColMapper(
        dataframe=new_col_mapper_data_fixture,
        mapper=new_col_mapper_map_fixture,
        input_col=new_col_mapper_col_fixture,
        output_col="col3",
    )
    transformer._create_new_column()
    exp_values = new_col_mapper_correct_vals_fixture
    obs_values = transformer.dataframe["col3"].values.tolist()
    assert exp_values == obs_values, f"Expected {exp_values}, got {obs_values}."


def test_instantiate_create_target_column(create_target_col_data_fixture):
    """
    Test if function instantiates CreateTargetColumn.
    """
    transformer = CreateTargetColumn(
        dataframe=create_target_col_data_fixture, service_col="IN_OUT", los_col="LOS"
    )
    assert isinstance(transformer, CreateTargetColumn)


def test_create_target_col_create(create_target_col_data_fixture):
    """
    test if function creates the correct column values.
    """
    transformer = CreateTargetColumn(
        dataframe=create_target_col_data_fixture,
        service_col="IN_OUT",
        los_col="LOS",
        target_col="TARGET_OBS",
    )
    transf_df = transformer.transform().dataframe
    msg = f"Expected {transf_df['TARGET_EXP'].values.tolist()}, got {transf_df['TARGET_OBS'].values.tolist()}."
    assert transf_df["TARGET_EXP"].equals(transf_df["TARGET_OBS"]), msg
