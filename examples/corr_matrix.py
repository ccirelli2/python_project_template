"""
TODO: Why is the correlation matrix run on categorical columns?
TODO :DISYR returns NaN in correlation matrix.
TODO: Need to create correlation matrix class and add to pipeline.
References
========================
- Pandas Correlation Matrix: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html
"""
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from decouple import config as d_config
from src import utils

# Settings
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)
matplotlib.use("Agg")

# Config Files
CONFIG_MASTER = utils.load_config()
FEATURES_BY_DATA_TYPE = CONFIG_MASTER["DATA_TYPE_CATEGORIES"]
CORR_MATRIX_FEATURES = CONFIG_MASTER["CORR_MATRIX_FEATURES"]

# Directories
DIR_DATA_ENHANCED = d_config("DIR_DATA_ENHANCED")
DIR_DATA_EDA = d_config("DIR_DATA_EDA_CORR")

# Globals
SAVE_RESULTS = True
SAMPLE = False
NROWS = 100_000


if __name__ == "__main__":
    # Load Data
    source_df = utils.load_dataframe(
        directory=DIR_DATA_ENHANCED,
        filename="TD_SD_ENH_TARGET_COL",
        sample=SAMPLE,
        nrows=NROWS,
    )
    # Validate Datatypes
    msg = "Feature set contains non numeric or non-categorical-ordinal features"
    assert not [
        x
        for x in CORR_MATRIX_FEATURES
        if x not in FEATURES_BY_DATA_TYPE["NUMERIC_ORDINAL"]
    ], msg

    # Limit DataFrame to Target Features
    data = source_df[CORR_MATRIX_FEATURES]

    # Get Data Types
    dtypes_df = data.dtypes
    dtypes_num = dtypes_df[(dtypes_df == "int64") | (dtypes_df == "float64")]
    assert dtypes_num.shape[0] == dtypes_df.shape[0], "Not all columns are numeric"

    # Get Data Levels
    levels_df = data.nunique()
    assert not levels_df[levels_df == 1].shape[0], "Some columns have only one level"

    # Log columns not included in correlation matrix calculation
    cols_excluded = set(source_df.columns) - set(data.columns)
    cols_excluded_df = pd.DataFrame(cols_excluded, columns=["COL_NAME"])
    if SAVE_RESULTS:
        utils.write_dataframe(
            pd_df=cols_excluded_df,
            directory=DIR_DATA_EDA,
            filename="CORR_MATRIX_EXCLUDED_COLS",
            index=True,
        )

    # Get Null Percentage By Column
    # See docs: By default pandas.corr ignores NaN values.
    null_df = data.isna().sum() / data.shape[0]
    if SAVE_RESULTS:
        utils.write_dataframe(
            pd_df=null_df,
            directory=DIR_DATA_EDA,
            filename="CORR_MATRIX_FEATURE_NULL_PCT",
            index=True,
        )

    # Calculate Correlation Matrix
    corr_matrix = data.corr(numeric_only=True, min_periods=100)
    if SAVE_RESULTS:
        utils.write_dataframe(
            pd_df=corr_matrix,
            directory=DIR_DATA_EDA,
            filename="CORR_MATRIX_RESULTS",
            index=True,
        )

    # Plot Correlation Matrix
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    # Draw the heatmap with the mask and correct aspect ratio
    heatmap = sns.heatmap(
        corr_matrix, mask=mask, square=True, linewidths=0.5, vmin=-1, vmax=1, annot=True
    )
    heatmap.set_title("Correlation Heatmap", fontdict={"fontsize": 12}, pad=12)
    if SAVE_RESULTS:
        plt.savefig(os.path.join(DIR_DATA_EDA, "CORR_MATRIX_PLOT.png"))
        plt.close()

    # Create Pairwise Correlation DataFrame
    corr_pair_dict = {"FEATURE-PARIS": [], "COEFFICIENTS": []}

    # Iterate through the correlation matrix and find highly correlated feature pairs.
    for i in range(corr_matrix.shape[0]):
        for j in range(i):
            feature1 = corr_matrix.columns[i]
            feature2 = corr_matrix.columns[j]
            coefficient = corr_matrix.iloc[i, j]
            corr_pair_dict["FEATURE-PARIS"].append(f"{feature1} - {feature2}")
            corr_pair_dict["COEFFICIENTS"].append(coefficient)

    corr_pair_df = pd.DataFrame(corr_pair_dict).sort_values(
        "COEFFICIENTS", ascending=False
    )

    # Plot Results
    f, ax = plt.subplots(figsize=(11, 9))
    pairwise_plot = sns.barplot(data=corr_pair_df, x="COEFFICIENTS", y="FEATURE-PARIS")
    pairwise_plot.set_title("Pairwise Feature Correlation")
    plt.grid(axis="both")

    if SAVE_RESULTS:
        utils.write_dataframe(
            pd_df=corr_pair_df,
            directory=DIR_DATA_EDA,
            filename="CORR_MATRIX_PAIRWISE",
            index=False,
        )

        plt.savefig(os.path.join(DIR_DATA_EDA, "CORR_MATRIX_PAIRWISE_PLOT.png"))
        plt.close()
