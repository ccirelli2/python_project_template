import os
import sys
import streamlit as st
from streamlit import session_state as state
import matplotlib.pyplot as plt
from decouple import config as d_config

# Directories
DIR_ROOT = d_config("DIR_ROOT")
DIR_DATA = d_config("DIR_DATA")
DIR_DATA_CLEAN = d_config("DIR_DATA_CLEAN")
sys.path.append(DIR_ROOT)

# Load Project Modules
from src.utils import load_config, load_dataframe
from src import describer

# Config Files
CONFIG_MASTER = load_config(directory=DIR_ROOT, filename="config.yaml")
FILE_NAMES = [None] + [
    x
    for x in os.listdir(DIR_DATA_CLEAN)
    if x.endswith(".csv") and "desc" not in x.lower()
]

########################################################################################################################
# Application
########################################################################################################################

st.title("Structural Analysis of TED-SD dataset")
st.write(
    "This page provides the ability to generate a data description for a given .csv data file in the clean zone."
)
st.divider()

st.subheader("Upload File")
option = st.selectbox("Select file to upload", FILE_NAMES)
st.divider()
st.subheader("Select Number of Rows to Display")

# Determine Sample Size
n_rows = st.slider(
    "Select number of rows to display",
    min_value=10_000,
    max_value=1_000_000,
    value=0,
    step=100_000,
)
if n_rows:
    st.session_state.n_rows = n_rows

# Button to Load File
load_file = st.button("Load File")
if load_file:
    st.session_state.load_file = True

# If N-rows selected & file selected
if all([option, n_rows, load_file]):
    # Load File
    st.write("Loading file: ", option)
    assert os.path.join(DIR_DATA_CLEAN, f"{option}.csv"), FileNotFoundError(
        "File not found"
    )
    pd_df = load_dataframe(
        directory=DIR_DATA_CLEAN, filename=option, sample=True, nrows=n_rows
    )
    st.write(f"File loaded successfully dimensions (rows/columns) {pd_df.shape}")
    st.divider()

    # Inspect DataFrame
    st.write("Inspect DataFrame")
    st.dataframe(pd_df)
    st.divider()

    # Generate Data Description
    st.subheader("Data Description")
    gen_dd = st.button("Generate Data Description")
    if gen_dd:
        st.session_state.generate_data_description = True
        st.write("Generating Data Description")
        desc_df = describer.DataDescriber(pd_df=pd_df).describe()
        st.markdown("#### Data Description")
        st.dataframe(desc_df)

        # Select Feature To Plot
        st.divider()
        st.subheader("Generate Plot of Pct. Null or Levels")
        feature = st.selectbox("Select Column", ["LEVELS", "NULL_PCT"])

        if feature:
            st.session_state.feature = feature
            generate_plot = st.button("Generate Plot")

            # Generate Plot
            if generate_plot:
                st.session_state.generate_plot = True

                if feature == "NULL_PCT":
                    min_null_pct = st.slider(
                        "Select minimum null percentage",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.0,
                        step=0.1,
                    )
                    if min_null_pct:
                        st.session_state.min_null_pct = min_null_pct
                        desc_df = desc_df[desc_df[feature] >= min_null_pct]

                if feature == "LEVELS":
                    max_num_levels = st.slider(
                        "Select maximum number of levels",
                        min_value=0,
                        max_value=1_000,
                        value=1_000,
                        step=10,
                    )
                    if max_num_levels:
                        st.session_state.max_num_levels = max_num_levels
                        desc_df = desc_df[
                            (desc_df.PRIMARY_KEY == False)
                            & (desc_df[desc_df[feature] <= max_num_levels])
                        ]

                # Plot
                desc_df.sort_values(by=feature, ascending=False, inplace=True)
                fig, ax = plt.subplots()
                plt.xticks(rotation=90)
                plt.grid(which="both")
                ax.bar(x=desc_df.COLUMN, height=desc_df[feature])
                st.pyplot(fig)
