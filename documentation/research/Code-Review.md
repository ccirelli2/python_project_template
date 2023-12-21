# Overview
- Purpose:  Review code notebooks.
- Author: Chris Cirelli
- Date: 11-01-2023


### Questions
- What is the correct order of the notebooks?
  - Assume: data-cleaning, feature-selection, ml-class, fairness, fairness-thres.
- Which is the correct dataset, 2019 or 2020?
- Was any EDA or structural analysis performed on the dataset?
- Are these the raw files or was some preprocessing done beforehand?
- Is this dataset a combination of HCUP and TED?
- Prior to the intended transformation shouldn't LOS be a continuous float variable?  The notebook shows a set of
  37 integers for this column.
- Need a data dictionary.  What does SERVICE_D and each sub category represent?
- What is the difference between inpatient and outpatient as it relates to the binary LOS target variable?
  - I see some analysis / work related to calculating the average LOS for these subgroups.  That said, there is
    no documentation for why this change was made.
  - Can you provide a list of required transformations for the data or is it all intended to be captured in this
    notebook?
---

### Notebook - Teds Data Cleaning
-
