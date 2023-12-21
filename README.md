
---
# Machine Learning Fairness

---
Machine learning fairness project for GSU University.

<br>


---
### Repository Structure

---
- analysis: exploratory analysis of data.
- data: contains the data used for the project.
- documentation: contains the documentation for the project.
- examples: all example scripts for how to run / execute source code and or transformations.
- notebooks: tutorials on how to reproduce study.
- tests: unit tests for source code.
- transform: pipeline of transformations to be applied to the data.


---
### Final ML Dataset

---
This section provides a summary of how the final ML dataset was constructed, in particular as it relates to
features that were dropped due to various qualifications of the project.

**Information**
- DISYR idicates the year of discharge.
- Regarind the current dataset, there is only one unique value for this column, 2019.
- Therefore, it adds no value in terms of predicting the target variable and is dropped.

**Target Leakage**
- Certain features were used to derive the target variable.
- Therefore, and in order to avoid target leakage, we drop these columns.
- Features dropped:
  - 'SERVICES',
  - 'LOS',
  - 'SERVICES_D',
  - 'IN_OUT'

**Null Percentages**
- For any given column we utilize a threshold of 70% null values to determine to drop the column.
- The following features were dropped due to null percentages (see '/data/TS_SD_CLEAN_DESC.csv' for details)
   -  ![img_2.png](data/imgs/img_2.png)

**Feature Importance & Selection**
- We utilize a majority vote approach to determine which independent variables are considered to be statistically
  significant for predicting our dependent variable.
- Based on our analysis, which included obtaining feature importance from four models
  (LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, and XGBClassifier), we determined that the
  the following features should be dropped from our feature set prior to training our models:
   - Features highlighted in yellow are demographic variables that for the purposes of this study
     did not receive a majority vote but will nevertheless be included in the final feature set.
   - ![img_3.png](data/imgs/img_3.png)

**Final Feature Set**
- Below is the list of features that had < 70% null values and received a majority vote:
   - ![img_4.png](data/imgs/img_4.png)

**Date Type**
- All features are of type integer.

<br>
<br>

---
### Executing Tests | Tools

---

**Tests**
- Execute `poetry run pytest -v -s tests/` from the terminal and repo root directory to run unit tests.
- Execute `poetry run pytest --cov=src/` from the terminal and repo root directory to run test coverage.

**Pre-Commit Hooks**
- Execute `poetry run pre-commit run --all-files` from the terminal and repo root directory.

**Streamlit Application**
- Execute `poetry run streamlit run Home.py` from the terminal and repo app/ directory.

<br>
<br>

---
### References

---

**ML Fairness Library**
- Fairlearn: https://fairlearn.org/

**Chi2 Test**
1. SelectBest:
    - Description: Select features according to the k highest scores.
    - Parameters
        - score_funct: function taking two arrays X and y, and returns a pair of arrays (scores, pvalues).
        - k: number of top features to select.
    - Attributes
        - scores_
        - pvalues_

    - Assumptions:
        - https://online.stat.psu.edu/stat200/book/export/html/230
        - https://is.muni.cz/el/fss/podzim2020/MEBn5033/105652634/Field_Miles_Field_2012_818-828.pdf
    - References:
        - https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html
        - https://github.com/scikit-learn/scikit-learn/blob/3f89022fa/sklearn/feature_selection/_univariate_selection.py#L177
        - https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html
