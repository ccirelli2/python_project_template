# Overview
- Purpose:  Review draft paper.
- Author: Chris Cirelli
- Date: 11-01-2023

---
### Paper
- Title: Demographics, Financials, and Fairness: How Disparities in  Predictive Models Impact Patients in Substance Use
Treatment for Length-of-Stay?
- Author: Urug,

### Questions (most of these were answered in later notes after reading report fully).
1. Were both HCUP and TEDS-D files provided?
2. If the data was concatenated we should probably start with the raw files pre-processing.
3. Why was the dataset limited to three states?
4. How does the python package fairlearn define unfairness and which features should be considered?
5. Does the report provide reference to any prior research for SUD patient financial condition relative to other
patient types?
6. Do we have knowledge of ML models being used in the medical field and for operational purposes?
7. Who defines what constitutes a protected variable?  Federal and state laws?
8. Is there a common / accepted definition of fairness for ML models?
9. How does one identify bias in the underlying dataset?  Unbalanced observations as it relates to protected classes?
10. Report references prior studies to predict LOS by Belderrar and Hazzab (2020) and 14 influential factors.  Do we
have access to this study and these factors?
11. Under methodology, why was a classification approach chosen over regression?
12. Target Variable: why is mean better than median for setting these thresholds?
13. Resampling - what about SMOTE + Tomek links?
14. Model Fitting - No HPO?
15. Need definition for fairness metrics listed in section 3.4 page 12.
16. Check LOS for outliers.  Validate dates.

### Key Topics
- Dataset: TEDS-D: Treatment Episode Data Set - Discharges
- Dataset: HCUP: Healthcare Cost and Utilization Project, Filtered for SUD Discharge Data
- Dataset-Scope: Limited to three US States: Arizona, Florida, and Maryland.
- Study Objective: "investigate the fairness of ML models developed for  prediction of LOS in SUD treatment to identify
social groups that may be adversely impacted from the model predictions."

### References
- Substance Abuse and Mental Health Services Administration
  - url: https://www.samhsa.gov/
- TEDS-D: Treatment Episode Data Set - Discharges
  - provides discharge data from detoxification, ambulatory, and residential SUD treatment programs
  - url: https://www.samhsa.gov/data/data-we-collect/teds-treatment-episode-data-set
- Healthcare Cost Utilization Project (HCUP) State In-Patient Databases (SID)
  - url:

### Notes
---

**Abstract**
- ML models used to predict when to discharge patients.
- Social biases can be encoded into model(s).
- SUD: substance use disorder.
- Consequences: financial burden.
- Model Target: Length of Stay (LOS)
- Unfairness Features: race, age, region, and primary payer.

**Introduction**
- Sources of unfairness:
  - data is not fully representative of all groups (eg unbalanced)
  - model development lifecycle
  - healthcare system inherent biases.
- Critical Decision
  - how long a patient needs to stay in a facility for observation and treatment (LOS)
  - late discharge can lead to financial burden.
  - late discharge particularly import for SUD may exacerbate financial condition.
- SUD
  - "Those diagnosed with a SUD are often either treated as in-patients within hospitals, and/or in SUD treatment
programs that can range from ambulatory to intensive residential programs"
- Study Structure
  - Develop ML models to predict LOS for SUD patients.
  - Select two best performing models.
  - Assess models for fairness.
  - Assess performance of models for protected variables (race, age, and primary payer), and
  - how fair they perform for each group within a variable (i.e. equally well predictions).
  - finish with discussion of fairness/accuracy tradeoff.

**2.0 Background Literature**

2.1 Mental Health and Substance Use Disorder Treatment and Operations
- Subset of mental health research focuses on SUD.
- SUD treatment focuses on assessment, treatment planning, and detoxification, typically followed by
  inpatient, residential, or outpatient treatment.
- Operations management publications for SUD are scarce.

2.2 Bias and AI Fairness in Substance Use Disorder Treatment
- Documented disparities in SUD wait times and treatment completion associated with race and ethnicity.
- Also, demographic disparities are present for LOS for other types of healthcare admissions.
- Disparities specifically associated with LOS for SUD not yet fully considered.
- LOS for SUD treatment is often longer than for general admissions and more resource intensive.
- Based on prior research we can assume that the data ML models are trained on is biased.

2.3 Bias and AI Fairness in Healthcare
- Fairness: author expects difference in resource use between groups.  They do not expect an equitable distribution of
  resources.
- Concern is for differences in health care resource use within groups with similar focal health needs where subgroups
  differ in ways unrelated to health needs.
- **Data Bias** """if a protected group has consistently received an inequitable amount of health care
  resources in the past, which is reflected in the data, algorithms that do not correct for such biases will likely
  perpetuate it (Wiens et al. 2020, Parikh et al. 2019)."""
- Groups: race, ethnicity, gender, age, income and access to insurance.
- **Study Focus**: algorithmic bias,resulting from input bias (originating from SUD discharge data) and consider
  multiple types of protected groups and associated mitigation strategies toward the goal of reducing output bias.

2.4 Bias and AI Fairness in General
- **Fairness** (defined): "the absence of any prejudice or favoritism toward an individual or group based on
  their inherent or acquired characteristics...an unfair algorithm is one whose decisions are skewed toward a particular
  group of people."
  - what do we mean by 'skewed'?  In what way?
- **Remove Bias**: two approached, 1.) Explainable AI, 2.) apply regularization terms or auxiliary datasets to
  mitigate prediction bias (fairlearn.org; Kallus et al 2022).  This study will use the latter approach.

3.0 Methodology
1. Develop classification model to predict length of stay got a given patient as short-term or long-term.
2. Assess the fairness of these models as to whether they perform equally well for all social groups without
  socially discriminating against specific groups.
3. Mitigate any potential unfairness.

3.1 Data
- Two datasets:
  - National Level: TEDS-D
  - State Level: HCUP SID
- HCUP-SID: Focus on hospital inpatient discharges for three states (Arizona, Florida, and Maryland).
- Date Range: Focus on 2019.

3.11 TEDS-D from SAMSHA
- SAMSHA administers online data collection system for SUD treatment facilities.
- Data is made available on an annual basis.
- N-Rows: 1.7m
- Features: 76 of which all are categorical.
- Two features removed due to 70% Null values.
- Feature Selection: reduced feature set to 44 variables.

3.1.2 Healthcare Cost and Utilization Project (HCUP) State In-Patient Databases (SID)
- Brings together databases from state data organizations, hospital associations, private data organizations, and
  the federal government.
- 97% of all US hospital discharges.
- Geography: Select 3 states from three different regions.
  - Focused on Arizona, Florida, and Maryland.
- Missing Values: new category for missing values was created (why?).
- Filters: filter for patients with one or more SUD diagnosis.
  - Variables: I10_DXn1, I10_PRn2
  - Codes: Selected all codes starting with F10-F19.

3.1.3 Target Variable: Length of Stay (LOS)
- LOS: Most reliable variable for SUD treatment outcomes in US.
- LOS-Levels:
  - TEDS-D Data:
    - Treatment facilities.
    - Average LOS: for SUD treatment is much longer than an in-patient admission in hospital.
    - Term: 1-30 days short term; > 30 days long term.
  - HCUP-SID Data:
    - In-patient hospital discharges.
    - Term: 1-5 days short term; > 5 days long term.

3.2 Feature Selection
- Techniques used in this study
  - chi-squared test
  - LASSO
  - Tree-based
  - Genetic Algorithm
- Majority Vote
  - Apply majority vote technique, i.e. if 3 models vote yes, include feature, otherwise exclude.

3.3 Computational Modeling
- Models: Logistic Regression, random forest, XGBoost, TabNet to predict LOS.
- Imbalance: Tried resampling. No performance improvement.  Author says resampling alter representation of groups in
  study which is critical to assessing fairness of model.  Chose not to use resampling.

3.4 Assessing and Mitigating Fairness in ML Models
- Assessment: measure the negative impacts for quality-of-service on the outcomes of these models for certain groups
  -   QUESTION: What do they mean by 'quality-of-service'
  - (i) assess which patient characteristics may be associated with negative impacts due to model predictions.
    - QUESTION: how is impact measured?
  - (ii) compare multiple models in terms of fairness.
- **Fairness Criteria**: "the probability of a positive outcome on the same item (i.e., variable, category)
  should be the same regardless of the population group membership"
  - QUESTION: Is this to say that after considering all other factors the only reason for the difference in the
    probability should not be the population group membership?
- **Criteria**: the ratio of predicted positive outcomes to ground truth positive outcomes should be equal for each
  group, which is called the selection rate.
- **Fairness Metrics**
    - selection rate
    - demographic parity
    - financial
  - each metric measure fairness relative to a protected class.

continue.. section 4.0 Results
