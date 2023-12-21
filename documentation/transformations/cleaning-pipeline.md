# Cleaning Pipeline
- Purpose: document build of cleaning pipeline
- Author: Chris Cirelli
- Date: 11-05-2023

### Action Items
- Obtain source copy of data dictionary.
- Extract data dictionary table from codebook.
- Confirm no cleaning transformations.
- Confirm rational for mappings used to create In/Out and L/S columns.

### Notes

**Transformations**:
- Document transformations in notebook: "Teds_data_cleaning"
- Cleaning
  - It does not appear that any cleaning steps were made. Confirm.
- Transformations
  - Creation of two new columns: In/Out & L/S
- In/Out
  - Where is the rationale for this transformation documented?
  - There are 1-8 Service-D levels.  Service-D refers to service at discharge.
  - Why are the first two values converted to 1, 2 and the others inpatient / outpatient?


**Data Dictionary**
- Page 36 of draft report contains a data dictionary.
- Does not appear to be the official copy.
- Does not contain data types.
- Does not contain LOS.
- Dos not contain descriptions of sub-category codes.
- Also only includes features from feature selection, no whole list.
- No stats on columns.
- source url: https://www.datafiles.samhsa.gov/dataset/teds-d-2019-ds0001-teds-d-2019-ds0001

TEDS CodeBook:
- data codebook: https://www.datafiles.samhsa.gov/sites/default/files/field-uploads-protected/studies/TEDS-D-2019/TEDS-D-2019-datasets/TEDS-D-2019-DS0001/TEDS-D-2019-DS0001-info/TEDS-D-2019-DS0001-info-codebook_V1.pdf
- provides background for TEDS files, as well as descriptive information for variables, frequencies of their values
 and limitations of the data.
- Appendix A contains the data dictionary.
- Appendix B contains the detailed variable descriptions.
