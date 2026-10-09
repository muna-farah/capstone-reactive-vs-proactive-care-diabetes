# Reactive vs. Proactive Care: Healthcare Inequities and Diabetes Readmission Prediction

Capstone project for the UC Riverside MS in Engineering (Data Science), 2026.

This project examines whether healthcare utilization patterns and health inequities are related to hospital readmission among patients with diabetes, and how well machine learning models can predict readmission. It combines exploratory analysis, K-Means clustering, two class-imbalance strategies (SMOTE and class weighting), and three classifiers (Logistic Regression as the baseline, Random Forest, XGBoost) with hyperparameter tuning.

## Data

- **Source:** Diabetes 130-US Hospitals for Years 1999-2008, UCI Machine Learning Repository (dataset id=296)
- **Size:** 101,766 inpatient encounters, 50 columns
- **Outcome:** readmission is coded as 1 if the patient was readmitted at any time after discharge (`<30` or `>30` days) and 0 otherwise (`NO`). About 46% of encounters are positive.
- **Citation:** Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Ventura, S., Cios, K. J., & Clore, J. N. (2014). Impact of HbA1c measurement on hospital readmission rates: Analysis of 70,000 clinical database patient records. *BioMed Research International, 2014*, 1-11.

## Methods

- **Feature engineering:** `med_change_count` and `med_steady_count`, which summarize 23 medication variables.
- **EDA:** age-by-race readmission heatmap, chi-square test of race vs. readmission, medication volatility vs. length of stay.
- **Clustering:** K-Means (k = 5, chosen with the elbow method) on length of stay, medication count, medication changes, and outpatient and inpatient visits; clusters compared against race, gender, and readmission using mutual information.
- **Modeling:** Logistic Regression, Random Forest, and XGBoost with and without cluster membership as a feature; SMOTE vs. class weighting; 80/20 stratified train/test split.
- **Tuning:** grid search and randomized search with 10-fold stratified cross-validation.

## Key Results

### Race and readmission

Readmission rate by race:

- Caucasian: 46.93% (n = 76,099)
- African American: 45.75% (n = 19,210)
- Hispanic: 41.92% (n = 2,037)
- Other: 39.24% (n = 1,506)
- Asian: 35.26% (n = 641)
- Unknown: 31.94% (n = 2,273)

The association between race and readmission was statistically significant (chi-square = 278.75, p < 0.001), but the relationship was negligible (Cramér's V = 0.052, on a scale where 0 means no relationship and 1 means a perfect one). Knowing a patient's race tells you very little about whether they will be readmitted. The smaller groups are less certain because there are fewer patients in them.

### Patient clusters

The five profiles include patients with many prior inpatient stays (Cluster 0), patients who rely heavily on outpatient care (Cluster 4), and patients whose medications changed the most (Cluster 1). Race differed only a little across clusters, and the clusters were only weakly related to readmission (AMI = 0.015).

### Model performance (untuned, no cluster features)

| Model | Method | Accuracy | AUC | F1 | Recall |
|---|---|---|---|---|---|
| Logistic Regression | SMOTE | 0.617 | 0.650 | 0.537 | 0.482 |
| Random Forest | SMOTE | 0.572 | 0.598 | 0.538 | 0.541 |
| XGBoost | SMOTE | 0.622 | 0.661 | 0.553 | 0.506 |
| Logistic Regression | Class weight | 0.618 | 0.651 | 0.537 | 0.480 |
| Random Forest | Class weight | 0.575 | 0.597 | 0.538 | 0.538 |
| XGBoost | Class weight (scale_pos_weight = 1.17) | 0.620 | 0.662 | 0.580 | 0.570 |

- The class-weighted XGBoost caught the most readmitted patients (recall of 0.570 vs. 0.506 with SMOTE) at a similar accuracy (0.620 vs. 0.622) and with somewhat more false positives (3,705 vs. 3,061), giving it the highest F1 in the table (0.580).
- Tuning changed little. XGBoost with class weights stayed at a recall of about 0.57 and F1 of 0.58, with AUC rising to 0.665; XGBoost with SMOTE reached a recall of about 0.52 and AUC of 0.665.
- Randomized search matched grid search for the SMOTE Random Forest (AUC 0.656 for both) and took 3.89 minutes instead of 26.11. Run times depend on the machine.

## Limitations

- The dataset contains multiple encounters per patient, and the train/test split was done by encounter, so the results may be a bit overestimated.
- The data are from 1999-2008 and may not reflect current practice.
- Race is observational, has an "Unknown" group, and the rates are unadjusted, so they say nothing on their own about access to care.

## Repository Contents

- `diabetes_readadmission_analysis.ipynb` — full analysis
- `figures/` — saved plots

## Author

Muna Farah · [LinkedIn](https://www.linkedin.com/in/muna-a-farah) · [Email](mailto:themunafarah@gmail.com)