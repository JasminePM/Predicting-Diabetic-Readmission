# Predicting Early Hospital Readmission in Diabetic Patients

**Author:** Paige Mitchell

**Course:** CISC 6745 

My project applies supervised machine learning to predict 30-day hospital readmission outcomes in diabetic patients across three classes: no readmission (Class 0), readmission after 30 days (Class 1), and early readmission within 30 days (Class 2). Prior to fitting, I used LASSO feature selection to create several model configurations. Three classifiers were trained and evaluated: Logistic Regression, Random Forest, and Support Vector Machines - with particular focus on Class 2 recall as the most clinically urgent and underrepresented outcome. Lastly, I applied Principal Component Analysis as a diagnostic tool to interpret the models' shared performance ceiling.

**Dataset:** Strack et al. (2014) Health Facts Dataset - 101,766 inpatient diabetic encounters across 130 U.S. hospitals (1999-2008), available through the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008).

---

## Skills Demonstrated

- Data cleaning and preprocessing (missing value imputation, ordinal mapping, ICD-9 grouping, one-hot encoding)
- Feature engineering and dimensionality reduction (LASSO L1 regularization, PCA)
- Multi-class classification with class imbalance handling
- Model evaluation using precision, recall, F1, and accuracy across three classes
- Data visualization (matplotlib, seaborn)

---

## Results

| Model | Features | Accuracy | Class 2 Recall |
|---|---|---|---|
| LR Baseline (balanced) | Full (98) | 0.4892 | 0.39 |
| LR LASSO C=0.01 | LASSO (83) | 0.4884 | 0.38 |
| RF Baseline (unbalanced) | Full (98) | 0.5810 | 0.01 |
| RF Balanced n=200 | Full (98) | 0.5825 | 0.01 |
| RF Balanced n=200 | LASSO (83) | 0.5829 | 0.01 |
| LinearSVC Baseline | Full (98) | 0.5709 | 0.00 |
| LinearSVC Balanced C=0.1 | Full (98) | 0.5604 | 0.12 |
| **RBF SVM C=0.1 (10k sub)** | **Full (98)** | **0.4898** | **0.30** |

---

## How to Run

Install dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn openpyxl jupyter
```

Run notebooks in order - each saves outputs the next one depends on:

1_DataCleaning_PM.ipynb

2_EDA_PM.ipynb
3_FeatureSelection_LogisticRegression_PM.ipynb

4_RandomForest_PM.ipynb

5_SVM_PM.ipynb (note: RBF cell takes several minutes)

6_PCA_PM.ipynb

7_CumulativeModelComparison_PM.ipynb

Place all notebooks and `diabetic_data.csv` in the same working directory before running.

---

## References

Strack, B. et al. (2014). Impact of HbA1c measurement on hospital readmission rates. *BioMed Research International*, Article 781670.
