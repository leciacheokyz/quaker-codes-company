### Earthquake & Tsunami Prediction
---

## Project Overview

This project applies machine learning to two interconnected seismic risk prediction tasks:

1. **Tsunami Classification** — predicting whether an earthquake will trigger a tsunami
2. **Earthquake Magnitude Regression** — estimating the magnitude of seismic events

Together, the models form a complementary risk assessment framework to support **government disaster response planning** and **insurance underwriting** decisions. Given that approximately 89% of tsunamis are triggered by earthquakes, accurately modelling the relationship between seismic characteristics and disaster outcomes has significant real-world value.

---

## Repository Structure

```
DBA3803-Earthquake-Tsunami-Prediction/
├── data/
│   ├── kaggle_earthquake_tsunami.csv     # Core dataset (Kaggle, 1995–2023)
│   ├── noaa_tsunami.csv                  # NOAA Tsunami Database
│   └── wikipedia_tsunami_summary.html    # Wikipedia tsunami event table
├── notebooks/
│   ├── 01_data_preprocessing.ipynb       # Cleaning, imputation, encoding
│   ├── 02_classification.ipynb           # Tsunami occurrence prediction
│   └── 03_regression.ipynb              # Earthquake magnitude estimation
├── report/
│   └── DBA3803_Group7_Report.pdf
└── README.md
```

---

## Data Sources

| Source | Description |
|---|---|
| [Kaggle Earthquake-Tsunami Dataset](https://www.kaggle.com/datasets/warcoder/earthquake-dataset) | 1,000 rows, 18 features; core dataset covering 1995–2023 |
| [NOAA Tsunami Database](https://www.ngdc.noaa.gov/hazel/view/hazards/tsunami/event-data) | Official NOAA records used to correct mislabeled tsunami events |
| [Wikipedia: List of Tsunamis](https://en.wikipedia.org/wiki/List_of_tsunamis) | Cross-reference for major historical events |

> **Note:** Due to known data quality issues prior to the post-2004 DART buoy deployment, classification was restricted to events from 2004 onwards.

---

## Methods

### Preprocessing
- Fuzzy-matching to recover mislabeled tsunami events from NOAA and Wikipedia sources
- Hierarchical median imputation for missing seismic station features (`nst`, `dmin`, `gap`)
- Log-normal distribution sampling to impute missing `cdi` values
- Seismic region groupings to reduce cardinality of `country`; temporal binning for `year`
- Log transformation (`log1p`) applied to skewed features (`sig`, `dmin`, `depth`, `gap`)
- SMOTE applied to pre-2013 subset to address class imbalance

### Classification (Tsunami Occurrence)
- **Primary metric:** Recall (minimising false negatives is critical in disaster contexts)
- **Secondary metric:** AUC-ROC
- Models evaluated: Logistic Regression (L1), Random Forest, KNN, SVM, XGBoost
- Feature selection via: L1 regularisation, forward selection (AIC/BIC), Random Forest importance
- **Best model: XGBoost (RandomizedSearch)** — Test Recall: **0.955**, Test AUC: **0.754**

### Regression (Earthquake Magnitude)
- **Metrics:** MAE, MSE, Adjusted R²
- Models evaluated: Linear Regression, Random Forest, SVR, KNN, XGBoost (Standard & Quantile)
- Feature selection via: RFECV, Lasso regularisation, Random Forest importance
- Log transformation applied to target variable to address heteroskedasticity
- **Best model: XGBoost (log-transformed target)** — Test MAE: **0.107**, Test MSE: **0.028**, Adj. R²: **0.854**

---

## Key Results

| Task | Best Model | Key Metric |
|---|---|---|
| Tsunami Classification | XGBoost (RandomizedSearch) | Recall = 0.955, AUC = 0.754 |
| Magnitude Regression | XGBoost (log-transformed) | MAE = 0.107, Adj. R² = 0.854 |

SHAP analysis confirmed that both models learned physically interpretable patterns:
- **Classification:** Earthquake magnitude and shallow depth were the strongest positive predictors of tsunami risk
- **Regression:** Depth-related features (`depth_magnitude`, `depth_log`) dominated predictions, consistent with seismological understanding that shallower earthquakes produce higher surface-level magnitudes

---

## Tools & Libraries

- **Python** — pandas, NumPy, scikit-learn, XGBoost, imbalanced-learn (SMOTE)
- **Visualisation** — matplotlib, seaborn, SHAP
- **Feature selection** — RFECV, Lasso, forward selection (AIC/BIC)
