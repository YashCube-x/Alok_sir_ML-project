# 🧵 Industrial Fabric Quality Inspection & Prediction
### **Machine Learning Practical Assignment: Data Preprocessing & Feature Selection**

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Complete-success.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 1. Project Title
**Industrial Fabric Quality Inspection and Prediction: From Theory to Implementation**

## 📖 2. Problem Statement
In industrial textile manufacturing, real-time quality grading of fabric rolls is vital to prevent defect propagation, minimize material waste, and guarantee compliance with tensile and durability specifications. Traditional manual fabric auditing is slow, subjective, and prone to inspector fatigue.

This project implements an end-to-end, leak-free Machine Learning preprocessing and feature selection pipeline from scratch to automatically classify industrial fabric quality into **High**, **Medium**, and **Low** grades based on physical, sensory, and manufacturing parameters.

---

## 👥 3. Group Members & Module Contributions

| Student Name | Roll Number | Module / Task Contribution |
| :--- | :---: | :--- |
| **Yash / Team Lead** | 2026-ML-001 | Problem Definition, Data Quality Audit, Duplicate & Negative Value Cleaning |
| **Team Member 2** | 2026-ML-002 | Categorical Encoding, Outlier Detection (IQR / Z-Score) & Winsorization Capping |
| **Team Member 3** | 2026-ML-003 | Feature Scaling (Min-Max/Standardization), Skewness Transformation & Leakage-Free Split |
| **Team Member 4** | 2026-ML-004 | Feature Selection from Scratch (Variance, ANOVA, Chi-Sq, MI) & Model Validation |

---

## 📊 4. Dataset Description
- **Dataset File:** `Industrial Fabric Quality Inspection Dataset.csv`
- **Total Records:** `25,750 rows` (Cleaned: `25,000 unique records`)
- **Total Features:** `23 Columns`
- **Target Variable:** `fabric_quality` (`High`, `Medium`, `Low`)
- **Data Attributes:**
  - *Continuous Numerical:* `tensile_strength`, `gsm`, `fabric_thickness`, `shrinkage_percent`, `elongation_percent`, `moisture_absorption`, `thread_count`, `machine_temperature`, `humidity_level`, `inspection_time_minutes`.
  - *Discrete Numerical:* `color_fastness` (1-5 rating), `defect_count` (0-14 count).
  - *Nominal Categorical:* `fabric_type`, `weave_type`, `finish_type`, `production_method`, `warehouse_id`, `inspection_shift`.
  - *Metadata/IDs:* `batch_id`, `roll_number`, `operator_name`, `inspection_notes`.

---

## 🌐 5. Dataset Source
- **Source:** Industrial Fabric Quality Inspection Dataset (Internal Manufacturing Audit Records)
- **Path in Repo:** [`Industrial Fabric Quality Inspection Dataset.csv`](Industrial%20Fabric%20Quality%20Inspection%20Dataset.csv)

---

## ⚙️ 6. Preprocessing Techniques Implemented

1. **Initial Exploratory Data Analysis (EDA):** Central tendency, dispersion, and distribution histograms for all 12 numerical features.
2. **Missing Value Handling:** Missing % calculation + From-scratch **Median Imputation** for numerical features and **Mode Imputation** for categorical attributes.
3. **Duplicate Record Cleaning:** Identification and removal of 750 duplicate rows (keeping first occurrence).
4. **Invalid & Negative Data Cleaning:** Detection of 24 impossible negative sensor values (e.g. negative tensile strength) and replacement with valid positive feature medians; removal of categorical whitespace and casing inconsistencies.
5. **Categorical Encoding:**
   - **Label Encoding:** Ordinal mapping for target variable (`Low: 0, Medium: 1, High: 2`).
   - **One-Hot Encoding:** Binary dummy indicator creation from scratch for nominal categories.
6. **Outlier Detection & Treatment:**
   - **IQR Method:** $Q_1 - 1.5\text{IQR}$ and $Q_3 + 1.5\text{IQR}$ thresholds.
   - **Z-Score Method:** $|Z| > 3$ threshold.
   - **Capping (Winsorization):** Extreme outliers clipped at boundary limits to retain data volume without destabilizing training.
7. **Feature Scaling:** From-scratch **Min-Max Normalization** ($[0, 1]$) and **Standardization** ($Z$-score $\mu=0, \sigma=1$).
8. **Train-Test Split & Data Leakage Prevention:** 80/20 train-test partition with parameters learned strictly on the training set and applied to the test set.

---

## 🎯 7. Feature-Selection Techniques Implemented

| Technique | Feature Type | Target Type | Mathematical Criterion | Role / Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **Variance Threshold** | Numerical | Unsupervised | $\text{Var}(X) = \frac{1}{n}\sum(x_i - \bar{x})^2 > \theta$ | Eliminates constant/quasi-constant variables. |
| **Pearson Correlation** | Numerical | Numerical | $r = \frac{\sum(x-\bar{x})(y-\bar{y})}{\sqrt{\sum(x-\bar{x})^2\sum(y-\bar{y})^2}}$ | Removes redundant multicollinear features ($|r| > 0.85$). |
| **Chi-Square Test ($\chi^2$)** | Categorical | Categorical | $\chi^2 = \sum \frac{(O - E)^2}{E}$ | Assesses statistical dependence between categorical inputs & quality grade. |
| **ANOVA F-Test** | Numerical | Categorical | $F = \frac{\text{MS}_{\text{between}}}{\text{MS}_{\text{within}}}$ | Tests whether mean physical feature values differ significantly across quality classes. |
| **Mutual Information** | Discrete/Binned | Categorical | $\text{MI}(X; Y) = H(Y) - H(Y\|X)$ | Measures non-linear shared entropy & information gain in bits. |

---

## 🧮 8. From-Scratch Implementations & Verification

Every core statistical computation was implemented using base Python, loops, and NumPy matrices before being verified against Scikit-Learn / SciPy:
- [x] Basic Statistics (`Mean`, `Median`, `Mode`, `Variance`, `Std`, `Range`)
- [x] Imputation logic (Median / Mode)
- [x] One-Hot & Label Encoders
- [x] IQR & Z-score Outlier Detectors
- [x] Min-Max Scaler & Standard Scaler
- [x] Train-Test Split with Random Shuffling
- [x] Variance Threshold, ANOVA F-statistic, & Mutual Information Entropy

---

## 🏆 9. Results & Model Validation

Comparison of Decision Tree Classifier trained on All Features vs Selected Features Subset:

| Parameter | Model A (All 23 Features) | Model B (Selected 6 Features) | Engineering Advantage |
| :--- | :---: | :---: | :--- |
| **Dimensionality** | 23 Features | **6 Core Features** | **74% Feature Reduction** |
| **Training Time** | 0.0482 s | **0.0118 s** | **4x Faster Inference & Training** |
| **Test Accuracy** | 94.80% | **95.20%** | Maintained & slightly enhanced generalizability |
| **Interpretability** | Low (Dense tree) | **High (Auditable rules)** | Clear operational rules for factory floor |

---

## 📌 10. Selected Features

The following 6 features were retained based on strong statistical significance:
1. `tensile_strength` (ANOVA F-statistic: High)
2. `gsm` (ANOVA F-statistic: High)
3. `fabric_thickness` (ANOVA F-statistic: Medium)
4. `defect_count` (Mutual Information: 0.42 bits)
5. `color_fastness` (Chi-Square: Significant)
6. `shrinkage_percent` (ANOVA F-statistic: Medium)

**Removed Features:** `batch_id` (Overfitting ID), `roll_number` (Counter ID), `operator_name` (Zero generalization), `inspection_notes` (Direct target leakage).

---

## 💡 11. Key Findings

1. **Physical properties dominate quality:** Tensile breaking force, fabric weight (GSM), and physical defect count account for over 85% of predictive information.
2. **Preventing Data Leakage is critical:** Preprocessing on the full dataset before splitting falsely boosts evaluation metrics by up to 4-6%; splitting first ensures true real-world generalizability.
3. **Outlier Capping outperforms Dropping:** Capping preserves sample integrity while mitigating gradient volatility.

---

## 🚀 12. Instructions to Run the Code

### Method 1: Google Colab
1. Open [Google Colab](https://colab.research.google.com).
2. Upload [`NEW_ML_project.ipynb`](NEW_ML_project.ipynb) via `File ➔ Upload notebook`.
3. Upload [`Industrial Fabric Quality Inspection Dataset.csv`](Industrial%20Fabric%20Quality%20Inspection%20Dataset.csv) in the session files.
4. Click `Runtime ➔ Run all`.

### Method 2: Local Jupyter Notebook
```bash
# Clone the repository
git clone https://github.com/YashCube-x/Alok_sir_ML-project.git
cd Alok_sir_ML-project

# Launch Jupyter
jupyter notebook NEW_ML_project.ipynb
```

---

## 📂 Repository Structure

```
Alok_sir_ML-project/
│
├── README.md                                          # Clean project overview and documentation
├── NEW_ML_project.ipynb                               # Master Executed Jupyter Notebook (with all outputs)
├── Industrial Fabric Quality Inspection Dataset.csv   # Dataset with injected real-world anomalies
│
├── dataset/
│   └── Industrial Fabric Quality Inspection Dataset.csv  # Direct Raw Dataset Link
│
└── notebooks/
    └── NEW_ML_project.ipynb                           # Direct Notebook Link
```
