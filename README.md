# Machine Learning Practical Assignment
## Industrial Fabric Quality Inspection and Classification

**Department of Computer Science & Engineering**  
**UIET, Chhatrapati Shahu Ji Maharaj University (CSJMU), Kanpur**  

---

### Project Overview

In this practical assignment, we worked on the **Industrial Fabric Quality Inspection Dataset** to analyze, preprocess, and select the most relevant features for classifying fabric rolls into three quality grades: **High**, **Medium**, and **Low**.

The main objective of the assignment was to implement all major preprocessing and feature selection techniques **from scratch using mathematical formulas in Python**, and then verify our manual calculations against standard libraries (`scipy`, `sklearn`, `numpy`).

---

### Team Details: Outlier Squad

- **Institution:** UIET, CSJMU Kanpur
- **Branch:** B.Tech Computer Science & Engineering
- **Year & Semester:** 4th Year, 7th Semester

| S.No | Student Name | University Roll Number | Email Address |
| :---: | :--- | :---: | :--- |
| 1 | **Suyash Prakash** | `CSJMA23001390046` | `yashcube07@gmail.com` |
| 2 | **Monika Singh** | `CSJMA23001390023` | `monikasingh20032110@gmail.com` |
| 3 | **Nikhilesh Gond** | `CSJMA23001390027` | `ngond7062@gmail.com` |
| 4 | **Md. Ahmad Raza** | `CSJMA23001390022` | `Ahmadrzaa45@gmail.com` |

---

### Dataset Summary

- **File Name:** `Industrial Fabric Quality Inspection Dataset.csv`
- **Total Records:** 25,750 rows (Raw) | 24,969 rows (Cleaned & Treated)
- **Total Features:** 23 columns (22 input features + 1 target variable)
- **Target Variable:** `fabric_quality` (`High`, `Medium`, `Low`)
- **Feature Types:**
  - *Numerical Features (12):* `tensile_strength`, `gsm`, `fabric_thickness`, `shrinkage_percent`, `elongation_percent`, `moisture_absorption`, `thread_count`, `machine_temperature`, `humidity_level`, `inspection_time_minutes`, `color_fastness`, `defect_count`
  - *Categorical Features (6):* `fabric_type`, `weave_type`, `finish_type`, `production_method`, `warehouse_id`, `inspection_shift`
  - *Identifier / Metadata Columns (4):* `batch_id`, `roll_number`, `operator_name`, `inspection_notes`

---

### Key Preprocessing & Feature Selection Steps Implemented

The completed notebook (`NEW_ML_project.ipynb`) contains 28 sequential steps implemented according to the practical assignment guidelines:

1. **Exploratory Data Analysis (EDA):** Inspected dataset shapes, data types, missing value percentages, and distributions for numerical features.
2. **Missing Value Treatment:** Implemented median imputation for skewed/continuous numerical features and mode imputation for categorical attributes.
3. **Duplicate Record Cleaning:** Detected and removed 750 duplicate entries while preserving first valid instances.
4. **Categorical Data Standardization:** Cleaned extraneous whitespaces, fixed inconsistent string casing, and handled invalid/unknown placeholder codes.
5. **Categorical Encoding:** Built manual Label Encoding for binary/ordinal categories and One-Hot Encoding for nominal variables.
6. **Outlier Detection & Capping:** Identified outliers using both the 1.5 × IQR rule and the |Z| > 3 method, and applied capping (Winsorization) to avoid data loss.
7. **Transformation & Feature Scaling:** Checked skewness of numerical variables; implemented Min-Max Normalization ([0, 1]) and Z-score Standardization ($\mu=0, \sigma=1$) from scratch.
8. **Train-Test Split (80/20):** Shuffled indices randomly and partitioned data into 19,975 training rows (80%) and 4,994 test rows (20%) without using external libraries.
9. **Data Leakage Prevention:** Ensured that all preprocessing parameters (means, standard deviations, mins, maxes, medians) were learned strictly on `X_train` and frozen before transforming `X_test`.
10. **Statistical Feature Selection (From Scratch):**
    - *Variance Threshold:* Calculated $\text{Var}(X) = \frac{1}{n}\sum(x_i - \bar{x})^2$ to filter out constant/near-zero variance features.
    - *Pearson Correlation:* Computed linear feature-target correlations and checked the feature-feature correlation matrix for multicollinearity.
    - *Chi-Square Test ($\chi^2$):* Constructed observed/expected contingency tables from scratch and calculated $\chi^2$ statistics and degrees of freedom for categorical predictors.
    - *ANOVA F-Test:* Calculated Between-Group Variance (MSB) and Within-Group Variance (MSW) to test whether continuous feature means differ across quality grades.
    - *Mutual Information:* Implemented Shannon Entropy $H(Y)$, Conditional Entropy $H(Y|X)$, and Information Gain in bits to capture non-linear dependencies.
11. **Library Verification:** Verified all manual from-scratch calculations against `scipy.stats` and `scikit-learn` (confirmed 100% numerical parity).

---

### Feature Selection Decisions

Based on statistical testing and textile manufacturing domain requirements, features were categorized as follows:

#### Retained Features (13 Core Predictors)
- **Mechanical & Physical Strength:** `tensile_strength`, `gsm`, `fabric_thickness`, `elongation_percent`, `shrinkage_percent`, `thread_count`, `moisture_absorption`
- **Quality & Flaw Ratings:** `defect_count`, `color_fastness`
- **Manufacturing Specifications:** `fabric_type`, `weave_type`, `finish_type`, `production_method`

#### Removed Features (9 Columns)
- `batch_id` & `roll_number`: Arbitrary identifiers that lead to model memorization and overfitting.
- `operator_name`: Auditor identity introduces personal bias without any generalizable predictive signal.
- `warehouse_id`: Storage location has no physical relationship to fabric quality.
- `inspection_notes`: Post-inspection audit remarks cause target leakage.
- `inspection_time_minutes`, `machine_temperature`, `humidity_level`, `inspection_shift`: Showed negligible statistical correlation / low F-statistics with the target quality class.

---

### How to Run the Project

#### Running in Google Colab:
1. Open Google Colab and upload `NEW_ML_project.ipynb` (or open directly from GitHub).
2. Upload `Industrial Fabric Quality Inspection Dataset.csv` into the session storage.
3. The first cell contains an automatic path detector that locates the dataset automatically.
4. Go to **Runtime -> Run all** (`Ctrl + F9`). All 98 cells will execute sequentially with all tables and plots generated.

#### Running Locally:
```bash
# Clone the repository
git clone https://github.com/YashCube-x/Alok_sir_ML-project.git
cd Alok_sir_ML-project

# Launch Jupyter Notebook
jupyter notebook NEW_ML_project.ipynb
```

---

### Repository Structure

```
Alok_sir_ML-project/
│
├── README.md                                          # Project documentation and team details
├── NEW_ML_project.ipynb                               # Master executed Jupyter Notebook (with all outputs)
├── Industrial Fabric Quality Inspection Dataset.csv   # Dataset file
│
├── dataset/
│   └── Industrial Fabric Quality Inspection Dataset.csv  # Direct dataset link
│
└── notebooks/
    └── NEW_ML_project.ipynb                           # Direct notebook link
```
