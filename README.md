# 📌 Bank Customer Churn Prediction using Support Vector Machine (SVM)

## 📖 Project Overview

This project builds a Machine Learning model to predict whether a bank customer will churn (leave the bank) or not.

The complete ML pipeline includes:
- Data preprocessing
- Feature engineering
- Handling imbalanced datasets
- Feature scaling
- Model training using SVM
- Hyperparameter tuning using GridSearchCV
- Performance comparison of multiple sampling strategies

---

## 📂 Dataset

**Dataset:** Bank Churn Modelling Dataset  
**Source:** YBI Foundation GitHub Repository  

### Features Included:
- CreditScore
- Geography
- Gender
- Age
- Tenure
- Balance
- Num Of Products
- Has Credit Card
- Is Active Member
- Estimated Salary
- Target: `Churn`

---

## 🛠️ Tech Stack

- Python 3.12
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Imbalanced-Learn

---

## ⚙️ Project Workflow

### 1️⃣ Data Preprocessing

- Checked and removed duplicate Customer IDs
- Encoded categorical variables:
  - Geography → Numerical encoding
  - Gender → Binary encoding
- Created new feature:
  - `Zero Balance`
- Standardized numerical features using `StandardScaler`

---

### 2️⃣ Handling Imbalanced Data

Since the dataset is imbalanced, three approaches were tested:

1. Normal dataset
2. Random Under Sampling (RUS)
3. Random Over Sampling (ROS)

Libraries used:

```python
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
