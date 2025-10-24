# 🎓 Student Performance Prediction and Fairness Analysis

This project explores **student academic performance prediction** using machine learning, combined with **data analytics and bias detection/mitigation techniques**.  
The goal is to understand how different demographic, academic, and social factors influence student grades — and ensure that the predictive model is **fair and unbiased** toward protected attributes like **gender** and **past academic failures**.

---

## 🧠 Project Overview

The dataset used here comes from the **UCI Machine Learning Repository**, containing student data from two Portuguese schools:

- **Gabriel Pereira (GP)**
- **Mousinho da Silveira (MS)**

The analysis includes:
- Data preprocessing and ETL  
- Exploratory Data Analysis (EDA)  
- Feature Engineering  
- Logistic Regression model training  
- Bias detection using **aif360**  
- Bias mitigation using the **Reweighing algorithm**

---

## 🗂️ Dataset Description

**File:** `student-mat.csv`  
**Rows:** 395  
**Columns:** 33  

Each record represents a student and includes:
- **Demographic information:** age, sex, address  
- **Family background:** parent jobs, education level  
- **Academic performance:** grades G1, G2, G3  
- **Social factors:** study time, failures, alcohol use, free time  
- **Final grade (G3)** — used to predict student success

---

## ⚙️ Data Processing and ETL Steps

### 1. **Data Import and Inspection**
```python
data = pd.read_csv('student-mat.csv')
data.head()
```

- No missing values were detected.  
- Added a new column `Grade` as a percentage:
  \[
  \text{Grade} = \frac{G3}{20} \times 100
  \]

---

### 2. **Data Cleaning**
- Verified for null values using a heatmap.  
- No imputation required — dataset is complete.  
- Renamed or encoded categorical variables for modeling.

---

### 3. **Feature Engineering**
- Created a binary variable `Result`:  
  `pass` if `G3 >= 10`, otherwise `fail`.  
- Added `failed_before = True` if `failures > 0`.  
- Performed **one-hot encoding** for nominal features:
  - `Mjob`, `Fjob`, `reason`, `guardian`  
- Used **Label Encoding** for binary features:
  - `sex`, `address`, `famsize`, `schoolsup`, etc.

---

## 📊 Exploratory Data Analysis (EDA)

Several insights were drawn through visualizations:

### 🏫 School Distribution
- ~88% of students belong to **Gabriel Pereira (GP)**.  
- ~12% belong to **Mousinho da Silveira (MS)**.

### 👩‍🦰 Gender Ratio
- Female students: **208**  
- Male students: **187**

### 🏠 Address Type
- **Urban:** ~72%  
- **Rural:** ~28%

### 📈 Performance Bias Indicators
- Male students have slightly higher average grades (**Mean: 54.57%**) than females (**Mean: 49.83%**).  
- Students who have **never failed before** show much higher performance (**Mean: 56.26%**) than those who failed before (**Mean: 36.32%**).

---

## 🤖 Machine Learning Model

### Model: Logistic Regression

A **binary classification** model was trained to predict whether a student passes or fails (`G3 >= 10`).

**Training-Test Split:**
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

### Performance Metrics
| Metric | Score |
|:-------|------:|
| Accuracy | 0.916 |
| Precision | 0.92 |
| Recall | 0.945 |
| F1-score | 0.932 |
| MAE | 0.08 |

**Confusion Matrix:**
```
[[40  6]
 [ 4 69]]
```

✅ The model performed well with **91.6% accuracy**, showing strong balance between precision and recall.

---

## ⚖️ Bias Detection & Mitigation

Bias was tested against two protected attributes:
- **Gender (sex)**
- **Previous academic failure**

---

### 1. **Bias Detection**

Using IBM’s **AI Fairness 360 (aif360)** toolkit:
```python
from aif360.metrics import BinaryLabelDatasetMetric
metric_orig_train = BinaryLabelDatasetMetric(
    bias_dataset,
    unprivileged_groups=[{'sex': 0}],
    privileged_groups=[{'sex': 1}]
)
metric_orig_train.mean_difference()
```

#### 📊 Result:
```
Difference in mean outcomes between unprivileged (female) and privileged (male) = -0.066
```
→ Indicates slight bias **favoring males**.

---

### 2. **Bias Mitigation – Reweighing Algorithm**

To correct unfair representation:
```python
from aif360.algorithms.preprocessing import Reweighing
RW = Reweighing(unprivileged_groups, privileged_groups)
RW.fit(bias_dataset)
train_tf_dataset = RW.transform(bias_dataset)
```

#### 📈 Post-Mitigation:
```
Difference in mean outcomes = 0.000000
```

✅ Bias successfully removed — fairness achieved between groups.

---

## 🧾 Key Insights

- Male students performed slightly better than female students in raw data.  
- Students with a history of failure had significantly lower grades.  
- The trained logistic regression model achieved high accuracy and generalization.  
- The **Reweighing algorithm** from `aif360` effectively neutralized observed gender bias.

---

## 🧰 Tech Stack

| Category | Tools / Libraries |
|-----------|------------------|
| **Language** | Python |
| **Data Handling** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Modeling** | scikit-learn |
| **Fairness & Ethics** | aif360 |
| **Metrics** | Accuracy, Precision, Recall, F1, MAE |

---

## 📦 Installation

```bash
# Clone this repository
git clone https://github.com/yourusername/student-performance-fairness.git
cd student-performance-fairness

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # (Linux/macOS)
# or
.venv\Scripts\activate      # (Windows)

# Install dependencies
pip install -r requirements.txt
```

---

## 📚 Dependencies

Include these in your `requirements.txt`:
```
numpy
pandas
matplotlib
seaborn
scikit-learn
aif360
```

---

## 📈 Future Work

- Explore **other ML models** (RandomForest, XGBoost) for better interpretability.  
- Implement **Fairness-aware post-processing** methods (e.g., Reject Option Classification).  
- Extend bias analysis to other features like **address**, **parent education**, or **internet access**.  
- Deploy model as a small web dashboard using **Streamlit** or **FastAPI**.

---

## 🧑‍💻 Author

**Anas Mustafa**  
AI/ML Developer | NLP & RAG Systems | Data Science Researcher  
📍 Cottbus, Germany  
🔗 [LinkedIn](https://linkedin.com/in/anasmustafa-ai) | [GitHub](https://github.com/anasmustafa-ai)
