# Telco Customer Churn Analysis and Prediction

This repository presents a complete end-to-end machine learning pipeline for analyzing customer churn data in the telecommunications industry. The project leverages clustering techniques for segmentation and supervised learning models to predict churn with performance evaluation using ROC and AUC metrics.

---


##  Project Description

Customer churn is a major concern for telecom companies. The ability to predict customer churn and understand why it happens can help in developing retention strategies. This project aims to:

- Clean and preprocess customer data.
- Perform exploratory analysis to uncover relationships between features.
- Create customer segments using KMeans clustering and PCA.
- Predict churn using multiple machine learning models.
- Compare model performance using AUC and classification metrics.
- Visualize important patterns and evaluation metrics.

---

##  Dataset

We use the **Telco Customer Churn** dataset. It contains various customer attributes such as services signed up for, tenure, billing information, and whether or not the customer has churned.

- **Source**: [Kaggle Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)
- **File**: `Telco-Customer-Churn.csv`

Key columns include:
- `gender`, `SeniorCitizen`, `Partner`, `tenure`
- `PhoneService`, `MultipleLines`, `InternetService`
- `MonthlyCharges`, `TotalCharges`
- `Churn` (Target)

---

##  Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/telco-churn-prediction.git
   cd telco-churn-prediction
   ```

2. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

---

##  Usage

Run the main script:
```bash
python churn_analysis.py
```

Ensure the dataset `Telco-Customer-Churn.csv` is in the same directory.

---

##  Project Structure

```
telco-churn-prediction/
│
├── Telco-Customer-Churn.csv          # Dataset
├── churn_analysis.py                 # Main analysis script
├── requirements.txt                  # Dependencies
└── README.md                         # Project documentation
```

---

##  Data Preprocessing

- Dropped `customerID` as it’s non-informative.
- Converted `TotalCharges` to numeric, handled missing values.
- Encoded categorical features using `LabelEncoder`.
- Transformed target `Churn` into binary (1 = Yes, 0 = No).

---

##  Exploratory Data Analysis (EDA)

- Correlation matrix using a heatmap.
- Identified strong relationships between features.
- Helped in feature selection and hypothesis building.

---

##  Feature Engineering

Added bucketized versions of key continuous variables:

- `TenureBucket`: Buckets based on tenure
- `MonthlyBucket`: Buckets for monthly charges
- `TotalBucket`: Buckets for total charges

These features enhance model interpretability and segmentation.

---

##  Customer Segmentation

- Applied `StandardScaler` for normalization.
- Reduced feature dimensionality with `PCA`.
- Clustered customers into 4 segments using `KMeans`.
- Visualized clusters on 2D PCA plane.

---

##  Churn Prediction

Tested 3 models using pipelines with scaling:
- **Random Forest Classifier**
- **Gradient Boosting Classifier**
- **Logistic Regression**

Trained and tested on a 80/20 split.

---

##  Model Evaluation

Each model was evaluated using:
- **AUC Score**
- **Confusion Matrix**
- **Precision, Recall, F1-Score**
- **ROC Curve Visualization**

---

##  Results

- **Best performing model**: `Gradient Boosting` (AUC = 0.8340)
- ROC Curves indicated strong performance for all models.
- Logistic Regression achieved comparable results with simpler interpretability.

Segmented customer insights can be used for targeted retention campaigns.

---

##  Dependencies

Install all required libraries using:

```bash
pip install -r requirements.txt
```

Dependencies include:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

