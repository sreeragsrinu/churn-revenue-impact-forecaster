# 📉 Churn Revenue Impact Forecaster — v3

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Enabled-green)
![Status](https://img.shields.io/badge/Status-Completed-success)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

An end-to-end machine learning pipeline for customer churn prediction, revenue risk forecasting, customer lifetime value estimation, and retention ROI analytics.

---

## Overview

The **Churn Revenue Impact Forecaster** is a business-focused machine learning and analytics project designed to identify customers at risk of churn and estimate the financial impact associated with customer loss.

Unlike traditional churn prediction systems that only classify customers as churned or retained, this project converts model predictions into actionable business intelligence through:

- Revenue loss forecasting
- Customer Lifetime Value (CLV) estimation
- Retention ROI analysis
- Risk-value segmentation
- Explainable AI insights

The project combines predictive analytics, financial modeling, and visualization into a production-style analytics workflow.

---

## Business Problem

Customer churn is one of the largest revenue risks for subscription-based businesses.

Most churn prediction systems stop at predicting whether a customer may leave. However, businesses also need to understand:

- Which customers are financially important
- How much revenue is at risk
- Whether retention campaigns are worth the cost
- Which customers should be prioritized first

This project solves that gap by integrating machine learning with revenue intelligence and retention analytics.

---

## Key Features

### Machine Learning

- Random Forest Classifier
- XGBoost Classifier
- Probability calibration using Isotonic Regression
- Stratified train-test split
- Cross-validation with ROC-AUC scoring
- F1-optimized threshold tuning

### Data Engineering

- Automated preprocessing pipeline
- Missing value handling
- One-hot encoding
- Feature engineering
- NaN-safe SMOTE oversampling

### Revenue Intelligence

- Discounted Customer Lifetime Value (CLV)
- Expected Revenue Loss forecasting
- Retention ROI estimation
- High-risk customer prioritization

### Explainability

- SHAP explainability integration
- Feature importance analysis
- Correlation heatmap analytics

### Visual Analytics

- 13 production-quality visualizations
- Dark analytics dashboard theme
- 300 DPI export-ready charts

---

## Project Structure

```bash
Churn Revenue Impact Forecaster/
│
├── data/
│   └── Telco-Customer-Churn.csv
│
├── images/
│   ├── charts/
│   │   ├── 01_churn_overview.png
│   │   ├── 02_monthly_charges.png
│   │   ├── ...
│   │   └── 13_shap_summary.png
│   │
│   ├── churn_revenue_forecast_results.csv
│   └── model_comparison.csv
│
├── powerbi/
│   └── churn_revenue_dashboard.pbix
│
├── churn_revenue_impact_forecaster.py
├── requirements.txt
├── README.md
└── .gitignore
````

---

## Dataset

Dataset used:

[Telco Customer Churn Dataset (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

The dataset contains telecom customer records including:

* Customer demographics
* Tenure information
* Billing details
* Contract types
* Service subscriptions
* Churn status

---

## Installation

Clone the repository:

```bash
git clone https://github.com/sreeragsrinu/churn-revenue-impact-forecaster.git
cd churn-revenue-impact-forecaster
```

Create virtual environment:

```bash
python -m venv .venv
```

Activate environment:

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Requirements

```txt
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
imbalanced-learn
shap
```

---

## Configuration

All configurable parameters are located at the top of the script.

```python
DATA_PATH        = r"path/to/Telco-Customer-Churn.csv"
OUTPUT_DIR       = r"path/to/output"

TEST_SIZE        = 0.20
RANDOM_STATE     = 42
N_ESTIMATORS     = 300

USE_SMOTE        = True

CLV_MONTHLY_CAP  = 24
DISCOUNT_RATE    = 0.10
RETENTION_COST   = 50

HIGH_RISK_THRESH = 0.60
DPI              = 300
```

---

## Running the Project

```bash
churn_revenue_impact_forecaster.py
```

---

## Workflow Pipeline

```text
Raw Dataset
    ↓
Data Cleaning
    ↓
Feature Engineering
    ↓
Encoding & Imputation
    ↓
Train/Test Split
    ↓
SMOTE Oversampling
    ↓
Model Training
    ↓
Probability Calibration
    ↓
Threshold Optimization
    ↓
Evaluation & Forecasting
    ↓
Revenue Intelligence
    ↓
Visualization Export
```

---

## Models Used

| Model                  | Purpose                      |
| ---------------------- | ---------------------------- |
| Random Forest          | Baseline ensemble classifier |
| XGBoost                | Gradient boosting classifier |
| CalibratedClassifierCV | Probability calibration      |
| SMOTE                  | Class imbalance handling     |

---

## Model Performance

| Model         | ROC-AUC | F1 Score | Average Precision | Accuracy |
| ------------- | ------- | -------- | ----------------- | -------- |
| Random Forest | 0.8302  | 0.6148   | 0.6291            | 0.7786   |
| XGBoost       | 0.8183  | 0.6117   | 0.6138            | 0.7594   |

---

## Revenue Intelligence Engine

### Customer Lifetime Value (CLV)

```text
CLV = MonthlyCharges × ((1 - (1 + r/12)^-n) / (r/12))
```

Where:

* `r` = annual discount rate
* `n` = projection months

### Expected Revenue Loss

```text
ExpectedRevenueLoss = ChurnProbability × CLV
```

### Retention ROI

```text
RetentionROI = (ChurnProbability × CLV) - RetentionCost
```

---

## Risk Segmentation

Customers are categorized into four business-focused segments:

| Risk Level | Customer Value | Segment                |
| ---------- | -------------- | ---------------------- |
| High Risk  | High CLV       | Priority Retention     |
| High Risk  | Low CLV        | Selective Intervention |
| Low Risk   | High CLV       | Relationship Nurturing |
| Low Risk   | Low CLV        | Low Priority           |

---


# 📈 Power BI Dashboard

An interactive business intelligence dashboard was developed in Power BI to complement the machine learning pipeline and provide executive-level customer retention insights.

The dashboard includes:

- Executive KPI monitoring
- Revenue-at-risk tracking
- Customer churn segmentation
- Retention ROI analysis
- CLV vs Churn Probability analytics
- Interactive filtering and drilldowns
- Model explainability integration

## Dashboard Preview
## 01 — Executive Overview
![Power BI Dashboard1](images/powerbi_dashboard1.png)
## 02 — Model Insights
![Power BI Dashboard2](images/powerbi_dashboard2.png)
---

# 📊 Analytics Dashboard Preview

## 01 — Churn Overview
![Churn Overview](images/charts/01_churn_overview.png)

---

## 02 — Monthly Charges Distribution
![Monthly Charges](images/charts/02_monthly_charges.png)

---

## 03 — Customer Tenure Analysis
![Tenure Analysis](images/charts/03_tenure.png)

---

## 04 — Churn Rate by Contract Type
![Contract Churn](images/charts/04_churn_by_contract.png)

---

## 05 — Correlation Heatmap
![Correlation Heatmap](images/charts/05_correlation_heatmap.png)

---

## 06 — ROC Curve Comparison
![ROC Curves](images/charts/06_roc_curves.png)

---

## 07 — Precision Recall Curves
![Precision Recall](images/charts/07_precision_recall.png)

---

## 08 — Confusion Matrices
![Confusion Matrices](images/charts/08_confusion_matrices.png)

---

## 09 — Feature Importance Analysis
![Feature Importance](images/charts/09_feature_importance.png)

---

## 10 — Churn Probability Distribution
![Churn Probability](images/charts/10_churn_probability_dist.png)

---

## 11 — Revenue Loss by Segment
![Revenue Loss](images/charts/11_revenue_loss_by_segment.png)

---

## 12 — Retention ROI Scatter
![Retention ROI](images/charts/12_retention_roi_scatter.png)

---

## 13 — SHAP Explainability Summary
![SHAP Summary](images/charts/13_shap_summary.png)

![Output1](images/output1.png)
![Output2](images/output2.png)
![Output3](images/output3.png)

## Generated Outputs

### Charts

* Churn distribution analysis
* Customer tenure analytics
* Monthly charges distribution
* Contract churn comparison
* Correlation heatmap
* ROC curve comparison
* Precision-Recall curves
* Confusion matrices
* Feature importance analysis
* Churn probability distribution
* Revenue loss forecasting
* Retention ROI visualization
* SHAP explainability summary

### CSV Exports

* `churn_revenue_forecast_results.csv`
* `model_comparison.csv`

---

## Technologies Used

| Category           | Tools                         |
| ------------------ | ------------------------------|         
| Programming        | Python                        |
| Data Analysis      | Pandas, NumPy                 |
| Visualization      | Matplotlib, Seaborn, Power BI |
| Machine Learning   | Scikit-learn, XGBoost         |
| Imbalance Handling | SMOTE                         |
| Explainability     | SHAP                          |

---

## Why This Project Matters

This project demonstrates practical skills in:

* Machine learning model development
* Probability calibration
* Business-focused analytics
* Revenue forecasting
* Customer segmentation
* Explainable AI (SHAP)
* Data visualization
* End-to-end analytics pipeline design

---

## Business Impact

This project demonstrates how machine learning can move beyond prediction into actionable business strategy.

Using churn probabilities combined with customer lifetime value estimation, the system helps businesses:

- Identify high-value customers at risk of churn
- Estimate revenue exposure before customer loss occurs
- Prioritize retention campaigns based on ROI
- Reduce inefficient marketing spend
- Improve customer retention decision-making

The integration of machine learning, financial analytics, and business intelligence makes the project suitable for real-world telecom, SaaS, subscription, and e-commerce retention use cases.

---

## Future Improvements

* Streamlit dashboard deployment
* Real-time prediction API
* Hyperparameter optimization
* Time-series churn forecasting
* Customer recommendation engine
* Automated PDF report generation

---

## Developer

 Sreerag S
 [ Data Scientist & Machine Learning Enthusiast ]
 GitHub: [https://github.com/sreeragsrinu](https://github.com/sreeragsrinu)
 
 LinkedIn: https://www.linkedin.com/in/sreerag-s-9b25162a7/

---

## License

This project is licensed under the MIT License.

```
