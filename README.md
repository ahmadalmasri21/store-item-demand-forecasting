# Store Item Sales Forecasting

A complete machine learning project for **time series forecasting**, aiming to predict daily sales for multiple items across multiple stores over a 3-month horizon using 5 years of historical data.

The project implements an end-to-end workflow including data loading, exploratory data analysis (EDA), feature engineering, baseline modeling, advanced model comparison, time-series cross-validation, and final submission preparation.

---

## Key Features

* **End-to-End Forecasting Pipeline:** From raw CSV → EDA → feature engineering → baseline → advanced modeling → cross-validation → final predictions.
* **Feature Engineering:** Created meaningful time-based features and generated **lag features** and **rolling statistics** to capture temporal dependencies.
* **Baseline Models:** Implemented simple (Last Year Sales) and **Linear Regression** baselines to benchmark performance.
* **Advanced Models:** Compared **XGBoost**, and **LSTM** deep learning models for forecasting.
* **Time-Series Cross-Validation:** Used **expanding-window folds** to evaluate models while respecting temporal order.

---

## Dataset

This project uses the **Store Item Demand Forecasting** dataset from Kaggle.

[Dataset Link](https://www.kaggle.com/competitions/demand-forecasting-kernels-only)

### Details

| Attribute | Value |
| :--- | :--- |
| **Type** | Tabular time-series dataset |
| **Target Variable** | `sales` |
| **Entities** | 50 items $\times$ 10 stores |
| **Date Range** | 5 years of daily sales |
| **Characteristics** | No holidays or store closures included; multiple items and stores; contains temporal patterns (trend, weekly/monthly seasonality). |

---

## Tech Stack

| Category | Components |
| :--- | :--- |
| **Language** | Python |
| **ML Libraries** | `pandas`, `numpy`, `scikit-learn`, `xgboost`,  `tensorflow` |
| **Visualization** | `matplotlib`, `seaborn` |
| **Tools** | Time-series cross-validation, lag & rolling feature engineering |

---

## Project Pipeline

1.  ### Data Preparation & EDA
    * Loaded `train.csv` and `test.csv`.
    * Converted the date column to datetime objects.
    * Visualized:
        * Total sales over time 
        * Yearly trend
        * Monthly and weekly/day-of-week patterns
        * Weekly/day-of-week patterns
    * Identified trend, seasonality, and variability across stores and items.

2.  ### Feature Engineering
    * Created time-based features:
        * `day_of_week`, `month`, `year`, `day_of_year`, `week_of_year`, `is_weekend`
    * Generated lag features to capture past sales: `lag_7`, `lag_30`, `lag_90`
    * Created rolling statistics: `rolling_mean_30`
    * Handled missing values (NaNs) created after lag/rolling calculation.

3.  ### Advanced Models & Comparison
    * Trained XGBoost, and LSTM models using engineered features.
    * **Baseline Modeling:**
        * Simple baseline: last year's same period sales.
        * Linear Regression baseline using only date features.
        * Reported RMSE for benchmarking (see Results Summary).
    * **Time-Series Cross-Validation (Expanding-Window):**
        * Fold 1: Train Year 1 $\to$ Predict Year 2
        * Fold 2: Train Years 1–2 $\to$ Predict Year 3
        * Fold 3: Train Years 1–3 $\to$ Predict Year 4
        * Fold 4: Train Years 1–4 $\to$ Predict Year 5
        * Average RMSE across folds: $5.35$

4.  ### Final Model & Submission
    * Retrained XGBoost on all training data using final tuned parameters.
    * Generated predictions for the 3-month test period.
    * Saved `submission.csv` in the required Kaggle format.

---

## Results Summary

| Model | RMSE | Notes |
| :--- | :---: | :--- |
| **Linear Regression Baseline** | $30.70$ | Poor fit due to complex temporal patterns. |
| **XGBoost (Initial Fit)** | $7.90$ | Strong tree-based performance. |
| **LSTM (Initial Fit)** | $5.86$ | Slightly better capture of sequential dependency. |
| **Time-Series CV (XGBoost)** | $5.35$ | Most reliable performance metric. |

> **Final Choice:** LSTM slightly outperformed XGBoost in RMSE but XGBoost was chosen for the final submission due to simplicity, reproducibility, and faster training time.
