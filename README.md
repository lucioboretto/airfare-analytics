# ✈️ Airfare Analytics

An end-to-end Data Science project focused on predicting flight prices and exploring the optimal booking window using Machine Learning.

## 🚀 Live Demo

👉 [Try the interactive Flight Price Predictor](https://airfare-analytics.streamlit.app/)

The application allows users to enter flight characteristics and explore:

- Estimated flight price
- Predicted best booking window
- Estimated price evolution from 1 to 180 days before departure
- Potential difference between the best predicted price and other booking points

> ⚠️ This application is experimental. Predictions are based on historical data and should not be interpreted as guaranteed future prices.

---

## 📌 Project Overview

Flight prices can change significantly depending on several factors, and one of the most common questions when booking a trip is:

> **"Should I buy the flight now, or should I wait?"**


Instead of simply checking prices repeatedly, I wanted to explore whether historical flight data could be used to build a model capable of estimating prices and identifying a potentially favorable booking window.

The project covers the complete Data Science workflow, from data exploration and feature engineering to Machine Learning, model interpretation and deployment.

---

## 🎯 Motivation

The main objective was not to build a production-level flight booking platform, but to create a practical Machine Learning project around a real-world problem.

The project aims to answer two questions:

1. **Can historical flight data be used to estimate flight prices?**
2. **Can the model be used to explore how predicted prices change depending on how far in advance a flight is booked?**

---

## 📊 Dataset

The project uses historical flight data containing information about:

- Airline
- Source airport
- Destination airport
- Number of stops
- Flight duration
- Days remaining before departure
- Departure day of the week
- Departure time / daypart
- Flight price

The dataset was cleaned and transformed before being used for Machine Learning.

---

## 🔍 Project Workflow

The project was developed through several stages:

### 1. Data Understanding

Initial exploration of the dataset, including:

- Dataset structure
- Data types
- Missing values
- Basic statistics
- Initial data quality checks

### 2. Data Cleaning

The data was prepared for analysis and modelling by handling:

- Missing values
- Inconsistent values
- Data types
- Feature formatting

### 3. Exploratory Data Analysis

Exploration of relationships between flight characteristics and price.

### 4. Feature Engineering

Several features were created or transformed to make the dataset more suitable for Machine Learning.

Examples include:

- Departure day of the week
- Departure daypart
- Days left before departure

### 5. Model Development and Evaluation

Several regression models were evaluated:

- Linear Regression
- Random Forest
- XGBoost
- CatBoost

Models were evaluated using:

- MAE
- RMSE
- R²

Random Forest achieved the best overall performance among the tested models.

---

## 🤖 Final Model

The final model selected for the application was a Random Forest Regressor.

### Performance

| Metric | Score |
|---|---:|
| MAE | €94.11 |
| RMSE | €192.02 |
| R² | 0.7821 |

The model was selected based on its performance compared with the other tested approaches.

A reduced hyperparameter tuning experiment was also performed. In this case, the tuned model performed worse than the initial model, illustrating that hyperparameter tuning does not necessarily improve performance.

---

## 🧠 Model Interpretability

Feature importance and SHAP were used to better understand the model.

The main features identified by the Random Forest included:

| Feature | Importance |
|---|---:|
| Airline | 0.315 |
| Stops | 0.174 |
| Duration | 0.155 |
| Source Airport | 0.139 |
| Destination Airport | 0.116 |
| Days Left | 0.052 |
| Departure Day of Week | 0.026 |
| Departure Daypart | 0.022 |

This analysis helped understand which variables the model relied on most when making predictions.

Importantly, feature importance should not be interpreted as causality. A feature being important to the model does not mean that it directly causes changes in flight prices.

---

## 💡 Booking Window Analysis

One of the main features of the project is the ability to evaluate predicted prices across different booking lead times.

Instead of testing only a few predefined scenarios, the application evaluates:

**1 to 180 days before departure.**

For each scenario, the model predicts the expected price while keeping the other flight characteristics constant.

The application then identifies the point with the lowest predicted price.

For example:

> **Best estimated booking window: 54 days before departure**  
> **Estimated price: €69.34**

This should be interpreted as a model-based scenario analysis rather than a guaranteed optimal booking date.

---

## 🖥️ Interactive Application

The final model was integrated into a Streamlit application.

Users can enter:

- Airline
- Origin airport
- Destination airport
- Travel date
- Number of stops
- Flight duration
- Departure time

The application automatically calculates derived features such as:

- Days left until departure
- Departure day of the week
- Departure daypart

It then provides:

- Estimated flight price
- Best estimated booking window
- Estimated price difference
- Price evolution chart



---

## ⚠️ Limitations

This project is primarily an educational and portfolio project.

There are several important limitations:

- The dataset represents historical flight prices and does not provide real-time market information.
- The model's MAE of approximately €94 indicates substantial prediction error.
- Flight prices are influenced by many external factors that are not included in the dataset.
- The predicted booking window should not be interpreted as a guaranteed optimal purchase date.
- Some model predictions can be counterintuitive, demonstrating the limitations of the available features and historical data.
- Feature importance describes model behavior and should not be interpreted as causal relationships.

The application should therefore be viewed as an **experimental decision-support tool**, rather than a replacement for real-time flight search platforms.

---

## 🗂️ Project Structure

```text
airfare-analytics/
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_modeling.ipynb
│   └── 05_application.ipynb
│
├── app.py
├── requirements.txt
├── .gitattributes
└── README.md

## 🛠️ Technologies

Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
XGBoost
CatBoost
SHAP
Streamlit
Git
GitHub
Git LFS