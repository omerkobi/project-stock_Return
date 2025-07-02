# Project_stock_return
 ## predicting the S&P 500 return & price using 8 years of daily prices
 **The models used in this project (ML & Deep Learning) : Xgboost, Random forest, LSTM, GRU**

### Project Structure
**1. import_stock_data**

Purpose: Initial data ingestion and preprocessing

What it does:

Downloads historical S&P 500 data from Yahoo Finance

Converts price series into daily returns

Imports external economic indicators

Generates basic features for downstream modeling

**2.EDA_external_data**

Purpose: Exploratory Data Analysis

What it does:

Computes and visualizes summary statistics for the S&P 500 and each external dataset

Uncovers patterns and relationships that informed later feature engineering

**3.prophet_arima**

Purpose: First-pass time-series modeling

What it does:

Fits Facebook Prophet and SARIMA models to the S&P 500 series

Evaluates forecast quality (Prophet underperforms; SARIMA delivers reasonable results after preprocessing)

**4.feature_selection**

Purpose: Feature engineering (Part 1)

What it does:

Applies Random Forest to assess initial feature importance

Migrates to XGBoost for improved predictive performance

Iteratively refines the feature set based on model feedback

**5.final_ML_models**

Purpose: Feature engineering (Part 2) & hyperparameter tuning

What it does:

Uses permutation importance and additional selection techniques to identify the strongest predictors

Conducts grid search to fine-tune hyperparameters for the final models

Prepares the end-to-end pipeline for production-ready predictions
