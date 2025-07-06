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

**3.prophet & arima**

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

**6.deep_learning_part1.ipynb**
Purpose: Sequence modeling with LSTM & GRU (Phase 1)
What it does:

Defines a reusable function to prepare time-series data for RNNs

Trains LSTM/GRU while searching over layer depth and hyperparameters

Produces four prediction series:

Price using own features

Price with external features

Daily return using own features

Daily return with external features

**7.deep_learning_part2.ipynb**
Purpose: Sequence modeling refinement (Phase 2)
What it does:

Continue hyperparameter tuning for GRU & LSTM to squeeze out top performance

Optimizes the feature set based on earlier importance analyses

Evaluates and visualizes final model outputs for both price and return forecasts

**8.app.py**
Purpose: Unified results dashboard
What it does:

Implements a Streamlit web app that loads all model outputs

Allows interactive selection of Random Forest, XGBoost, GRU, and LSTM results

Displays metrics tables and Actual vs. Predicted plots for each model


**This workflow—from raw data ingestion through classical ML baselines to advanced deep-learning architectures
and a live demo—provides a complete end-to-end pipeline for time-series prediction on the S&P 500.**
