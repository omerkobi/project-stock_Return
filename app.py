import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Time Series Predictions Demo", layout="wide")
st.title("📈 Time Series Model Predictions Demo")
st.subheader("predicting the S&P500 price & daily return. ")
st.subheader("Start date : 01.01.2017 , end date : 17.04.2025 ")

# --- Sidebar for selection ---
option = st.sidebar.selectbox(
    "Select prediction to display:",
    [
        "Daily Return - RandomForest",
        "Daily Return - XGBoost",
        "Daily Return - Own Features",
        "Daily Return - External Features",
        "Price - Own Features",
        "Price - External Features"
    ]
)

# --- Load DataFrames ---
# (Replace with your actual loading code)
# df_rf_return  = pd.read_csv('raw_files/random_forest.csv')
# df_xgb_return = pd.read_csv('raw_files/xgboost.csv')
# df_return_own = pd.read_csv('raw_files/return1_deep_models.csv')
# df_return_ext = pd.read_csv('raw_files/reuturn_deep_ext.csv')
# df_price_own  = pd.read_csv('raw_files/price_own_feat.csv')
# df_price_ext  = pd.read_csv('raw_files/price_ext_feat.csv')


directory = 'raw_files'
df_price_own = pd.read_csv(rf'{directory}/price_own_feat.csv')
df_return_own = pd.read_csv(rf'{directory}/return1_deep_models.csv')
df_rf_return = pd.read_csv(rf'{directory}/random_forest.csv')
df_xgb_return = pd.read_csv(rf'{directory}/xgboost.csv')
df_return_ext = pd.read_csv(rf'{directory}/reuturn_deep_ext.csv')


# --- Load precomputed metrics ---
metrics_df = pd.read_csv('raw_files/metrics.csv', index_col='model')

# Map option to its DataFrame
dfs = {
    "Daily Return - RandomForest": df_rf_return,
    "Daily Return - XGBoost":      df_xgb_return,
    "Daily Return - Own Features": df_return_own,
    "Daily Return - External Features": df_return_ext,
    "Price - Own Features":        df_price_own,
    #"Price - External Features":    df_price_ext
}

# Validate selection
if option not in dfs:
    st.error(f"DataFrame for '{option}' not found. Please load your data.")
    st.stop()

df = dfs[option].copy()

# Ensure Date index
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').set_index('Date')

# --- Display Metrics ---
st.subheader("📊 Metrics")
if option in metrics_df.index:
    # Grab the metrics row and drop all NaN columns
    row = metrics_df.loc[option].dropna()
    st.table(row.to_frame().T)
else:
    st.write("Metrics not found for this selection.")

# --- Plot Actual vs Predicted ---
st.subheader("📈 Actual vs Predicted Plot")
fig, ax = plt.subplots(figsize=(12, 5))

# Plot Actual
if 'Actual' in df.columns:
    ax.plot(df.index, df['Actual'], label='Actual', color='black', linewidth=2)

# Plot each prediction column
for col in df.columns:
    if col.lower().startswith('pred'):
        ax.plot(df.index, df[col], label=col.replace('_', ' '), linestyle='--')

ax.set_title(option)
ax.set_xlabel('Date')
ax.set_ylabel('Value')
ax.legend()
st.pyplot(fig)

st.markdown('---')
st.write('📌 Run this app with: streamlit run app.py')