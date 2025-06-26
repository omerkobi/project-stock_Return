import streamlit as st
import pandas as pd
# sklearn.metrics imports can be removed since metrics are precomputed
import matplotlib.pyplot as plt


directory = 'raw_files'
df_price_own = pd.read_csv(rf'{directory}/price_own_feat.csv')
df_return_own = pd.read_csv(rf'{directory}/return1_deep_models.csv')
df_rf_return = pd.read_csv(rf'{directory}/random_forest.csv')
df_xgb_return = pd.read_csv(rf'{directory}/xgboost.csv')
df_return_ext = pd.read_csv(rf'{directory}/reuturn_deep_ext.csv')




st.set_page_config(page_title="Time Series Predictions Demo", layout="wide")
st.title("📈 Time Series Model Predictions Demo")

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
# Your actual loading logic here
# df_rf_return      = pd.read_pickle('raw_files/rf_return.pkl')
# df_xgb_return     = pd.read_pickle('raw_files/xgb_return.pkl')
# df_return_own     = pd.read_pickle('raw_files/gru_lstm_return_own.pkl')
# df_return_ext     = pd.read_pickle('raw_files/gru_lstm_return_ext.pkl')
# df_price_own      = pd.read_pickle('raw_files/gru_lstm_price_own.pkl')
# df_price_ext      = pd.read_pickle('raw_files/gru_lstm_price_ext.pkl')

# --- Load precomputed metrics ---
# metrics.csv should have a 'model' column matching the sidebar options
# and metric columns like mse_lstm, r2_lstm, mse_gru, r2_gru, mse, r2
metrics_df = pd.read_csv('raw_files/metrics.csv')
metrics_df = metrics_df.set_index('model')

# Map option to its DataFrame
dfs = {
    "Daily Return - RandomForest": df_rf_return,
    "Daily Return - XGBoost": df_xgb_return,
    "Daily Return - Own Features": df_return_own,
    "Daily Return - External Features": df_return_ext,
    "Price - Own Features": df_price_own,
    #"Price - External Features": df_price_ext
}

# Validate selection
if option not in dfs :
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
    # select and show the row for the chosen model
    row = metrics_df.loc[option]
    # Convert to DataFrame for nicer display
    st.table(row.to_frame().T)
else:
    st.write("Metrics not found for this selection.")

# --- Plot Actual vs Predicted ---
st.subheader("📈 Actual vs Predicted Plot")
fig, ax = plt.subplots(figsize=(12, 5))
# Plot Actual if exists
if 'Actual' in df.columns:
    ax.plot(df.index, df['Actual'], label='Actual', color='black', linewidth=2)
# Plot prediction columns
for col in df.columns:
    if col.lower().startswith('pred'):
        ax.plot(df.index, df[col], label=col.replace('_', ' '), linestyle='--')
ax.set_title(option)
ax.set_xlabel('Date')
ax.set_ylabel('Value')
ax.legend()
st.pyplot(fig)

st.markdown('---')
st.write('📌 Run this app with: `streamlit run app.py`')
