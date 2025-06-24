import pandas as pd
from prophet import Prophet
import pickle
import os
# import matplotlib.pyplot as plt # Not needed in this script anymore
# from sklearn.metrics import mean_squared_error # Not needed in this script anymore
# import numpy as np # Not needed in this script anymore

# Define paths
PROCESSED_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "AAPL_preprocessed_data.csv")
PROPHET_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "trained_models", "prophet_model.pkl")


def train_prophet_model(df_prophet_input):
    """
    Trains a Prophet model.
    df_prophet_input must already be a DataFrame with 'ds' (datestamp) and 'y' (value) columns.
    """
    print("Training Prophet model...")
    try:
        df_prophet = df_prophet_input.copy() # Work on a copy
        
        # Explicitly ensure 'ds' is datetime
        df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
        
        # Explicitly ensure 'y' is numeric and handle NaNs thoroughly
        df_prophet['y'] = pd.to_numeric(df_prophet['y'], errors='coerce')
        df_prophet['y'] = df_prophet['y'].ffill().bfill() # Apply ffill then bfill
        
        # Final fallback for NaNs (e.g., if the column was entirely NaN)
        if df_prophet['y'].isnull().any():
            print("Warning: NaNs still present in 'y' column after fill for Prophet. Filling with mean/0.")
            df_prophet['y'].fillna(df_prophet['y'].mean(), inplace=True)
            df_prophet['y'].fillna(0, inplace=True) # Final fill with 0 if mean is also NaN

        # --- CRITICAL DEBUG PRINTS FOR PROPHET (BEFORE MODEL FIT) ---
        print("\n--- Prophet Debug Info BEFORE model.fit() ---")
        print("df_prophet head:")
        print(df_prophet.head())
        print("\ndf_prophet info:")
        df_prophet.info()
        print(f"\nNumber of NaNs in 'ds': {df_prophet['ds'].isnull().sum()}")
        print(f"Number of NaNs in 'y': {df_prophet['y'].isnull().sum()}")
        print(f"Earliest 'ds': {df_prophet['ds'].min()}")
        print(f"Latest 'ds': {df_prophet['ds'].max()}")
        print(f"Number of unique 'ds' values: {df_prophet['ds'].nunique()}")
        print(f"DataFrame shape: {df_prophet.shape}")
        print("--- End Prophet Debug Info (BEFORE model.fit()) ---")
        # --- END DEBUG PRINTS ---

        model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
        model.fit(df_prophet) # THIS IS THE LINE THAT FAILS
        print("Prophet model trained successfully.")
        return model
    except Exception as e:
        print(f"Error training Prophet model: {e}")
        return None

if __name__ == "__main__":
    print("\n--- Running scripts/prophet_model.py ---")
    os.makedirs(os.path.dirname(PROPHET_MODEL_PATH), exist_ok=True)

    try:
        # Load processed data
        df = pd.read_csv(PROCESSED_DATA_PATH, header=None, index_col=0, parse_dates=True)
        df.columns = ['Close'] # Assign column name after loading

        # Ensure index is datetime (important for Prophet)
        df.index = pd.to_datetime(df.index)

        # --- Initial DataFrame Load Info ---
        print("\n--- Prophet Initial DataFrame Load Info ---")
        print("df head:")
        print(df.head())
        print("\ndf info:")
        df.info()
        print(f"\nNaNs in 'Close' column after initial load: {df['Close'].isnull().sum()}")
        print("--- End Prophet Initial DataFrame Load Info ---")

        # Split data into training and testing sets
        train_size = int(len(df) * 0.8)
        train_df, test_df = df[0:train_size], df[train_size:len(df)]

        # --- NEW DEBUG PRINTS FOR train_df and train_df_prophet ---
        print("\n--- Prophet train_df Info Before Transformation ---")
        print("train_df head:")
        print(train_df.head())
        print("\ntrain_df info:")
        train_df.info()
        print(f"\nNaNs in train_df['Close']: {train_df['Close'].isnull().sum()}")
        print("--- End Prophet train_df Info Before Transformation ---")
        # --- END NEW DEBUG PRINTS ---

        # Prepare train_df for Prophet: 'ds' and 'y' columns
        # Change this line:
        # train_df_prophet = train_df.reset_index().rename(columns={'index': 'ds', 'Close': 'y'})
        # To this:
        train_df_prophet = train_df.reset_index().rename(columns={train_df.index.name if train_df.index.name else 0: 'ds', 'Close': 'y'})
        
        # --- NEW DEBUG PRINTS FOR train_df_prophet AFTER TRANSFORMATION ---
        print("\n--- Prophet train_df_prophet Info After Transformation ---")
        print("train_df_prophet head:")
        print(train_df_prophet.head())
        print("\ntrain_df_prophet info:")
        train_df_prophet.info()
        print(f"\nNaNs in train_df_prophet['ds']: {train_df_prophet['ds'].isnull().sum()}")
        print(f"NaNs in train_df_prophet['y']: {train_df_prophet['y'].isnull().sum()}")
        print("--- End Prophet train_df_prophet Info After Transformation ---")
        # --- END NEW DEBUG PRINTS ---

        prophet_model_fit = train_prophet_model(train_df_prophet)
        
        if prophet_model_fit:
            # Save the trained model
            with open(PROPHET_MODEL_PATH, 'wb') as f:
                pickle.dump(prophet_model_fit, f)
            print(f"Prophet model saved to {PROPHET_MODEL_PATH}")
        else:
            print("Prophet model training failed, skipping save and evaluation.")

    except Exception as e:
        print(f"An error occurred during Prophet model training script execution: {e}")