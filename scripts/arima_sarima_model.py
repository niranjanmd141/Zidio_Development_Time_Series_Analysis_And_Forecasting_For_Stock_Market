# In scripts/arima_sarima_model.py

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np

# Define paths
PROCESSED_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "AAPL_preprocessed_data.csv")
ARIMA_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "trained_models", "arima_model.pkl")


def train_arima_model(train_data, order=(5, 1, 0)):
    """
    Trains an ARIMA model. The input train_data should ideally have a frequency set.
    """
    print(f"Training ARIMA model with order {order}...")
    try:
        # The frequency should now be set by preprocessing, so no need to infer here
        model = ARIMA(train_data, order=order) # train_data should now have a frequency (e.g., 'B')
        model_fit = model.fit()
        print("ARIMA model trained successfully.")
        return model_fit
    except Exception as e:
        print(f"Error training ARIMA model: {e}")
        return None

# Rest of arima_sarima_model.py remains the same, as the prediction logic was already fine
# given a correctly indexed model. The warnings about "No supported index" should disappear
# if the model is trained on data with a proper frequency.
if __name__ == "__main__":
    print("\n--- Running scripts/arima_sarima_model.py ---")
    os.makedirs(os.path.dirname(ARIMA_MODEL_PATH), exist_ok=True)

    try:
        # Read the preprocessed data, which now has a consistent 'B' frequency
        df = pd.read_csv(PROCESSED_DATA_PATH, header=None, index_col=0, parse_dates=True)
        df.columns = ['Close']
        df.index = pd.to_datetime(df.index)
        
        # Ensure 'Close' column is numeric and fill NaNs (if any remain, though preprocessing should handle this)
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        # Use modern fillna syntax
        df['Close'] = df['Close'].ffill()
        df['Close'] = df['Close'].bfill()


        # Explicitly set frequency for df if it somehow got lost (redundant but safe after preprocessing)
        if df.index.freq is None:
            df = df.asfreq('B') # Force Business Day frequency
            # Use modern fillna syntax
            df['Close'] = df['Close'].ffill()
            df['Close'] = df['Close'].bfill()

        # Split data
        train_size = int(len(df) * 0.8)
        train_df, test_df = df[0:train_size], df[train_size:len(df)]

        # Train ARIMA
        arima_order = (5, 1, 0)
        arima_model = train_arima_model(train_df['Close'], order=arima_order)

        if arima_model:
            with open(ARIMA_MODEL_PATH, 'wb') as f:
                joblib.dump(arima_model, f)
            print(f"ARIMA model saved to {ARIMA_MODEL_PATH}")

            if not test_df.empty:
                # The prediction should now work correctly with dates if the model was trained with frequency
                # The start and end indices will now map correctly to dates due to the frequency in the model's data.
                arima_forecast_values = arima_model.predict(start=len(train_df), end=len(df)-1)
                
                # Align to test_df index
                if len(arima_forecast_values) == len(test_df.index):
                    arima_forecast_series = pd.Series(arima_forecast_values, index=test_df.index)
                else:
                    # This fallback should ideally not be needed if frequency is handled well
                    print(f"Warning: ARIMA forecast length ({len(arima_forecast_values)}) does not match test_df length ({len(test_df.index)}). Reindexing.")
                    arima_forecast_series = pd.Series(arima_forecast_values, 
                                                      index=pd.date_range(start=test_df.index.min(), 
                                                                          periods=len(arima_forecast_values), 
                                                                          freq=test_df.index.freq if test_df.index.freq else 'B')) # Use 'B' as fallback
                    arima_forecast_series = arima_forecast_series.reindex(test_df.index)
                    arima_forecast_series.ffill(inplace=True)
                    arima_forecast_series.bfill(inplace=True)

                common_index_arima = arima_forecast_series.dropna().index.intersection(test_df['Close'].dropna().index)
                if not common_index_arima.empty:
                    rmse = np.sqrt(mean_squared_error(test_df['Close'].loc[common_index_arima], arima_forecast_series.loc[common_index_arima]))
                    print(f"ARIMA RMSE: {rmse:.4f}")

                    plt.figure(figsize=(12, 6))
                    plt.plot(train_df.index, train_df['Close'], label='Training Data')
                    plt.plot(test_df.index, test_df['Close'], label='Actual Prices')
                    plt.plot(arima_forecast_series.index, arima_forecast_series, label='ARIMA Forecast')
                    plt.title('ARIMA Model Forecast vs Actual Prices')
                    plt.xlabel('Date')
                    plt.ylabel('Close Price')
                    plt.legend()
                    plt.grid(True)
                    plt.show()
                else:
                    print("ARIMA: No common index for RMSE calculation after prediction and alignment.")
            else:
                print("Test DataFrame is empty, skipping ARIMA forecasting and evaluation.")
        else:
            print("ARIMA model training failed, skipping save and evaluation.")

    except Exception as e:
        print(f"An error occurred during ARIMA model script execution: {e}")