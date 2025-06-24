# In scripts/evaluation.py

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib # For loading ARIMA models
from tensorflow.keras.models import load_model # For loading LSTM model
from prophet import Prophet # For loading Prophet model
import pickle # For Prophet
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os

# Define paths (ensure these are correctly defined at the top of evaluation.py)
PROCESSED_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "AAPL_preprocessed_data.csv")
ARIMA_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "trained_models", "arima_model.pkl")
PROPHET_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "trained_models", "prophet_model.pkl")
LSTM_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "trained_models", "lstm_model.h5")


def evaluate_models():
    print("\n--- Running scripts/evaluation.py ---")
    try:
        # Load processed data with robust reading
        df = None
        try:
            df = pd.read_csv(PROCESSED_DATA_PATH, header=None, index_col=0, parse_dates=True)
            df.columns = ['Close']
            df.index = pd.to_datetime(df.index) # Ensure index is datetime after read
            
            # Since preprocessing now ensures 'B' frequency, set it here explicitly
            if df.index.freq is None:
                df = df.asfreq('B')
                df['Close'].fillna(method='ffill', inplace=True)
                df['Close'].fillna(method='bfill', inplace=True)
                print("Set 'B' frequency for full DataFrame index in evaluation.")

        except Exception as e:
            print(f"Error reading processed data in evaluation: {e}")
            return # Exit if data can't be loaded

        # Ensure the 'Close' column is numeric and fill NaNs robustly (redundant but safe after preprocessing)
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        # Use modern fillna syntax
        df['Close'] = df['Close'].ffill()
        df['Close'] = df['Close'].bfill()
        if df['Close'].isnull().any():
            print("Warning: NaNs in 'Close' column after ffill/bfill in evaluation. Filling with mean/0.")
            # Use modern fillna syntax
            df['Close'] = df['Close'].fillna(df['Close'].mean())
            df['Close'] = df['Close'].fillna(0)

        # The rest of evaluation.py ARIMA section should be fine as it was
        # since the model itself will now have a frequency-aware index.
        # The `common_index` logic is good for robustness.

        # --- IMPORTANT: Ensure 'df' has a frequency for ARIMA/Prophet consistency ---
        # This df should ideally be the same as the one ARIMA/Prophet trained on.
        # Let's try to infer and set frequency here if not already set by preprocessing.
        if df.index.freq is None:
            inferred_freq_df = pd.infer_freq(df.index)
            if inferred_freq_df:
                df = df.asfreq(inferred_freq_df)
                print(f"Inferred and set frequency for full DataFrame index in evaluation: {inferred_freq_df}")
            else:
                print("Warning: Could not infer frequency for full DataFrame index in evaluation. This might affect models that rely on frequency.")

        # Split data (same as in other scripts)
        train_size = int(len(df) * 0.8)
        train_df, test_df = df[0:train_size], df[train_size:len(df)]

        # Check if test_df is empty
        if test_df.empty:
            print("Error: test_df is empty. Cannot perform evaluation.")
            return

        # Initialize results dictionary
        results = {}
        all_forecasts = pd.DataFrame(index=test_df.index)

        # --- Evaluate ARIMA/SARIMA Model ---
        try:
            with open(ARIMA_MODEL_PATH, 'rb') as f:
                arima_model = joblib.load(f)
            
            # --- REVERT ARIMA PREDICTION TO INTEGER POSITIONS ---
            # But calculate positions relative to the *original full dataset* that the training was based on.
            # The model was trained on `train_df`, which is `df[0:train_size]`.
            # So, to predict the test set, we predict from `train_size` up to `len(df)-1`.
            start_predict_idx = len(train_df) 
            end_predict_idx = len(df) - 1

            # Ensure the test_df index is available to apply to the predictions
            # The statsmodels warning about "No supported index" will still appear,
            # but we are manually assigning the index after prediction.
            
            # Safely predict and align
            if start_predict_idx <= end_predict_idx:
                arima_forecast_values_raw = arima_model.predict(start=start_predict_idx, end=end_predict_idx)
                
                # Align predictions with the test set's index
                if len(arima_forecast_values_raw) == len(test_df.index):
                    arima_forecast_series = pd.Series(arima_forecast_values_raw, index=test_df.index)
                else:
                    print(f"Mismatch in ARIMA prediction length ({len(arima_forecast_values_raw)}) and test_df length ({len(test_df.index)}). Attempting reindex.")
                    # Fallback for length mismatch: Create series, then reindex to target test_df length.
                    # This implies the prediction might not cover all exact test_df dates or covers more.
                    # Reindexing will fill NaNs where test_df dates are not covered by raw prediction.
                    arima_forecast_series = pd.Series(arima_forecast_values_raw, 
                                                      index=pd.date_range(start=test_df.index.min(), 
                                                                          periods=len(arima_forecast_values_raw), 
                                                                          freq=test_df.index.freq if test_df.index.freq else 'D')) # Use test_df freq if available, else 'D'
                    arima_forecast_series = arima_forecast_series.reindex(test_df.index)
                    arima_forecast_series.ffill(inplace=True) # Fill NaNs
                    arima_forecast_series.bfill(inplace=True) # Fill NaNs
            else:
                print("ARIMA prediction range is invalid (start_predict_idx > end_predict_idx). No ARIMA predictions generated.")
                arima_forecast_series = pd.Series([], index=test_df.index) # Empty series for consistent type
            
            # Ensure arima_forecast_series has the same length as test_df['Close'] for RMSE
            # and handle potential NaNs that might arise from alignment
            common_index = arima_forecast_series.dropna().index.intersection(test_df['Close'].dropna().index)
            if not common_index.empty:
                all_forecasts['ARIMA'] = arima_forecast_series.reindex(test_df.index) # Ensure full alignment before RMSE
                rmse_arima = np.sqrt(mean_squared_error(test_df['Close'].loc[common_index], all_forecasts['ARIMA'].loc[common_index]))
                results['ARIMA'] = rmse_arima
                print(f"ARIMA RMSE: {rmse_arima:.4f}")
            else:
                print("No common data points for ARIMA RMSE calculation after prediction and alignment. Skipping RMSE for ARIMA.")
                
        except FileNotFoundError:
            print(f"ARIMA model not found at {ARIMA_MODEL_PATH}. Skipping ARIMA evaluation.")
        except Exception as e:
            print(f"Error loading/evaluating ARIMA: {e}")
            print(f"ARIMA evaluation error details: {e}") 

        # --- Evaluate Prophet Model ---
        try:
            with open(PROPHET_MODEL_PATH, 'rb') as f:
                prophet_model = pickle.load(f)
            
            # Prophet requires a DataFrame with 'ds' column for prediction
            future_prophet_df = pd.DataFrame({'ds': test_df.index})
            future_prophet_df['ds'] = pd.to_datetime(future_prophet_df['ds'])
            
            prophet_forecast_df = prophet_model.predict(future_prophet_df)
            prophet_forecast_series = prophet_forecast_df.set_index('ds')['yhat'].reindex(test_df.index)
            
            if prophet_forecast_series.isnull().any():
                print("Warning: NaNs in Prophet predictions after alignment in evaluation. Filling with last valid prediction.")
                prophet_forecast_series.ffill(inplace=True) # Fill NaNs forward
                prophet_forecast_series.bfill(inplace=True) # Fill NaNs backward (for leading NaNs)
                prophet_forecast_series.fillna(0, inplace=True) # Fallback if all are NaN

            all_forecasts['Prophet'] = prophet_forecast_series
            
            rmse_prophet = np.sqrt(mean_squared_error(test_df['Close'], prophet_forecast_series))
            results['Prophet'] = rmse_prophet
            print(f"Prophet RMSE: {rmse_prophet:.4f}")
        except FileNotFoundError:
            print(f"Prophet model not found at {PROPHET_MODEL_PATH}. Skipping Prophet evaluation.")
        except Exception as e:
            print(f"Error loading/evaluating Prophet: {e}")
            print(f"Prophet evaluation error details: {e}")


        # --- Evaluate LSTM Model ---
        try:
            lstm_model = load_model(LSTM_MODEL_PATH)
            
            scaler = MinMaxScaler(feature_range=(0, 1))
            train_data_for_scaler = train_df['Close'].values.reshape(-1, 1)
            scaler.fit(train_data_for_scaler)

            combined_data_for_lstm = df['Close'].values.reshape(-1, 1)
            scaled_combined_data = scaler.transform(combined_data_for_lstm)
            
            lstm_predictions_scaled = []
            look_back = 60 
            
            if len(test_df) > 0 and len(train_df) >= look_back:
                start_test_data_in_scaled = len(train_df) 

                for i in range(len(test_df)):
                    input_seq_start = start_test_data_in_scaled + i - look_back
                    input_seq_end = start_test_data_in_scaled + i
                    
                    if input_seq_start < 0: 
                         lstm_predictions_scaled.append(np.nan)
                         continue 
                        
                    input_seq = scaled_combined_data[input_seq_start : input_seq_end]
                    input_seq = input_seq.reshape(1, look_back, 1)
                    
                    predicted_scaled_value = lstm_model.predict(input_seq, verbose=0)[0, 0]
                    lstm_predictions_scaled.append(predicted_scaled_value)
                
                valid_lstm_predictions_scaled = np.array([p for p in lstm_predictions_scaled if not np.isnan(p)])
                if len(valid_lstm_predictions_scaled) > 0:
                    lstm_forecast_values_original = scaler.inverse_transform(valid_lstm_predictions_scaled.reshape(-1, 1)).flatten()
                    
                    lstm_forecast_series = pd.Series(lstm_forecast_values_original, 
                                                     index=test_df.index[:len(lstm_forecast_values_original)])
                    all_forecasts['LSTM'] = lstm_forecast_series
                    
                    common_index = lstm_forecast_series.dropna().index.intersection(test_df['Close'].dropna().index)
                    if not common_index.empty:
                        rmse_lstm = np.sqrt(mean_squared_error(test_df['Close'].loc[common_index], lstm_forecast_series.loc[common_index]))
                        results['LSTM'] = rmse_lstm
                        print(f"LSTM RMSE: {rmse_lstm:.4f}")
                    else:
                        print("No common data points for LSTM RMSE calculation after walk-forward prediction.")
                else:
                    print("No valid LSTM predictions generated.")
            else:
                print("Skipping LSTM evaluation: Not enough data for walk-forward prediction or empty test set.")


        except FileNotFoundError:
            print(f"LSTM model not found at {LSTM_MODEL_PATH}. Skipping LSTM evaluation.")
        except Exception as e:
            print(f"Error loading/evaluating LSTM: {e}")
            print(e) 

        print("\n--- Model Evaluation Summary ---")
        for model_name, rmse_val in results.items():
            print(f"{model_name} RMSE: {rmse_val:.4f}")
        
        # --- Plotting All Forecasts vs Actual ---
        plt.figure(figsize=(18, 9)) 
        plt.plot(df.index, df['Close'], label='Full Historical Data', color='gray', linestyle=':', alpha=0.7)
        plt.plot(test_df.index, test_df['Close'], label='Actual Test Prices', color='blue', linewidth=2)
        
        if 'ARIMA' in all_forecasts.columns:
            plt.plot(all_forecasts.index, all_forecasts['ARIMA'], label='ARIMA Forecast', color='red', linestyle='--')
        
        if 'Prophet' in all_forecasts.columns:
            plt.plot(all_forecasts.index, all_forecasts['Prophet'], label='Prophet Forecast', color='green', linestyle='-.')

        if 'LSTM' in all_forecasts.columns:
            plt.plot(all_forecasts.index, all_forecasts['LSTM'], label='LSTM Forecast', color='purple', linestyle=':')

        plt.title('Comparison of Model Forecasts vs Actual Prices')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend()
        plt.grid(True)
        plt.tight_layout() 
        plt.show()

    except Exception as e:
        print(f"General error during evaluation script execution: {e}")

if __name__ == "__main__":
    evaluate_models()