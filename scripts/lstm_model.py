import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os

# Define paths
PROCESSED_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "AAPL_preprocessed_data.csv")
LSTM_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "trained_models", "lstm_model.h5")

def create_sequences(data, sequence_length):
    """
    Creates sequences for LSTM input.
    `data` should be a 2D numpy array (e.g., reshaped Close prices).
    """
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length), 0])
        y.append(data[i + sequence_length, 0])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    """
    Builds and compiles an LSTM model.
    """
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

if __name__ == "__main__":
    os.makedirs(os.path.dirname(LSTM_MODEL_PATH), exist_ok=True)

    try:
        # Load processed data
        df = pd.read_csv(PROCESSED_DATA_PATH, header=None, index_col=0, parse_dates=True)
        df.columns = ['Close'] # Assign column name after loading
        
        # Ensure the 'Close' column is numeric and fill NaNs
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df['Close'] = df['Close'].ffill()
        df['Close'] = df['Close'].bfill()

        # Select the 'Close' column as a numpy array for LSTM processing
        data_to_scale = df['Close'].values.reshape(-1, 1) # Reshape for scaler

        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data_to_scale)

        # Split into training and testing sets
        train_size = int(len(scaled_data) * 0.8)
        train_data = scaled_data[0:train_size, :]
        test_data = scaled_data[train_size:len(scaled_data), :]

        sequence_length = 60 # Number of previous days to consider for prediction

        # Create sequences for training
        X_train, y_train = create_sequences(train_data, sequence_length)
        # Reshape X for LSTM [samples, time_steps, features]
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        # Build and train LSTM model
        lstm_model = build_lstm_model((X_train.shape[1], 1))
        print("Training LSTM model...")
        # verbose=0 means no epoch output, verbose=1 shows progress bar
        lstm_model.fit(X_train, y_train, epochs=25, batch_size=32, verbose=0)
        print("LSTM model trained successfully.")

        # Save the model
        lstm_model.save(LSTM_MODEL_PATH)
        print(f"LSTM model saved to {LSTM_MODEL_PATH}")

        # --- Evaluate the model on the test set ---
        # Create sequences for test data (note: need enough data points)
        # Handle cases where test_data might be too short for sequence_length
        if len(test_data) > sequence_length + 1:
            X_test, y_test = create_sequences(test_data, sequence_length)
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

            predictions_scaled = lstm_model.predict(X_test)
            predictions_original = scaler.inverse_transform(predictions_scaled)
            y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))

            rmse = np.sqrt(mean_squared_error(y_test_original, predictions_original))
            print(f"LSTM Test RMSE: {rmse:.4f}")

            # Plot results (optional, will show a popup plot)
            plt.figure(figsize=(12, 6))
            # Align test_dates correctly, skipping initial sequence_length points from test_data
            test_dates = df.index[train_size + sequence_length: train_size + sequence_length + len(y_test_original)]
            plt.plot(test_dates, y_test_original, label='Actual Prices')
            plt.plot(test_dates, predictions_original, label='LSTM Forecast')
            plt.title('LSTM Model Forecast vs Actual Prices (Test Set)')
            plt.xlabel('Date')
            plt.ylabel('Close Price')
            plt.legend()
            plt.grid(True)
            plt.show()
        else:
            print("Not enough test data points to create sequences for LSTM evaluation.")

    except Exception as e:
        print(f"Error in main LSTM script execution: {e}")