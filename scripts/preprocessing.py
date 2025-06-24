# In scripts/preprocessing.py

import pandas as pd
import numpy as np
import os

RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "AAPL_historical_data.csv")
PROCESSED_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "AAPL_preprocessed_data.csv")

def preprocess_data(input_path, output_path):
    """
    Loads raw stock data, handles missing values, ensures a consistent business day frequency,
    and prepares it for time series analysis. This version handles the unusual multiple
    header rows (if present) and ensures the 'Close' column is numeric.
    """
    print(f"\n--- Running scripts/preprocessing.py ---")
    print(f"Preprocessing data from {input_path}")
    if not os.path.exists(input_path):
        print(f"Raw data file not found for preprocessing: {input_path}")
        return None

    df = None
    try:
        # Try reading with default yfinance CSV structure first
        df = pd.read_csv(input_path, index_col='Date', parse_dates=True)
        print("Successfully read CSV with default yfinance structure.")
    except Exception as e:
        print(f"Attempting alternative read method due to: {e}")
        try:
            # Fallback for the unusual CSV format with multiple header rows (as per your snippet)
            # Assuming Date is in the first column (index 0) *after* skipping rows.
            # No header means pandas will assign numeric columns (0, 1, 2, 3...)
            df = pd.read_csv(input_path, skiprows=3, header=None, index_col=0, parse_dates=True)
            print("Successfully read CSV with skiprows=3 and no header.")
            
            # After skiprows and header=None, columns are typically numeric indices.
            # We need to map these to meaningful names.
            # Assuming the order: Date (index), Open, High, Low, Close, Adj Close, Volume
            # So, Close would be at column index 4 (since 0 is Date, 1 Open, 2 High, 3 Low, 4 Close)
            
            # Check if column 4 exists (which would be the 'Close' column after skipping header)
            if 4 in df.columns:
                df.rename(columns={4: 'Close'}, inplace=True)
                # You might want to rename other columns too for clarity, but 'Close' is essential
                # col_map_after_skip = {1: 'Open', 2: 'High', 3: 'Low', 5: 'Adj Close', 6: 'Volume'}
                # for old, new in col_map_after_skip.items():
                #     if old in df.columns:
                #         df.rename(columns={old: new}, inplace=True)
            else:
                raise ValueError("Could not find expected 'Close' column (index 4) after skipping 3 rows and no header.")

            # Final check to ensure 'Close' column exists after all attempts
            if 'Close' not in df.columns:
                raise ValueError("Could not find or reliably identify 'Close' column after reading.")

        except Exception as inner_e:
            print(f"Failed to read CSV with alternative method: {inner_e}")
            return None

    if df is None or df.empty:
        print("DataFrame is empty after reading or preprocessing failed.")
        return None
        
    # Ensure index is datetime (already tried with parse_dates=True, but re-confirm for safety)
    df.index = pd.to_datetime(df.index)

    # --- Robust conversion of 'Close' column to numeric ---
    if 'Close' in df.columns:
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    else:
        print("Error: 'Close' column not found in the DataFrame after initial parsing.")
        return None

    # Handle missing numerical values using ffill then bfill
    # Updated to avoid FutureWarning on inplace=True and Series.fillna with method
    df['Close'] = df['Close'].ffill() 
    df['Close'] = df['Close'].bfill()
    
    # If there are still NaNs after bfill (e.g., completely empty column)
    if df['Close'].isnull().any():
        print("Warning: NaNs still present in 'Close' column after ffill/bfill. Filling with mean/0.")
        df['Close'] = df['Close'].fillna(df['Close'].mean())
        df['Close'] = df['Close'].fillna(0) # Final fallback for all NaNs case

    # Select relevant columns for forecasting (e.g., 'Close' price)
    df_processed = df[['Close']].copy()

    # --- IMPORTANT: Reindex to a Business Day frequency and fill any new NaNs ---
    print(f"Original date range: {df_processed.index.min()} to {df_processed.index.max()}")
    print(f"Original number of entries: {len(df_processed)}")

    full_date_range = pd.date_range(start=df_processed.index.min(), end=df_processed.index.max(), freq='B')
    df_processed = df_processed.reindex(full_date_range)
    
    # After reindexing, there will be new NaNs for days that weren't in the original data (e.g., holidays).
    # We need to fill these. Forward-fill, then back-fill for robustness.
    print(f"Number of entries after reindexing to Business Day frequency: {len(df_processed)}")
    print(f"NaNs in 'Close' before post-reindex fill: {df_processed['Close'].isnull().sum()}")

    # Updated to avoid FutureWarning on inplace=True and Series.fillna with method
    df_processed['Close'] = df_processed['Close'].ffill()
    df_processed['Close'] = df_processed['Close'].bfill()
    
    # Final check for any stubborn NaNs (should not happen if data is reasonable)
    if df_processed['Close'].isnull().any():
        print("CRITICAL WARNING: NaNs still present after reindexing and all fillna attempts. Investigating data is recommended.")
        df_processed['Close'] = df_processed['Close'].fillna(df_processed['Close'].mean())
        df_processed['Close'] = df_processed['Close'].fillna(0)

    # Save the processed data
    # Save with header=False and index=True, as read by other scripts
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_processed.to_csv(output_path, header=False, index=True) 
    print(f"Successfully preprocessed data and saved to {output_path}")
    return df_processed

if __name__ == "__main__":
    preprocess_data(RAW_DATA_PATH, PROCESSED_DATA_PATH)