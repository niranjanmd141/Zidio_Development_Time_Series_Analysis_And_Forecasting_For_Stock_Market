import yfinance as yf
import pandas as pd
import os

def download_stock_data(ticker, start_date, end_date, output_path):
    """
    Downloads historical stock data for a given ticker and saves it to a CSV.
    The CSV format from yfinance will be handled by the preprocessing script.
    """
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if not data.empty:
            # Save without index=True as the date is usually the index,
            # and our preprocessing script will handle the specific format.
            # However, if yfinance adds a header row for the index, that's fine too.
            data.to_csv(output_path, index=True) 
            print(f"Successfully downloaded data for {ticker} to {output_path}")
        else:
            print(f"No data found for {ticker} in the specified range. Check ticker or dates.")
            # Create an empty file to prevent FileNotFoundError in preprocessing
            with open(output_path, 'w') as f:
                f.write('') 
            return False
        return True
    except Exception as e:
        print(f"Error downloading data for {ticker}: {e}")
        # Create an empty file to prevent FileNotFoundError in preprocessing
        with open(output_path, 'w') as f:
            f.write('')
        return False

if __name__ == "__main__":
    TICKER = "AAPL" # Apple Inc. - You can change this ticker
    START_DATE = "2010-01-01"
    END_DATE = "2025-06-14" # Adjust to current date minus a day or so, or future date for test
    
    # Ensure relative paths work correctly from the main.py context
    # Get the directory of the current script, then navigate to data/raw
    script_dir = os.path.dirname(__file__)
    RAW_DATA_DIR = os.path.join(script_dir, "..", "data", "raw")
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    OUTPUT_FILE = os.path.join(RAW_DATA_DIR, f"{TICKER}_historical_data.csv")

    download_stock_data(TICKER, START_DATE, END_DATE, OUTPUT_FILE)