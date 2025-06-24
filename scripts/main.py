import os
import subprocess
import shutil
import sys # Import the sys module

def run_script(script_path):
    """Helper function to run a Python script using the current Python executable."""
    print(f"\n--- Running {script_path} ---")
    try:
        # Use sys.executable to get the path to the current Python interpreter
        # This will be the python.exe from your active virtual environment
        python_executable = sys.executable
        
        # Pass the full path to python.exe to subprocess.run
        # Use shell=True for windows if paths with spaces are causing issues, but it's generally less secure.
        # Direct execution with the full path should be robust.
        subprocess.run([python_executable, script_path], check=True, cwd=os.path.dirname(script_path))
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_path}: {e}")
        # exit(1) # Do not exit, allow subsequent scripts to run for full report
    except FileNotFoundError:
        print(f"Error: Script not found at {script_path}. Check path.")
    except Exception as e:
        print(f"An unexpected error occurred while running {script_path}: {e}")

def setup_directories():
    """Creates necessary directories."""
    # Get the directory of the main.py script
    script_dir = os.path.dirname(__file__)
    
    # Define paths relative to the script directory
    data_raw_dir = os.path.join(script_dir, "..", "data", "raw")
    data_processed_dir = os.path.join(script_dir, "..", "data", "processed")
    trained_models_dir = os.path.join(script_dir, "..", "trained_models")
    reports_dir = os.path.join(script_dir, "..", "reports")
    dashboards_dir = os.path.join(script_dir, "..", "dashboards")
    notebooks_dir = os.path.join(script_dir, "..", "notebooks")

    dirs = [data_raw_dir, data_processed_dir, trained_models_dir, reports_dir, dashboards_dir, notebooks_dir]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("Directories set up.")

def clean_up_generated_files():
    """Removes generated files and directories (optional, for clean runs)."""
    script_dir = os.path.dirname(__file__)
    data_raw_dir = os.path.join(script_dir, "..", "data", "raw")
    data_processed_dir = os.path.join(script_dir, "..", "data", "processed")
    trained_models_dir = os.path.join(script_dir, "..", "trained_models")

    if os.path.exists(data_raw_dir): shutil.rmtree(data_raw_dir)
    if os.path.exists(data_processed_dir): shutil.rmtree(data_processed_dir)
    if os.path.exists(trained_models_dir): shutil.rmtree(trained_models_dir)
    print("Cleaned up generated files.")

if __name__ == "__main__":
    # Optional: Clean up previous runs if you want a fresh start each time
    # This will delete existing downloaded data, processed data, and trained models.
    # Uncomment the line below if you want to enable this.
    # clean_up_generated_files()

    setup_directories()

    # Define script paths relative to the main.py script location
    script_dir = os.path.dirname(__file__)
    data_collection_script = os.path.join(script_dir, "data_collection.py")
    preprocessing_script = os.path.join(script_dir, "preprocessing.py")
    arima_script = os.path.join(script_dir, "arima_sarima_model.py")
    prophet_script = os.path.join(script_dir, "prophet_model.py")
    lstm_script = os.path.join(script_dir, "lstm_model.py")
    evaluation_script = os.path.join(script_dir, "evaluation.py")

    # Step 1: Data Collection
    run_script(data_collection_script)

    # Step 2: Preprocessing
    run_script(preprocessing_script)

    # Step 3: Model Training (individual scripts will save models)
    run_script(arima_script)
    run_script(prophet_script)
    run_script(lstm_script)

    # Step 4: Model Evaluation and Comparison
    run_script(evaluation_script)

    print("\nProject execution complete. Check 'reports' and 'dashboards' for outputs.")
    print("Look for generated plots from individual model scripts and evaluation.py.")