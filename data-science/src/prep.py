# data-science/src/prep.py
import os
import argparse
import datetime
import time
from pathlib import Path
import pandas as pd
import numpy as np
import mlflow
import mltable # Import mltable

# Import shared functions from utils.py
from utils import (
    is_weekend,
    is_night,
    get_customer_spending_behaviour_features,
    get_count_risk_rolling_window
)

# REMOVED: MAX_LOOKBACK_DAYS - Lookback should be handled when creating the input MLTable

def parse_args():
    parser = argparse.ArgumentParser("prep_mltable") # Changed name slightly
    # UPDATED: Input is now the path to the mounted MLTable definition
    parser.add_argument("--input_tabular_data", type=str, help="Path to input MLTable directory (mounted)")
    # UPDATED: Output is the path where AML expects the output MLTable definition to be saved
    parser.add_argument("--output_mltable_path", type=str, help="Path to output directory for the final MLTable")

    # Keep date args if needed for filtering the loaded table
    parser.add_argument("--output_start_date", type=str, help="Start date for filtering the transformed data (YYYY-MM-DD)")
    parser.add_argument("--output_end_date", type=str, help="End date for filtering the transformed data (YYYY-MM-DD)")
    parser.add_argument("--baseline_date_str", type=str, required=False, help="Optional fixed baseline date for TX_TIME_DAYS calculation if not already present (YYYY-MM-DD)")

    args = parser.parse_args()
    return args

# REMOVED: read_raw_files function is no longer needed

def main(args):
    mlflow.start_run()
    print("Preparation script started (MLTable input/output)")
    print(f"Args: {args}")

    # Log parameters
    mlflow.log_param("input_mltable_path", args.input_tabular_data)
    mlflow.log_param("output_mltable_path", args.output_mltable_path)
    mlflow.log_param("filter_output_start_date", args.output_start_date)
    mlflow.log_param("filter_output_end_date", args.output_end_date)
    mlflow.log_param("baseline_date_str_provided", args.baseline_date_str if args.baseline_date_str else "None")

    # --- Load Input MLTable ---
    print(f"Loading input MLTable from: {args.input_tabular_data}")
    try:
        # Load the MLTable definition file (e.g., MLTable) inside the mounted path
        input_tbl = mltable.load(args.input_tabular_data)
        # Load the data into a pandas DataFrame
        transactions_df = input_tbl.to_pandas_dataframe()
        print(f"Loaded MLTable into DataFrame. Shape: {transactions_df.shape}")
        if transactions_df.empty:
             print("ERROR: Loaded DataFrame is empty. Exiting.")
             mlflow.log_metric("input_rows_loaded_prep", 0)
             mlflow.end_run(status="FAILED")
             return
        mlflow.log_metric("input_rows_loaded_prep", len(transactions_df))

        # --- Data Validation and Time Column Handling ---
        required_cols = ['TRANSACTION_ID', 'TX_DATETIME', 'CUSTOMER_ID', 'TERMINAL_ID', 'TX_AMOUNT', 'TX_FRAUD']
        missing_cols = [col for col in required_cols if col not in transactions_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in input MLTable data: {missing_cols}")

        # Ensure TX_DATETIME is datetime type
        if not pd.api.types.is_datetime64_any_dtype(transactions_df['TX_DATETIME']):
            print("Converting TX_DATETIME to datetime objects...")
            transactions_df['TX_DATETIME'] = pd.to_datetime(transactions_df['TX_DATETIME'])

        # Calculate TX_TIME_DAYS/SECONDS if baseline provided AND they don't exist
        if args.baseline_date_str and ('TX_TIME_DAYS' not in transactions_df.columns or 'TX_TIME_SECONDS' not in transactions_df.columns):
            print(f"Calculating time columns relative to fixed baseline: {args.baseline_date_str}")
            try:
                baseline_dt_date = datetime.datetime.strptime(args.baseline_date_str, "%Y-%m-%d").date()
                baseline_dt_datetime = datetime.datetime.strptime(args.baseline_date_str, "%Y-%m-%d")
                transactions_df['TX_TIME_DAYS'] = (transactions_df['TX_DATETIME'].dt.date - baseline_dt_date).apply(lambda x: x.days)
                transactions_df['TX_TIME_SECONDS'] = (transactions_df['TX_DATETIME'] - baseline_dt_datetime).dt.total_seconds()
                mlflow.set_tag("time_columns_calculated", "True")
            except Exception as e:
                 print(f"ERROR calculating time columns: {e}. Proceeding without them if possible.")
                 mlflow.set_tag("time_columns_calculated", "Error")
        elif 'TX_TIME_DAYS' not in transactions_df.columns or 'TX_TIME_SECONDS' not in transactions_df.columns:
             print("Warning: TX_TIME_DAYS or TX_TIME_SECONDS missing and baseline_date_str not provided. Some features/splitting might fail.")
             mlflow.set_tag("time_columns_calculated", "Missing")
        else:
             mlflow.set_tag("time_columns_calculated", "Already Present")


    except Exception as e:
        print(f"ERROR loading or initially processing input MLTable: {e}")
        import traceback; traceback.print_exc()
        mlflow.log_param("prep_error", f"Input loading failed: {e}")
        mlflow.end_run(status="FAILED")
        return

    # --- Apply Transformations ---
    # Transformations run on the entire loaded DataFrame.
    # Lookback is inherently handled by the data included in the input MLTable.
    print("Applying feature transformations (this may take time)...")
    start_transform_time = time.time()

    # Apply existing transformations (assuming they work on the DataFrame columns)
    try:
        transactions_df['TX_DURING_WEEKEND'] = transactions_df.TX_DATETIME.apply(is_weekend)
        transactions_df['TX_DURING_NIGHT'] = transactions_df.TX_DATETIME.apply(is_night)
        mlflow.set_tag("transformation_datetime", "Success")

        transactions_df = transactions_df.groupby('CUSTOMER_ID', group_keys=False).apply(
            lambda x: get_customer_spending_behaviour_features(x, windows_size_in_days=[1, 7, 30])
        )
        transactions_df = transactions_df.sort_values('TRANSACTION_ID').reset_index(drop=True) # Ensure order
        mlflow.set_tag("transformation_customer", "Success")

        transactions_df = transactions_df.groupby('TERMINAL_ID', group_keys=False).apply(
            lambda x: get_count_risk_rolling_window(x, delay_period=7, windows_size_in_days=[1, 7, 30], feature="TERMINAL_ID")
        )
        transactions_df = transactions_df.sort_values('TRANSACTION_ID').reset_index(drop=True) # Ensure order
        mlflow.set_tag("transformation_terminal", "Success")

    except Exception as e:
        print(f"ERROR during transformations: {e}")
        import traceback; traceback.print_exc()
        mlflow.log_param("prep_error", f"Transformation failed: {e}")
        mlflow.set_tag("Transformation Status", f"Failed: {e}")
        mlflow.end_run(status="FAILED")
        return

    transform_time = time.time() - start_transform_time
    print(f"Transformations applied in {transform_time:.2f} seconds.")
    mlflow.log_metric("prep_transform_time_sec", transform_time)

    # --- Filter Transformed Data (Optional based on dates) ---
    transactions_df_filtered = transactions_df # Start with the full transformed df
    if args.output_start_date and args.output_end_date:
        print(f"Filtering transformed data to output range: {args.output_start_date} to {args.output_end_date}")
        try:
            output_start_dt = datetime.datetime.strptime(args.output_start_date, "%Y-%m-%d")
            output_end_dt = datetime.datetime.strptime(args.output_end_date, "%Y-%m-%d")

            transactions_df_filtered = transactions_df[
                (transactions_df['TX_DATETIME'] >= output_start_dt) &
                (transactions_df['TX_DATETIME'] < output_end_dt + datetime.timedelta(days=1)) # Make end date inclusive
            ].copy()

            print(f"Filtered data shape: {transactions_df_filtered.shape}")
            if transactions_df_filtered.empty:
                print("ERROR: No data remains after filtering. No MLTable will be saved.")
                mlflow.log_metric("final_prepared_rows", 0)
                mlflow.set_tag("Data Saving Status", "Failed - Empty after filter")
                mlflow.end_run(status="FAILED")
                return
            mlflow.set_tag("Data Filtering", "Applied")

        except Exception as e:
            print(f"ERROR filtering data for output range: {e}")
            mlflow.log_param("prep_error", f"Filtering failed: {e}")
            mlflow.set_tag("Data Filtering", "Error")
            mlflow.end_run(status="FAILED")
            return
    else:
        print("No output date range specified, using all transformed data.")
        mlflow.set_tag("Data Filtering", "Skipped")


        # --- Save Final DataFrame and Create MLTable Definition ---
    print(f"Saving final DataFrame to Parquet and creating MLTable definition in: {args.output_mltable_path}")
    output_dir = Path(args.output_mltable_path)
    # Azure ML creates the output directory structure for you
    # output_dir.mkdir(parents=True, exist_ok=True) # No need to explicitly create

    # Define the path for the data file *inside* the output directory
    # Using a simple name like 'data.parquet' is common
    data_file_path = output_dir / "data.parquet"
    # Define the relative path used *within* the MLTable definition file
    relative_data_path = "./data.parquet"

    try:
        # --- Step 1: Save the DataFrame to a Parquet file ---
        print(f"Saving DataFrame to Parquet file: {data_file_path}")
        transactions_df_filtered.to_parquet(data_file_path, index=False, engine='pyarrow')
        print("DataFrame saved successfully.")

        # --- Step 2: Create the MLTable definition referencing the saved file ---
        # Define the paths for the MLTable definition. It should point to the
        # data file relative to the MLTable file location (which is output_dir).
        paths = [{'file': relative_data_path}]

        # Create the MLTable object from the saved Parquet file path
        final_mltable = mltable.from_parquet_files(paths=paths)

        # --- Step 3: Save the MLTable definition file (MLTable YAML) ---
        # Save the MLTable file itself into the output directory.
        # mltable_obj.save() saves the definition file to the specified folder.
        print(f"Saving MLTable definition file to directory: {output_dir}")
        final_mltable.save(str(output_dir)) # Pass the directory path

        print(f"MLTable definition saved successfully in {output_dir}.")
        print("Azure ML pipeline will handle the registration based on pipeline output definition.")
        mlflow.log_metric("final_prepared_rows", len(transactions_df_filtered))
        mlflow.set_tag("Data Saving Status", "Success (MLTable Definition + Parquet)")

    except Exception as e:
        print(f"ERROR saving final data and creating MLTable: {e}")
        import traceback; traceback.print_exc()
        mlflow.log_param("prep_error", f"MLTable saving/creation failed: {e}")
        mlflow.set_tag("Data Saving Status", "Failed")
        mlflow.end_run(status="FAILED")
        return

    mlflow.end_run()
    print("Preparation script finished successfully.")


if __name__ == "__main__":
    args = parse_args()
    main(args)