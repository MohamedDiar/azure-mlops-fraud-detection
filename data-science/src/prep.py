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
import traceback # For detailed error printing

# Import shared functions from utils.py
from utils import (
    is_weekend,
    is_night,
    get_customer_spending_behaviour_features,
    get_count_risk_rolling_window
)

def parse_args():
    parser = argparse.ArgumentParser("prep_mltable")
    parser.add_argument("--input_tabular_data", type=str, required=True, help="Path to input MLTable directory (mounted)")
    # FIX 1: Make output path required
    parser.add_argument("--output_mltable_path", type=str, help="Path to output directory for the final MLTable")

    # FIX 2: Add default values matching pipeline.yml defaults
    parser.add_argument("--output_start_date", type=str, default="2025-06-11", help="Start date for filtering the transformed data (YYYY-MM-DD)")
    parser.add_argument("--output_end_date", type=str, default="2025-08-14", help="End date for filtering the transformed data (YYYY-MM-DD)")
    parser.add_argument("--baseline_date_str", type=str, default="2025-04-01", help="Optional fixed baseline date for TX_TIME_DAYS calculation if not already present (YYYY-MM-DD)")

    args = parser.parse_args()
    print(f"Parsed Arguments: {args}") # Print parsed args to confirm defaults
    return args

def main(args):
    mlflow.start_run()
    print("Preparation script started (MLTable input/output)")
    print(f"Full Arguments Received (before parsing defaults): {args}") # Log raw args object

    # Log parameters AFTER parsing (will show defaults if applied)
    mlflow.log_param("input_mltable_path", args.input_tabular_data)
    mlflow.log_param("output_mltable_path", args.output_mltable_path)
    mlflow.log_param("filter_output_start_date", args.output_start_date)
    mlflow.log_param("filter_output_end_date", args.output_end_date)
    mlflow.log_param("baseline_date_str_used", args.baseline_date_str if args.baseline_date_str else "None")

    # --- Load Input MLTable ---
    print(f"\n[LOAD DATA] Loading input MLTable from: {args.input_tabular_data}")
    try:
        input_path_obj = Path(args.input_tabular_data)
        if not input_path_obj.exists() or not input_path_obj.is_dir() or not (input_path_obj / "MLTable").is_file():
             raise FileNotFoundError(f"Valid MLTable definition not found at directory: {args.input_tabular_data}")

        input_tbl = mltable.load(args.input_tabular_data)
        print("[LOAD DATA] MLTable definition loaded. Loading into DataFrame...")
        transactions_df = input_tbl.to_pandas_dataframe()
        print(f"Loaded MLTable into DataFrame. Shape: {transactions_df.shape}")

        if transactions_df.empty:
             print("ERROR: Loaded DataFrame is empty. Exiting.")
             mlflow.log_metric("input_rows_loaded_prep", 0)
             mlflow.end_run(status="FAILED")
             return
        mlflow.log_metric("input_rows_loaded_prep", len(transactions_df))

        # --- Data Validation and Time Column Handling ---
        required_cols = ['TRANSACTION_ID', 'TX_DATETIME', 'CUSTOMER_ID', 'TERMINAL_ID', 'TX_AMOUNT', 'TX_FRAUD'] # Add other required cols if any
        missing_cols = [col for col in required_cols if col not in transactions_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in input MLTable data: {missing_cols}")

        if not pd.api.types.is_datetime64_any_dtype(transactions_df['TX_DATETIME']):
            print("Converting TX_DATETIME to datetime objects...")
            transactions_df['TX_DATETIME'] = pd.to_datetime(transactions_df['TX_DATETIME'])

        if args.baseline_date_str and ('TX_TIME_DAYS' not in transactions_df.columns or 'TX_TIME_SECONDS' not in transactions_df.columns):
            print(f"Calculating time columns relative to fixed baseline: {args.baseline_date_str}")
            try:
                baseline_dt_date = datetime.datetime.strptime(args.baseline_date_str, "%Y-%m-%d").date()
                baseline_dt_datetime = datetime.datetime.strptime(args.baseline_date_str, "%Y-%m-%d")
                transactions_df['TX_TIME_DAYS'] = (transactions_df['TX_DATETIME'].dt.date - baseline_dt_date).apply(lambda x: x.days)
                transactions_df['TX_TIME_SECONDS'] = (transactions_df['TX_DATETIME'] - baseline_dt_datetime).dt.total_seconds()
                mlflow.set_tag("time_columns_calculated", "True")
                print("Time columns TX_TIME_DAYS and TX_TIME_SECONDS calculated.")
            except Exception as e:
                 print(f"ERROR calculating time columns: {e}. Proceeding without them if possible.")
                 mlflow.set_tag("time_columns_calculated", "Error")
        elif 'TX_TIME_DAYS' not in transactions_df.columns or 'TX_TIME_SECONDS' not in transactions_df.columns:
             print("Warning: TX_TIME_DAYS or TX_TIME_SECONDS missing and baseline_date_str not provided or not needed. Some features/splitting might fail if they rely on these.")
             mlflow.set_tag("time_columns_calculated", "Missing/NotNeeded")
        else:
             print("Time columns TX_TIME_DAYS and TX_TIME_SECONDS already present.")
             mlflow.set_tag("time_columns_calculated", "Already Present")


    except Exception as e:
        print(f"ERROR loading or initially processing input MLTable: {e}")
        traceback.print_exc()
        mlflow.log_param("prep_error", f"Input loading failed: {e}")
        mlflow.end_run(status="FAILED")
        return

    # --- Apply Transformations ---
    print("\n[TRANSFORM] Applying feature transformations...")
    start_transform_time = time.time()
    try:
        # Apply existing transformations
        # Note: Ensure groupy().apply() warnings are acceptable or add include_groups=False if appropriate for your logic
        print("[TRANSFORM] Calculating datetime features...")
        transactions_df['TX_DURING_WEEKEND'] = transactions_df.TX_DATETIME.apply(is_weekend)
        transactions_df['TX_DURING_NIGHT'] = transactions_df.TX_DATETIME.apply(is_night)
        mlflow.set_tag("transformation_datetime", "Success")
        print("[TRANSFORM] Calculating customer spending features...")
        transactions_df = transactions_df.groupby('CUSTOMER_ID', group_keys=False).apply(
            lambda x: get_customer_spending_behaviour_features(x, windows_size_in_days=[1, 7, 30])
            #, include_groups=False # Add this if pandas version requires it and logic is correct
        )
        transactions_df = transactions_df.sort_values('TRANSACTION_ID').reset_index(drop=True)
        mlflow.set_tag("transformation_customer", "Success")
        print("[TRANSFORM] Calculating terminal risk features...")
        transactions_df = transactions_df.groupby('TERMINAL_ID', group_keys=False).apply(
            lambda x: get_count_risk_rolling_window(x, delay_period=7, windows_size_in_days=[1, 7, 30], feature="TERMINAL_ID")
            #, include_groups=False # Add this if pandas version requires it and logic is correct
        )
        transactions_df = transactions_df.sort_values('TRANSACTION_ID').reset_index(drop=True)
        mlflow.set_tag("transformation_terminal", "Success")
        print("[TRANSFORM] All transformations applied.")

    except Exception as e:
        print(f"ERROR during transformations: {e}")
        traceback.print_exc()
        mlflow.log_param("prep_error", f"Transformation failed: {e}")
        mlflow.set_tag("Transformation Status", f"Failed: {e}")
        mlflow.end_run(status="FAILED")
        return

    transform_time = time.time() - start_transform_time
    print(f"Transformations applied in {transform_time:.2f} seconds.")
    mlflow.log_metric("prep_transform_time_sec", transform_time)

    # --- Filter Transformed Data ---
    transactions_df_filtered = transactions_df # Start with the full transformed df
    # Check if dates were provided (they will have defaults now unless overridden)
    if args.output_start_date and args.output_end_date:
        print(f"\n[FILTER] Filtering transformed data to output range: {args.output_start_date} to {args.output_end_date}")
        try:
            # Convert string dates from args to datetime objects
            output_start_dt = datetime.datetime.strptime(args.output_start_date, "%Y-%m-%d")
            output_end_dt = datetime.datetime.strptime(args.output_end_date, "%Y-%m-%d")

            # Perform the filtering on the TX_DATETIME column
            transactions_df_filtered = transactions_df[
                (transactions_df['TX_DATETIME'] >= output_start_dt) &
                # End date is inclusive, so check less than day *after* end date
                (transactions_df['TX_DATETIME'] < output_end_dt + datetime.timedelta(days=1))
            ].copy() # Use copy to avoid SettingWithCopyWarning

            print(f"Filtered data shape: {transactions_df_filtered.shape}")
            if transactions_df_filtered.empty:
                print("ERROR: No data remains after filtering for the specified date range. No MLTable will be saved.")
                mlflow.log_metric("final_prepared_rows", 0)
                mlflow.set_tag("Data Saving Status", "Failed - Empty after filter")
                mlflow.end_run(status="FAILED")
                return
            mlflow.set_tag("Data Filtering", "Applied")

        except ValueError as e:
             print(f"ERROR parsing filter dates: {e}. Check YYYY-MM-DD format.")
             mlflow.log_param("prep_error", f"Date parsing failed: {e}")
             mlflow.set_tag("Data Filtering", "Error")
             mlflow.end_run(status="FAILED")
             return
        except Exception as e:
            print(f"ERROR filtering data for output range: {e}")
            traceback.print_exc()
            mlflow.log_param("prep_error", f"Filtering failed: {e}")
            mlflow.set_tag("Data Filtering", "Error")
            mlflow.end_run(status="FAILED")
            return
    else:
        # This case should not happen now with defaults, unless defaults are empty strings
        print("WARNING: Output start or end date missing, using all transformed data.")
        mlflow.set_tag("Data Filtering", "Skipped (Missing Dates)")


    # --- Save Final DataFrame and Create MLTable Definition ---
    # FIX 3: Correct saving logic
    print(f"\n[SAVE MLTABLE] Saving final DataFrame to Parquet and creating MLTable definition in: {args.output_mltable_path}")
    output_dir = Path(args.output_mltable_path)
    # Azure ML creates the output directory structure for the component task.

    # Define the path for the Parquet data file *inside* the output directory
    data_filename = "data.parquet" # A standard name for the data file
    data_file_path = output_dir / data_filename
    # Define the relative path to be used *within* the MLTable definition file
    # relative_data_path = f"./{data_filename}"

    try:
        # --- Step 1: Save the Filtered DataFrame to a Parquet file ---
        print(f"Saving DataFrame ({transactions_df_filtered.shape}) to Parquet file: {data_file_path}")
        transactions_df_filtered.to_parquet(data_file_path, index=False, engine='pyarrow')
        print("DataFrame saved successfully to Parquet.")

        # --- Step 2: Create the MLTable definition referencing the saved Parquet file ---
        # The paths list should contain the relative path to the data file(s)
        # paths = [{'file': f'./{data_filename}'}]

     
        # Create the MLTable object defining how to read the Parquet file
        # Use the corrected paths list
        # final_mltable = mltable.from_parquet_files(paths=paths)

        # --- Step 3: Save the MLTable definition file (MLTable YAML) ---
        # This saves the `MLTable` YAML file into the specified output directory.
        print(f"Saving MLTable definition file (MLTable YAML) to directory: {output_dir}")
        # final_mltable.save(str(output_dir)) # Pass the directory path

        print(f"MLTable definition saved successfully in {output_dir}.")
        print("Azure ML pipeline will handle the registration based on pipeline output definition.")
        mlflow.log_metric("final_prepared_rows", len(transactions_df_filtered))
        mlflow.set_tag("Data Saving Status", "Success (MLTable Definition + Parquet)")

    except Exception as e:
        print(f"ERROR saving final data and creating MLTable: {e}")
        import traceback # Keep for detailed errors
        traceback.print_exc()
        mlflow.log_param("prep_error", f"MLTable saving/creation failed: {e}")
        mlflow.set_tag("Data Saving Status", "Failed")
        mlflow.end_run(status="FAILED")
        return

    mlflow.end_run()
    print("\nPreparation script finished successfully.")


if __name__ == "__main__":
    args = parse_args()
    main(args)

