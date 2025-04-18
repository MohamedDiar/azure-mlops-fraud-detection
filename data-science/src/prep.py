# data-science/src/prep.py
import os
import argparse
import datetime 
from time import time
from pathlib import Path
import pandas as pd
import numpy as np
import mlflow

# Import shared functions from utils.py
from utils import (
    is_weekend,
    is_night,
    get_customer_spending_behaviour_features,
    get_count_risk_rolling_window
)

# Define maximum lookback needed for feature calculation (30 days + 7 day delay)
MAX_LOOKBACK_DAYS = 37

def parse_args():
    parser = argparse.ArgumentParser("prep")
    parser.add_argument("--raw_data", type=str, help="Path to raw data directory (contains daily .pkl files)")
    parser.add_argument("--transformed_data", type=str, help="Path to output directory for transformed data (.pkl files)")

    # Date range for the DESIRED TRANSFORMED OUTPUT
    parser.add_argument("--output_start_date", type=str, help="Start date for the transformed data output (YYYY-MM-DD)")
    parser.add_argument("--output_end_date", type=str, help="End date for the transformed data output (YYYY-MM-DD)")

    args = parser.parse_args()
    return args

def read_raw_files(DIR_INPUT, BEGIN_DATE_STR, END_DATE_STR):
    """Reads raw pickle files within a calculated date range, handling lookback."""
    try:
        # Calculate the actual raw data start date needed including lookback
        output_start_dt = datetime.datetime.strptime(BEGIN_DATE_STR, "%Y-%m-%d")
        raw_load_start_dt = output_start_dt - datetime.timedelta(days=MAX_LOOKBACK_DAYS)
        raw_load_start_str = raw_load_start_dt.strftime("%Y-%m-%d")
        raw_load_end_str = END_DATE_STR # Load up to the desired output end date

        print(f"Prep: Required transformed output range: {BEGIN_DATE_STR} to {END_DATE_STR}")
        print(f"Prep: Calculated raw data load range (including lookback): {raw_load_start_str} to {raw_load_end_str}")

        files = [os.path.join(DIR_INPUT, f) for f in os.listdir(DIR_INPUT) if f.endswith('.pkl') and raw_load_start_str + '.pkl' <= f <= raw_load_end_str + '.pkl']
        files.sort()
    except FileNotFoundError:
         print(f"ERROR: Raw data input directory not found: {DIR_INPUT}")
         return pd.DataFrame()
    except ValueError as e:
        print(f"ERROR: Invalid date format provided for output range: {e}")
        return pd.DataFrame()

    if not files:
        print(f"WARNING: No raw '.pkl' files found in {DIR_INPUT} for calculated range {raw_load_start_str} to {raw_load_end_str}")
        return pd.DataFrame()

    print(f"Found {len(files)} raw files to load for transformation.")
    frames = []
    required_cols = ['TRANSACTION_ID', 'TX_DATETIME', 'CUSTOMER_ID', 'TERMINAL_ID', 'TX_AMOUNT', 'TX_FRAUD']
    for f_path in files:
        try:
            df = pd.read_pickle(f_path)
            if not all(col in df.columns for col in required_cols):
                 print(f"Warning: Required columns missing in {Path(f_path).name}. Skipping file.")
                 continue
            frames.append(df)
        except Exception as e:
            print(f"Error reading raw file {Path(f_path).name}: {e}")

    if not frames:
        print("No raw dataframes were successfully loaded.")
        return pd.DataFrame()

    df_final = pd.concat(frames, ignore_index=True)
    df_final = df_final.sort_values('TRANSACTION_ID')
    df_final.reset_index(drop=True, inplace=True)

    # Convert TX_DATETIME and add time columns
    if 'TX_DATETIME' in df_final.columns and not pd.api.types.is_datetime64_any_dtype(df_final['TX_DATETIME']):
        try:
            df_final['TX_DATETIME'] = pd.to_datetime(df_final['TX_DATETIME'])
        except Exception as e:
            print(f"Warning: Could not convert TX_DATETIME to datetime in raw data: {e}. This may cause errors.")
            df_final['TX_TIME_SECONDS'] = np.nan
            df_final['TX_TIME_DAYS'] = np.nan
            return df_final # Return early if critical conversion fails

    if 'TX_DATETIME' in df_final.columns and pd.api.types.is_datetime64_any_dtype(df_final['TX_DATETIME']):
         # IMPORTANT: Calculate days relative to the *true minimum* date in the loaded raw data
         # to ensure rolling windows work correctly across the entire loaded span.
         min_raw_date = df_final['TX_DATETIME'].dt.date.min()
         if min_raw_date:
             df_final['TX_TIME_SECONDS'] = (df_final['TX_DATETIME'] - df_final['TX_DATETIME'].min()).dt.total_seconds()
             df_final['TX_TIME_DAYS'] = (df_final['TX_DATETIME'].dt.date - min_raw_date).apply(lambda x: x.days)
             print(f"Generated 'TX_TIME_SECONDS' and 'TX_TIME_DAYS' relative to raw data start: {min_raw_date.strftime('%Y-%m-%d')}")
         else:
              print("Warning: Could not determine minimum date from loaded raw data. Cannot generate time columns.")
              df_final['TX_TIME_SECONDS'] = np.nan
              df_final['TX_TIME_DAYS'] = np.nan
    else:
         print("Warning: Cannot generate time-based columns ('TX_TIME_SECONDS', 'TX_TIME_DAYS') as TX_DATETIME is missing or not datetime type.")
         df_final['TX_TIME_SECONDS'] = np.nan
         df_final['TX_TIME_DAYS'] = np.nan

    return df_final


def main(args):
    mlflow.start_run() # Start MLflow run for this component
    print("Preparation script started")
    print(f"Args: {args}")

    # Log parameters
    mlflow.log_param("raw_data_input_path", args.raw_data)
    mlflow.log_param("transformed_data_output_path", args.transformed_data)
    mlflow.log_param("output_start_date", args.output_start_date)
    mlflow.log_param("output_end_date", args.output_end_date)
    mlflow.log_param("max_lookback_days", MAX_LOOKBACK_DAYS)

    # --- Load Raw Data (Handles lookback inside the function) ---
    print(f"Loading raw data from: {args.raw_data} to cover output range {args.output_start_date} to {args.output_end_date} with lookback")
    transactions_df = read_raw_files(args.raw_data, args.output_start_date, args.output_end_date)

    if transactions_df.empty:
        print("ERROR: No raw data loaded after handling lookback. Exiting.")
        mlflow.log_metric("raw_rows_loaded_prep", 0)
        mlflow.end_run(status="FAILED")
        return

    print(f"Loaded {len(transactions_df)} raw transactions for processing.")
    mlflow.log_metric("raw_rows_loaded_prep", len(transactions_df))

    # --- Apply Transformations ---
    # Important: Transformations run on the *entire* loaded df (including lookback period)
    # to ensure correct rolling calculations for the *target* output dates.

    print("Applying feature transformations (this may take time)...")
    start_transform_time = time.time()

    # 1. Date/Time Transformations
    if 'TX_DATETIME' in transactions_df.columns:
        transactions_df['TX_DURING_WEEKEND'] = transactions_df.TX_DATETIME.apply(is_weekend)
        transactions_df['TX_DURING_NIGHT'] = transactions_df.TX_DATETIME.apply(is_night)
        mlflow.set_tag("transformation_datetime", "Success")
    else: mlflow.set_tag("transformation_datetime", "Skipped - Missing Column")

    # 2. Customer Spending Behavior Transformations
    if 'CUSTOMER_ID' in transactions_df.columns and 'TX_DATETIME' in transactions_df.columns:
        try:
            # Apply transformation per group
            transactions_df = transactions_df.groupby('CUSTOMER_ID', group_keys=False).apply(
                lambda x: get_customer_spending_behaviour_features(x, windows_size_in_days=[1, 7, 30])
            )
            # Groupby+apply can mess up index/order, ensure sorting
            transactions_df = transactions_df.sort_values('TRANSACTION_ID').reset_index(drop=True)
            mlflow.set_tag("transformation_customer", "Success")
        except Exception as e:
            print(f"  ERROR during customer spending transformations: {e}")
            mlflow.set_tag("transformation_customer", f"Failed: {e}")
    else: mlflow.set_tag("transformation_customer", "Skipped - Missing Column")

    # 3. Terminal Risk Transformations
    if 'TERMINAL_ID' in transactions_df.columns and 'TX_DATETIME' in transactions_df.columns and 'TX_FRAUD' in transactions_df.columns:
        try:
            # Apply transformation per group
            transactions_df = transactions_df.groupby('TERMINAL_ID', group_keys=False).apply(
                lambda x: get_count_risk_rolling_window(x, delay_period=7, windows_size_in_days=[1, 7, 30], feature="TERMINAL_ID")
            )
            # Groupby+apply can mess up index/order, ensure sorting
            transactions_df = transactions_df.sort_values('TRANSACTION_ID').reset_index(drop=True)
            mlflow.set_tag("transformation_terminal", "Success")
        except Exception as e:
            print(f"  ERROR during terminal risk transformations: {e}")
            mlflow.set_tag("transformation_terminal", f"Failed: {e}")
    else: mlflow.set_tag("transformation_terminal", "Skipped - Missing Column")

    transform_time = time.time() - start_transform_time
    print(f"Transformations applied in {transform_time:.2f} seconds.")
    mlflow.log_metric("prep_transform_time_sec", transform_time)

    # --- Filter and Save Transformed Data ---
    # Now, filter the dataframe to keep only the desired output date range before saving daily files.
    print(f"Filtering transformed data to output range: {args.output_start_date} to {args.output_end_date}")
    try:
        output_start_dt = datetime.datetime.strptime(args.output_start_date, "%Y-%m-%d")
        output_end_dt = datetime.datetime.strptime(args.output_end_date, "%Y-%m-%d")

        # Filter based on TX_DATETIME
        transactions_df_filtered = transactions_df[
            (transactions_df['TX_DATETIME'] >= output_start_dt) &
            (transactions_df['TX_DATETIME'] < output_end_dt + datetime.timedelta(days=1)) # Make end date inclusive
        ].copy()

        print(f"Filtered data shape: {transactions_df_filtered.shape}")
        if transactions_df_filtered.empty:
            print("ERROR: No data remains after filtering for the desired output date range. No files will be saved.")
            mlflow.log_metric("transformed_files_saved", 0)
            mlflow.log_metric("transformed_rows_saved", 0)
            mlflow.set_tag("Data Saving Status", "Failed - Empty after filter")
            mlflow.end_run(status="FAILED")
            return

    except Exception as e:
        print(f"ERROR filtering data for output range: {e}")
        mlflow.set_tag("Data Saving Status", "Failed - Filtering error")
        mlflow.end_run(status="FAILED")
        return

    # --- Save Filtered Data to Daily Files ---
    print(f"Saving filtered transformed data to: {args.transformed_data}")
    output_path = Path(args.transformed_data)
    output_path.mkdir(parents=True, exist_ok=True)

    # Need TX_TIME_DAYS relative to the output start date for saving correctly named files
    # Re-calculate TX_TIME_DAYS based on the filtered data's min date (should be output_start_date)
    min_output_date = transactions_df_filtered['TX_DATETIME'].dt.date.min()
    if min_output_date:
        transactions_df_filtered['OUTPUT_TX_TIME_DAYS'] = (transactions_df_filtered['TX_DATETIME'].dt.date - min_output_date).apply(lambda x: x.days)
        print(f"Generated 'OUTPUT_TX_TIME_DAYS' relative to output start: {min_output_date.strftime('%Y-%m-%d')}")

        start_save_time = time.time()
        max_output_days = transactions_df_filtered.OUTPUT_TX_TIME_DAYS.max()
        days_processed = 0
        total_rows_saved = 0

        # Use the newly calculated relative days for saving loop
        for day_idx in range(max_output_days + 1):
            transactions_day = transactions_df_filtered[transactions_df_filtered.OUTPUT_TX_TIME_DAYS == day_idx].sort_values('TX_TIME_SECONDS')

            if not transactions_day.empty:
                actual_date = min_output_date + datetime.timedelta(days=day_idx)
                filename_output = actual_date.strftime("%Y-%m-%d") + '.pkl'
                file_path = output_path / filename_output
                try:
                    # Drop the temporary day column before saving
                    transactions_day_to_save = transactions_day.drop(columns=['OUTPUT_TX_TIME_DAYS'])
                    transactions_day_to_save.to_pickle(str(file_path), protocol=4)
                    days_processed += 1
                    total_rows_saved += len(transactions_day_to_save)
                except Exception as e:
                    print(f"  Error saving file {filename_output}: {e}")

        save_time = time.time() - start_save_time
        print(f"Saved transformed data for {days_processed} days ({total_rows_saved} rows) in {args.transformed_data} ({save_time:.2f} seconds)")
        mlflow.log_metric("transformed_files_saved", days_processed)
        mlflow.log_metric("transformed_rows_saved", total_rows_saved)
        mlflow.log_metric("prep_save_time_sec", save_time)
        mlflow.set_tag("Data Saving Status", "Success")
    else:
         print("ERROR: Cannot save daily files because min output date could not be determined.")
         mlflow.set_tag("Data Saving Status", "Failed - Min Date Error")
         mlflow.end_run(status="FAILED")
         return

    mlflow.end_run()
    print("Preparation script finished.")


if __name__ == "__main__":
    args = parse_args()
    main(args)