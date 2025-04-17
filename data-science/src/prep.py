# data-science/src/prep.py
import os
import argparse
import datetime
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

def parse_args():
    parser = argparse.ArgumentParser("prep")
    parser.add_argument("--raw_data", type=str, help="Path to raw data directory (contains daily .pkl files)")
    parser.add_argument("--transformed_data", type=str, help="Path to output directory for transformed data (.pkl files)")

    # Date range for processing raw data
    parser.add_argument("--start_date", type=str, help="Start date for processing raw data (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, help="End date for processing raw data (YYYY-MM-DD)")

    args = parser.parse_args()
    return args

def read_raw_files(DIR_INPUT, BEGIN_DATE, END_DATE):
    """Reads raw pickle files within a date range."""
    try:
        files = [os.path.join(DIR_INPUT, f) for f in os.listdir(DIR_INPUT) if f.endswith('.pkl') and BEGIN_DATE+'.pkl' <= f <= END_DATE+'.pkl']
        files.sort()
    except FileNotFoundError:
         print(f"ERROR: Input directory not found: {DIR_INPUT}")
         return pd.DataFrame()

    if not files:
        print(f"WARNING: No '.pkl' files found in {DIR_INPUT} for date range {BEGIN_DATE} to {END_DATE}")
        return pd.DataFrame()

    print(f"Found {len(files)} raw files to load.")
    frames = []
    for f in files:
        try:
            df = pd.read_pickle(f)
            # Basic validation
            required_cols = ['TRANSACTION_ID', 'TX_DATETIME', 'CUSTOMER_ID', 'TERMINAL_ID', 'TX_AMOUNT', 'TX_FRAUD']
            if not all(col in df.columns for col in required_cols):
                 print(f"Warning: Required columns missing in {f}. Skipping file.")
                 continue
            frames.append(df)
        except Exception as e:
            print(f"Error reading raw file {f}: {e}")

    if not frames:
        print("No raw dataframes were successfully loaded.")
        return pd.DataFrame()

    df_final = pd.concat(frames, ignore_index=True)
    df_final = df_final.sort_values('TRANSACTION_ID')
    df_final.reset_index(drop=True, inplace=True)
    # Convert TX_DATETIME to datetime objects if not already
    if 'TX_DATETIME' in df_final.columns and not pd.api.types.is_datetime64_any_dtype(df_final['TX_DATETIME']):
        try:
            df_final['TX_DATETIME'] = pd.to_datetime(df_final['TX_DATETIME'])
        except Exception as e:
            print(f"Warning: Could not convert TX_DATETIME to datetime in raw data: {e}")
            # Consider raising an error if this conversion is critical

    # Add TX_TIME_SECONDS and TX_TIME_DAYS if not present (needed for transformations)
    if 'TX_DATETIME' in df_final.columns and pd.api.types.is_datetime64_any_dtype(df_final['TX_DATETIME']):
         df_final['TX_TIME_SECONDS'] = (df_final['TX_DATETIME'] - df_final['TX_DATETIME'].min()).dt.total_seconds()
         df_final['TX_TIME_DAYS'] = (df_final['TX_DATETIME'].dt.date - df_final['TX_DATETIME'].dt.date.min()).apply(lambda x: x.days)
         print("Generated 'TX_TIME_SECONDS' and 'TX_TIME_DAYS'.")
    else:
         print("Warning: Cannot generate time-based columns ('TX_TIME_SECONDS', 'TX_TIME_DAYS') as TX_DATETIME is missing or not datetime type.")
         # Raise error if these columns are strictly needed downstream
         # raise ValueError("Missing required TX_DATETIME column or incorrect type for time calculations.")

    return df_final


def main(args):
    mlflow.start_run() # Start MLflow run for this component
    print("Preparation script started")
    print(f"Args: {args}")

    # --- Load Raw Data ---
    print(f"Loading raw data from: {args.raw_data} between {args.start_date} and {args.end_date}")
    transactions_df = read_raw_files(args.raw_data, args.start_date, args.end_date)

    if transactions_df.empty:
        print("ERROR: No raw data loaded. Exiting.")
        mlflow.log_metric("raw_rows_loaded", 0)
        mlflow.end_run(status="FAILED")
        return

    print(f"Loaded {len(transactions_df)} raw transactions.")
    mlflow.log_metric("raw_rows_loaded", len(transactions_df))
    mlflow.log_param("raw_data_start_date", args.start_date)
    mlflow.log_param("raw_data_end_date", args.end_date)

    # --- Apply Transformations (from BaselineFeatureTransformation notebook) ---

    # 1. Date/Time Transformations
    print("Applying date/time transformations...")
    if 'TX_DATETIME' in transactions_df.columns:
        transactions_df['TX_DURING_WEEKEND'] = transactions_df.TX_DATETIME.apply(is_weekend)
        transactions_df['TX_DURING_NIGHT'] = transactions_df.TX_DATETIME.apply(is_night)
        print("  Applied TX_DURING_WEEKEND and TX_DURING_NIGHT.")
        mlflow.set_tag("transformation_datetime", "Success")
    else:
        print("  Skipping date/time transformations: TX_DATETIME column missing.")
        mlflow.set_tag("transformation_datetime", "Skipped - Missing Column")


    # 2. Customer Spending Behavior Transformations
    print("Applying customer spending behavior transformations...")
    if 'CUSTOMER_ID' in transactions_df.columns and 'TX_DATETIME' in transactions_df.columns:
        try:
            # Use apply with a lambda function to call the transformation function per group
            transactions_df = transactions_df.groupby('CUSTOMER_ID').apply(
                lambda x: get_customer_spending_behaviour_features(x, windows_size_in_days=[1, 7, 30])
            )
            # Groupby might change the order, sort back and reset index
            transactions_df = transactions_df.sort_values('TX_DATETIME').reset_index(drop=True)
            print("  Applied customer spending features.")
            mlflow.set_tag("transformation_customer", "Success")
        except Exception as e:
            print(f"  ERROR during customer spending transformations: {e}")
            mlflow.set_tag("transformation_customer", f"Failed: {e}")
    else:
        print("  Skipping customer spending transformations: CUSTOMER_ID or TX_DATETIME missing.")
        mlflow.set_tag("transformation_customer", "Skipped - Missing Column")


    # 3. Terminal Risk Transformations
    print("Applying terminal risk transformations...")
    if 'TERMINAL_ID' in transactions_df.columns and 'TX_DATETIME' in transactions_df.columns and 'TX_FRAUD' in transactions_df.columns:
        try:
            # Use apply with a lambda function to call the transformation function per group
            transactions_df = transactions_df.groupby('TERMINAL_ID').apply(
                lambda x: get_count_risk_rolling_window(x, delay_period=7, windows_size_in_days=[1, 7, 30], feature="TERMINAL_ID")
            )
            # Groupby might change the order, sort back and reset index
            transactions_df = transactions_df.sort_values('TX_DATETIME').reset_index(drop=True)
            print("  Applied terminal risk features.")
            mlflow.set_tag("transformation_terminal", "Success")
        except Exception as e:
            print(f"  ERROR during terminal risk transformations: {e}")
            mlflow.set_tag("transformation_terminal", f"Failed: {e}")
    else:
        print("  Skipping terminal risk transformations: TERMINAL_ID, TX_DATETIME or TX_FRAUD missing.")
        mlflow.set_tag("transformation_terminal", "Skipped - Missing Column")

    # --- Save Transformed Data ---
    print(f"Saving transformed data to: {args.transformed_data}")
    output_path = Path(args.transformed_data)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save data back into daily .pkl files, similar to original notebook's output format
    # Ensure 'TX_TIME_DAYS' and 'TX_TIME_SECONDS' exist
    if 'TX_TIME_DAYS' in transactions_df.columns and 'TX_TIME_SECONDS' in transactions_df.columns:
        start_date_obj = datetime.datetime.strptime(args.start_date, "%Y-%m-%d")
        max_days = transactions_df.TX_TIME_DAYS.max()
        days_processed = 0
        for day in range(max_days + 1):
            transactions_day = transactions_df[transactions_df.TX_TIME_DAYS == day].sort_values('TX_TIME_SECONDS')

            if not transactions_day.empty:
                actual_date = start_date_obj + datetime.timedelta(days=day)
                filename_output = actual_date.strftime("%Y-%m-%d") + '.pkl'
                file_path = output_path / filename_output
                try:
                    transactions_day.to_pickle(str(file_path), protocol=4) # Protocol 4 for compatibility
                    days_processed += 1
                except Exception as e:
                    print(f"  Error saving file {filename_output}: {e}")
            # else:
                # print(f"  No transactions for day {day}, skipping file.")

        print(f"Saved transformed data for {days_processed} days in {args.transformed_data}")
        mlflow.log_metric("transformed_files_saved", days_processed)
        mlflow.log_metric("transformed_rows_saved", len(transactions_df))
        mlflow.set_tag("Data Saving Status", "Success")

    else:
         print("ERROR: Cannot save daily files because 'TX_TIME_DAYS' or 'TX_TIME_SECONDS' is missing.")
         mlflow.set_tag("Data Saving Status", "Failed - Missing Time Columns")
         mlflow.end_run(status="FAILED")
         return


    mlflow.end_run()
    print("Preparation script finished.")


if __name__ == "__main__":
    args = parse_args()
    main(args)