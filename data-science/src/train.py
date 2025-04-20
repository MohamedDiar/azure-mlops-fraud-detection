# data-science/src/train.py
# --- UPDATED to consume MLTable input ---

import os
import argparse
import datetime
import time
import pickle
import json
from pathlib import Path
import pandas as pd
import numpy as np
import traceback # Import traceback for detailed error printing
import sys
import mltable # Import mltable

import sklearn
from sklearn import model_selection, metrics, pipeline, preprocessing, tree, ensemble
import xgboost

import mlflow
import mlflow.sklearn

# Import shared functions from utils.py
from utils import (
    card_precision_top_k_custom,
    get_summary_performances,
    model_selection_wrapper,
    prequentialSplit_with_dates,
    get_train_test_set
)

def parse_args():
    # UPDATED Argument Parser for MLTable input
    parser = argparse.ArgumentParser("train_mltable") # Renamed for clarity
    parser.add_argument("--input_mltable_data", type=str, required=True, help="Path to folder containing the prepared MLTable data (mounted)")
    parser.add_argument("--model_output", type=str, required=True, help="Path to save output model artifact (MLflow format)")
    parser.add_argument("--test_data_output", type=str, required=True, help="Path to save the final test data split (as folder containing pkl)")

    # REMOVED: --transformed_data, --train_load_start_date, --train_load_end_date

    # Keep arguments needed for splitting the loaded data
    parser.add_argument("--anchor_date_str", type=str, required=True, help="Anchor date for deriving training/validation splits (YYYY-MM-DD)")
    parser.add_argument("--delta_train", type=int, default=7, help="Duration of training period in days")
    parser.add_argument("--delta_delay", type=int, default=7, help="Duration of delay period in days")
    parser.add_argument("--delta_assessment", type=int, default=7, help="Duration of assessment period in days")
    parser.add_argument("--n_folds", type=int, default=4, help="Number of folds for prequential validation")

    # Other Parameters
    parser.add_argument("--top_k_value", type=int, default=100, help="Value K for Card Precision@k")
    parser.add_argument("--n_jobs", type=int, default=5, help="Number of parallel jobs for GridSearchCV")

    args = parser.parse_args()
    print(f"Parsed Arguments: {args}") # Print args early for debugging
    return args

# REMOVED: load_transformed_data_window function is no longer needed

def main(args):
    mlflow.start_run()
    run_id = mlflow.active_run().info.run_id
    print(f"===== Training Script Started (MLTable Input - MLflow Run ID: {run_id}) =====")
    print(f"Full Arguments Received: {args}") # Log full args object

    # Log parameters (exclude path inputs if desired)
    print("[PARAM LOG] Logging input parameters...")
    loggable_params = {k: v for k, v in vars(args).items() if k not in ['input_mltable_data', 'model_output', 'test_data_output']}
    mlflow.log_params(loggable_params)
    # Log path parameters separately for clarity
    mlflow.log_param("input_mltable_path_param", args.input_mltable_data)
    mlflow.log_param("model_output_path_param", args.model_output)
    mlflow.log_param("test_data_output_path_param", args.test_data_output)


    # --- Load Prepared MLTable Data ---
    print(f"\n[LOAD DATA] Loading prepared MLTable from path '{args.input_mltable_data}'...")
    load_start_time = time.time()
    transactions_df = pd.DataFrame() # Initialize
    try:
        # Check if the input path exists
        input_path_obj = Path(args.input_mltable_data)
        if not input_path_obj.exists():
            raise FileNotFoundError(f"Input MLTable path does not exist: {args.input_mltable_data}")
        if not input_path_obj.is_dir():
             raise ValueError(f"Input MLTable path must be a directory: {args.input_mltable_data}")
        # Check for MLTable file inside the directory
        if not (input_path_obj / "MLTable").is_file():
             raise FileNotFoundError(f"MLTable definition file not found inside directory: {input_path_obj}")

        # Load the MLTable definition
        print(f"[LOAD DATA] Attempting to load MLTable definition from {input_path_obj}...")
        input_tbl = mltable.load(str(input_path_obj)) # Ensure path is string
        print("[LOAD DATA] MLTable definition loaded. Attempting to load into DataFrame...")
        # Load the full data into pandas
        transactions_df = input_tbl.to_pandas_dataframe()
        print(f"[LOAD DATA] Successfully loaded MLTable into DataFrame. Shape: {transactions_df.shape}")

        # Validate essential columns after loading
        if transactions_df.empty:
             print("[LOAD DATA] WARNING: Loaded DataFrame is empty. No training possible.")
             mlflow.log_metric("input_rows_loaded_train", 0)
             mlflow.set_tag("Training Status", "Failed - Empty DataFrame Loaded")
             mlflow.end_run(status="FAILED")
             sys.exit(1)

        if 'TX_DATETIME' not in transactions_df.columns:
            raise ValueError("Loaded DataFrame must contain 'TX_DATETIME' column for splitting.")
        # Ensure TX_DATETIME is datetime type
        if not pd.api.types.is_datetime64_any_dtype(transactions_df['TX_DATETIME']):
             print("[LOAD DATA] Converting TX_DATETIME to datetime objects...")
             transactions_df['TX_DATETIME'] = pd.to_datetime(transactions_df['TX_DATETIME'])
             print("[LOAD DATA] TX_DATETIME conversion complete.")

        # Log actual date range loaded
        min_date_loaded = transactions_df['TX_DATETIME'].min()
        max_date_loaded = transactions_df['TX_DATETIME'].max()
        print(f"[LOAD DATA] Actual loaded TX_DATETIME range: {min_date_loaded} to {max_date_loaded}")
        mlflow.log_param("actual_train_data_min_date", min_date_loaded.strftime('%Y-%m-%d %H:%M:%S') if pd.notna(min_date_loaded) else "N/A")
        mlflow.log_param("actual_train_data_max_date", max_date_loaded.strftime('%Y-%m-%d %H:%M:%S') if pd.notna(max_date_loaded) else "N/A")

    except FileNotFoundError as e:
        print(f"[LOAD DATA] ERROR: {e}")
        traceback.print_exc()
        mlflow.set_tag("Training Status", "Failed - Input Path/MLTable Not Found")
        mlflow.end_run(status="FAILED")
        sys.exit(1)
    except Exception as e:
        print(f"[LOAD DATA] ERROR loading input MLTable or converting to DataFrame: {e}")
        traceback.print_exc()
        mlflow.set_tag("Training Status", "Failed - Data Loading Error")
        mlflow.end_run(status="FAILED")
        sys.exit(1) # Exit script if loading failed

    load_time = time.time() - load_start_time
    print(f"[LOAD DATA] Data loading finished in {load_time:.2f} seconds.")
    mlflow.log_metric("input_rows_loaded_train", len(transactions_df))
    mlflow.log_metric("data_load_time_sec", load_time)


    # Define features and output (Assuming these columns exist in the prepared MLTable)
    OUTPUT_FEATURE = "TX_FRAUD"
    INPUT_FEATURES = ['TX_AMOUNT','TX_DURING_WEEKEND', 'TX_DURING_NIGHT', 'CUSTOMER_ID_NB_TX_1DAY_WINDOW',
                      'CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW', 'CUSTOMER_ID_NB_TX_7DAY_WINDOW',
                      'CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW', 'CUSTOMER_ID_NB_TX_30DAY_WINDOW',
                      'CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW', 'TERMINAL_ID_NB_TX_1DAY_WINDOW',
                      'TERMINAL_ID_RISK_1DAY_WINDOW', 'TERMINAL_ID_NB_TX_7DAY_WINDOW',
                      'TERMINAL_ID_RISK_7DAY_WINDOW', 'TERMINAL_ID_NB_TX_30DAY_WINDOW',
                      'TERMINAL_ID_RISK_30DAY_WINDOW']

    # --- Validate features exist in the loaded DataFrame ---
    print("\n[CONFIG] Validating required features in loaded DataFrame...")
    missing_input_features = [col for col in INPUT_FEATURES if col not in transactions_df.columns]
    if missing_input_features:
        print(f"[CONFIG] ERROR: Missing required input features in loaded data: {missing_input_features}")
        mlflow.set_tag("Training Status", "Failed - Missing Input Features")
        mlflow.end_run(status="FAILED")
        sys.exit(1)
    if OUTPUT_FEATURE not in transactions_df.columns:
        print(f"[CONFIG] ERROR: Missing required output feature '{OUTPUT_FEATURE}' in loaded data.")
        mlflow.set_tag("Training Status", "Failed - Missing Output Feature")
        mlflow.end_run(status="FAILED")
        sys.exit(1)

    print(f"[CONFIG] All required input features found: {INPUT_FEATURES}")
    print(f"[CONFIG] Output Feature found: {OUTPUT_FEATURE}")
    mlflow.log_param("input_features_used", json.dumps(INPUT_FEATURES))
    mlflow.log_param("output_feature_used", OUTPUT_FEATURE)


    # --- Derive Dates (Based on Anchor Date arg) ---
    print("\n[DATE DERIVATION] Calculating split dates based on anchor date...")
    try:
        start_date_training_anchor = datetime.datetime.strptime(args.anchor_date_str, "%Y-%m-%d")
        start_date_validation = start_date_training_anchor - datetime.timedelta(days=(args.delta_delay + args.delta_assessment))
        start_date_test_estimation = start_date_training_anchor
        final_train_start_date = start_date_training_anchor

        # Calculate approximate required range based on derived dates
        earliest_gs_train_start = start_date_validation - datetime.timedelta(days=(args.n_folds - 1) * args.delta_assessment)
        latest_final_test_end = final_train_start_date + datetime.timedelta(days=args.delta_train + args.delta_delay + args.delta_assessment)

        print(f"  Anchor Date: {start_date_training_anchor.strftime('%Y-%m-%d')}")
        print(f"  Derived Validation GridSearch Start: {start_date_validation.strftime('%Y-%m-%d')}")
        print(f"  Derived Test Estimation GridSearch Start: {start_date_test_estimation.strftime('%Y-%m-%d')}")
        print(f"  Derived Final Training Start: {final_train_start_date.strftime('%Y-%m-%d')}")
        print(f"  Approx. Earliest Date Needed for GridSearch Train: {earliest_gs_train_start.strftime('%Y-%m-%d')}")
        print(f"  Approx. Latest Date Needed for Final Test: {latest_final_test_end.strftime('%Y-%m-%d')}")

        mlflow.log_param("derived_validation_start_date", start_date_validation.strftime('%Y-%m-%d'))
        mlflow.log_param("derived_test_estimation_start_date", start_date_test_estimation.strftime('%Y-%m-%d'))
        mlflow.log_param("derived_final_train_start_date", final_train_start_date.strftime('%Y-%m-%d'))

        # --- REVISED DATE VALIDATION ---
        # Check if the derived splitting dates fall within the actual data loaded
        min_date_loaded = transactions_df['TX_DATETIME'].min()
        max_date_loaded = transactions_df['TX_DATETIME'].max()
        if pd.isna(min_date_loaded) or pd.isna(max_date_loaded):
             print("[DATE VALIDATION] WARNING: Could not determine date range from loaded data (contains NaNs?). Cannot validate split coverage.")
             mlflow.set_tag("Data Window Warning", "NaN dates in loaded data, split coverage unknown")
        elif earliest_gs_train_start < min_date_loaded or latest_final_test_end > max_date_loaded:
             print("[DATE VALIDATION] WARNING: Derived split date range extends beyond the actual dates present in the loaded MLTable data!")
             print(f"  Loaded Data Actual Range: {min_date_loaded.strftime('%Y-%m-%d')} to {max_date_loaded.strftime('%Y-%m-%d')}")
             print(f"  Derived Split Range Needed: Approx {earliest_gs_train_start.strftime('%Y-%m-%d')} to {latest_final_test_end.strftime('%Y-%m-%d')}")
             print("  Splitting functions might generate empty or smaller-than-expected sets. Ensure input MLTable covers the required period.")
             mlflow.set_tag("Data Window Warning", "Derived splits might exceed loaded data range")
        else:
             print("[DATE VALIDATION] OK: Loaded MLTable data date range appears to cover the derived split periods.")

    except ValueError as e:
        print(f"[DATE DERIVATION] ERROR parsing anchor date '{args.anchor_date_str}': {e}")
        mlflow.set_tag("Training Status", "Failed - Date Parsing Error")
        mlflow.end_run(status="FAILED")
        sys.exit(1) # Exit on critical date parse error
    except Exception as e:
        print(f"[DATE DERIVATION] ERROR during date calculation or validation: {e}")
        traceback.print_exc()
        mlflow.set_tag("Training Status", "Failed - Date Calculation Error")
        mlflow.end_run(status="FAILED")
        sys.exit(1) # Exit on critical date calc error

    # --- Model Selection (Grid Search for XGBoost ONLY) ---
    print("\n===== Starting Model Selection (XGBoost Only) =====")
    start_selection_time = time.time()

    classifier_xgb = xgboost.XGBClassifier(random_state=0, use_label_encoder=False, eval_metric='logloss', n_jobs=1)
    parameters_xgb = {
        'clf__max_depth': [3, 6, 9], 'clf__n_estimators': [25, 50, 100], 'clf__learning_rate': [0.1, 0.3],
        'clf__random_state':[0], 'clf__n_jobs':[1], 'clf__verbosity':[0],
        'clf__use_label_encoder':[False], 'clf__eval_metric':['logloss']
    }
    print(f"[GRID SEARCH] XGBoost Parameter Grid: {parameters_xgb}")
    mlflow.log_param("xgboost_param_grid", json.dumps({k:str(v) for k,v in parameters_xgb.items()}))

    # Prepare scorer dataframe subset (needs TX_TIME_DAYS)
    transactions_df_scorer = pd.DataFrame()
    print("[GRID SEARCH] Preparing scorer helper dataframe...")
    try:
        # Ensure TX_TIME_DAYS exists from loading or calculation in prep
        # This column is crucial for the custom scorer logic
        scorer_cols = ['CUSTOMER_ID', 'TX_FRAUD', 'TX_TIME_DAYS']
        if not all(col in transactions_df.columns for col in scorer_cols):
             missing_scorer_cols = [col for col in scorer_cols if col not in transactions_df.columns]
             raise ValueError(f"Missing columns required for scorer helper DataFrame: {missing_scorer_cols}. Ensure they are present in the prepared MLTable.")

        transactions_df_scorer = transactions_df[scorer_cols].copy()
        print(f"[GRID SEARCH] Scorer helper DataFrame created. Shape: {transactions_df_scorer.shape}")
    except Exception as e:
        print(f"[GRID SEARCH] ERROR creating scorer helper DataFrame: {e}")
        mlflow.set_tag("Training Status", "Failed - Scorer DF Creation Error")
        mlflow.end_run(status="FAILED")
        sys.exit(1) # Exit if scorer cannot be prepared

    # Create custom scorer
    card_precision_top_k_scorer = None
    if not transactions_df_scorer.empty:
        print("[GRID SEARCH] Creating custom scorer...")
        try:
            # Ensure the scorer function and its dependencies are correct
            card_precision_top_k_scorer = sklearn.metrics.make_scorer(
                card_precision_top_k_custom, needs_proba=True,
                top_k=args.top_k_value, transactions_df=transactions_df_scorer)
            print(f"[GRID SEARCH] Custom scorer 'card_precision@{args.top_k_value}' created successfully.")
        except Exception as e:
            print(f"[GRID SEARCH] Warning: Failed to create custom scorer: {e}. Card Precision@k metric will be unavailable in GridSearchCV results.")
            # Continue without the custom scorer if it fails
            card_precision_top_k_scorer = None # Explicitly set to None

    # Define scoring dictionary for GridSearchCV
    scoring = {'roc_auc': 'roc_auc', 'average_precision': 'average_precision'}
    if card_precision_top_k_scorer:
        scoring[f'card_precision@{args.top_k_value}'] = card_precision_top_k_scorer
    # List used later to extract results from cv_results_
    performance_metrics_list_grid = list(scoring.keys())
    # List used for the summary table generation
    performance_metrics_list = ['AUC ROC', 'Average precision']
    # Add CP@k to summary list ONLY if scorer was successfully created
    if card_precision_top_k_scorer:
        performance_metrics_list.append(f'Card Precision@{args.top_k_value}')
    print(f"[GRID SEARCH] Scoring metrics for GridSearchCV: {performance_metrics_list_grid}")
    print(f"[GRID SEARCH] Metrics for summary table: {performance_metrics_list}")


    # Run model selection wrapper FOR XGBOOST ONLY
    performances_df_xgb = pd.DataFrame()
    print("[GRID SEARCH] Starting model_selection_wrapper for XGBoost...")
    try:
        # Pass the main transactions_df loaded from the MLTable
        performances_df_xgb = model_selection_wrapper(
            transactions_df, # Use the DataFrame loaded from MLTable
            classifier_xgb, INPUT_FEATURES, OUTPUT_FEATURE,
            parameters_xgb, scoring, # Pass the potentially modified scoring dict
            start_date_validation, start_date_test_estimation, # Use DERIVED dates
            n_folds=args.n_folds,
            delta_train=args.delta_train, delta_delay=args.delta_delay, delta_assessment=args.delta_assessment,
            performance_metrics_list_grid=performance_metrics_list_grid, # Pass grid keys
            performance_metrics_list=performance_metrics_list, # Pass summary keys
            n_jobs=args.n_jobs
        )
        selection_time = time.time() - start_selection_time
        print(f"[GRID SEARCH] XGBoost GridSearchCV finished in {selection_time:.2f} seconds.")
        mlflow.log_metric("xgb_gridsearch_time_sec", selection_time)

        # Log GridSearch results artifact
        if not performances_df_xgb.empty:
             print(f"[GRID SEARCH] GridSearchCV results obtained. Shape: {performances_df_xgb.shape}")
             perf_artifact_path = "xgboost_grid_search_results.csv"
             performances_df_xgb.round(5).to_csv(perf_artifact_path, index=False) # Round for consistency
             mlflow.log_artifact(perf_artifact_path)
             print(f"[GRID SEARCH] Logged XGBoost grid search results to MLflow artifact: {perf_artifact_path}")
             mlflow.set_tag("XGBoost Grid Search Status", "Completed - Success")
        else:
             print("[GRID SEARCH] Warning: XGBoost grid search returned empty results DataFrame.")
             mlflow.set_tag("XGBoost Grid Search Status", "Completed - No Results")

    except Exception as e:
        print(f"[GRID SEARCH] ERROR during XGBoost model_selection_wrapper execution: {e}")
        traceback.print_exc()
        mlflow.log_metric("xgb_gridsearch_time_sec", time.time() - start_selection_time) # Log time even on failure
        mlflow.set_tag("XGBoost Grid Search Status", f"Failed: {e}")
        # Decide if failure here is critical. For now, continue to default params.
        print("[GRID SEARCH] Proceeding with default parameters due to GridSearchCV failure.")


    # --- Determine Best Parameters for XGBoost ---
    # This logic relies on the structure of performances_df_xgb generated above
    best_params_xgb_dict = None
    best_params_xgb_summary = "Defaults"
    primary_metric = 'Average precision' # Metric to optimize (must match a name in performance_metrics_list)
    print(f"\n[BEST PARAMS] Determining best XGBoost parameters based on Validation '{primary_metric}'...")

    if not performances_df_xgb.empty:
        try:
            # Ensure 'Parameters summary' column exists (fallback if needed)
            if 'Parameters summary' not in performances_df_xgb.columns:
                 if 'Parameters' in performances_df_xgb.columns:
                     # Define fallback for creating summary string if missing
                     def params_to_str_fallback(params):
                         try:
                             if isinstance(params, dict): items = [f"{k.split('__')[1]}={v}" for k, v in sorted(params.items())]; return ", ".join(items)
                             else: return str(params) # Return raw string if not dict
                         except Exception: return str(params) # Final fallback
                     performances_df_xgb['Parameters summary'] = performances_df_xgb['Parameters'].apply(params_to_str_fallback)
                     print("[BEST PARAMS] Created fallback 'Parameters summary' column.")
                 else:
                     # If neither 'Parameters' nor 'Parameters summary' exists, we can't determine best params
                     raise KeyError("Missing 'Parameters' and 'Parameters summary' columns in GridSearchCV results.")

            # Generate summary performance table using the defined metrics list
            summary_xgb = get_summary_performances(performances_df_xgb, parameter_column_name="Parameters summary")
            print("[BEST PARAMS] Generated performance summary table:")
            print(summary_xgb) # Print the summary table
            # Log summary as artifact
            summary_artifact_path = "grid_search_summary.csv"
            summary_xgb.to_csv(summary_artifact_path)
            mlflow.log_artifact(summary_artifact_path)
            print(f"[BEST PARAMS] Logged performance summary to MLflow artifact: {summary_artifact_path}")


            # Find best parameters based on the primary validation metric
            if primary_metric in summary_xgb.columns:
                best_params_xgb_summary = summary_xgb.loc["Best estimated parameters", primary_metric]
                validation_perf_str = summary_xgb.loc["Validation performance", primary_metric]

                # Check if grid search actually found a best parameter set (not 'N/A')
                if best_params_xgb_summary != 'N/A':
                    # Find the original parameter dictionary corresponding to the best summary string
                    best_row = performances_df_xgb[performances_df_xgb['Parameters summary'] == best_params_xgb_summary]
                    if not best_row.empty:
                        best_params_xgb_dict = best_row['Parameters'].iloc[0] # Get the actual dict
                        print(f"[BEST PARAMS] Found Best XGBoost parameters (based on Validation {primary_metric}):")
                        print(f"  Summary String: {best_params_xgb_summary}")
                        print(f"  Parameter Dict: {best_params_xgb_dict}")
                        mlflow.set_tag("best_xgb_params_source", "GridSearch")
                        mlflow.log_param("best_xgb_params_summary", best_params_xgb_summary)
                        # Log the best validation score achieved
                        try:
                             if isinstance(validation_perf_str, str) and validation_perf_str not in ['N/A', 'NaN']:
                                 # Extract numeric part (before potential '+/-')
                                 best_val_score_num = float(validation_perf_str.split('+/-')[0].strip())
                                 mlflow.log_metric(f"best_validation_{primary_metric.lower().replace(' ','_').replace('@','_at_')}", best_val_score_num)
                                 print(f"[BEST PARAMS] Best Validation {primary_metric} Score: {best_val_score_num:.4f}")
                             else:
                                 print(f"[BEST PARAMS] Best Validation {primary_metric} score string is invalid or N/A: '{validation_perf_str}'")
                        except Exception as score_e:
                             print(f"[BEST PARAMS] Warning: Could not parse/log best validation score from '{validation_perf_str}': {score_e}")
                    else:
                        print(f"[BEST PARAMS] Warning: Could not find original parameters row matching best summary string '{best_params_xgb_summary}'. Using defaults.")
                        best_params_xgb_dict = None # Reset to ensure defaults are used
                else:
                    print(f"[BEST PARAMS] Best parameters summary is 'N/A' in the summary table. Using defaults.")
                    best_params_xgb_dict = None # Reset to ensure defaults are used
            else:
                print(f"[BEST PARAMS] Warning: Primary metric '{primary_metric}' not found in performance summary columns. Using defaults.")
                best_params_xgb_dict = None # Reset to ensure defaults are used

        except KeyError as e:
             print(f"[BEST PARAMS] Error: Missing expected column during best parameter determination: {e}. Using defaults.")
             best_params_xgb_dict = None
        except Exception as e:
            print(f"[BEST PARAMS] Error determining best XGBoost parameters from results: {e}. Using defaults.")
            traceback.print_exc()
            best_params_xgb_dict = None # Reset to ensure defaults are used
    else:
        print("[BEST PARAMS] XGBoost grid search results DataFrame is empty. Using default parameters.")
        best_params_xgb_dict = None # Ensure defaults are used

    # Fallback to default parameters if grid search failed or didn't find better params
    if best_params_xgb_dict is None:
        # Define the default parameters explicitly
        best_params_xgb_dict = {
            'clf__max_depth': 6, 'clf__n_estimators': 100, 'clf__learning_rate': 0.3,
            'clf__random_state': 0, 'clf__n_jobs': 1, 'clf__verbosity': 0,
            'clf__use_label_encoder': False, 'clf__eval_metric': 'logloss'
        }
        # Regenerate summary string from defaults for consistency in logging/tags
        try:
             items = [f"{k.split('__')[1]}={v}" for k, v in sorted(best_params_xgb_dict.items())]
             best_params_xgb_summary = ", ".join(items)
        except Exception: best_params_xgb_summary = str(best_params_xgb_dict) # Fallback

        print(f"[BEST PARAMS] Using default XGBoost parameters: {best_params_xgb_summary}")
        mlflow.set_tag("best_xgb_params_source", "Default")
        mlflow.log_param("best_xgb_params_summary", best_params_xgb_summary) # Log the defaults used


    # --- Train Final XGBoost Model ---
    print("\n===== Training Final XGBoost Model =====")

    # Prepare final training data split using the full loaded DataFrame and derived dates
    final_train_df = pd.DataFrame()
    final_test_df = pd.DataFrame()
    print(f"[FINAL SPLIT] Creating final train/test split using derived start date: {final_train_start_date.strftime('%Y-%m-%d')}")
    try:
        # Ensure the split function uses the DataFrame loaded from MLTable
        (final_train_df, final_test_df) = get_train_test_set(
            transactions_df, # Use the DataFrame loaded from MLTable
            start_date_training=final_train_start_date, # Use derived date
            delta_train=args.delta_train, delta_delay=args.delta_delay, delta_test=args.delta_assessment
        )
        print(f"[FINAL SPLIT] Final training set shape: {final_train_df.shape}")
        print(f"[FINAL SPLIT] Final test set shape: {final_test_df.shape}")
        mlflow.log_metric("final_train_rows", final_train_df.shape[0])
        mlflow.log_metric("final_test_rows", final_test_df.shape[0])
        if not final_train_df.empty: mlflow.log_metric("final_train_fraud_rate", final_train_df[OUTPUT_FEATURE].mean())
        if not final_test_df.empty: mlflow.log_metric("final_test_fraud_rate", final_test_df[OUTPUT_FEATURE].mean())

        # --- CRITICAL CHECK: Ensure final training data is not empty ---
        if final_train_df.empty:
            print("[FINAL SPLIT] ERROR: Final training set is empty after splitting. Cannot train model.")
            mlflow.set_tag("Training Status", "Failed - Empty Final Train Split")
            mlflow.end_run(status="FAILED")
            sys.exit(1) # Exit script with failure code
        else:
             print("[FINAL SPLIT] Final training set is not empty. Proceeding with training.")

    except Exception as e:
        print(f"[FINAL SPLIT] ERROR creating final train/test split: {e}")
        traceback.print_exc()
        mlflow.set_tag("Training Status", "Failed - Final Split Error")
        mlflow.end_run(status="FAILED")
        sys.exit(1) # Exit script with failure code

    # Create final pipeline with selected/default parameters
    print("[FINAL TRAIN] Creating final pipeline with selected/default XGBoost parameters...")
    final_classifier_xgb = xgboost.XGBClassifier() # Base instance
    # Prepare parameters dictionary (remove 'clf__' prefix needed for GridSearchCV)
    final_params_xgb_filtered = {}
    try:
        final_params_xgb_filtered = {k.split('__', 1)[1]: v for k, v in best_params_xgb_dict.items() if k.startswith('clf__')}
        # Ensure boolean/int types are correct if they came from strings (less likely now but good practice)
        if 'use_label_encoder' in final_params_xgb_filtered: final_params_xgb_filtered['use_label_encoder'] = bool(str(final_params_xgb_filtered['use_label_encoder']).lower() == 'true')
        if 'verbosity' in final_params_xgb_filtered: final_params_xgb_filtered['verbosity'] = int(final_params_xgb_filtered['verbosity'])
        if 'n_jobs' in final_params_xgb_filtered: final_params_xgb_filtered['n_jobs'] = int(final_params_xgb_filtered['n_jobs'])
        if 'random_state' in final_params_xgb_filtered: final_params_xgb_filtered['random_state'] = int(final_params_xgb_filtered['random_state'])
        # Set parameters on the classifier instance
        final_classifier_xgb.set_params(**final_params_xgb_filtered)
        print(f"[FINAL TRAIN] Set final XGBoost classifier parameters: {final_params_xgb_filtered}")
    except Exception as e:
        print(f"[FINAL TRAIN] Warning: Error processing/setting final XGBoost params: {e}. Using defaults.")
        # Reset to known defaults if setting params failed
        final_classifier_xgb = xgboost.XGBClassifier(random_state=0, use_label_encoder=False, eval_metric='logloss', n_jobs=1)

    # Define the final scikit-learn pipeline (Scaler + Classifier)
    final_pipeline = sklearn.pipeline.Pipeline([
        ('scaler', sklearn.preprocessing.StandardScaler()), # Standard scaler remains
        ('clf', final_classifier_xgb) # Use the configured classifier
    ])
    print(f"[FINAL TRAIN] Final scikit-learn pipeline created: {final_pipeline}")

    # Fit final pipeline on the final training data
    print("[FINAL TRAIN] Starting final model fitting...")
    start_fit_time = time.time()
    try:
        X_train_final = final_train_df[INPUT_FEATURES]
        y_train_final = final_train_df[OUTPUT_FEATURE]
        print(f"[FINAL TRAIN] Input features shape for final fit: {X_train_final.shape}")
        print(f"[FINAL TRAIN] Output feature shape for final fit: {y_train_final.shape}")

        # Check for NaNs before fitting (Scaler should handle, but good practice)
        if X_train_final.isnull().values.any():
            nan_cols = X_train_final.columns[X_train_final.isnull().any()].tolist()
            print(f"[FINAL TRAIN] Warning: NaNs detected in final training features before fitting (Columns: {nan_cols}). Scaler should handle or raise error.")
            # Optional: Add explicit imputation here if pipeline doesn't handle it
            # Example: X_train_final = SimpleImputer(strategy='mean').fit_transform(X_train_final)

        # Fit the pipeline
        final_pipeline.fit(X_train_final, y_train_final)
        final_fit_time = time.time() - start_fit_time
        print(f"[FINAL TRAIN] Final XGBoost model pipeline fitting completed successfully in {final_fit_time:.2f} seconds.")
        mlflow.log_metric("final_model_train_time_sec", final_fit_time)
        mlflow.set_tag("Final Model Training Status", "Success")

        # --- Log and Save Final Model ---
        print(f"\n[MODEL LOGGING] Attempting to log final pipeline to MLflow run...")
        # The output path is provided by the Azure ML pipeline runtime
        print(f"[MODEL LOGGING] Target output path variable for model: {args.model_output}")
        try:
            # Save the scikit-learn pipeline using mlflow.sklearn
            mlflow.sklearn.save_model(
                sk_model=final_pipeline,
                path=args.model_output,  # Save to the path specified by the pipeline output
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
                # Optionally include signature and input example for better tracking/deployment
                # signature=infer_signature(X_train_final, y_train_final),
                # input_example=X_train_final.head(5)
            )
            print(f"[MODEL LOGGING] mlflow.sklearn.save_model call completed. Model saved to path: {args.model_output}")
            mlflow.set_tag("Model Logging Status", "Success")
            # Note: Azure ML automatically uploads the content of args.model_output path
            # as the pipeline output artifact named 'model_output'.

        except Exception as log_e:
             print(f"[MODEL LOGGING] ERROR during mlflow.sklearn.save_model: {log_e}")
             traceback.print_exc()
             mlflow.set_tag("Model Logging Status", f"Failed: {log_e}")
             mlflow.set_tag("Training Status", "Failed - Model Logging Error")
             mlflow.end_run(status="FAILED")
             sys.exit(1) # Exit script with failure code

        # --- Save Final Test Data Split to Output Path ---
        # This part remains the same, as evaluate.py still expects a .pkl file
        print(f"\n[TEST DATA SAVE] Saving final test data split to output path: {args.test_data_output}")
        test_data_output_path = Path(args.test_data_output)
        test_data_output_path.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        test_data_file = test_data_output_path / "final_test_data.pkl"
        try:
            if not final_test_df.empty:
                # Save the test split DataFrame as a pickle file
                final_test_df.to_pickle(test_data_file)
                print(f"[TEST DATA SAVE] Final test data DataFrame saved successfully to {test_data_file}")
                mlflow.set_tag("Test Data Save Status", "Success")
            else:
                print("[TEST DATA SAVE] Warning: Final test data split is empty, not saving the pickle file.")
                # Create an empty file perhaps, or just log? Creating empty might be safer downstream.
                test_data_file.touch() # Create empty file marker
                print(f"[TEST DATA SAVE] Created empty marker file at {test_data_file}")
                mlflow.set_tag("Test Data Save Status", "Skipped - Empty Split")
        except Exception as e:
            print(f"[TEST DATA SAVE] ERROR saving final test data pickle file: {e}")
            traceback.print_exc()
            mlflow.set_tag("Test Data Save Status", f"Failed: {e}")
            # Consider if this failure is critical. Maybe not, if eval can handle missing file?
            # For now, log error but don't fail the run.

    except Exception as e:
        print(f"[FINAL TRAIN] ERROR fitting final XGBoost model pipeline: {e}")
        traceback.print_exc()
        mlflow.log_metric("final_model_train_time_sec", time.time() - start_fit_time) # Log time even on failure
        mlflow.set_tag("Final Model Training Status", f"Failed: {e}")
        mlflow.set_tag("Training Status", "Failed - Final Fit Error")
        mlflow.end_run(status="FAILED")
        sys.exit(1) # Exit script with failure code

    # --- End of Script ---
    mlflow.set_tag("Training Status", "Completed Successfully") # Set status if reached here
    mlflow.end_run()
    print(f"===== Training Script Finished Successfully (MLflow Run ID: {run_id}) =====")

if __name__ == "__main__":
    # This block executes when the script is run directly
    print("--- Script execution started ---")
    args = parse_args()
    main(args)
    print("--- Script execution finished ---")
