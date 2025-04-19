# data-science/src/train.py
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
import sys # Keep sys import for potential future use, but remove exit(1) here

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
    prequentialSplit_with_dates, # Ensure this is the one with prints if debugging splits
    get_train_test_set
)

def parse_args():
    parser = argparse.ArgumentParser("train")
    parser.add_argument("--transformed_data", type=str, help="Path to folder containing ALL transformed data files (.pkl) from prep step")
    # parser.add_argument("--model_output", type=str, help="Path to save output model artifact (MLflow format)")
    parser.add_argument("--test_data_output", type=str, help="Path to save the final test data split (as folder containing pkl)")

    # Dates defining the specific window of TRANSFORMED data to load for this step
    parser.add_argument("--train_load_start_date", type=str, required=True, help="Start date of transformed data window to load (YYYY-MM-DD)")
    parser.add_argument("--train_load_end_date", type=str, required=True, help="End date of transformed data window to load (YYYY-MM-DD)")
    # Anchor date for splitting the loaded window
    parser.add_argument("--anchor_date_str", type=str, required=True, help="Anchor date for deriving training/validation splits (YYYY-MM-DD)")

    # DELTA ARGUMENTS
    parser.add_argument("--delta_train", type=int, default=7, help="Duration of training period in days")
    parser.add_argument("--delta_delay", type=int, default=7, help="Duration of delay period in days")
    parser.add_argument("--delta_assessment", type=int, default=7, help="Duration of assessment period in days")
    parser.add_argument("--n_folds", type=int, default=4, help="Number of folds for prequential validation")

    # Other Parameters
    parser.add_argument("--top_k_value", type=int, default=100, help="Value K for Card Precision@k")
    parser.add_argument("--n_jobs", type=int, default=5, help="Number of parallel jobs for GridSearchCV")

    args = parser.parse_args()
    return args

def load_transformed_data_window(data_path, start_date_str, end_date_str):
    """Loads transformed pickle files ONLY within the specified window."""
    print(f"[LOAD DATA] Loading transformed data from path '{data_path}' for window: {start_date_str} to {end_date_str}")
    data_path_obj = Path(data_path)
    if not data_path_obj.is_dir():
        print(f"[LOAD DATA] ERROR: Input directory not found: {data_path}")
        return pd.DataFrame()

    all_files = sorted([f for f in data_path_obj.glob('*.pkl') if f.is_file()])
    print(f"[LOAD DATA] Found {len(all_files)} total .pkl files in source directory.")

    target_files = [
        f for f in all_files
        if start_date_str <= f.stem <= end_date_str # Filter based on dates in filename
    ]

    if not target_files:
        print(f"[LOAD DATA] ERROR: No transformed files found in {data_path} for required training window {start_date_str} to {end_date_str}")
        return pd.DataFrame()

    print(f"[LOAD DATA] Found {len(target_files)} transformed files matching date window.")
    frames = []
    for f_path in target_files:
         try:
             df = pd.read_pickle(f_path)
             # Basic check after loading each file
             if df.empty:
                 print(f"[LOAD DATA] Warning: Loaded empty dataframe from {f_path.name}")
             frames.append(df)
         except Exception as e:
              print(f"[LOAD DATA] Error reading transformed file {f_path.name}: {e}")

    if not frames:
        print("[LOAD DATA] ERROR: No dataframes were successfully loaded for the window.")
        return pd.DataFrame()

    try:
        df_final = pd.concat(frames, ignore_index=True)
        print(f"[LOAD DATA] Concatenated data shape: {df_final.shape}")
        df_final = df_final.sort_values('TRANSACTION_ID').reset_index(drop=True)

        # Ensure TX_DATETIME is datetime
        if 'TX_DATETIME' in df_final.columns:
            if not pd.api.types.is_datetime64_any_dtype(df_final['TX_DATETIME']):
                print("[LOAD DATA] Converting TX_DATETIME to datetime objects...")
                df_final['TX_DATETIME'] = pd.to_datetime(df_final['TX_DATETIME'])
            # Log the actual date range loaded
            min_date_loaded = df_final['TX_DATETIME'].min()
            max_date_loaded = df_final['TX_DATETIME'].max()
            print(f"[LOAD DATA] Actual loaded TX_DATETIME range: {min_date_loaded} to {max_date_loaded}")
            if mlflow.active_run():
                 mlflow.log_param("actual_train_data_min_date", min_date_loaded.strftime('%Y-%m-%d %H:%M:%S') if min_date_loaded else "N/A")
                 mlflow.log_param("actual_train_data_max_date", max_date_loaded.strftime('%Y-%m-%d %H:%M:%S') if max_date_loaded else "N/A")
        else:
            print("[LOAD DATA] ERROR: TX_DATETIME column missing after concatenation.")
            return pd.DataFrame() # Critical column missing

    except Exception as e:
        print(f"[LOAD DATA] ERROR during final processing/concatenation: {e}")
        traceback.print_exc()
        return pd.DataFrame()

    return df_final

def main(args):
    mlflow.start_run()
    run_id = mlflow.active_run().info.run_id
    print(f"===== Training Script Started (MLflow Run ID: {run_id}) =====")
    print(f"Arguments: {args}")

    # Log parameters
    print("[PARAM LOG] Logging input parameters...")
    mlflow.log_params({k: v for k, v in vars(args).items() if k not in ['transformed_data', 'test_data_output']})
    mlflow.log_param("train_data_load_start", args.train_load_start_date)
    mlflow.log_param("train_data_load_end", args.train_load_end_date)

    # --- Load SPECIFIC Window of Transformed Data ---
    load_start_time = time.time()
    transactions_df = load_transformed_data_window(
        Path(args.transformed_data),
        args.train_load_start_date,
        args.train_load_end_date
    )
    load_time = time.time() - load_start_time
    print(f"[LOAD DATA] Data loading finished in {load_time:.2f} seconds.")

    if transactions_df.empty:
        print("[LOAD DATA] ERROR: No transformed data loaded. Stopping execution.")
        mlflow.log_metric("transformed_rows_loaded_train", 0)
        mlflow.set_tag("Training Status", "Failed - No Data Loaded")
        mlflow.end_run(status="FAILED")
        sys.exit(1) # Exit script if loading failed

    print(f"[LOAD DATA] Loaded {len(transactions_df)} transformed transactions.")
    mlflow.log_metric("transformed_rows_loaded_train", len(transactions_df))
    mlflow.log_metric("data_load_time_sec", load_time)

    # Define features and output
    OUTPUT_FEATURE = "TX_FRAUD"
    INPUT_FEATURES = ['TX_AMOUNT','TX_DURING_WEEKEND', 'TX_DURING_NIGHT', 'CUSTOMER_ID_NB_TX_1DAY_WINDOW',
                      'CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW', 'CUSTOMER_ID_NB_TX_7DAY_WINDOW',
                      'CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW', 'CUSTOMER_ID_NB_TX_30DAY_WINDOW',
                      'CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW', 'TERMINAL_ID_NB_TX_1DAY_WINDOW',
                      'TERMINAL_ID_RISK_1DAY_WINDOW', 'TERMINAL_ID_NB_TX_7DAY_WINDOW',
                      'TERMINAL_ID_RISK_7DAY_WINDOW', 'TERMINAL_ID_NB_TX_30DAY_WINDOW',
                      'TERMINAL_ID_RISK_30DAY_WINDOW']
    print(f"[CONFIG] Input Features: {INPUT_FEATURES}")
    print(f"[CONFIG] Output Feature: {OUTPUT_FEATURE}")
    mlflow.log_param("input_features", json.dumps(INPUT_FEATURES))
    mlflow.log_param("output_feature", OUTPUT_FEATURE)

    # --- Derive Dates (Based on Anchor Date arg) ---
    print("\n[DATE DERIVATION] Calculating split dates...")
    try:
        start_date_training_anchor = datetime.datetime.strptime(args.anchor_date_str, "%Y-%m-%d")
        start_date_validation = start_date_training_anchor - datetime.timedelta(days=(args.delta_delay + args.delta_assessment))
        start_date_test_estimation = start_date_training_anchor
        final_train_start_date = start_date_training_anchor

        # Calculate approximate required range based on derived dates
        earliest_gs_train_start = start_date_validation - datetime.timedelta(days=(args.n_folds - 1) * args.delta_assessment)
        latest_final_test_end = final_train_start_date + datetime.timedelta(days=args.delta_train + args.delta_delay + args.delta_assessment)

        print(f"  Anchor Date: {start_date_training_anchor.strftime('%Y-%m-%d')}")
        print(f"  Validation GridSearch Start: {start_date_validation.strftime('%Y-%m-%d')}")
        print(f"  Test Estimation GridSearch Start: {start_date_test_estimation.strftime('%Y-%m-%d')}")
        print(f"  Final Training Start: {final_train_start_date.strftime('%Y-%m-%d')}")
        print(f"  Approx. Earliest Date Needed for GridSearch Train: {earliest_gs_train_start.strftime('%Y-%m-%d')}")
        print(f"  Approx. Latest Date Needed for Final Test: {latest_final_test_end.strftime('%Y-%m-%d')}")

        mlflow.log_param("derived_validation_start_date", start_date_validation.strftime('%Y-%m-%d'))
        mlflow.log_param("derived_test_estimation_start_date", start_date_test_estimation.strftime('%Y-%m-%d'))
        mlflow.log_param("derived_final_train_start_date", final_train_start_date.strftime('%Y-%m-%d'))

        # --- DATE VALIDATION ---
        min_date_loaded = transactions_df['TX_DATETIME'].min()
        max_date_loaded = transactions_df['TX_DATETIME'].max()
        # --- MODIFICATION START ---
        # Change ERROR to WARNING and remove exit
        if min_date_loaded > earliest_gs_train_start or max_date_loaded < latest_final_test_end:
             print("[DATE VALIDATION] WARNING: Loaded data window does NOT cover the full range required by the derived split dates!")
             print(f"  Loaded: {min_date_loaded} to {max_date_loaded}")
             print(f"  Required Approx: {earliest_gs_train_start} to {latest_final_test_end}")
             print("  Proceeding, but splitting functions might truncate periods based on available data (like notebook behavior).")
             mlflow.set_tag("Data Window Warning", "Loaded data may truncate requested split periods")
        # --- MODIFICATION END ---
        else:
             print("[DATE VALIDATION] OK: Loaded data window covers the required date range.")

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

    # Prepare scorer dataframe subset
    transactions_df_scorer = pd.DataFrame()
    print("[GRID SEARCH] Preparing scorer helper dataframe...")
    try:
        scorer_cols = ['CUSTOMER_ID', 'TX_FRAUD', 'TX_TIME_DAYS']
        if not all(col in transactions_df.columns for col in scorer_cols):
             # Add TX_TIME_DAYS if missing (might happen if prep failed silently)
             if 'TX_DATETIME' in transactions_df.columns and 'TX_TIME_DAYS' not in transactions_df.columns:
                  print("[GRID SEARCH] Warning: TX_TIME_DAYS missing, attempting recalculation for scorer...")
                  min_date_temp = transactions_df['TX_DATETIME'].dt.date.min()
                  if min_date_temp:
                       transactions_df['TX_TIME_DAYS'] = (transactions_df['TX_DATETIME'].dt.date - min_date_temp).apply(lambda x: x.days)
                       print("[GRID SEARCH] Recalculated TX_TIME_DAYS.")
                       if 'TX_TIME_DAYS' not in transactions_df.columns: # Check again
                            raise ValueError("Failed to recalculate TX_TIME_DAYS")
                  else: raise ValueError("Cannot recalculate TX_TIME_DAYS, min date unknown")
             else:
                  raise ValueError(f"Missing columns required for scorer DF: {scorer_cols}")

        transactions_df_scorer = transactions_df[scorer_cols].copy()
        print(f"[GRID SEARCH] Scorer helper DataFrame shape: {transactions_df_scorer.shape}")
    except Exception as e:
        print(f"[GRID SEARCH] ERROR creating scorer helper DF: {e}")
        mlflow.set_tag("Training Status", "Failed - Scorer DF Creation Error")
        mlflow.end_run(status="FAILED")
        sys.exit(1) # Exit if scorer cannot be prepared

    # Create custom scorer
    card_precision_top_k_scorer = None
    if not transactions_df_scorer.empty:
        print("[GRID SEARCH] Creating custom scorer...")
        try:
            card_precision_top_k_scorer = sklearn.metrics.make_scorer(
                card_precision_top_k_custom, needs_proba=True,
                top_k=args.top_k_value, transactions_df=transactions_df_scorer)
            print(f"[GRID SEARCH] Custom scorer 'card_precision@{args.top_k_value}' created successfully.")
        except Exception as e:
            print(f"[GRID SEARCH] Warning: Failed to create custom scorer: {e}. CP@k metric will be unavailable.")

    # Define scoring dictionary
    scoring = {'roc_auc': 'roc_auc', 'average_precision': 'average_precision'}
    if card_precision_top_k_scorer:
        scoring[f'card_precision@{args.top_k_value}'] = card_precision_top_k_scorer
    performance_metrics_list_grid = list(scoring.keys())
    performance_metrics_list = ['AUC ROC', 'Average precision']
    if card_precision_top_k_scorer: performance_metrics_list.append(f'Card Precision@{args.top_k_value}')
    print(f"[GRID SEARCH] Scoring metrics for GridSearchCV: {performance_metrics_list_grid}")

    # Run model selection wrapper FOR XGBOOST ONLY
    performances_df_xgb = pd.DataFrame()
    print("[GRID SEARCH] Starting model_selection_wrapper for XGBoost...")
    try:
        performances_df_xgb = model_selection_wrapper(
            transactions_df, # Use the loaded window
            classifier_xgb, INPUT_FEATURES, OUTPUT_FEATURE,
            parameters_xgb, scoring,
            start_date_validation, start_date_test_estimation, # Use DERIVED dates
            n_folds=args.n_folds,
            delta_train=args.delta_train, delta_delay=args.delta_delay, delta_assessment=args.delta_assessment,
            performance_metrics_list_grid=performance_metrics_list_grid,
            performance_metrics_list=performance_metrics_list, n_jobs=args.n_jobs
        )
        selection_time = time.time() - start_selection_time
        print(f"[GRID SEARCH] XGBoost GridSearchCV finished in {selection_time:.2f} seconds.")
        mlflow.log_metric("xgb_gridsearch_time_sec", selection_time)

        if not performances_df_xgb.empty:
             print(f"[GRID SEARCH] GridSearchCV results shape: {performances_df_xgb.shape}")
             perf_artifact_path = "xgboost_grid_search_results.csv"
             performances_df_xgb.to_csv(perf_artifact_path, index=False)
             mlflow.log_artifact(perf_artifact_path)
             print(f"[GRID SEARCH] Logged XGBoost grid search results to MLflow artifact: {perf_artifact_path}")
             mlflow.set_tag("XGBoost Grid Search Status", "Completed - Success")
        else:
             print("[GRID SEARCH] Warning: XGBoost grid search returned empty results.")
             mlflow.set_tag("XGBoost Grid Search Status", "Completed - No Results")

    except Exception as e:
        print(f"[GRID SEARCH] ERROR during XGBoost GridSearchCV: {e}")
        traceback.print_exc()
        mlflow.log_metric("xgb_gridsearch_time_sec", time.time() - start_selection_time)
        mlflow.set_tag("XGBoost Grid Search Status", f"Failed: {e}")
        # Continue to default params, but log the failure

    # --- Determine Best Parameters for XGBoost ---
    best_params_xgb_dict = None
    best_params_xgb_summary = "Defaults"
    primary_metric = 'Average precision' # As defined in original notebook
    print(f"\n[BEST PARAMS] Determining best XGBoost parameters based on Validation '{primary_metric}'...")

    if not performances_df_xgb.empty:
        try:
            # Ensure 'Parameters summary' column exists
            if 'Parameters summary' not in performances_df_xgb.columns:
                 if 'Parameters' in performances_df_xgb.columns:
                     def params_to_str_fallback(params):
                         try:
                             if isinstance(params, dict): items = [f"{k.split('__')[1]}={v}" for k, v in sorted(params.items())]; return ", ".join(items)
                             else: return str(params)
                         except Exception: return str(params)
                     performances_df_xgb['Parameters summary'] = performances_df_xgb['Parameters'].apply(params_to_str_fallback)
                     print("[BEST PARAMS] Created fallback 'Parameters summary' column.")
                 else: raise KeyError("Missing parameter summary columns.")

            summary_xgb = get_summary_performances(performances_df_xgb, parameter_column_name="Parameters summary")
            print("[BEST PARAMS] Generated performance summary:")
            print(summary_xgb) # Print the summary table

            if primary_metric in summary_xgb.columns:
                best_params_xgb_summary = summary_xgb.loc["Best estimated parameters", primary_metric]
                validation_perf_str = summary_xgb.loc["Validation performance", primary_metric]

                # Check if best params were actually found (not 'N/A')
                if best_params_xgb_summary != 'N/A':
                    best_row = performances_df_xgb[performances_df_xgb['Parameters summary'] == best_params_xgb_summary]
                    if not best_row.empty:
                        best_params_xgb_dict = best_row['Parameters'].iloc[0] # Get the actual dict
                        print(f"[BEST PARAMS] Found Best XGBoost parameters (Summary): {best_params_xgb_summary}")
                        print(f"[BEST PARAMS] Corresponding parameter dict: {best_params_xgb_dict}")
                        mlflow.set_tag("best_xgb_params_source", "GridSearch")
                        mlflow.log_param("best_xgb_params_summary", best_params_xgb_summary)
                        # Log the validation score for the best params
                        try:
                             if validation_perf_str not in ['N/A', 'NaN']:
                                 best_val_score = float(validation_perf_str.split('+/-')[0])
                                 mlflow.log_metric(f"best_validation_{primary_metric.lower().replace(' ','_')}", best_val_score)
                                 print(f"[BEST PARAMS] Best Validation {primary_metric}: {best_val_score}")
                             else: print(f"[BEST PARAMS] Best Validation {primary_metric} score is N/A or NaN.")
                        except Exception as score_e: print(f"[BEST PARAMS] Warning: Could not parse/log best validation score: {score_e}")
                    else: print(f"[BEST PARAMS] Warning: Could not find row matching best summary string '{best_params_xgb_summary}'. Using defaults.")
                else: print(f"[BEST PARAMS] Best parameters summary is 'N/A'. Using defaults.")
            else: print(f"[BEST PARAMS] Warning: Primary metric '{primary_metric}' not in summary. Using defaults.")
        except Exception as e:
            print(f"[BEST PARAMS] Error determining best XGBoost parameters: {e}. Using defaults.")
            traceback.print_exc()
    else:
        print("[BEST PARAMS] XGBoost grid search results empty. Using default parameters.")

    # Fallback to default parameters if needed
    if best_params_xgb_dict is None:
        best_params_xgb_dict = {
            'clf__max_depth': 6, 'clf__n_estimators': 100, 'clf__learning_rate': 0.3,
            'clf__random_state': 0, 'clf__n_jobs': 1, 'clf__verbosity': 0,
            'clf__use_label_encoder': False, 'clf__eval_metric': 'logloss' }
        # Regenerate summary string from defaults for consistency
        try: best_params_xgb_summary = ", ".join([f"{k.split('__')[1]}={v}" for k, v in sorted(best_params_xgb_dict.items())])
        except: best_params_xgb_summary = str(best_params_xgb_dict)
        print(f"[BEST PARAMS] Using default XGBoost parameters: {best_params_xgb_summary}")
        mlflow.set_tag("best_xgb_params_source", "Default")
        mlflow.log_param("best_xgb_params_summary", best_params_xgb_summary)

    # --- Train Final XGBoost Model ---
    print("\n===== Training Final XGBoost Model =====")

    # Prepare final training data split
    final_train_df = pd.DataFrame()
    final_test_df = pd.DataFrame()
    print(f"[FINAL SPLIT] Creating final train/test split using start date: {final_train_start_date.strftime('%Y-%m-%d')}")
    try:
        (final_train_df, final_test_df) = get_train_test_set(
            transactions_df, # Use the window loaded at the start
            start_date_training=final_train_start_date, # Use derived date
            delta_train=args.delta_train, delta_delay=args.delta_delay, delta_test=args.delta_assessment
        )
        print(f"[FINAL SPLIT] Final training set shape: {final_train_df.shape}")
        print(f"[FINAL SPLIT] Final test set shape: {final_test_df.shape}")
        mlflow.log_metric("final_train_rows", final_train_df.shape[0])
        mlflow.log_metric("final_test_rows", final_test_df.shape[0])

        # --- CRITICAL CHECK: Ensure final training data is not empty ---
        if final_train_df.empty:
            print("[FINAL SPLIT] ERROR: Final training set is empty after split. Cannot train model.")
            mlflow.set_tag("Training Status", "Failed - Empty Final Train Split")
            mlflow.end_run(status="FAILED")
            sys.exit(1) # Exit script with failure code
        else:
             print("[FINAL SPLIT] Final training set is not empty.")

    except Exception as e:
        print(f"[FINAL SPLIT] ERROR creating final train/test split: {e}")
        traceback.print_exc()
        mlflow.set_tag("Training Status", "Failed - Final Split Error")
        mlflow.end_run(status="FAILED")
        sys.exit(1) # Exit script with failure code

    # Create final pipeline
    print("[FINAL TRAIN] Creating final pipeline with selected/default parameters...")
    final_classifier_xgb = xgboost.XGBClassifier() # Base instance
    # Prepare parameters (remove 'clf__' prefix)
    final_params_xgb_filtered = {k.split('__', 1)[1]: v for k, v in best_params_xgb_dict.items() if k.startswith('clf__')}
    # Ensure boolean/int types are correct if coming from string logs/summaries
    if 'use_label_encoder' in final_params_xgb_filtered: final_params_xgb_filtered['use_label_encoder'] = bool(str(final_params_xgb_filtered['use_label_encoder']).lower() == 'true')
    if 'verbosity' in final_params_xgb_filtered: final_params_xgb_filtered['verbosity'] = int(final_params_xgb_filtered['verbosity'])
    # Set parameters
    try:
        final_classifier_xgb.set_params(**final_params_xgb_filtered)
        print(f"[FINAL TRAIN] Set final XGBoost params: {final_params_xgb_filtered}")
    except Exception as e:
        print(f"[FINAL TRAIN] Warning: Error setting final XGBoost params: {e}. Using defaults.")
        final_classifier_xgb = xgboost.XGBClassifier(random_state=0, use_label_encoder=False, eval_metric='logloss', n_jobs=1) # Reset to known defaults

    # Define the pipeline
    final_pipeline = sklearn.pipeline.Pipeline([
        ('scaler', sklearn.preprocessing.StandardScaler()), ('clf', final_classifier_xgb)
    ])
    print(f"[FINAL TRAIN] Final pipeline created: {final_pipeline}")

    # Fit final pipeline
    print("[FINAL TRAIN] Starting final model fitting...")
    start_fit_time = time.time()
    try:
        X_train_final = final_train_df[INPUT_FEATURES]
        y_train_final = final_train_df[OUTPUT_FEATURE]
        print(f"[FINAL TRAIN] Input features shape for fit: {X_train_final.shape}")
        print(f"[FINAL TRAIN] Output feature shape for fit: {y_train_final.shape}")

        # Check for NaNs before fitting
        if X_train_final.isnull().values.any():
            print("[FINAL TRAIN] Warning: NaNs detected in final training features before fitting. Scaler should handle or raise error.")
            # If explicit imputation is needed, add it here.

        final_pipeline.fit(X_train_final, y_train_final)
        final_fit_time = time.time() - start_fit_time
        print(f"[FINAL TRAIN] Final XGBoost model fitting completed successfully in {final_fit_time:.2f} seconds.")
        mlflow.log_metric("final_model_train_time_sec", final_fit_time)
        mlflow.set_tag("Final Model Training Status", "Success")

        # --- Log and Save Final Model ---
        print(f"[MODEL LOGGING] Attempting to log final pipeline to MLflow run...")
        print(f"[MODEL LOGGING] Target artifact path: 'model'")
        # print(f"[MODEL LOGGING] Target output path variable: {args.model_output}")
        try:
            mlflow.sklearn.log_model(
                sk_model=final_pipeline,
                artifact_path="model", # Standard artifact path name
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
            )
            print("[MODEL LOGGING] mlflow.sklearn.log_model call completed.")
            mlflow.set_tag("Model Logging Status", "Success")
            # Note: The actual saving to args.model_output happens automatically
            # when the pipeline step finishes and uploads the 'model' artifact.
            # print(f"[MODEL LOGGING] Model artifact will be available at pipeline output path: {args.model_output}")

        except Exception as log_e:
             print(f"[MODEL LOGGING] ERROR during mlflow.sklearn.log_model: {log_e}")
             traceback.print_exc()
             mlflow.set_tag("Model Logging Status", f"Failed: {log_e}")
             mlflow.set_tag("Training Status", "Failed - Model Logging Error")
             mlflow.end_run(status="FAILED")
             sys.exit(1) # Exit script with failure code

        # --- Save Final Test Data Split to Output Path ---
        print(f"[TEST DATA SAVE] Saving final test data split to output path: {args.test_data_output}")
        test_data_output_path = Path(args.test_data_output)
        test_data_output_path.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        test_data_file = test_data_output_path / "final_test_data.pkl"
        try:
            if not final_test_df.empty:
                final_test_df.to_pickle(test_data_file)
                print(f"[TEST DATA SAVE] Final test data saved successfully to {test_data_file}")
                mlflow.set_tag("Test Data Save Status", "Success")
            else:
                print("[TEST DATA SAVE] Warning: Final test data is empty, not saving file.")
                mlflow.set_tag("Test Data Save Status", "Skipped - Empty")
        except Exception as e:
            print(f"[TEST DATA SAVE] ERROR saving final test data: {e}")
            traceback.print_exc()
            mlflow.set_tag("Test Data Save Status", f"Failed: {e}")
            # Don't necessarily fail the whole run for this, but log it.

    except Exception as e:
        print(f"[FINAL TRAIN] ERROR fitting final XGBoost model: {e}")
        traceback.print_exc()
        mlflow.set_tag("Final Model Training Status", f"Failed: {e}")
        mlflow.set_tag("Training Status", "Failed - Final Fit Error")
        mlflow.end_run(status="FAILED")
        sys.exit(1) # Exit script with failure code

    # --- End of Script ---
    mlflow.set_tag("Training Status", "Completed Successfully") # If reached here
    mlflow.end_run()
    print(f"===== Training Script Finished Successfully (MLflow Run ID: {run_id}) =====")

if __name__ == "__main__":
    args = parse_args()
    main(args)