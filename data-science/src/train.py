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
    parser = argparse.ArgumentParser("train")
    parser.add_argument("--transformed_data", type=str, help="Path to folder containing ALL transformed data files (.pkl) from prep step")
    parser.add_argument("--model_output", type=str, help="Path to save output model artifact")
    parser.add_argument("--test_data_output", type=str, help="Path to save the final test data split (as folder containing pkl)")

    # --- CORRECTED DATE ARGUMENTS ---
    # Dates defining the specific window of TRANSFORMED data to load for this step
    parser.add_argument("--train_load_start_date", type=str, required=True, help="Start date of transformed data window to load (YYYY-MM-DD)")
    parser.add_argument("--train_load_end_date", type=str, required=True, help="End date of transformed data window to load (YYYY-MM-DD)")
    # Anchor date for splitting the loaded window
    parser.add_argument("--anchor_date_str", type=str, required=True, help="Anchor date for deriving training/validation splits (YYYY-MM-DD)")

    # --- DELTA ARGUMENTS ---
    parser.add_argument("--delta_train", type=int, default=7, help="Duration of training period in days")
    parser.add_argument("--delta_delay", type=int, default=7, help="Duration of delay period in days")
    parser.add_argument("--delta_assessment", type=int, default=7, help="Duration of assessment period in days")
    parser.add_argument("--n_folds", type=int, default=4, help="Number of folds for prequential validation")

    # --- Other Parameters ---
    parser.add_argument("--top_k_value", type=int, default=100, help="Value K for Card Precision@k")
    parser.add_argument("--n_jobs", type=int, default=5, help="Number of parallel jobs for GridSearchCV")

    args = parser.parse_args()
    return args

def load_transformed_data_window(data_path, start_date_str, end_date_str):
    """Loads transformed pickle files ONLY within the specified window."""
    print(f"Train: Loading transformed data from path '{data_path}' for window: {start_date_str} to {end_date_str}")
    all_files = sorted([f for f in Path(data_path).glob('*.pkl') if f.is_file()])
    target_files = [
        f for f in all_files
        if start_date_str <= f.stem <= end_date_str # Filter based on dates
    ]

    if not target_files:
        print(f"ERROR: No transformed files found in {data_path} for required training window {start_date_str} to {end_date_str}")
        return pd.DataFrame()

    print(f"Found {len(target_files)} transformed files for training window.")
    frames = []
    for f_path in target_files:
         try:
             df = pd.read_pickle(f_path)
             frames.append(df)
         except Exception as e:
              print(f"Error reading transformed file {f_path.name}: {e}")

    if not frames:
        print("No transformed dataframes were successfully loaded for the window.")
        return pd.DataFrame()

    df_final = pd.concat(frames, ignore_index=True)
    df_final = df_final.sort_values('TRANSACTION_ID').reset_index(drop=True)
    if 'TX_DATETIME' in df_final.columns and not pd.api.types.is_datetime64_any_dtype(df_final['TX_DATETIME']):
         df_final['TX_DATETIME'] = pd.to_datetime(df_final['TX_DATETIME'])
    return df_final

def main(args):
    mlflow.start_run()
    print("Training script started")
    print(f"Args: {args}")

    # Log parameters (including the newly passed dates)
    mlflow.log_params({k: v for k, v in vars(args).items() if k not in ['transformed_data', 'model_output', 'test_data_output']})
    # Explicitly log the train window dates passed as args
    mlflow.log_param("train_data_load_start", args.train_load_start_date)
    mlflow.log_param("train_data_load_end", args.train_load_end_date)


    # --- Load SPECIFIC Window of Transformed Data using arguments ---
    print(f"Loading specific window of transformed data ({args.train_load_start_date} to {args.train_load_end_date}) from input path: {args.transformed_data}")
    transactions_df = load_transformed_data_window(
        Path(args.transformed_data),
        args.train_load_start_date, # Use argument
        args.train_load_end_date    # Use argument
    )

    if transactions_df.empty:
        print(f"ERROR: No transformed data loaded for the required window {args.train_load_start_date} to {args.train_load_end_date}. Exiting.")
        mlflow.log_metric("transformed_rows_loaded_train", 0)
        mlflow.end_run(status="FAILED")
        return

    print(f"Loaded {len(transactions_df)} transformed transactions for training/selection window.")
    mlflow.log_metric("transformed_rows_loaded_train", len(transactions_df))

    # Define features and output
    OUTPUT_FEATURE = "TX_FRAUD"
    INPUT_FEATURES = ['TX_AMOUNT','TX_DURING_WEEKEND', 'TX_DURING_NIGHT', 'CUSTOMER_ID_NB_TX_1DAY_WINDOW',
                      'CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW', 'CUSTOMER_ID_NB_TX_7DAY_WINDOW',
                      'CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW', 'CUSTOMER_ID_NB_TX_30DAY_WINDOW',
                      'CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW', 'TERMINAL_ID_NB_TX_1DAY_WINDOW',
                      'TERMINAL_ID_RISK_1DAY_WINDOW', 'TERMINAL_ID_NB_TX_7DAY_WINDOW',
                      'TERMINAL_ID_RISK_7DAY_WINDOW', 'TERMINAL_ID_NB_TX_30DAY_WINDOW',
                      'TERMINAL_ID_RISK_30DAY_WINDOW']
    mlflow.log_param("input_features", json.dumps(INPUT_FEATURES))
    mlflow.log_param("output_feature", OUTPUT_FEATURE)


    # --- Derive Dates (Based on Anchor Date arg) ---
    try:
        start_date_training_anchor = datetime.datetime.strptime(args.anchor_date_str, "%Y-%m-%d")
        start_date_validation = start_date_training_anchor - datetime.timedelta(days=(args.delta_delay + args.delta_assessment))
        start_date_test_estimation = start_date_training_anchor
        final_train_start_date = start_date_training_anchor

        print(f"\nDerived Dates:")
        print(f"  Anchor Date: {start_date_training_anchor.strftime('%Y-%m-%d')}")
        print(f"  Validation GridSearch Start: {start_date_validation.strftime('%Y-%m-%d')}")
        print(f"  Test Estimation GridSearch Start: {start_date_test_estimation.strftime('%Y-%m-%d')}")
        print(f"  Final Training Start: {final_train_start_date.strftime('%Y-%m-%d')}")

        mlflow.log_param("derived_validation_start_date", start_date_validation.strftime('%Y-%m-%d'))
        mlflow.log_param("derived_test_estimation_start_date", start_date_test_estimation.strftime('%Y-%m-%d'))

    except ValueError as e:
        print(f"ERROR parsing anchor date '{args.anchor_date_str}': {e}")
        mlflow.end_run(status="FAILED")
        return


    # --- Model Selection (Grid Search for XGBoost ONLY) ---
    print("\n===== Starting Model Selection (XGBoost Only) =====")
    start_selection_time = time.time()

    classifier_xgb = xgboost.XGBClassifier(random_state=0, use_label_encoder=False, eval_metric='logloss', n_jobs=1)
    parameters_xgb = {
        'clf__max_depth': [3, 6, 9], 'clf__n_estimators': [25, 50, 100], 'clf__learning_rate': [0.1, 0.3],
        'clf__random_state':[0], 'clf__n_jobs':[1], 'clf__verbosity':[0],
        'clf__use_label_encoder':[False], 'clf__eval_metric':['logloss']
    }
    mlflow.log_param("xgboost_param_grid", json.dumps({k:str(v) for k,v in parameters_xgb.items()}))

    # Prepare scorer dataframe subset from the loaded window
    transactions_df_scorer = pd.DataFrame()
    try:
        scorer_cols = ['CUSTOMER_ID', 'TX_FRAUD', 'TX_TIME_DAYS']
        if not all(col in transactions_df.columns for col in scorer_cols):
            raise ValueError(f"Missing columns required for scorer DF: {scorer_cols}")
        transactions_df_scorer = transactions_df[scorer_cols].copy()
        print(f"Created scorer helper DataFrame with shape: {transactions_df_scorer.shape}")
    except Exception as e:
        print(f"ERROR creating scorer helper DF: {e}")
        mlflow.end_run(status="FAILED")
        return

    # Create custom scorer
    card_precision_top_k_scorer = None
    if not transactions_df_scorer.empty:
        try:
            card_precision_top_k_scorer = sklearn.metrics.make_scorer(
                card_precision_top_k_custom, needs_proba=True,
                top_k=args.top_k_value, transactions_df=transactions_df_scorer)
            print(f"Custom scorer 'card_precision@{args.top_k_value}' created.")
        except Exception as e:
            print(f"Warning: Failed to create custom scorer: {e}.")

    # Define scoring dictionary
    scoring = {'roc_auc': 'roc_auc', 'average_precision': 'average_precision'}
    if card_precision_top_k_scorer:
        scoring[f'card_precision@{args.top_k_value}'] = card_precision_top_k_scorer
    performance_metrics_list_grid = list(scoring.keys())
    performance_metrics_list = ['AUC ROC', 'Average precision']
    if card_precision_top_k_scorer: performance_metrics_list.append(f'Card Precision@{args.top_k_value}')

    # Run model selection wrapper FOR XGBOOST ONLY
    performances_df_xgb = pd.DataFrame()
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
        print(f"XGBoost GridSearchCV finished in {selection_time:.2f} seconds.")
        mlflow.log_metric("xgb_gridsearch_time_sec", selection_time)

        if not performances_df_xgb.empty:
             perf_artifact_path = "xgboost_grid_search_results.csv"
             performances_df_xgb.to_csv(perf_artifact_path, index=False)
             mlflow.log_artifact(perf_artifact_path)
             print(f"Logged XGBoost grid search results to {perf_artifact_path}")
        else:
             print("Warning: XGBoost grid search returned empty results.")
             mlflow.set_tag("XGBoost Grid Search Status", "Completed - No Results")

    except Exception as e:
        print(f"ERROR during XGBoost GridSearchCV: {e}")
        import traceback; traceback.print_exc()
        mlflow.log_metric("xgb_gridsearch_time_sec", time.time() - start_selection_time)
        mlflow.set_tag("XGBoost Grid Search Status", f"Failed: {e}")

    # --- Determine Best Parameters for XGBoost ---
    best_params_xgb_dict = None
    best_params_xgb_summary = "Defaults"
    primary_metric = 'Average precision'

    if not performances_df_xgb.empty:
        print("\nDetermining best XGBoost parameters...")
        try:
            if 'Parameters summary' not in performances_df_xgb.columns:
                 if 'Parameters' in performances_df_xgb.columns:
                     def params_to_str_fallback(params):
                         try:
                             if isinstance(params, dict): items = [f"{k.split('__')[1]}={v}" for k, v in sorted(params.items())]; return ", ".join(items)
                             else: return str(params)
                         except Exception: return str(params)
                     performances_df_xgb['Parameters summary'] = performances_df_xgb['Parameters'].apply(params_to_str_fallback)
                     print("Created fallback 'Parameters summary' column.")
                 else: raise KeyError("Missing parameter summary columns.")

            summary_xgb = get_summary_performances(performances_df_xgb, parameter_column_name="Parameters summary")

            if primary_metric in summary_xgb.columns:
                best_params_xgb_summary = summary_xgb.loc["Best estimated parameters", primary_metric]
                validation_perf_str = summary_xgb.loc["Validation performance", primary_metric]
                best_row = performances_df_xgb[performances_df_xgb['Parameters summary'] == best_params_xgb_summary]
                if not best_row.empty:
                    best_params_xgb_dict = best_row['Parameters'].iloc[0]
                    print(f"Best XGBoost parameters (Summary): {best_params_xgb_summary}")
                    mlflow.set_tag("best_xgb_params_source", "GridSearch")
                    mlflow.log_param("best_xgb_params_summary", best_params_xgb_summary)
                    try:
                         best_val_score = float(validation_perf_str.split('+/-')[0])
                         mlflow.log_metric(f"best_validation_{primary_metric.lower().replace(' ','_')}", best_val_score)
                    except: pass
                else: print(f"Warning: Could not find row matching best '{best_params_xgb_summary}'. Using defaults.")
            else: print(f"Warning: Primary metric '{primary_metric}' not in summary. Using defaults.")
        except Exception as e: print(f"Error determining best XGBoost parameters: {e}. Using defaults.")
    else: print("XGBoost grid search results empty. Using default parameters.")

    # Fallback to default parameters
    if best_params_xgb_dict is None:
        best_params_xgb_dict = {
            'clf__max_depth': 6, 'clf__n_estimators': 100, 'clf__learning_rate': 0.3,
            'clf__random_state': 0, 'clf__n_jobs': 1, 'clf__verbosity': 0,
            'clf__use_label_encoder': False, 'clf__eval_metric': 'logloss' }
        best_params_xgb_summary = "eval_metric=logloss, learning_rate=0.3, max_depth=6, n_estimators=100, n_jobs=1, random_state=0, use_label_encoder=False, verbosity=0"
        print(f"Using default XGBoost parameters: {best_params_xgb_summary}")
        mlflow.set_tag("best_xgb_params_source", "Default")
        mlflow.log_param("best_xgb_params_summary", best_params_xgb_summary)

    # --- Train Final XGBoost Model ---
    print("\n===== Training Final XGBoost Model =====")

    # Prepare final training data split using the DERIVED final_train_start_date
    final_train_df = pd.DataFrame()
    final_test_df = pd.DataFrame()
    try:
        # Use get_train_test_set on the ALREADY LOADED data window (transactions_df)
        (final_train_df, final_test_df) = get_train_test_set(
            transactions_df, # Use the window loaded at the start of this script
            start_date_training=final_train_start_date, # Use derived date
            delta_train=args.delta_train, delta_delay=args.delta_delay, delta_test=args.delta_assessment
        )
        if final_train_df.empty: raise ValueError("Final training set is empty after split.")
        print(f"Final training set shape: {final_train_df.shape}")
        print(f"Final test set shape: {final_test_df.shape}")
        mlflow.log_metric("final_train_rows", final_train_df.shape[0])
        mlflow.log_metric("final_test_rows", final_test_df.shape[0])
    except Exception as e:
        print(f"ERROR creating final train/test split: {e}")
        mlflow.end_run(status="FAILED")
        return

    # Create final pipeline
    final_classifier_xgb = xgboost.XGBClassifier()
    final_params_xgb_filtered = {k.split('__', 1)[1]: v for k, v in best_params_xgb_dict.items() if k.startswith('clf__')}
    if 'use_label_encoder' in final_params_xgb_filtered: final_params_xgb_filtered['use_label_encoder'] = bool(str(final_params_xgb_filtered['use_label_encoder']).lower() == 'true')
    if 'verbosity' in final_params_xgb_filtered: final_params_xgb_filtered['verbosity'] = int(final_params_xgb_filtered['verbosity'])
    try:
        final_classifier_xgb.set_params(**final_params_xgb_filtered)
        print(f"Set final XGBoost params: {final_params_xgb_filtered}")
    except Exception as e:
        print(f"Warning: Error setting final XGBoost params: {e}. Using defaults.")
        final_classifier_xgb = xgboost.XGBClassifier(random_state=0, use_label_encoder=False, eval_metric='logloss', n_jobs=1)

    final_pipeline = sklearn.pipeline.Pipeline([
        ('scaler', sklearn.preprocessing.StandardScaler()), ('clf', final_classifier_xgb)
    ])

    # Fit final pipeline
    start_fit_time = time.time()
    try:
        X_train_final = final_train_df[INPUT_FEATURES]
        y_train_final = final_train_df[OUTPUT_FEATURE]
        if X_train_final.isnull().values.any(): print("Warning: NaNs detected in final training features.")

        final_pipeline.fit(X_train_final, y_train_final)
        final_fit_time = time.time() - start_fit_time
        print(f"Final XGBoost model fitted in {final_fit_time:.2f} seconds.")
        mlflow.log_metric("final_model_train_time_sec", final_fit_time)

        # --- Log and Save Final Model ---
        print(f"Logging final model to MLflow run and preparing output path: {args.model_output}")
        mlflow.sklearn.log_model(
            sk_model=final_pipeline, artifact_path="model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE )
        mlflow.set_tag("Final Model Training Status", "Success")
        print(f"Model artifact expected at pipeline output path: {args.model_output}")

        # --- Save Final Test Data Split to Output Path ---
        print(f"Saving final test data split to output path: {args.test_data_output}")
        test_data_output_path = Path(args.test_data_output)
        test_data_output_path.mkdir(parents=True, exist_ok=True)
        test_data_file = test_data_output_path / "final_test_data.pkl"
        try:
            if not final_test_df.empty:
                final_test_df.to_pickle(test_data_file)
                print(f"Final test data saved to {test_data_file}")
            else: print("Warning: Final test data is empty, not saving file.")
        except Exception as e: print(f"ERROR saving final test data: {e}")

    except Exception as e:
        print(f"ERROR fitting final XGBoost model: {e}")
        import traceback; traceback.print_exc()
        mlflow.set_tag("Final Model Training Status", f"Failed: {e}")
        mlflow.end_run(status="FAILED")
        return

    mlflow.end_run()
    print("Training script finished.")

if __name__ == "__main__":
    args = parse_args()
    main(args)
