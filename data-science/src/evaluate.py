# data-science/src/evaluate.py
import os
import argparse
import datetime
from pathlib import Path
import pickle
import json
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

# Import shared functions from utils.py
from utils import (
    performance_assessment,
    get_class_from_fraud_probability,
    threshold_based_metrics,
    card_precision_top_k,
    get_train_test_set # Needed to regenerate the exact test set
)

def parse_args():
    parser = argparse.ArgumentParser("evaluate")
    parser.add_argument("--model_input", type=str, help="Path to input model directory")
    parser.add_argument("--transformed_data", type=str, help="Path to folder containing transformed data files (.pkl)")
    parser.add_argument("--evaluation_output", type=str, help="Path to save evaluation results")
    parser.add_argument("--model_name", type=str, help="Name of the model for registration comparison", default="fraud-detection-model")

    # Parameters needed to recreate the specific test split used during training
    parser.add_argument("--final_train_start_date_str", type=str, help="Start date of the final training period (YYYY-MM-DD)")
    parser.add_argument("--delta_train", type=int, help="Duration of training period in days")
    parser.add_argument("--delta_delay", type=int, help="Duration of delay period in days")
    parser.add_argument("--delta_test", type=int, help="Duration of test period in days")

    # Threshold for deploy flag based on a primary metric (e.g., Average Precision)
    # Set to -1 to always deploy, or a value like 0.7
    parser.add_argument("--deploy_threshold_metric", type=str, default="Average precision", help="Metric to check for deployment threshold")
    parser.add_argument("--deploy_threshold_value", type=float, default=0.7, help="Threshold value for deployment metric")

    args = parser.parse_args()
    return args

def load_data_for_split(data_path, start_date_str, end_date_str):
    """Loads pickle files within a date range from the transformed data path."""
    all_files = sorted([f for f in Path(data_path).glob('*.pkl') if f.is_file()])
    target_files = [
        f for f in all_files
        if start_date_str <= f.stem <= end_date_str
    ]

    if not target_files:
        print(f"Warning: No transformed files found in {data_path} for range {start_date_str} to {end_date_str}")
        return pd.DataFrame()

    frames = [pd.read_pickle(f) for f in target_files]
    if not frames:
        return pd.DataFrame()

    df_final = pd.concat(frames, ignore_index=True)
    df_final = df_final.sort_values('TRANSACTION_ID').reset_index(drop=True)
    if 'TX_DATETIME' in df_final.columns and not pd.api.types.is_datetime64_any_dtype(df_final['TX_DATETIME']):
        df_final['TX_DATETIME'] = pd.to_datetime(df_final['TX_DATETIME'])
    return df_final

def main(args):
    mlflow.start_run() # Start MLflow run for this component

    print("Evaluation script started")
    print(f"Args: {args}")

    # Ensure output directory exists
    evaluation_output_path = Path(args.evaluation_output)
    evaluation_output_path.mkdir(parents=True, exist_ok=True)

    # --- Load Model ---
    print(f"Loading model from: {args.model_input}")
    try:
        model = mlflow.sklearn.load_model(args.model_input)
        print("Model loaded successfully.")
        mlflow.log_param("model_input_path", args.model_input)
    except Exception as e:
        print(f"ERROR: Failed to load model from {args.model_input}: {e}")
        mlflow.log_param("evaluation_error", f"Model loading failed: {e}")
        mlflow.end_run(status="FAILED")
        return

    # --- Load and Prepare Test Data ---
    print("Preparing test data...")
    try:
        final_train_start_date = datetime.datetime.strptime(args.final_train_start_date_str, "%Y-%m-%d")

        # Calculate the full date range needed for the split function
        # The split function needs data from the training start up to the end of the test period
        split_start_date = final_train_start_date
        split_end_date = final_train_start_date + datetime.timedelta(days=args.delta_train + args.delta_delay + args.delta_test -1) # inclusive end

        print(f"Loading transformed data from {split_start_date.strftime('%Y-%m-%d')} to {split_end_date.strftime('%Y-%m-%d')} for split generation...")
        transformed_df_full = load_data_for_split(
            args.transformed_data,
            split_start_date.strftime('%Y-%m-%d'),
            split_end_date.strftime('%Y-%m-%d')
        )

        if transformed_df_full.empty:
             raise ValueError("Failed to load any transformed data for the required date range.")

        print("Generating final test set using get_train_test_set...")
        _, test_df_final = get_train_test_set(
            transformed_df_full,
            start_date_training=final_train_start_date,
            delta_train=args.delta_train,
            delta_delay=args.delta_delay,
            delta_test=args.delta_test
        )

        if test_df_final.empty:
            print("Warning: Final test set is empty after split generation.")
            mlflow.log_metric("test_set_rows", 0)
            # Decide if this is acceptable or an error
            # For now, create an empty metrics file and set deploy_flag to 0
            performance_metrics = pd.DataFrame(columns=['AUC ROC', 'Average precision', f'Card Precision@{100}'])
            performance_metrics.loc[0] = [np.nan, np.nan, np.nan]
            deploy_flag = 0
        else:
            print(f"Final test set generated with shape: {test_df_final.shape}")
            mlflow.log_metric("test_set_rows", test_df_final.shape[0])
            mlflow.log_metric("test_set_fraud_rate", test_df_final['TX_FRAUD'].mean())

            # --- Get Predictions ---
            print("Making predictions on test set...")
            # Define input features (should match training)
            INPUT_FEATURES = ['TX_AMOUNT','TX_DURING_WEEKEND', 'TX_DURING_NIGHT', 'CUSTOMER_ID_NB_TX_1DAY_WINDOW',
                              'CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW', 'CUSTOMER_ID_NB_TX_7DAY_WINDOW',
                              'CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW', 'CUSTOMER_ID_NB_TX_30DAY_WINDOW',
                              'CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW', 'TERMINAL_ID_NB_TX_1DAY_WINDOW',
                              'TERMINAL_ID_RISK_1DAY_WINDOW', 'TERMINAL_ID_NB_TX_7DAY_WINDOW',
                              'TERMINAL_ID_RISK_7DAY_WINDOW', 'TERMINAL_ID_NB_TX_30DAY_WINDOW',
                              'TERMINAL_ID_RISK_30DAY_WINDOW']
            OUTPUT_FEATURE = "TX_FRAUD"

            X_test = test_df_final[INPUT_FEATURES]
            y_test = test_df_final[OUTPUT_FEATURE]

            # Handle potential NaNs in test data (use scaler/imputer from pipeline if possible)
            # Since we loaded the pipeline, it should handle scaling/imputation.
            # If NaNs occurred *after* transformation in prep step, they need handling here.
            if X_test.isnull().values.any():
                 print("Warning: NaNs detected in test features before prediction. Pipeline should handle this if imputer/scaler included.")
                 # If pipeline doesn't impute, apply mean imputation as a fallback:
                 # from sklearn.impute import SimpleImputer
                 # print("Applying fallback mean imputation to test set...")
                 # imputer = SimpleImputer(strategy='mean')
                 # X_test = imputer.fit_transform(X_test)
                 # X_test = pd.DataFrame(X_test, columns=INPUT_FEATURES, index=test_df_final.index)

            predictions = model.predict_proba(X_test)[:, 1]
            test_df_final['predictions'] = predictions
            print("Predictions generated.")

            # --- Evaluate Performance ---
            print("Calculating performance metrics...")
            # Use the performance_assessment function from utils
            performance_metrics = performance_assessment(
                test_df_final,
                output_feature=OUTPUT_FEATURE,
                prediction_feature='predictions',
                top_k_list=[100], # Or get K from args
                rounded=False # Get raw values
            )
            print("Performance Metrics on Test Set:")
            print(performance_metrics)

            # --- Log Metrics ---
            print("Logging metrics to MLflow...")
            for metric in performance_metrics.columns:
                value = performance_metrics[metric].iloc[0]
                if not pd.isna(value):
                    # Sanitize metric name for MLflow
                    mlflow_metric_name = metric.lower().replace(" ", "_").replace("@", "_at_")
                    mlflow.log_metric(f"test_{mlflow_metric_name}", value)
            print("Metrics logged.")

            # --- Determine Deploy Flag ---
            print("Determining deploy flag...")
            deploy_flag = 0 # Default to not deploying
            if args.deploy_threshold_value >= 0 and args.deploy_threshold_metric in performance_metrics.columns:
                metric_value = performance_metrics[args.deploy_threshold_metric].iloc[0]
                if not pd.isna(metric_value) and metric_value >= args.deploy_threshold_value:
                    deploy_flag = 1
                    print(f"Deploy flag set to 1 ({args.deploy_threshold_metric} {metric_value:.4f} >= {args.deploy_threshold_value})")
                else:
                    print(f"Deploy flag remains 0 ({args.deploy_threshold_metric} {metric_value:.4f} < {args.deploy_threshold_value} or NaN)")
            elif args.deploy_threshold_value < 0:
                 deploy_flag = 1 # Always deploy if threshold is negative
                 print("Deploy flag set to 1 (threshold is negative).")
            else:
                 print(f"Deploy flag remains 0 (metric '{args.deploy_threshold_metric}' not found or threshold <= 0)")

            mlflow.log_metric("deploy_flag", deploy_flag)
            mlflow.log_param("deploy_threshold_metric", args.deploy_threshold_metric)
            mlflow.log_param("deploy_threshold_value", args.deploy_threshold_value)

    except Exception as e:
        print(f"ERROR during data preparation or evaluation: {e}")
        import traceback
        traceback.print_exc()
        mlflow.log_param("evaluation_error", f"Data prep/eval failed: {e}")
        # Create empty metrics and set deploy flag to 0
        performance_metrics = pd.DataFrame(columns=['AUC ROC', 'Average precision', f'Card Precision@{100}'])
        performance_metrics.loc[0] = [np.nan, np.nan, np.nan]
        deploy_flag = 0
        mlflow.end_run(status="FAILED")
        # Still save the (empty) metrics and deploy flag files
        # return # Optional: Stop execution vs saving empty files

    # --- Save Evaluation Results ---
    print(f"Saving evaluation results to {evaluation_output_path}")
    # Save performance metrics
    metrics_file = evaluation_output_path / "test_performance_metrics.csv"
    try:
        performance_metrics.round(5).to_csv(metrics_file, index=False)
        mlflow.log_artifact(str(metrics_file))
        print(f"Metrics saved to {metrics_file}")
    except Exception as e:
        print(f"Error saving metrics file: {e}")

    # Save deploy flag
    deploy_flag_file = evaluation_output_path / "deploy_flag"
    try:
        with open(deploy_flag_file, 'w') as f:
            f.write(str(deploy_flag))
        mlflow.log_artifact(str(deploy_flag_file))
        print(f"Deploy flag ({deploy_flag}) saved to {deploy_flag_file}")
    except Exception as e:
        print(f"Error saving deploy flag file: {e}")

    # Optionally save predictions dataframe
    # predictions_file = evaluation_output_path / "test_predictions.csv"
    # try:
    #     if not test_df_final.empty:
    #         test_df_final[['TRANSACTION_ID', 'TX_DATETIME', 'CUSTOMER_ID', 'TX_FRAUD', 'predictions']].to_csv(predictions_file, index=False)
    #         mlflow.log_artifact(str(predictions_file))
    # except Exception as e:
    #      print(f"Error saving predictions file: {e}")

    mlflow.end_run()
    print("Evaluation script finished.")

if __name__ == "__main__":
    args = parse_args()
    main(args)