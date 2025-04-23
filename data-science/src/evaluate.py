
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


from utils import (
    performance_assessment 

)

def parse_args():
    parser = argparse.ArgumentParser("evaluate")
    parser.add_argument("--model_input", type=str, help="Path to input model directory")
    parser.add_argument("--test_data", type=str, help="Path to the DIRECTORY containing the final test data file (final_test_data.pkl)") # Updated help text
    parser.add_argument("--evaluation_output", type=str, help="Path to save evaluation results")
    parser.add_argument("--model_name", type=str, help="Name of the model for registration comparison", default="fraud-detection-model")


    parser.add_argument("--deploy_threshold_metric", type=str, default="Average precision", help="Metric to check for deployment threshold")
    parser.add_argument("--deploy_threshold_value", type=float, default=0.65, help="Threshold value for deployment metric")

    args = parser.parse_args()
    return args

def main(args):
    mlflow.start_run()
    print("Evaluation script started")
    print(f"Args: {args}")

    evaluation_output_path = Path(args.evaluation_output)
    evaluation_output_path.mkdir(parents=True, exist_ok=True)


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


    print(f"Looking for final test data PKL inside directory: {args.test_data}")
    test_data_dir = Path(args.test_data)
    test_data_file = test_data_dir / "final_test_data.pkl" # Construct the full path


    test_df_final = pd.DataFrame() # Initialize
    if test_data_file.is_file():
        try:
            test_df_final = pd.read_pickle(test_data_file)
            if test_df_final.empty:
                print("Warning: Loaded test data file is empty.")
            else:

                print(f"Loaded final test set '{test_data_file.name}' with shape: {test_df_final.shape}")
            mlflow.log_metric("test_set_rows", test_df_final.shape[0])
            if not test_df_final.empty and 'TX_FRAUD' in test_df_final.columns:
                 mlflow.log_metric("test_set_fraud_rate", test_df_final['TX_FRAUD'].mean())

        except Exception as e:
            print(f"ERROR loading test data file {test_data_file}: {e}")
            mlflow.log_param("evaluation_error", f"Test data loading failed: {e}")
            mlflow.end_run(status="FAILED")
            return
    else:

        print(f"ERROR: Test data file not found at expected location: {test_data_file}")
        mlflow.log_param("evaluation_error", "Test data file not found.")
        mlflow.end_run(status="FAILED")
        return


    if test_df_final.empty:
        print("Warning: Final test set is empty. Skipping evaluation.")
        performance_metrics = pd.DataFrame(columns=['AUC ROC', 'Average precision', f'Card Precision@{100}'])
        performance_metrics.loc[0] = [np.nan, np.nan, np.nan]
        deploy_flag = 0 # Cannot evaluate, so do not deploy
    else:
        try:

            print("Making predictions on test set...")

            INPUT_FEATURES = ['TX_AMOUNT','TX_DURING_WEEKEND', 'TX_DURING_NIGHT', 'CUSTOMER_ID_NB_TX_1DAY_WINDOW',
                              'CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW', 'CUSTOMER_ID_NB_TX_7DAY_WINDOW',
                              'CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW', 'CUSTOMER_ID_NB_TX_30DAY_WINDOW',
                              'CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW', 'TERMINAL_ID_NB_TX_1DAY_WINDOW',
                              'TERMINAL_ID_RISK_1DAY_WINDOW', 'TERMINAL_ID_NB_TX_7DAY_WINDOW',
                              'TERMINAL_ID_RISK_7DAY_WINDOW', 'TERMINAL_ID_NB_TX_30DAY_WINDOW',
                              'TERMINAL_ID_RISK_30DAY_WINDOW']
            OUTPUT_FEATURE = "TX_FRAUD"


            missing_input_features = [col for col in INPUT_FEATURES if col not in test_df_final.columns]
            if missing_input_features:
                raise ValueError(f"Missing input features in test data: {missing_input_features}")
            if OUTPUT_FEATURE not in test_df_final.columns:
                raise ValueError(f"Missing output feature '{OUTPUT_FEATURE}' in test data")


            X_test = test_df_final[INPUT_FEATURES]
            y_test = test_df_final[OUTPUT_FEATURE] # Defined for clarity, used in performance_assessment



            if X_test.isnull().values.any():
                 print("Warning: NaNs detected in test features before prediction. Model's pipeline should handle imputation/scaling.")


                





            predictions = model.predict_proba(X_test)[:, 1]
            test_df_final['predictions'] = predictions # Add predictions column for assessment function
            print("Predictions generated.")


            print("Calculating performance metrics...")
            performance_metrics = performance_assessment(
                test_df_final, 
                output_feature=OUTPUT_FEATURE,
                prediction_feature='predictions',
                top_k_list=[100], 
                rounded=False 
            )
            print("Performance Metrics on Test Set:")
            print(performance_metrics)


            print("Logging metrics to MLflow...")
            for metric in performance_metrics.columns:
                value = performance_metrics[metric].iloc[0]
                if not pd.isna(value):

                    mlflow_metric_name = metric.lower().replace(" ", "_").replace("@", "_at_").replace("%","perc")
                    mlflow.log_metric(f"test_{mlflow_metric_name}", value)
            print("Metrics logged.")


            print("Determining deploy flag...")
            deploy_flag = 0

            if args.deploy_threshold_value >= 0 and args.deploy_threshold_metric in performance_metrics.columns:
                metric_value = performance_metrics[args.deploy_threshold_metric].iloc[0]

                if not pd.isna(metric_value) and metric_value >= args.deploy_threshold_value:
                    deploy_flag = 1
                    print(f"Deploy flag set to 1 ({args.deploy_threshold_metric} {metric_value:.4f} >= {args.deploy_threshold_value})")
                else:

                    reason = "NaN" if pd.isna(metric_value) else f"{metric_value:.4f} < {args.deploy_threshold_value}"
                    print(f"Deploy flag remains 0 ({args.deploy_threshold_metric} is {reason})")

            elif args.deploy_threshold_value < 0:
                 deploy_flag = 1
                 print("Deploy flag set to 1 (deploy_threshold_value is negative).")

            else:
                 reason = f"metric '{args.deploy_threshold_metric}' not found" if args.deploy_threshold_metric not in performance_metrics.columns else f"threshold is {args.deploy_threshold_value}"
                 print(f"Deploy flag remains 0 ({reason})")

            mlflow.log_metric("deploy_flag", deploy_flag)
            mlflow.log_param("deploy_threshold_metric", args.deploy_threshold_metric)
            mlflow.log_param("deploy_threshold_value", args.deploy_threshold_value)

        except Exception as e:
            print(f"ERROR during prediction or evaluation: {e}")
            import traceback; traceback.print_exc()
            mlflow.log_param("evaluation_error", f"Prediction/eval failed: {e}")

            performance_metrics = pd.DataFrame(columns=['AUC ROC', 'Average precision', f'Card Precision@{100}'])
            performance_metrics.loc[0] = [np.nan, np.nan, np.nan]
            deploy_flag = 0 
            mlflow.end_run(status="FAILED")


    print(f"Saving evaluation results to {evaluation_output_path}")
    metrics_file = evaluation_output_path / "test_performance_metrics.csv"
    try:
        performance_metrics.round(5).to_csv(metrics_file, index=False)
        mlflow.log_artifact(str(metrics_file))
        print(f"Metrics saved to {metrics_file}")
    except Exception as e: print(f"Error saving metrics file: {e}")

    deploy_flag_file = evaluation_output_path / "deploy_flag"
    try:
        with open(deploy_flag_file, 'w') as f: f.write(str(deploy_flag))
        mlflow.log_artifact(str(deploy_flag_file))
        print(f"Deploy flag ({deploy_flag}) saved to {deploy_flag_file}")
    except Exception as e: print(f"Error saving deploy flag file: {e}")

    mlflow.end_run()
    print("Evaluation script finished.")

if __name__ == "__main__":
    args = parse_args()
    main(args)