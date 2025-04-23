# data-science/src/register.py
import argparse
from pathlib import Path
import pickle
import json
import os
import traceback

import mlflow
import mlflow.sklearn

def parse_args():
    parser = argparse.ArgumentParser("register")
    parser.add_argument('--model_name', type=str, help='Name under which model will be registered', default="fraud-detection-model")
    parser.add_argument('--model_path', type=str, help='Path to trained model directory (local path in job)')
    parser.add_argument('--evaluation_output', type=str, help='Path of evaluation results directory (contains deploy_flag)')
    parser.add_argument('--model_info_output_path', type=str, help="Path to write model info JSON")
    args, _ = parser.parse_known_args()
    print(f'Arguments: {args}')
    return args

def main(args):
    mlflow.start_run()
    active_run = mlflow.active_run()
    if active_run:
        run_id = active_run.info.run_id
        print(f"Register script started (MLflow Run ID: {run_id})")
    else:
        print("Register script started (WARNING: No active MLflow run found!)")
        run_id = None # Handle cases where run might not be active
    deploy_flag = 0 # Default to not deploying
    deploy_flag_file = Path(args.evaluation_output) / "deploy_flag"
    if deploy_flag_file.exists():
        try:
            with open(deploy_flag_file, 'r') as f:
                deploy_flag = int(f.read().strip())
            print(f"Read deploy_flag: {deploy_flag}")
        except Exception as e:
            print(f"Warning: Could not read deploy flag file at {deploy_flag_file}. Defaulting to 0. Error: {e}")
    else:
        print(f"Warning: Deploy flag file not found at {deploy_flag_file}. Defaulting to 0.")

    mlflow.log_metric("deploy_flag_read", deploy_flag)

    if deploy_flag == 1:
        print(f"Attempting to register model '{args.model_name}'...")
        print(f"Model artifact downloaded by AML to local path: '{args.model_path}'")

        try:
            print(f"Loading model from local path: {args.model_path}")
            local_model_path = Path(args.model_path)
            if not local_model_path.is_dir() or not (local_model_path / "MLmodel").exists():
                 raise ValueError(f"Provided model_path '{args.model_path}' is not a valid MLflow model directory.")
            print("Loading model object for re-logging...")
            model_object = mlflow.sklearn.load_model(args.model_path)

            print(f"Logging model '{args.model_name}' within registration step's run ({run_id})...")
            mlflow.sklearn.log_model(
                 sk_model=model_object,
                 artifact_path=args.model_name # Log it under the model name within this run's artifacts
            )
            print("Model logged.")
            model_uri_for_registration = f'runs:/{run_id}/{args.model_name}'
            print(f"Constructed URI for registration: {model_uri_for_registration}")
            print(f"Registering model using URI: {model_uri_for_registration}")
            registered_model = mlflow.register_model(
                model_uri=model_uri_for_registration, # Use the constructed URI
                name=args.model_name
            )
            model_version = registered_model.version
            print(f"Successfully registered model '{args.model_name}' version {model_version}")
            mlflow.log_param("registered_model_name", args.model_name)
            mlflow.log_param("registered_model_version", model_version)
            print("Writing model info JSON...")
            model_info = {"id": f"{args.model_name}:{model_version}"}
            output_dir = Path(args.model_info_output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "model_info.json"
            try:
                with open(output_path, "w") as of:
                    json.dump(model_info, fp=of, indent=4)
                print(f"Model info saved to {output_path}")
                mlflow.log_artifact(str(output_path))
            except Exception as json_e:
                 print(f"Error writing model info JSON: {json_e}")

        except Exception as e:
            print(f"ERROR during model registration: {e}")

            error_message = str(e)
            max_len = 499 # Max length for MLflow param value is 500
            truncated_error_message = (error_message[:max_len] + '...') if len(error_message) > max_len else error_message
            print(f"Logging truncated error: {truncated_error_message}")
            try:
                mlflow.log_param("registration_error", truncated_error_message)
            except Exception as log_err:
                print(f"Could not log registration error parameter: {log_err}")
            mlflow.end_run(status="FAILED")
            print("Register script failed.")

            raise e # Re-raise the original exception to fail the step

    else:
        print("Deploy flag is 0. Model will not be registered.")
        print("Creating empty model info JSON.")
        model_info = {"id": "None:0"} # Indicate no model registered
        output_dir = Path(args.model_info_output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "model_info.json"
        try:
            with open(output_path, "w") as of:
                json.dump(model_info, fp=of, indent=4)
            print(f"Empty model info saved to {output_path}")
        except Exception as e:
             print(f"Error writing empty model info JSON: {e}")

    mlflow.end_run()
    print("Register script finished.")

if __name__ == "__main__":
    args = parse_args()
    main(args)