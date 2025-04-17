# data-science/src/register.py
import argparse
from pathlib import Path
import pickle
import json
import os
import mlflow
import mlflow.sklearn

def parse_args():
    parser = argparse.ArgumentParser("register")
    parser.add_argument('--model_name', type=str, help='Name under which model will be registered', default="fraud-detection-model")
    parser.add_argument('--model_path', type=str, help='Path to trained model directory')
    parser.add_argument('--evaluation_output', type=str, help='Path of evaluation results directory (contains deploy_flag)')
    parser.add_argument('--model_info_output_path', type=str, help="Path to write model info JSON")
    args, _ = parser.parse_known_args()
    print(f'Arguments: {args}')
    return args

def main(args):
    mlflow.start_run() # Start MLflow run for this component
    print("Register script started")

    # --- Read Deploy Flag ---
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

    # --- Register Model Condition ---
    # Set deploy_flag=1 to force registration for testing/simplicity if needed
    # deploy_flag = 1
    # print(f"Overriding deploy_flag to: {deploy_flag}")

    if deploy_flag == 1:
        print(f"Attempting to register model '{args.model_name}' from path '{args.model_path}'...")
        try:
            # Load model (optional, but good practice to check loading)
            # model = mlflow.sklearn.load_model(args.model_path)
            # print("Model loaded successfully for verification.")

            # Register the model artifact logged in the parent run (usually from train step)
            # Assuming the model was logged as an artifact named 'model'
            # Need the parent run_id if this is a separate run, or use active run if logged here.
            # For simplicity in AML pipelines, assume model was logged by train step.
            # We reference the *path* passed as input, which *is* the logged model artifact dir.

            # The path passed to --model_path *is* the MLflow model directory
            model_uri = args.model_path

            # Check if URI looks like an MLflow artifact path (optional)
            if not Path(model_uri, "MLmodel").exists():
                 print(f"Warning: MLmodel file not found in {model_uri}. Registration might fail or use unexpected format.")

            print(f"Registering model from URI: {model_uri}")
            registered_model = mlflow.register_model(
                model_uri=model_uri,
                name=args.model_name
            )
            model_version = registered_model.version
            print(f"Successfully registered model '{args.model_name}' version {model_version}")
            mlflow.log_param("registered_model_name", args.model_name)
            mlflow.log_param("registered_model_version", model_version)

            # Write model info JSON output for downstream steps (like deployment workflows)
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
            except Exception as e:
                 print(f"Error writing model info JSON: {e}")

        except Exception as e:
            print(f"ERROR during model registration: {e}")
            mlflow.log_param("registration_error", str(e))
            # Optionally, fail the run explicitly
            mlflow.end_run(status="FAILED")
            return # Stop execution

    else:
        print("Deploy flag is 0. Model will not be registered.")
        # Create an empty model_info.json to avoid downstream errors if file is expected
        print("Creating empty model info JSON.")
        model_info = {"id": "None:0"} # Indicate no model registered
        output_dir = Path(args.model_info_output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "model_info.json"
        try:
            with open(output_path, "w") as of:
                json.dump(model_info, fp=of, indent=4)
            print(f"Empty model info saved to {output_path}")
            # mlflow.log_artifact(str(output_path)) # Optionally log the empty file
        except Exception as e:
             print(f"Error writing empty model info JSON: {e}")

    mlflow.end_run()
    print("Register script finished.")

if __name__ == "__main__":
    args = parse_args()
    main(args)