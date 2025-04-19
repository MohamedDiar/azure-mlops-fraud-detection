# data-science/src/register.py
import argparse
from pathlib import Path
import pickle
import json
import os
import mlflow
import mlflow.sklearn
import traceback # Import traceback for logging detailed errors

def parse_args():
    parser = argparse.ArgumentParser("register")
    # --- CORRECTED: model_path argument will now receive the MODEL URI string ---
    parser.add_argument('--model_path', type=str, help='URI of the trained model artifact (e.g., runs:/... or azureml://...)')
    parser.add_argument('--model_name', type=str, help='Name under which model will be registered', default="fraud-detection-model")
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
            mlflow.log_param("deploy_flag_status", "read_error")
    else:
        print(f"Warning: Deploy flag file not found at {deploy_flag_file}. Defaulting to 0.")
        mlflow.log_param("deploy_flag_status", "not_found")


    mlflow.log_metric("deploy_flag_read", deploy_flag)

    # --- Register Model Condition ---
    # Set deploy_flag=1 to force registration for testing/simplicity if needed
    # deploy_flag = 1
    # print(f"Overriding deploy_flag to: {deploy_flag}")

    if deploy_flag == 1:
        # --- CORRECTED: Use model URI directly ---
        # The model_path argument now contains the URI string passed from the pipeline
        model_uri = args.model_path
        print(f"Attempting to register model '{args.model_name}' from URI '{model_uri}'...")

        try:
            # Optional: Check if URI starts with expected prefixes
            if not (model_uri.startswith("runs:/") or model_uri.startswith("azureml://")):
                print(f"Warning: Model URI '{model_uri}' doesn't look like a standard MLflow run artifact or Azure ML artifact URI.")

            print(f"Registering model from URI: {model_uri}")
            registered_model = mlflow.register_model(
                model_uri=model_uri, # Pass the URI directly
                name=args.model_name
            )
            model_version = registered_model.version
            print(f"Successfully registered model '{args.model_name}' version {model_version}")
            mlflow.log_param("registered_model_name", args.model_name)
            mlflow.log_param("registered_model_version", model_version)
            mlflow.log_param("registration_status", "succeeded")

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
                 mlflow.log_param("model_info_status", "write_error")


        # --- UPDATED EXCEPTION HANDLING ---
        except Exception as e:
            error_message = f"ERROR during model registration: {e}"
            print(error_message)

            # Log short status as parameter
            mlflow.log_param("registration_status", "failed")

            # Create a file for the full error log
            error_log_path = Path("./registration_error.log")
            try:
                with open(error_log_path, "w") as f:
                    f.write(error_message)
                    # Also include traceback
                    f.write("\n\nTraceback:\n")
                    traceback.print_exc(file=f)
                mlflow.log_artifact(str(error_log_path))
                print(f"Full registration error logged to artifact: {error_log_path.name}")
            except Exception as log_e:
                print(f"Warning: Failed to log registration error artifact: {log_e}")
                # Fallback: Log truncated error as param if artifact logging fails
                # This might still fail if the original error 'e' string itself is huge,
                # but it's better than trying to log the full traceback as param.
                mlflow.log_param("registration_error_short", error_message[:490] + "...")

            # Fail the run explicitly
            mlflow.end_run(status="FAILED")
            return # Stop execution

    else:
        print("Deploy flag is 0. Model will not be registered.")
        mlflow.log_param("registration_status", "skipped_deploy_flag_0")
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
             mlflow.log_param("model_info_status", "write_error_empty")

    mlflow.end_run()
    print("Register script finished.")

if __name__ == "__main__":
    args = parse_args()
    main(args)