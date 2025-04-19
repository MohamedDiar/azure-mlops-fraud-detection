# data-science/src/register.py
import argparse
from pathlib import Path
import pickle # Keep pickle import if needed elsewhere, though not directly used here for registration
import json
import os
import traceback # Import traceback for better error details if needed

import mlflow
import mlflow.sklearn

def parse_args():
    parser = argparse.ArgumentParser("register")
    parser.add_argument('--model_name', type=str, help='Name under which model will be registered', default="fraud-detection-model")
    # Input model_path is the LOCAL path where the artifact was downloaded by AML
    parser.add_argument('--model_path', type=str, help='Path to trained model directory (local path in job)')
    parser.add_argument('--evaluation_output', type=str, help='Path of evaluation results directory (contains deploy_flag)')
    parser.add_argument('--model_info_output_path', type=str, help="Path to write model info JSON")
    args, _ = parser.parse_known_args()
    print(f'Arguments: {args}')
    return args

def main(args):
    # --- Start MLflow Run ---
    # An MLflow run is automatically started when a pipeline step executes.
    # Using mlflow.start_run() is often redundant here but doesn't hurt.
    # We'll get the active run ID later.
    mlflow.start_run()
    active_run = mlflow.active_run()
    if active_run:
        run_id = active_run.info.run_id
        print(f"Register script started (MLflow Run ID: {run_id})")
    else:
        print("Register script started (WARNING: No active MLflow run found!)")
        run_id = None # Handle cases where run might not be active

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
    # Uncomment below to force registration for testing
    # deploy_flag = 1
    # print(f"Overriding deploy_flag to: {deploy_flag}")

    if deploy_flag == 1:
        print(f"Attempting to register model '{args.model_name}'...")
        print(f"Model artifact downloaded by AML to local path: '{args.model_path}'")

        try:
            # --- FIX: Construct the correct model URI ---
            # Method: Use the artifact logged by the *train_model* step.
            # We assume the train_model step logged the model using mlflow.sklearn.save_model
            # which implicitly logs it to MLflow under args.model_name within *that* step's run.
            # The input args.model_path IS the locally downloaded artifact.
            # We need to register using a valid MLflow URI. The most reliable way
            # in this pattern is often to log it *again* within this step's run,
            # or construct the 'runs:/...' URI relative to the *training* run.
            # Let's try the re-logging approach first, as it's simpler contextually.

            # 1. Load the model from the local path provided by AML input binding
            print(f"Loading model from local path: {args.model_path}")
            # Check if it's a directory and contains MLmodel
            local_model_path = Path(args.model_path)
            if not local_model_path.is_dir() or not (local_model_path / "MLmodel").exists():
                 raise ValueError(f"Provided model_path '{args.model_path}' is not a valid MLflow model directory.")
            # No need to load the model object itself unless validation is desired

            # NOTE: We are NOT re-logging here. The pipeline definition itself
            # implies the input 'model_path' *is* the model. We just need to tell
            # mlflow.register_model the *correct* reference URI format.
            # The value passed via `${{parent.jobs.train_model.outputs.model_output}}`
            # *should* be interpretable by the backend if the pipeline is set up correctly.
            # The error indicates the value being passed *to the script* is the mount path.

            # --- REVISED FIX: Use the input path directly but ensure it's treated correctly ---
            # The error message implies it expects azureml:// format for file sources.
            # However, the standard practice is to use runs:/<run_id>/path
            # Let's try constructing the runs:/ URI based on the CURRENT run_id, assuming
            # the model was logged within the *training* step, and the artifact path is just the model name.
            # This is brittle as it assumes the training step's artifact path.

            # --- SAFEST FIX (adopted from original template logic): Log Within This Step ---
            # This avoids needing info from the parent run.
            print("Loading model object for re-logging...")
            model_object = mlflow.sklearn.load_model(args.model_path)

            print(f"Logging model '{args.model_name}' within registration step's run ({run_id})...")
            mlflow.sklearn.log_model(
                 sk_model=model_object,
                 artifact_path=args.model_name # Log it under the model name within this run's artifacts
            )
            print("Model logged.")

            # Construct the URI based on the *current* run's ID and the artifact path we just used
            model_uri_for_registration = f'runs:/{run_id}/{args.model_name}'
            print(f"Constructed URI for registration: {model_uri_for_registration}")

            # 2. Register the model using the correctly formatted URI
            print(f"Registering model using URI: {model_uri_for_registration}")
            registered_model = mlflow.register_model(
                model_uri=model_uri_for_registration, # Use the constructed URI
                name=args.model_name
            )
            model_version = registered_model.version
            print(f"Successfully registered model '{args.model_name}' version {model_version}")
            mlflow.log_param("registered_model_name", args.model_name)
            mlflow.log_param("registered_model_version", model_version)

            # 3. Write model info JSON output
            print("Writing model info JSON...")
            model_info = {"id": f"{args.model_name}:{model_version}"}
            output_dir = Path(args.model_info_output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "model_info.json"
            try:
                with open(output_path, "w") as of:
                    json.dump(model_info, fp=of, indent=4)
                print(f"Model info saved to {output_path}")
                # Optionally log this JSON as an artifact of the registration step
                mlflow.log_artifact(str(output_path))
            except Exception as json_e:
                 print(f"Error writing model info JSON: {json_e}")
                 # Don't fail the whole step for this minor error

        except Exception as e:
            print(f"ERROR during model registration: {e}")
            # --- FIX for secondary error: Truncate the error message ---
            error_message = str(e)
            max_len = 499 # Max length for MLflow param value is 500
            truncated_error_message = (error_message[:max_len] + '...') if len(error_message) > max_len else error_message
            print(f"Logging truncated error: {truncated_error_message}")
            try:
                mlflow.log_param("registration_error", truncated_error_message)
            except Exception as log_err:
                print(f"Could not log registration error parameter: {log_err}")
            # Optionally, fail the run explicitly
            mlflow.end_run(status="FAILED")
            print("Register script failed.")
            # Use raise to ensure the step fails in Azure ML
            raise e # Re-raise the original exception to fail the step

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