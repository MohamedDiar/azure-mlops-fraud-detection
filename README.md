# Azure MLOps - Fraud Detection Example

This repository provides a structured example of implementing an MLOps pipeline for a fraud detection machine learning model using Azure Machine Learning and GitHub Actions. It adapts the concepts from the [Azure MLOps (v2) solution accelerator](https://github.com/Azure/mlops-v2) and applies them to a specific fraud detection scenario based on provided notebooks.

## Project Structure

*   **`.github/workflows`**: Contains GitHub Actions workflows for:
    *   `tf-gha-deploy-infra.yml`: Deploying Azure infrastructure (AML Workspace, Storage, etc.) using Terraform.
    *   `deploy-model-training-pipeline.yml`: Orchestrating the model training pipeline in Azure ML.
    *   `deploy-online-endpoint-pipeline.yml`: Deploying the trained model to a managed online endpoint.
    *   `deploy-batch-endpoint-pipeline.yml`: Deploying the trained model to a batch endpoint.
*   **`data-science`**: Holds the core data science code.
    *   `environment/train-conda.yml`: Conda environment definition for training.
    *   `src/`: Python scripts for individual pipeline steps (`prep.py`, `train.py`, `evaluate.py`, `register.py`) and shared utilities (`utils.py`).
*   **`infrastructure`**: Contains Terraform modules and configuration for deploying Azure resources.
*   **`mlops/azureml`**: Azure Machine Learning specific configurations.
    *   `train/`: Definitions for the training pipeline (`pipeline.yml`), data asset (`data.yml`), and environment (`environment.yml`).
    *   `deploy/`: Definitions for online and batch endpoints and deployments.
*   **`config-infra-prod.yml`**: Configuration file defining resource names, locations, and feature flags for the production environment infrastructure.

## Scenario Overview

The pipeline implements the following steps:

1.  **Data Preparation (`prep.py`)**:
    *   Reads raw daily transaction data (pickle files).
    *   Applies feature transformations:
        *   Date/Time features (weekend, night).
        *   Customer spending behavior (aggregates over time windows).
        *   Terminal risk scores (aggregates over time windows with delay).
    *   Saves the transformed data back into daily pickle files.
2.  **Model Training (`train.py`)**:
    *   Loads the transformed data.
    *   Performs model selection using prequential cross-validation **specifically for a Decision Tree classifier** (as requested). Hyperparameter tuning uses GridSearchCV.
    *   Trains the final Decision Tree model on the designated training split using the best found hyperparameters (or defaults).
    *   Logs the trained model artifact using MLflow.
    *   Saves the defined test data split for the evaluation step.
3.  **Model Evaluation (`evaluate.py`)**:
    *   Loads the trained model artifact.
    *   Loads the test data split.
    *   Calculates performance metrics (AUC ROC, Average Precision, Card Precision@k).
    *   Determines a deployment flag based on whether a primary metric (e.g., Average Precision) meets a predefined threshold.
    *   Logs metrics to MLflow.
4.  **Model Registration (`register.py`)**:
    *   Checks the deployment flag from the evaluation step.
    *   If the flag is set, registers the trained model artifact in the Azure ML Model Registry.
    *   Outputs model registration information (name and version).

## Prerequisites

1.  **Azure Subscription**: Access to an Azure subscription.
2.  **Azure ML Workspace**: An existing Azure Machine Learning workspace (or deploy one using the `tf-gha-deploy-infra.yml` workflow).
3.  **GitHub Repository**: A GitHub repository based on this template.
4.  **Service Principal**: An Azure Service Principal with `Contributor` rights on the subscription (or target resource group).
5.  **GitHub Secrets**: Configure the following secrets in your GitHub repository settings:
    *   `AZURE_CREDENTIALS`: The JSON output from creating the service principal (`az ad sp create-for-rbac --role Contributor --sdk-auth`).
    *   `ARM_CLIENT_ID`: The Client ID of the service principal.
    *   `ARM_CLIENT_SECRET`: The Client Secret of the service principal.
    *   `ARM_SUBSCRIPTION_ID`: Your Azure Subscription ID.
    *   `ARM_TENANT_ID`: Your Azure Tenant ID.
6.  **Terraform State Storage**: An Azure Storage Account and container created beforehand to store Terraform state files (configure names in `config-infra-prod.yml`).
7.  **Raw Data**: Upload your raw daily transaction pickle files to the location specified in `mlops/azureml/train/data.yml` (default: `raw-fraud-data` folder in the default workspace datastore).

## Getting Started

1.  **Fork/Clone**: Fork or clone this repository.
2.  **Configure**:
    *   Update `config-infra-prod.yml` with your desired `namespace`, `postfix`, `location`, and Terraform backend details.
    *   Review `mlops/azureml/train/data.yml` and ensure the `path` points to your raw data location in Azure ML.
    *   Set up the required GitHub Secrets.
    *   Ensure the Terraform state storage account/container exists.
3.  **Deploy Infrastructure (Optional)**: Run the `tf-gha-deploy-infra` workflow from the GitHub Actions tab to provision Azure resources.
4.  **Run Training Pipeline**: Run the `deploy-model-training-pipeline` workflow from the GitHub Actions tab. This will execute the prep, train, evaluate, and register steps in Azure ML.
5.  **Deploy Model (Optional)**:
    *   Run the `deploy-online-endpoint-pipeline` workflow to deploy the registered model to a real-time endpoint.
    *   Run the `deploy-batch-endpoint-pipeline` workflow to deploy the registered model to a batch scoring endpoint.

## Customization

*   **Data Source**: Modify `mlops/azureml/train/data.yml` if your raw data format or location differs.
*   **Feature Engineering**: Update `data-science/src/prep.py` and potentially `data-science/src/utils.py` to change feature transformations.
*   **Model Selection**: Modify `data-science/src/train.py` to include different classifiers or change the hyperparameter grids. Remember the current version is hardcoded for Decision Tree grid search only.
*   **Evaluation**: Adjust metrics or the deployment logic in `data-science/src/evaluate.py`.
*   **Environment**: Update `data-science/environment/train-conda.yml` to add or remove dependencies.
*   **Infrastructure**: Modify Terraform files in the `infrastructure` directory if different Azure resources are needed.