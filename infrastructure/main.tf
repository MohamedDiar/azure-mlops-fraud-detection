terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      # Pinning the provider version is recommended for stability
      # Check latest compatible version: https://registry.terraform.io/providers/hashicorp/azurerm/latest
      version = "~> 3.0" # Example: Use any 3.x version
    }
    # Add other providers if needed (e.g., random, time)
    # random = {
    #   source = "hashicorp/random"
    #   version = "~> 3.1"
    # }
  }

  # Configure the Azure Remote Backend for storing Terraform state
  # Ensure the storage account, container, and access key/SAS token/identity are configured correctly.
  backend "azurerm" {
    # These backend configurations are typically placeholders here.
    # They are initialized during the CI/CD pipeline run (e.g., `terraform init -backend-config=...`)
    # using values from the `config-infra-*.yml` file or environment variables/secrets.

    # resource_group_name  = "PLACEHOLDER_RG_NAME_FOR_TFSTATE"
    # storage_account_name = "PLACEHOLDER_STORAGE_ACCOUNT_NAME_FOR_TFSTATE"
    # container_name       = "PLACEHOLDER_CONTAINER_NAME_FOR_TFSTATE" # e.g., "tfstate"
    # key                  = "PLACEHOLDER_STATE_FILE_NAME.tfstate" # e.g., "fraud-detection-prod.tfstate"

    # Recommended: Use Managed Identity or Service Principal authentication for the backend
    # use_oidc = true # For GitHub Actions OIDC
    # or
    # use_msi = true # If running Terraform from an Azure resource with Managed Identity
    # or SP details (less secure if hardcoded):
    # subscription_id = "PLACEHOLDER_SUBSCRIPTION_ID"
    # tenant_id = "PLACEHOLDER_TENANT_ID"
    # client_id = "PLACEHOLDER_CLIENT_ID"
    # client_secret = "PLACEHOLDER_CLIENT_SECRET" # Should be injected securely
  }
}

# Configure the Azure Provider
provider "azurerm" {
  features {}

  # Authentication for the provider (separate from the backend)
  # This is typically handled by environment variables set by `az login` or CI/CD service connection secrets.
  # Examples (DO NOT HARDCODE CREDENTIALS):
  # subscription_id = var.subscription_id # From input variables or env vars
  # tenant_id       = var.tenant_id       # From input variables or env vars
  # client_id       = var.client_id       # From input variables or env vars
  # client_secret   = var.client_secret   # From input variables or env vars (sensitive)
  # Or using Managed Identity:
  # use_msi = true
  # Or using OIDC (e.g., with GitHub Actions):
  # use_oidc = true
}

# Optional: Data source to get information about the current Azure subscription/client
# data "azurerm_client_config" "current" {}
# data "azurerm_subscription" "primary" {}