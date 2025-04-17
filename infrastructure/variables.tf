# --- Core Deployment Variables ---

variable "location" {
  type        = string
  description = "Azure region where all resources will be deployed."
  # Example default (consider removing or setting to a specific region)
  # default = "WestUS2"
}

variable "prefix" {
  type        = string
  description = "Prefix for resource names (e.g., project name, department)."
}

variable "postfix" {
  type        = string
  description = "Postfix for resource names (e.g., unique identifier, random string)."
}

variable "environment" {
  type        = string
  description = "Environment identifier (e.g., 'dev', 'staging', 'prod')."
}

# --- Feature Flags ---

variable "enable_aml_computecluster" {
  type        = bool
  description = "Flag to enable or disable the creation of a default AML compute cluster."
  default     = true
}

variable "enable_monitoring" {
  type        = bool
  description = "Flag to enable or disable the deployment of Azure Data Explorer resources for monitoring."
  default     = false
}

variable "enable_acr" {
  type        = bool
  description = "Flag to enable or disable the creation of an Azure Container Registry."
  default     = true # Recommended for custom environments
}

# --- Tagging Variables ---

variable "tag_owner" {
  type        = string
  description = "Value for the 'Owner' tag."
  default     = "MLOpsTeam"
}

variable "tag_project" {
  type        = string
  description = "Value for the 'Project' tag."
  default     = "FraudDetection"
}


# --- Monitoring Service Principal Credentials (only needed if enable_monitoring is true) ---
# These should be provided securely, e.g., via GitHub Secrets or Azure DevOps variable groups.

variable "monitoring_sp_client_id" {
  type        = string
  description = "The Client ID of the Service Principal for monitoring data ingestion."
  default     = "" # Default to empty, module handles conditional creation of secrets
  sensitive   = true
}

variable "monitoring_sp_client_secret" {
  type        = string
  description = "The Client Secret of the Service Principal for monitoring data ingestion."
  default     = "" # Default to empty
  sensitive   = true
}

variable "monitoring_sp_tenant_id" {
  type        = string
  description = "The Tenant ID of the Service Principal for monitoring data ingestion."
  default     = "" # Default to empty
  sensitive   = true
}

# --- Authentication Variables for Terraform Provider (Optional) ---
# It's generally better to rely on environment variables (ARM_CLIENT_ID, etc.)
# or Managed Identity / OIDC for authentication rather than defining these here.
# variable "subscription_id" {
#   type        = string
#   description = "Azure Subscription ID (if not using environment variables)."
# }
# variable "tenant_id" {
#   type        = string
#   description = "Azure Tenant ID (if not using environment variables)."
# }
# variable "client_id" {
#   type        = string
#   description = "Client ID for Azure Service Principal authentication (if not using environment variables)."
# }
# variable "client_secret" {
#   type        = string
#   description = "Client Secret for Azure Service Principal authentication (if not using environment variables)."
#   sensitive   = true
# }