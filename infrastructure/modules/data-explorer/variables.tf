variable "enable_monitoring" {
  type        = bool
  description = "Flag to enable or disable the deployment of Azure Data Explorer resources for monitoring."
  default     = false
}

variable "rg_name" {
  type        = string
  description = "The name of the resource group where ADX resources will be created."
}

variable "location" {
  type        = string
  description = "The Azure region where ADX resources will be created."
}

variable "prefix" {
  type        = string
  description = "Prefix used in the naming convention for ADX resources."
}

variable "postfix" {
  type        = string
  description = "Postfix used in the naming convention for ADX resources."
}

variable "env" {
  type        = string
  description = "Environment identifier (e.g., 'dev', 'prod') used in the naming convention."
}

variable "tags" {
  type        = map(string)
  default     = {}
  description = "A map of tags to assign to the ADX resources."
}

# ADX Configuration (only relevant if enable_monitoring is true)
variable "adx_sku_name" {
  type        = string
  description = "The SKU name for the Kusto cluster (e.g., 'Dev(No SLA)_Standard_D11_v2', 'Standard_D11_v2')."
  default     = "Dev(No SLA)_Standard_D11_v2" # Use Dev SKU for cost savings by default
}

variable "adx_sku_capacity" {
  type        = number
  description = "The number of instances for the Kusto cluster SKU."
  default     = 1
}

variable "adx_database_name" {
  type        = string
  description = "The name for the Kusto database."
  default     = "mlmonitoring"
}

# Key Vault ID for storing secrets
variable "key_vault_id" {
  type        = string
  description = "The Resource ID of the Key Vault where ADX connection secrets will be stored."
  # No default, must be provided if monitoring is enabled
}

# Monitoring Service Principal Credentials (required by monitoring agent/collector)
# These should be passed securely, e.g., via environment variables or pipeline secrets.
variable "monitoring_sp_client_id" {
  type        = string
  description = "The Client ID of the Service Principal used for monitoring data ingestion."
  default     = ""
  sensitive   = true
}

variable "monitoring_sp_client_secret" {
  type        = string
  description = "The Client Secret of the Service Principal used for monitoring data ingestion."
  default     = ""
  sensitive   = true
}

variable "monitoring_sp_tenant_id" {
  type        = string
  description = "The Tenant ID of the Service Principal used for monitoring data ingestion."
  default     = "" # If empty, will attempt to use the current Terraform context's tenant ID
  sensitive   = true
}

# Optional: Object ID for role assignment (if assigning roles via Terraform)
# variable "monitoring_sp_object_id" {
#   type        = string
#   description = "The Object ID (not Client ID) of the Service Principal used for monitoring data ingestion (required for ADX role assignment)."
#   default     = ""
#   sensitive   = true
# }


# Removed client_secret variable as it's now specific (monitoring_sp_client_secret)
# variable "client_secret" {
#   description = "client secret" # Too generic
#   default     = false
# }