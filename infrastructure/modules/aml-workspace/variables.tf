variable "rg_name" {
  type        = string
  description = "Resource group name where the workspace will be created."
}

variable "location" {
  type        = string
  description = "Azure region where the workspace will be created."
}

variable "prefix" {
  type        = string
  description = "Prefix used for naming the workspace and potentially related resources."
}

variable "postfix" {
  type        = string
  description = "Postfix used for naming the workspace and potentially related resources."
}

variable "env" {
  type        = string
  description = "Environment identifier (e.g., 'dev', 'prod') used in naming."
}

variable "tags" {
  type        = map(string)
  default     = {}
  description = "A map of tags to assign to the Azure Machine Learning workspace."
}

# Linked Resource IDs
variable "storage_account_id" {
  type        = string
  description = "The resource ID of the primary Azure Storage Account linked to the AML workspace."
}

variable "key_vault_id" {
  type        = string
  description = "The resource ID of the Azure Key Vault linked to the AML workspace."
}

variable "application_insights_id" {
  type        = string
  description = "The resource ID of the Azure Application Insights instance linked to the AML workspace."
}

variable "container_registry_id" {
  type        = string
  description = "The resource ID of the Azure Container Registry linked to the AML workspace (optional)."
  default     = ""
}

# Feature Flags
variable "enable_aml_computecluster" {
  type        = bool
  description = "Flag to enable or disable the creation of a default compute cluster."
  default     = true
}

# variable "enable_aml_secure_workspace" {
#   type        = bool
#   description = "Flag to configure network settings for a secure workspace (e.g., disable public access)."
#   default     = false # Default to public access for simplicity
# }

# Compute Cluster Configuration (only used if enable_aml_computecluster is true)
variable "compute_cluster_name" {
  type        = string
  description = "Name for the compute cluster."
  default     = "cpu-cluster"
}

variable "compute_cluster_vm_size" {
  type        = string
  description = "VM size for the compute cluster nodes."
  default     = "Standard_DS3_v2"
}

variable "compute_cluster_vm_priority" {
  type        = string
  description = "VM priority for the compute cluster ('Dedicated' or 'LowPriority')."
  default     = "LowPriority"
}

variable "compute_cluster_min_nodes" {
  type        = number
  description = "Minimum number of nodes for the compute cluster."
  default     = 0
}

variable "compute_cluster_max_nodes" {
  type        = number
  description = "Maximum number of nodes for the compute cluster."
  default     = 4
}

variable "compute_cluster_idle_shutdown" {
  type        = string
  description = "Idle time before scaling down nodes (ISO 8601 duration format, e.g., 'PT5M' for 5 minutes)."
  default     = "PT10M" # Default to 10 minutes
}

variable "compute_cluster_subnet_id" {
  type        = string
  description = "Resource ID of the subnet to deploy the compute cluster into (optional)."
  default     = ""
}

# Removed storage_account_name as ID is sufficient and preferred
# variable "storage_account_name" {
#   type        = string
#   description = "The Name of the Storage Account linked to AML workspace (used for datastore creation via ARM)"
# }