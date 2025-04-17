variable "rg_name" {
  type        = string
  description = "The name of the resource group where the Key Vault will be created."
}

variable "location" {
  type        = string
  description = "The Azure region where the Key Vault will be created."
}

variable "prefix" {
  type        = string
  description = "Prefix used in the naming convention for the Key Vault resource."
}

variable "postfix" {
  type        = string
  description = "Postfix used in the naming convention for the Key Vault resource."
}

variable "env" {
  type        = string
  description = "Environment identifier (e.g., 'dev', 'prod') used in the naming convention."
}

variable "tags" {
  type        = map(string)
  default     = {}
  description = "A map of tags to assign to the Key Vault resource."
}

variable "kv_sku" {
  type        = string
  description = "The SKU (tier) of the Key Vault ('standard' or 'premium')."
  default     = "standard"
  validation {
    condition     = contains(["standard", "premium"], var.kv_sku)
    error_message = "Allowed values for kv_sku are 'standard', 'premium'."
  }
}

# Optional: If assigning RBAC roles or access policies to other principals
# variable "aml_workspace_principal_id" {
#   type        = string
#   description = "The Principal ID of the Azure ML Workspace Managed Identity (if granting access)."
#   default     = ""
# }