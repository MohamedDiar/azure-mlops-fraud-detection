variable "rg_name" {
  type        = string
  description = "The name of the resource group where the Container Registry will be created."
}

variable "location" {
  type        = string
  description = "The Azure region where the Container Registry will be created."
}

variable "prefix" {
  type        = string
  description = "Prefix used in the naming convention for the Container Registry resource."
}

variable "postfix" {
  type        = string
  description = "Postfix used in the naming convention for the Container Registry resource."
}

variable "env" {
  type        = string
  description = "Environment identifier (e.g., 'dev', 'prod') used in the naming convention."
}

variable "tags" {
  type        = map(string)
  default     = {}
  description = "A map of tags to assign to the Container Registry resource."
}

variable "acr_sku" {
  type        = string
  description = "The SKU (tier) of the Container Registry (e.g., 'Basic', 'Standard', 'Premium')."
  default     = "Standard"
  validation {
    condition     = contains(["Basic", "Standard", "Premium"], var.acr_sku)
    error_message = "Allowed values for acr_sku are 'Basic', 'Standard', 'Premium'."
  }
}

variable "acr_admin_enabled" {
  type        = bool
  description = "Enable the admin user account for the Container Registry."
  default     = true # Consider setting to false for production environments
}