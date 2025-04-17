variable "rg_name" {
  type        = string
  description = "The name of the resource group where the Storage Account will be created."
}

variable "location" {
  type        = string
  description = "The Azure region where the Storage Account will be created."
}

variable "prefix" {
  type        = string
  description = "Prefix used in the naming convention for the Storage Account resource."
}

variable "postfix" {
  type        = string
  description = "Postfix used in the naming convention for the Storage Account resource."
}

variable "env" {
  type        = string
  description = "Environment identifier (e.g., 'dev', 'prod') used in the naming convention."
}

variable "tags" {
  type        = map(string)
  default     = {}
  description = "A map of tags to assign to the Storage Account resource."
}

# Storage Account Configuration
variable "account_tier" {
  type        = string
  description = "The tier of the Storage Account ('Standard' or 'Premium')."
  default     = "Standard"
  validation {
    condition     = contains(["Standard", "Premium"], var.account_tier)
    error_message = "Allowed values for account_tier are 'Standard', 'Premium'."
  }
}

variable "account_replication_type" {
  type        = string
  description = "The replication type ('LRS', 'GRS', 'RAGRS', 'ZRS', 'GZRS', 'RAGZRS')."
  default     = "LRS" # Local redundancy is often sufficient and cheapest for AML default storage
  validation {
    condition     = contains(["LRS", "GRS", "RAGRS", "ZRS", "GZRS", "RAGZRS"], var.account_replication_type)
    error_message = "Invalid account_replication_type specified."
  }
}

variable "account_kind" {
  type        = string
  description = "The kind of storage account ('StorageV2', 'BlobStorage', 'BlockBlobStorage', 'FileStorage', 'Storage')."
  default     = "StorageV2" # General purpose v2 is standard for AML
  validation {
    condition     = contains(["StorageV2", "BlobStorage", "BlockBlobStorage", "FileStorage", "Storage"], var.account_kind)
    error_message = "Invalid account_kind specified."
  }
}

variable "hns_enabled" {
  type        = bool
  description = "Enable Hierarchical Namespace (Data Lake Storage Gen2 features)."
  default     = false # Default to false unless ADLS features are specifically needed
}

# Firewall Configuration (Optional)
variable "enable_firewall" {
  type        = bool
  description = "Flag to enable network rules (firewall) for the storage account."
  default     = false # Default to public access
}

variable "firewall_default_action" {
  type        = string
  description = "Default action when no rules match ('Allow' or 'Deny'). Only used if enable_firewall is true."
  default     = "Deny"
  validation {
    condition     = var.enable_firewall ? contains(["Allow", "Deny"], var.firewall_default_action) : true
    error_message = "Allowed values for firewall_default_action are 'Allow', 'Deny'."
  }
}

variable "firewall_ip_rules" {
  type        = list(string)
  description = "List of IP addresses or CIDR ranges allowed access. Only used if enable_firewall is true."
  default     = []
}

variable "firewall_virtual_network_subnet_ids" {
  type        = list(string)
  description = "List of subnet resource IDs allowed access. Only used if enable_firewall is true."
  default     = []
}

variable "firewall_bypass" {
  type        = list(string)
  description = "Specifies which Azure services can bypass network rules ('None', 'Logging', 'Metrics', 'AzureServices'). Only used if enable_firewall is true."
  default     = ["AzureServices"] # Common setting to allow other Azure services (like AML) access
}