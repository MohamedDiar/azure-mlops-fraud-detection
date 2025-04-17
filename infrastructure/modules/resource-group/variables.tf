variable "location" {
  type        = string
  # No default - location should be specified at the root module level
  description = "The Azure region where the resource group will be created."
}

variable "prefix" {
  type        = string
  description = "Prefix used in the naming convention for the resource group."
}

variable "postfix" {
  type        = string
  description = "Postfix used in the naming convention for the resource group."
}

variable "env" {
  type        = string
  description = "Environment identifier (e.g., 'dev', 'prod') used in the naming convention."
}

variable "tags" {
  type        = map(string)
  default     = {}
  description = "A map of tags to assign to the Resource Group."
}