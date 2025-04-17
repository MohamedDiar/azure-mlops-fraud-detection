variable "rg_name" {
  type        = string
  description = "The name of the resource group where Application Insights will be created."
}

variable "location" {
  type        = string
  description = "The Azure region where Application Insights will be created."
}

variable "prefix" {
  type        = string
  description = "Prefix used in the naming convention for the Application Insights resource."
}

variable "postfix" {
  type        = string
  description = "Postfix used in the naming convention for the Application Insights resource."
}

variable "env" {
  type        = string
  description = "Environment identifier (e.g., 'dev', 'prod') used in the naming convention."
}

variable "tags" {
  type        = map(string)
  default     = {}
  description = "A map of tags to assign to the Application Insights resource."
}

# Optional: Link to Log Analytics Workspace
# variable "log_analytics_workspace_id" {
#   type        = string
#   description = "The Resource ID of the Log Analytics Workspace to link Application Insights to (optional)."
#   default     = null
# }