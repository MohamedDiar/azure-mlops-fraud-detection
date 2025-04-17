# Define common tags and potentially other local variables

locals {
  common_tags = {
    Owner       = var.tag_owner       # Example: Get owner from variable
    Project     = var.tag_project     # Example: Get project from variable
    Environment = var.environment     # Use environment variable
    ManagedBy   = "Terraform"         # Indicate infrastructure management tool
    Automation  = "GitHubActions"     # Indicate the CI/CD tool used
  }
}

# You can define other locals here, for example, complex naming conventions:
# locals {
#   resource_group_name = "rg-${var.prefix}-${var.postfix}-${var.environment}"
#   # ... other resource names
# }