data "azurerm_client_config" "current" {}

locals {
  # Key Vault name constraints: 3-24 alphanumeric chars and hyphens, start/end with alphanumeric
  safe_prefix = lower(replace(var.prefix, "_", "-")) # Replace underscores just in case
  safe_postfix = lower(var.postfix)
  safe_env = lower(var.env)
  # Construct base name
  kv_base_name = "kv-${safe_prefix}-${safe_postfix}${safe_env}"
  # Ensure length and character constraints (simple truncation example)
  kv_name = substr(replace(lower(kv_base_name), "/[^a-z0-9-]/", ""), 0, 24)
}


resource "azurerm_key_vault" "kv" {
  name                       = local.kv_name
  location                   = var.location
  resource_group_name        = var.rg_name
  tenant_id                  = data.azurerm_client_config.current.tenant_id
  sku_name                   = var.kv_sku # e.g., "standard" or "premium"

  # Network ACLs (optional, default allows public access)
  # public_network_access_enabled = true # or false
  # network_acls {
  #   bypass         = "AzureServices"
  #   default_action = "Deny" # or "Allow"
  #   ip_rules       = [] # List of allowed IPs/CIDRs
  #   virtual_network_subnet_ids = [] # List of allowed subnet IDs
  # }

  # Access Policies (consider using RBAC via `azurerm_role_assignment` for more granular control)
  # This policy grants the Terraform runner/user permissions. Add policies for other principals as needed.
  access_policy {
    tenant_id = data.azurerm_client_config.current.tenant_id
    object_id = data.azurerm_client_config.current.object_id # The identity running Terraform

    # Adjust permissions as needed for the Terraform runner
    key_permissions = [
      "Get",
      "List",
      # "Create", "Delete", "Purge", "Recover" # Add if Terraform manages keys directly
    ]
    secret_permissions = [
      "Get",
      "List",
      "Set",    # Needed if Terraform modules write secrets (like ADX module does)
      # "Delete", "Purge", "Recover" # Add if Terraform manages secrets directly
    ]
    certificate_permissions = [
      "Get",
      "List",
      # "Create", "Import", "Delete", "Purge", "Recover" # Add if Terraform manages certificates
    ]
  }

  # Add other access policies if required, e.g., for the AML Workspace MSI
  # access_policy {
  #   tenant_id = data.azurerm_client_config.current.tenant_id # Assuming same tenant
  #   object_id = var.aml_workspace_principal_id # Pass the MSI principal ID as a variable
  #
  #   secret_permissions = [
  #     "Get",
  #     "List",
  #   ]
  #   # Add other permissions as needed
  # }

  # Soft delete and purge protection are recommended
  enabled_for_deployment          = true # Allows VMs to retrieve certs if needed
  enabled_for_disk_encryption     = true # Allows OS disk encryption key storage
  enabled_for_template_deployment = true # Allows ARM templates to retrieve secrets
  soft_delete_retention_days      = 7    # Or desired retention (7-90)
  enable_rbac_authorization       = false # Set to true if *only* using RBAC and not access policies

  tags = var.tags
}