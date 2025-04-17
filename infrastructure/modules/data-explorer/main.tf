# Note: This module deploys Azure Data Explorer (Kusto) resources,
# which are primarily used for the optional monitoring solution
# based on AzureML-Observability. It is only deployed if
# the `enable_monitoring` variable is set to true.

data "azurerm_client_config" "current" {}

locals {
  # Ensure names are compliant (alphanumeric, lowercase, length limits)
  safe_prefix  = lower(replace(var.prefix, "-", ""))
  safe_postfix = lower(replace(var.postfix, "-", ""))
  safe_env     = lower(var.env)
  # Kusto cluster name: 4-22 chars, lowercase letters/numbers
  adx_cluster_base_name = "adx${local.safe_prefix}${local.safe_postfix}${local.safe_env}"
  adx_cluster_name      = substr(local.adx_cluster_base_name, 0, min(length(local.adx_cluster_base_name), 22))
  # Database name: 1-260 chars
  adx_database_name = var.adx_database_name != "" ? var.adx_database_name : "mlmonitoring"
}

resource "azurerm_kusto_cluster" "adx_cluster" {
  count               = var.enable_monitoring ? 1 : 0 # Only create if monitoring is enabled

  name                = local.adx_cluster_name
  location            = var.location
  resource_group_name = var.rg_name

  sku {
    name     = var.adx_sku_name     # e.g., "Dev(No SLA)_Standard_D11_v2" or "Standard_D11_v2"
    capacity = var.adx_sku_capacity # e.g., 1 or 2
  }

  # Enable features needed by AzureML-Observability
  enable_streaming_ingest = true
  enable_purge            = true # Often useful for managing monitoring data
  # enable_double_encryption = false # Optional

  # Optional: Configure networking, identity, etc.
  # public_network_access_enabled = true
  # identity { type = "SystemAssigned" }

  tags = var.tags
}

resource "azurerm_kusto_database" "adx_database" {
  count               = var.enable_monitoring ? 1 : 0 # Only create if monitoring is enabled

  name                = local.adx_database_name
  resource_group_name = var.rg_name
  location            = var.location # Must be same location as cluster
  cluster_name        = azurerm_kusto_cluster.adx_cluster[0].name

  # Optional: Define retention and caching policies
  # hot_cache_period = "P7D" # 7 days
  # soft_delete_period = "P30D" # 30 days

  tags = var.tags
}

# Store connection details in Key Vault if monitoring is enabled
resource "azurerm_key_vault_secret" "adx_uri_secret" {
  count = var.enable_monitoring ? 1 : 0

  name         = "kvmonitoringadxuri" # Consistent name from example
  value        = azurerm_kusto_cluster.adx_cluster[0].uri
  key_vault_id = var.key_vault_id

  tags = {
    "Description" = "ADX Cluster Ingest URI for Monitoring"
  }
}

resource "azurerm_key_vault_secret" "adx_db_secret" {
  count = var.enable_monitoring ? 1 : 0

  name         = "kvmonitoringadxdb" # Consistent name from example
  value        = azurerm_kusto_database.adx_database[0].name
  key_vault_id = var.key_vault_id

  tags = {
    "Description" = "ADX Database Name for Monitoring"
  }
}

# Secrets for Service Principal (SP) used by the monitoring agent/collector
# Ensure the SP whose details are provided via variables has appropriate access to ADX.
resource "azurerm_key_vault_secret" "monitoring_sp_id_secret" {
  count = var.enable_monitoring && var.monitoring_sp_client_id != "" ? 1 : 0

  name         = "kvmonitoringspid" # Consistent name from example
  value        = var.monitoring_sp_client_id
  key_vault_id = var.key_vault_id

  tags = {
    "Description" = "Monitoring Service Principal Client ID"
  }
}

resource "azurerm_key_vault_secret" "monitoring_sp_key_secret" {
  count = var.enable_monitoring && var.monitoring_sp_client_secret != "" ? 1 : 0

  name         = "kvmonitoringspkey" # Consistent name from example
  value        = var.monitoring_sp_client_secret
  key_vault_id = var.key_vault_id

  tags = {
    "Description" = "Monitoring Service Principal Client Secret"
  }
  # Mark sensitive value
  lifecycle {
    ignore_changes = [value] # Or prevent_destroy if secret shouldn't be deleted
  }
}

resource "azurerm_key_vault_secret" "monitoring_sp_tenant_secret" {
  count = var.enable_monitoring && var.monitoring_sp_tenant_id != "" ? 1 : 0

  name         = "kvmonitoringadxtenantid" # Consistent name from example
  value        = var.monitoring_sp_tenant_id != "" ? var.monitoring_sp_tenant_id : data.azurerm_client_config.current.tenant_id # Use provided or current tenant
  key_vault_id = var.key_vault_id

  tags = {
    "Description" = "Monitoring Service Principal Tenant ID"
  }
}

# Example Role Assignment: Give the Monitoring SP access to ingest data into the ADX database
# Requires the SP object ID, which is often harder to get dynamically in TF.
# Consider assigning roles outside Terraform or using managed identity if possible.
# resource "azurerm_kusto_database_principal_assignment" "monitoring_sp_ingestor" {
#   count = var.enable_monitoring && var.monitoring_sp_object_id != "" ? 1 : 0
#
#   name                = "MonitoringSPIngestorAssignment"
#   resource_group_name = var.rg_name
#   cluster_name        = azurerm_kusto_cluster.adx_cluster[0].name
#   database_name       = azurerm_kusto_database.adx_database[0].name
#
#   tenant_id    = var.monitoring_sp_tenant_id != "" ? var.monitoring_sp_tenant_id : data.azurerm_client_config.current.tenant_id
#   principal_id = var.monitoring_sp_object_id # MUST provide the Object ID (not Client ID)
#   principal_type = "App"
#   role           = "Ingestor"
# }