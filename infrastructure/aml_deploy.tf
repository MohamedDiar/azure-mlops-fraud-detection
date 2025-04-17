# Main deployment orchestration for MLOps infrastructure

# Resource group - Foundation for all resources
module "resource_group" {
  source   = "./modules/resource-group"
  location = var.location
  prefix   = var.prefix
  postfix  = var.postfix
  env      = var.environment
  tags     = local.common_tags
}

# Storage account - Required for AML workspace
module "storage_account_aml" {
  source = "./modules/storage-account"

  rg_name  = module.resource_group.name
  location = module.resource_group.location
  prefix   = var.prefix
  postfix  = var.postfix
  env      = var.environment

  account_tier             = "Standard"
  account_replication_type = "LRS" # Local redundancy usually sufficient for AML
  account_kind             = "StorageV2"
  hns_enabled              = false # Typically false for default AML storage

  enable_firewall           = false # Keep public by default for simplicity
  # firewall_default_action = "Deny"
  # firewall_bypass         = ["AzureServices"]

  tags = local.common_tags
}

# Key vault - Required for AML workspace
module "key_vault" {
  source = "./modules/key-vault"

  rg_name  = module.resource_group.name
  location = module.resource_group.location
  prefix   = var.prefix
  postfix  = var.postfix
  env      = var.environment
  kv_sku   = "standard"

  tags = local.common_tags
}

# Application insights - Required for AML workspace
module "application_insights" {
  source = "./modules/application-insights"

  rg_name  = module.resource_group.name
  location = module.resource_group.location
  prefix   = var.prefix
  postfix  = var.postfix
  env      = var.environment

  tags = local.common_tags
}

# Container registry (Optional but recommended for custom environments)
module "container_registry" {
  source = "./modules/container-registry"
  count = var.enable_acr ? 1 : 0 # Only create if enabled

  rg_name           = module.resource_group.name
  location          = module.resource_group.location
  prefix            = var.prefix
  postfix           = var.postfix
  env               = var.environment
  acr_sku           = "Standard"
  acr_admin_enabled = true # Consider false for production

  tags = local.common_tags
}

# Azure Machine Learning workspace - Core MLOps resource
module "aml_workspace" {
  source = "./modules/aml-workspace"

  rg_name                 = module.resource_group.name
  location                = module.resource_group.location
  prefix                  = var.prefix
  postfix                 = var.postfix
  env                     = var.environment

  # Link dependent resources
  storage_account_id      = module.storage_account_aml.id
  key_vault_id            = module.key_vault.id
  application_insights_id = module.application_insights.id
  container_registry_id   = var.enable_acr ? module.container_registry[0].id : "" # Link ACR if created

  # Optional compute cluster creation
  enable_aml_computecluster = var.enable_aml_computecluster
  compute_cluster_name      = "cpu-cluster" # Customize if needed
  compute_cluster_vm_size   = "Standard_DS3_v2" # Customize if needed
  # Add other compute cluster vars if defaults in module are not suitable

  tags = local.common_tags

  depends_on = [
    module.storage_account_aml,
    module.key_vault,
    module.application_insights,
    module.container_registry # Add dependency even if count is 0
  ]
}

# Azure Data Explorer (Kusto) resources for monitoring (Optional)
module "data_explorer" {
  source = "./modules/data-explorer"
  count = var.enable_monitoring ? 1 : 0 # Only create if enabled

  enable_monitoring = var.enable_monitoring # Pass the flag down
  rg_name           = module.resource_group.name
  location          = module.resource_group.location
  prefix            = var.prefix
  postfix           = var.postfix
  env               = var.environment
  key_vault_id      = module.key_vault.id # For storing connection secrets

  # ADX SKU configuration (can be customized via root variables if needed)
  # adx_sku_name     = "Dev(No SLA)_Standard_D11_v2"
  # adx_sku_capacity = 1

  # Monitoring SP credentials (passed from root variables)
  monitoring_sp_client_id     = var.monitoring_sp_client_id
  monitoring_sp_client_secret = var.monitoring_sp_client_secret
  monitoring_sp_tenant_id     = var.monitoring_sp_tenant_id

  tags = local.common_tags

  depends_on = [
    module.key_vault # Depends on Key Vault for storing secrets
  ]
}