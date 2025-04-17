data "azurerm_client_config" "current" {}
# Optional: Get current IP if needed for firewall rules (use cautiously)
# data "http" "ip" {
#   url = "https://ifconfig.me"
# }

locals {
  # Storage account names: 3-24 lowercase alphanumeric chars
  safe_prefix  = lower(replace(var.prefix, "-", ""))
  safe_postfix = lower(replace(var.postfix, "-", ""))
  safe_env     = lower(var.env)
  # Construct base name
  st_base_name = "st${local.safe_prefix}${local.safe_postfix}${local.safe_env}"
  # Ensure length constraints
  st_name      = substr(local.st_base_name, 0, min(length(local.st_base_name), 24))
}

resource "azurerm_storage_account" "st" {
  name                     = local.st_name
  resource_group_name      = var.rg_name
  location                 = var.location
  account_tier             = var.account_tier             # e.g., "Standard" or "Premium"
  account_replication_type = var.account_replication_type # e.g., "LRS", "GRS", "ZRS"
  account_kind             = var.account_kind             # e.g., "StorageV2", "BlobStorage"

  # Hierarchical Namespace (ADLS Gen2) - Enable if needed, especially for data lakes
  is_hns_enabled = var.hns_enabled

  # Security settings (recommended defaults)
  allow_nested_items_to_be_public = false # Disable public access for nested items
  # min_tls_version                 = "TLS1_2" # Enforce minimum TLS version

  # Optional: Configure Lifecycle Management, Encryption, Networking, etc.
  # blob_properties { ... }
  # network_rules { ... } - see below
  # identity { type = "SystemAssigned" } # Assign identity if needed for accessing other resources

  tags = var.tags
}

# Optional: Configure Network Rules / Firewall
resource "azurerm_storage_account_network_rules" "firewall_rules" {
  # Only configure if firewall rules are specified
  count = var.enable_firewall ? 1 : 0

  storage_account_id = azurerm_storage_account.st.id

  default_action             = var.firewall_default_action # "Allow" or "Deny"
  ip_rules                   = var.firewall_ip_rules       # List of allowed IP addresses/CIDRs
  virtual_network_subnet_ids = var.firewall_virtual_network_subnet_ids # List of allowed subnet IDs
  bypass                     = var.firewall_bypass         # e.g., ["AzureServices", "Logging", "Metrics"]

  # Optional: Define private link access rules
  # private_link_access { ... }
}