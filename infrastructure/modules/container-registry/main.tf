locals {
  # Ensure ACR name is alphanumeric and meets length constraints (3-50 chars)
  # Remove hyphens, ensure lowercase
  safe_prefix  = lower(replace(var.prefix, "-", ""))
  safe_postfix = lower(replace(var.postfix, "-", ""))
  safe_env     = lower(var.env)
  # Construct name and truncate if necessary
  acr_base_name = "cr${local.safe_prefix}${local.safe_postfix}${local.safe_env}"
  acr_name      = substr(local.acr_base_name, 0, min(length(local.acr_base_name), 50))
}

resource "azurerm_container_registry" "cr" {
  name                = local.acr_name
  resource_group_name = var.rg_name
  location            = var.location
  sku                 = var.acr_sku # e.g., "Basic", "Standard", "Premium"
  admin_enabled       = var.acr_admin_enabled # Usually true for simplicity in dev/test, consider false for prod

  # Optional: Configure network rules, encryption, identity, etc.
  # public_network_access_enabled = true
  # network_rule_set { ... }
  # encryption { ... }
  # identity { ... }

  tags = var.tags
}