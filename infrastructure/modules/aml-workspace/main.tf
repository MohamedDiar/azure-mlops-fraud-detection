resource "azurerm_machine_learning_workspace" "mlw" {
  name                    = "mlw-${var.prefix}-${var.postfix}${var.env}"
  location                = var.location
  resource_group_name     = var.rg_name
  application_insights_id = var.application_insights_id
  key_vault_id            = var.key_vault_id
  storage_account_id      = var.storage_account_id
  container_registry_id   = var.container_registry_id != "" ? var.container_registry_id : null # Optional CR

  sku_name = "Basic" # Or "Enterprise" if needed

  # Use System Assigned Identity for simplicity, User Assigned is also possible
  identity {
    type = "SystemAssigned"
  }

  # Optionally configure Networking (Public is default)
  # public_network_access_enabled = !var.enable_aml_secure_workspace
  # primary_user_assigned_identity_id = (condition) ? value : null # If using UAI

  # Optionally configure encryption, managed vnet, etc.
  # encryption { ... }
  # managed_network { ... }

  tags = var.tags
}

# Compute cluster (Optional, can be created via SDK/CLI/UI later too)
resource "azurerm_machine_learning_compute_cluster" "aml_compute_cluster" {
  # Example name, can be customized
  name                          = var.compute_cluster_name != "" ? var.compute_cluster_name : "cpu-cluster"
  location                      = var.location # Should be same region as workspace
  vm_priority                   = var.compute_cluster_vm_priority # LowPriority or Dedicated
  vm_size                       = var.compute_cluster_vm_size     # e.g., Standard_DS3_v2
  machine_learning_workspace_id = azurerm_machine_learning_workspace.mlw.id

  count = var.enable_aml_computecluster ? 1 : 0 # Only create if enabled

  subnet_resource_id = var.compute_cluster_subnet_id != "" ? var.compute_cluster_subnet_id : null # Optional: Deploy into VNet

  scale_settings {
    min_node_count                       = var.compute_cluster_min_nodes
    max_node_count                       = var.compute_cluster_max_nodes
    scale_down_nodes_after_idle_duration = var.compute_cluster_idle_shutdown # e.g., "PT5M" for 5 minutes
  }

  # Optional: Use identity for accessing resources if needed (e.g., secure datastores)
  # identity {
  #   type = "SystemAssigned" # or "UserAssigned"
  #   identity_ids = [...] # If UserAssigned
  # }

  tags = var.tags
}

# Add role assignments if necessary (e.g., give Workspace MSI access to storage/keyvault)
# Example: Assign Storage Blob Data Contributor to Workspace MSI on the default storage account
# resource "azurerm_role_assignment" "mlw_msi_storage_access" {
#   scope                = var.storage_account_id
#   role_definition_name = "Storage Blob Data Contributor"
#   principal_id         = azurerm_machine_learning_workspace.mlw.identity[0].principal_id
# }