output "id" {
  description = "The Resource ID of the Azure Container Registry."
  value       = azurerm_container_registry.cr.id
}

output "name" {
  description = "The name of the Azure Container Registry."
  value       = azurerm_container_registry.cr.name
}

output "login_server" {
  description = "The login server hostname of the Azure Container Registry."
  value       = azurerm_container_registry.cr.login_server
}

output "admin_username" {
  description = "The admin username for the Azure Container Registry (only available if admin_enabled is true)."
  value       = var.acr_admin_enabled ? azurerm_container_registry.cr.admin_username : "N/A (Admin Disabled)"
  sensitive   = true
}

output "admin_password" {
  description = "The admin password for the Azure Container Registry (only available if admin_enabled is true)."
  value       = var.acr_admin_enabled ? azurerm_container_registry.cr.admin_password : "N/A (Admin Disabled)"
  sensitive   = true
}