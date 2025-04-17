output "name" {
  description = "The name of the resource group."
  value       = azurerm_resource_group.rg.name
}

output "location" {
  description = "The Azure region of the resource group."
  value       = azurerm_resource_group.rg.location
}

output "id" {
  description = "The Resource ID of the resource group."
  value       = azurerm_resource_group.rg.id
}