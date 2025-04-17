output "id" {
  description = "The Resource ID of the Key Vault."
  value       = azurerm_key_vault.kv.id
}

output "name" {
  description = "The name of the Key Vault."
  value       = azurerm_key_vault.kv.name
}

output "uri" {
  description = "The URI of the Key Vault."
  value       = azurerm_key_vault.kv.vault_uri
}