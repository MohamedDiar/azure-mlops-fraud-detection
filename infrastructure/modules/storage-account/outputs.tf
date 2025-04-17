output "id" {
  description = "The Resource ID of the Storage Account."
  value       = azurerm_storage_account.st.id
}

output "name" {
  description = "The name of the Storage Account."
  value       = azurerm_storage_account.st.name
}

output "primary_blob_endpoint" {
  description = "The primary Blob storage endpoint for the Storage Account."
  value       = azurerm_storage_account.st.primary_blob_endpoint
}

output "primary_dfs_endpoint" {
  description = "The primary Data Lake Storage Gen2 endpoint (if HNS is enabled)."
  value       = var.hns_enabled ? azurerm_storage_account.st.primary_dfs_endpoint : "N/A (HNS Disabled)"
}

output "primary_access_key" {
  description = "The primary access key for the Storage Account."
  value       = azurerm_storage_account.st.primary_access_key
  sensitive   = true
}

output "secondary_access_key" {
  description = "The secondary access key for the Storage Account."
  value       = azurerm_storage_account.st.secondary_access_key
  sensitive   = true
}

output "primary_connection_string" {
  description = "The primary connection string for the Storage Account."
  value       = azurerm_storage_account.st.primary_connection_string
  sensitive   = true
}