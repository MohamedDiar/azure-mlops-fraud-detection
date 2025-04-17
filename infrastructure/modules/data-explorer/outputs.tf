output "adx_cluster_name" {
  description = "The name of the Azure Data Explorer (Kusto) cluster created for monitoring (if enabled)."
  value       = var.enable_monitoring ? azurerm_kusto_cluster.adx_cluster[0].name : "Not Created"
}

output "adx_cluster_uri" {
  description = "The URI of the Azure Data Explorer (Kusto) cluster created for monitoring (if enabled)."
  value       = var.enable_monitoring ? azurerm_kusto_cluster.adx_cluster[0].uri : "Not Created"
}

output "adx_database_name" {
  description = "The name of the Azure Data Explorer (Kusto) database created for monitoring (if enabled)."
  value       = var.enable_monitoring ? azurerm_kusto_database.adx_database[0].name : "Not Created"
}