output "aml_workspace_name" {
  description = "The name of the Azure Machine Learning workspace."
  value       = azurerm_machine_learning_workspace.mlw.name
}

output "aml_workspace_id" {
  description = "The ID of the Azure Machine Learning workspace."
  value       = azurerm_machine_learning_workspace.mlw.id
}

output "aml_workspace_url" {
  description = "The URL of the Azure Machine Learning workspace studio."
  value       = azurerm_machine_learning_workspace.mlw.workspace_url
}

output "aml_workspace_principal_id" {
  description = "The Principal ID of the System Assigned Identity for the AML Workspace."
  value       = azurerm_machine_learning_workspace.mlw.identity[0].principal_id # Accessing the first identity block
}

output "compute_cluster_name" {
  description = "The name of the created compute cluster (if enabled)."
  value       = var.enable_aml_computecluster ? azurerm_machine_learning_compute_cluster.aml_compute_cluster[0].name : "Not Created"
}