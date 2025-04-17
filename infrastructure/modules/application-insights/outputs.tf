output "id" {
  description = "The Resource ID of the Application Insights instance."
  value       = azurerm_application_insights.appi.id
}

output "instrumentation_key" {
  description = "The Instrumentation Key for the Application Insights instance."
  value       = azurerm_application_insights.appi.instrumentation_key
  sensitive   = true # Mark as sensitive as it can be used to send data
}

output "connection_string" {
  description = "The Connection String for the Application Insights instance."
  value       = azurerm_application_insights.appi.connection_string
  sensitive   = true # Mark as sensitive
}