resource "azurerm_application_insights" "appi" {
  # Naming convention example: appi-<prefix>-<postfix><env>
  name                = "appi-${var.prefix}-${var.postfix}${var.env}"
  location            = var.location
  resource_group_name = var.rg_name
  application_type    = "web" # Standard type for general monitoring

  # Optional: Configure retention, sampling, workspace linkage
  # retention_in_days = 90
  # sampling_percentage = 100
  # workspace_id = var.log_analytics_workspace_id # Link to Log Analytics Workspace if desired

  tags = var.tags
}