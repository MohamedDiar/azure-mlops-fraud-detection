# Creates or references an Azure Resource Group

resource "azurerm_resource_group" "rg" {
  # Naming convention example: rg-<prefix>-<postfix><env>
  name     = "rg-${var.prefix}-${var.postfix}${var.env}"
  location = var.location

  tags = var.tags

  # Optional: Configure management locks if needed
  # lifecycle {
  #   prevent_destroy = true # Protects the resource group from accidental deletion
  # }
}