# Terraform Variables for Archangel Infrastructure
# Comprehensive deployment configuration variables

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
  
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

variable "project_name" {
  description = "Project name"
  type        = string
  default     = "archangel"
  
  validation {
    condition     = can(regex("^[a-z0-9-]+$", var.project_name))
    error_message = "Project name must contain only lowercase letters, numbers, and hyphens."
  }
}

variable "network_prefix" {
  description = "Network CIDR prefix for simulation networks"
  type        = string
  default     = "192.168"
  
  validation {
    condition     = can(regex("^\\d+\\.\\d+$", var.network_prefix))
    error_message = "Network prefix must be in format 'X.Y' (e.g., '192.168')."
  }
}

variable "enable_monitoring" {
  description = "Enable monitoring stack (Prometheus, Grafana, AlertManager)"
  type        = bool
  default     = true
}

variable "enable_security_tools" {
  description = "Enable security tools and vulnerability scanners"
  type        = bool
  default     = true
}

variable "enable_honeypots" {
  description = "Enable honeypot services for deception"
  type        = bool
  default     = true
}

variable "enable_elk_stack" {
  description = "Enable ELK stack for logging and analysis"
  type        = bool
  default     = true
}

variable "agent_replicas" {
  description = "Number of autonomous agent replicas to deploy"
  type        = number
  default     = 3
  
  validation {
    condition     = var.agent_replicas >= 1 && var.agent_replicas <= 10
    error_message = "Agent replicas must be between 1 and 10."
  }
}

variable "vulnerability_services" {
  description = "Map of vulnerable services to deploy"
  type = map(object({
    enabled     = bool
    image       = string
    ports       = list(number)
    environment = list(string)
    volumes     = list(string)
  }))
  
  default = {
    wordpress = {
      enabled     = true
      image       = "wordpress:5.8-apache"
      ports       = [80]
      environment = ["WORDPRESS_DB_HOST=archangel-mysql"]
      volumes     = ["/var/www/html"]
    }
    mysql = {
      enabled     = true
      image       = "mysql:5.7"
      ports       = [3306]
      environment = ["MYSQL_ROOT_PASSWORD=root123"]
      volumes     = ["/var/lib/mysql"]
    }
    postgresql = {
      enabled     = true
      image       = "postgres:12"
      ports       = [5432]
      environment = ["POSTGRES_PASSWORD=admin123"]
      volumes     = ["/var/lib/postgresql/data"]
    }
  }
}

variable "resource_limits" {
  description = "Resource limits for containers"
  type = map(object({
    memory_mb = number
    cpu_cores = number
  }))
  
  default = {
    agent = {
      memory_mb = 512
      cpu_cores = 1
    }
    database = {
      memory_mb = 1024
      cpu_cores = 2
    }
    monitoring = {
      memory_mb = 2048
      cpu_cores = 2
    }
  }
}

variable "network_segmentation" {
  description = "Network segmentation configuration"
  type = map(object({
    subnet     = string
    internal   = bool
    enable_nat = bool
  }))
  
  default = {
    dmz = {
      subnet     = "10.0/24"
      internal   = false
      enable_nat = true
    }
    internal = {
      subnet     = "20.0/24"
      internal   = true
      enable_nat = false
    }
    management = {
      subnet     = "40.0/24"
      internal   = false
      enable_nat = true
    }
    deception = {
      subnet     = "50.0/24"
      internal   = true
      enable_nat = false
    }
  }
}

variable "security_policies" {
  description = "Security policy configuration"
  type = object({
    enable_network_policies = bool
    enable_rbac            = bool
    enable_pod_security    = bool
    scan_images           = bool
    enforce_tls           = bool
  })
  
  default = {
    enable_network_policies = true
    enable_rbac            = true
    enable_pod_security    = true
    scan_images           = true
    enforce_tls           = true
  }
}

variable "backup_configuration" {
  description = "Backup and disaster recovery configuration"
  type = object({
    enabled             = bool
    retention_days     = number
    backup_schedule    = string
    remote_storage     = bool
  })
  
  default = {
    enabled         = true
    retention_days  = 30
    backup_schedule = "0 2 * * *"  # Daily at 2 AM
    remote_storage  = false
  }
}

variable "auto_scaling" {
  description = "Auto-scaling configuration"
  type = object({
    enabled     = bool
    min_agents  = number
    max_agents  = number
    cpu_target  = number
    mem_target  = number
  })
  
  default = {
    enabled    = false
    min_agents = 1
    max_agents = 5
    cpu_target = 70
    mem_target = 80
  }
}

variable "deployment_strategy" {
  description = "Deployment strategy configuration"
  type = object({
    strategy          = string
    max_unavailable   = number
    max_surge        = number
    health_check_url  = string
    readiness_timeout = number
  })
  
  default = {
    strategy          = "rolling"
    max_unavailable   = 1
    max_surge        = 1
    health_check_url  = "/health"
    readiness_timeout = 60
  }
}

# Local computed values
locals {
  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
    Component   = "archangel-infrastructure"
    Version     = "1.0.0"
  }
  
  # Network CIDR calculation
  networks = {
    for name, config in var.network_segmentation :
    name => "${var.network_prefix}.${config.subnet}"
  }
  
  # Service discovery configuration
  service_discovery = {
    core_endpoint    = "http://${var.project_name}-core:8888"
    metrics_endpoint = var.enable_monitoring ? "http://${var.project_name}-prometheus:9090" : null
    logs_endpoint    = var.enable_elk_stack ? "http://${var.project_name}-elasticsearch:9200" : null
  }
  
  # Security context
  security_context = {
    run_as_user        = 1000
    run_as_group       = 1000
    run_as_non_root    = true
    read_only_root_fs  = true
    allow_escalation   = false
  }
  
  # Resource requests and limits
  resource_config = {
    for service, limits in var.resource_limits :
    service => {
      requests = {
        memory = "${floor(limits.memory_mb * 0.5)}Mi"
        cpu    = "${floor(limits.cpu_cores * 0.5 * 1000)}m"
      }
      limits = {
        memory = "${limits.memory_mb}Mi"
        cpu    = "${limits.cpu_cores * 1000}m"
      }
    }
  }
}

# Output computed values for use in main configuration
output "computed_networks" {
  value = local.networks
}

output "computed_tags" {
  value = local.common_tags
}

output "service_discovery_config" {
  value = local.service_discovery
}

output "security_context_config" {
  value = local.security_context
}

output "resource_configuration" {
  value = local.resource_config
}