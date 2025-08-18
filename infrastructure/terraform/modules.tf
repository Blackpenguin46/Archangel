# Terraform Modules for Archangel Infrastructure
# Modular infrastructure components for reusability and maintainability

# Monitoring Stack Module
module "monitoring" {
  source = "./modules/monitoring"
  
  count = var.enable_monitoring ? 1 : 0
  
  project_name    = var.project_name
  environment     = var.environment
  network_config  = local.networks
  resource_limits = local.resource_config.monitoring
  
  prometheus_config = {
    retention_time    = "15d"
    storage_size     = "10Gi"
    scrape_interval  = "15s"
    external_labels  = local.common_tags
  }
  
  grafana_config = {
    admin_password   = "archangel_admin_${var.environment}"
    allow_signup     = false
    plugins         = ["grafana-worldmap-panel", "grafana-piechart-panel"]
    data_retention  = "30d"
  }
  
  alertmanager_config = {
    smtp_host        = "localhost:587"
    smtp_from        = "alerts@archangel.local"
    webhook_url      = "http://${var.project_name}-core:8888/alerts"
    resolve_timeout  = "5m"
  }
  
  depends_on = [docker_network.management_network]
}

# Security Tools Module
module "security_tools" {
  source = "./modules/security"
  
  count = var.enable_security_tools ? 1 : 0
  
  project_name   = var.project_name
  environment    = var.environment
  network_config = local.networks
  
  vulnerability_scanners = {
    nmap = {
      enabled = true
      image   = "instrumentisto/nmap:latest"
      targets = [local.networks.dmz, local.networks.internal]
    }
    zap = {
      enabled = true
      image   = "owasp/zap2docker-stable:latest"
      targets = ["http://wordpress", "http://nginx"]
    }
    nikto = {
      enabled = true
      image   = "sullo/nikto:latest"
      targets = ["wordpress", "nginx"]
    }
  }
  
  ids_config = {
    suricata = {
      enabled     = true
      image      = "jasonish/suricata:latest"
      rules_path = "/etc/suricata/rules"
      log_level  = "info"
    }
  }
  
  depends_on = [docker_network.dmz_network]
}

# Honeypot Services Module
module "honeypots" {
  source = "./modules/honeypots"
  
  count = var.enable_honeypots ? 1 : 0
  
  project_name   = var.project_name
  environment    = var.environment
  network_config = local.networks
  
  honeypot_services = {
    cowrie = {
      enabled = true
      image   = "cowrie/cowrie:latest"
      ports   = [22, 23, 2222, 2223]
    }
    dionaea = {
      enabled = true
      image   = "dinotools/dionaea:latest"
      ports   = [21, 42, 135, 443, 445, 1433, 3306, 5060]
    }
    conpot = {
      enabled = true
      image   = "honeynet/conpot:latest"
      ports   = [80, 102, 502, 161]
    }
  }
  
  deception_config = {
    fake_services = true
    honeytokens   = true
    canary_files  = true
    dns_sinkhole  = true
  }
  
  depends_on = [docker_network.deception_network]
}

# Logging and Analytics Module
module "logging" {
  source = "./modules/logging"
  
  count = var.enable_elk_stack ? 1 : 0
  
  project_name   = var.project_name
  environment    = var.environment
  network_config = local.networks
  
  elasticsearch_config = {
    cluster_name     = "${var.project_name}-${var.environment}"
    node_name       = "elasticsearch-master"
    discovery_type  = "single-node"
    heap_size      = "1g"
    data_retention = "30d"
  }
  
  logstash_config = {
    pipeline_workers = 2
    pipeline_batch   = 125
    heap_size       = "512m"
    config_reload   = true
  }
  
  kibana_config = {
    server_name        = "${var.project_name}-kibana"
    elasticsearch_host = "http://archangel-elasticsearch:9200"
    logging_level     = "info"
  }
  
  filebeat_config = {
    log_paths = [
      "/var/log/nginx/*.log",
      "/var/log/containers/*.log",
      "/var/log/suricata/*.log"
    ]
    multiline_patterns = ["^\\d{4}-\\d{2}-\\d{2}"]
  }
  
  depends_on = [docker_network.management_network]
}

# Agent Deployment Module
module "agents" {
  source = "./modules/agents"
  
  project_name   = var.project_name
  environment    = var.environment
  network_config = local.networks
  
  agent_config = {
    replicas      = var.agent_replicas
    image         = "${var.project_name}-agent:latest"
    cpu_limit     = local.resource_config.agent.limits.cpu
    memory_limit  = local.resource_config.agent.limits.memory
    restart_policy = "unless-stopped"
  }
  
  autonomous_config = {
    decision_interval    = 30
    learning_enabled    = true
    ethics_enforcement  = true
    boundary_checks     = true
    anomaly_detection   = true
  }
  
  communication_config = {
    core_endpoint     = local.service_discovery.core_endpoint
    metrics_endpoint  = local.service_discovery.metrics_endpoint
    heartbeat_interval = 10
    timeout          = 30
  }
  
  depends_on = [
    docker_network.internal_network,
    docker_network.dmz_network,
    docker_container.archangel_core
  ]
}

# Backup and Recovery Module
module "backup" {
  source = "./modules/backup"
  
  count = var.backup_configuration.enabled ? 1 : 0
  
  project_name   = var.project_name
  environment    = var.environment
  
  backup_config = {
    schedule       = var.backup_configuration.backup_schedule
    retention_days = var.backup_configuration.retention_days
    remote_storage = var.backup_configuration.remote_storage
    compression   = true
    encryption    = true
  }
  
  backup_targets = [
    {
      name = "mysql_data"
      type = "mysql"
      host = "archangel-mysql"
      databases = ["wordpress", "corporate"]
    },
    {
      name = "postgresql_data"
      type = "postgresql"
      host = "archangel-postgresql"
      databases = ["corporate", "logs"]
    },
    {
      name = "elasticsearch_data"
      type = "elasticsearch"
      host = "archangel-elasticsearch"
      indices = ["logs-*", "metrics-*"]
    }
  ]
  
  depends_on = [
    docker_container.mysql,
    docker_container.postgresql,
    docker_container.elasticsearch
  ]
}

# Network Security Module
module "network_security" {
  source = "./modules/network_security"
  
  count = var.security_policies.enable_network_policies ? 1 : 0
  
  project_name   = var.project_name
  environment    = var.environment
  network_config = local.networks
  
  firewall_rules = {
    dmz_to_internal = {
      source      = local.networks.dmz
      destination = local.networks.internal
      ports       = [3306, 5432, 6379]
      protocol    = "tcp"
      action      = "allow"
    }
    
    management_to_all = {
      source      = local.networks.management
      destination = "any"
      ports       = ["any"]
      protocol    = "any"
      action      = "allow"
    }
    
    external_to_dmz = {
      source      = "0.0.0.0/0"
      destination = local.networks.dmz
      ports       = [80, 443]
      protocol    = "tcp"
      action      = "allow"
    }
  }
  
  network_policies = {
    isolation_enabled    = true
    default_deny        = true
    log_violations      = true
    anomaly_detection   = true
  }
  
  depends_on = [
    docker_network.dmz_network,
    docker_network.internal_network,
    docker_network.management_network
  ]
}

# Load Testing Module
module "load_testing" {
  source = "./modules/load_testing"
  
  count = var.environment == "dev" ? 1 : 0
  
  project_name   = var.project_name
  environment    = var.environment
  
  load_testing_config = {
    tools = ["artillery", "k6", "jmeter"]
    scenarios = [
      {
        name = "web_traffic_simulation"
        target = "http://nginx"
        duration = "5m"
        rate = "10/s"
      },
      {
        name = "database_stress_test"
        target = "mysql:3306"
        duration = "2m"
        connections = 50
      }
    ]
  }
  
  depends_on = [
    docker_container.nginx_lb,
    docker_container.mysql
  ]
}

# Output module information
output "module_deployment_status" {
  value = {
    monitoring_enabled     = var.enable_monitoring
    security_tools_enabled = var.enable_security_tools
    honeypots_enabled     = var.enable_honeypots
    logging_enabled       = var.enable_elk_stack
    backup_enabled        = var.backup_configuration.enabled
    network_security      = var.security_policies.enable_network_policies
    load_testing         = var.environment == "dev"
  }
}

output "module_endpoints" {
  value = {
    monitoring = var.enable_monitoring ? {
      prometheus   = "http://localhost:9090"
      grafana     = "http://localhost:3000"
      alertmanager = "http://localhost:9093"
    } : null
    
    logging = var.enable_elk_stack ? {
      elasticsearch = "http://localhost:9200"
      kibana       = "http://localhost:5601"
      logstash     = "http://localhost:9600"
    } : null
    
    security = var.enable_security_tools ? {
      vulnerability_scanner = "http://localhost:8080"
      ids_dashboard        = "http://localhost:8443"
    } : null
  }
}

output "agent_deployment_info" {
  value = {
    total_agents      = var.agent_replicas
    core_endpoint     = local.service_discovery.core_endpoint
    metrics_endpoint  = local.service_discovery.metrics_endpoint
    network_segments = length(local.networks)
  }
}