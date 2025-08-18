# Archangel Infrastructure as Code - Enhanced Main Configuration
# Complete environment deployment with comprehensive automation

terraform {
  required_version = ">= 1.0"
  required_providers {
    docker = {
      source  = "kreuzwerker/docker"
      version = "~> 3.0"
    }
    local = {
      source  = "hashicorp/local"
      version = "~> 2.4"
    }
    null = {
      source  = "hashicorp/null"
      version = "~> 3.2"
    }
  }
  
  backend "local" {
    path = "terraform.tfstate"
  }
}

# Provider Configuration
provider "docker" {
  host = "unix:///var/run/docker.sock"
}

# Data sources for existing resources
data "docker_network" "bridge" {
  name = "bridge"
}

# Local values from variables
locals {
  # Import computed values from variables.tf
  networks      = data.external.compute_networks.result
  common_tags   = data.external.compute_tags.result
  service_urls  = data.external.service_discovery.result
}

# External data sources for computed values
data "external" "compute_networks" {
  program = ["echo", jsonencode({
    dmz        = "${var.network_prefix}.10.0/24"
    internal   = "${var.network_prefix}.20.0/24"
    management = "${var.network_prefix}.40.0/24"
    deception  = "${var.network_prefix}.50.0/24"
  })]
}

data "external" "compute_tags" {
  program = ["echo", jsonencode({
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
    Component   = "archangel-infrastructure"
    Version     = "1.0.0"
  })]
}

data "external" "service_discovery" {
  program = ["echo", jsonencode({
    core_api = "http://localhost:8888"
    prometheus = var.enable_monitoring ? "http://localhost:9090" : ""
    grafana = var.enable_monitoring ? "http://localhost:3000" : ""
  })]
}

# Enhanced Network Configuration
resource "docker_network" "dmz_network" {
  name = "${var.project_name}-dmz-${var.environment}"
  driver = "bridge"
  
  ipam_config {
    subnet = "${var.network_prefix}.10.0/24"
    gateway = "${var.network_prefix}.10.1"
  }
  
  options = {
    "com.docker.network.bridge.enable_icc"           = "true"
    "com.docker.network.bridge.enable_ip_masquerade" = "true"
    "com.docker.network.driver.mtu"                  = "1500"
  }
  
  labels {
    label = "archangel.zone"
    value = "dmz"
  }
  
  labels {
    label = "archangel.environment"
    value = var.environment
  }
  
  labels {
    label = "archangel.managed-by"
    value = "terraform"
  }
}

resource "docker_network" "internal_network" {
  name = "${var.project_name}-internal-${var.environment}"
  driver = "bridge"
  internal = true  # Internal network for security
  
  ipam_config {
    subnet = "${var.network_prefix}.20.0/24"
    gateway = "${var.network_prefix}.20.1"
  }
  
  labels {
    label = "archangel.zone"
    value = "internal"
  }
  
  labels {
    label = "archangel.environment"
    value = var.environment
  }
}

resource "docker_network" "management_network" {
  name = "${var.project_name}-management-${var.environment}"
  driver = "bridge"
  
  ipam_config {
    subnet = "${var.network_prefix}.40.0/24"
    gateway = "${var.network_prefix}.40.1"
  }
  
  labels {
    label = "archangel.zone"
    value = "management"
  }
  
  labels {
    label = "archangel.environment"
    value = var.environment
  }
}

# Additional deception network for honeypots
resource "docker_network" "deception_network" {
  name = "${var.project_name}-deception-${var.environment}"
  driver = "bridge"
  internal = true  # Isolated deception network
  
  ipam_config {
    subnet = "${var.network_prefix}.50.0/24"
    gateway = "${var.network_prefix}.50.1"
  }
  
  labels {
    label = "archangel.zone"
    value = "deception"
  }
  
  labels {
    label = "archangel.environment"
    value = var.environment
  }
}

# Volumes
resource "docker_volume" "mysql_data" {
  name = "archangel_mysql_data"
}

resource "docker_volume" "postgresql_data" {
  name = "archangel_postgresql_data"
}

resource "docker_volume" "elasticsearch_data" {
  name = "archangel_elasticsearch_data"
}

resource "docker_volume" "wordpress_data" {
  name = "archangel_wordpress_data"
}

# Frontend Services (DMZ)
resource "docker_container" "nginx_lb" {
  image = "nginx:alpine"
  name  = "archangel-nginx-lb"
  
  ports {
    internal = 80
    external = 80
  }
  
  ports {
    internal = 443
    external = 443
  }
  
  networks_advanced {
    name         = docker_network.dmz_network.name
    ipv4_address = "192.168.10.10"
  }
  
  volumes {
    host_path      = "${path.cwd}/../config/nginx/nginx.conf"
    container_path = "/etc/nginx/nginx.conf"
  }
  
  labels {
    label = "archangel.service"
    value = "load-balancer"
  }
  
  labels {
    label = "archangel.zone"
    value = "dmz"
  }
}

resource "docker_container" "wordpress" {
  image = "wordpress:5.8-apache"
  name  = "archangel-wordpress"
  
  env = [
    "WORDPRESS_DB_HOST=archangel-mysql",
    "WORDPRESS_DB_USER=wordpress",
    "WORDPRESS_DB_PASSWORD=vulnerable123",
    "WORDPRESS_DB_NAME=wordpress"
  ]
  
  networks_advanced {
    name         = docker_network.dmz_network.name
    ipv4_address = "192.168.10.20"
  }
  
  networks_advanced {
    name = docker_network.internal_network.name
  }
  
  volumes {
    volume_name    = docker_volume.wordpress_data.name
    container_path = "/var/www/html"
  }
  
  labels {
    label = "archangel.service"
    value = "wordpress"
  }
  
  labels {
    label = "archangel.zone"
    value = "dmz"
  }
  
  depends_on = [docker_container.mysql]
}

# Backend Services (Internal)
resource "docker_container" "mysql" {
  image = "mysql:5.7"
  name  = "archangel-mysql"
  
  env = [
    "MYSQL_ROOT_PASSWORD=root123",
    "MYSQL_DATABASE=wordpress",
    "MYSQL_USER=wordpress",
    "MYSQL_PASSWORD=vulnerable123"
  ]
  
  ports {
    internal = 3306
    external = 3306
  }
  
  networks_advanced {
    name         = docker_network.internal_network.name
    ipv4_address = "192.168.20.10"
  }
  
  volumes {
    volume_name    = docker_volume.mysql_data.name
    container_path = "/var/lib/mysql"
  }
  
  volumes {
    host_path      = "${path.cwd}/../config/mysql/vulnerable.cnf"
    container_path = "/etc/mysql/conf.d/vulnerable.cnf"
  }
  
  volumes {
    host_path      = "${path.cwd}/../config/mysql/init.sql"
    container_path = "/docker-entrypoint-initdb.d/init.sql"
  }
  
  labels {
    label = "archangel.service"
    value = "database"
  }
  
  labels {
    label = "archangel.zone"
    value = "internal"
  }
}

resource "docker_container" "postgresql" {
  image = "postgres:12"
  name  = "archangel-postgresql"
  
  env = [
    "POSTGRES_DB=corporate",
    "POSTGRES_USER=admin",
    "POSTGRES_PASSWORD=admin123",
    "POSTGRES_HOST_AUTH_METHOD=trust"
  ]
  
  ports {
    internal = 5432
    external = 5432
  }
  
  networks_advanced {
    name         = docker_network.internal_network.name
    ipv4_address = "192.168.20.20"
  }
  
  volumes {
    volume_name    = docker_volume.postgresql_data.name
    container_path = "/var/lib/postgresql/data"
  }
  
  labels {
    label = "archangel.service"
    value = "database"
  }
  
  labels {
    label = "archangel.zone"
    value = "internal"
  }
}

# Management Services (ELK Stack)
resource "docker_container" "elasticsearch" {
  image = "docker.elastic.co/elasticsearch/elasticsearch:7.15.0"
  name  = "archangel-elasticsearch"
  
  env = [
    "discovery.type=single-node",
    "ES_JAVA_OPTS=-Xms512m -Xmx512m",
    "xpack.security.enabled=false"
  ]
  
  ports {
    internal = 9200
    external = 9200
  }
  
  networks_advanced {
    name         = docker_network.management_network.name
    ipv4_address = "192.168.40.10"
  }
  
  volumes {
    volume_name    = docker_volume.elasticsearch_data.name
    container_path = "/usr/share/elasticsearch/data"
  }
  
  labels {
    label = "archangel.service"
    value = "elasticsearch"
  }
  
  labels {
    label = "archangel.zone"
    value = "management"
  }
}

resource "docker_container" "kibana" {
  image = "docker.elastic.co/kibana/kibana:7.15.0"
  name  = "archangel-kibana"
  
  env = [
    "ELASTICSEARCH_HOSTS=http://archangel-elasticsearch:9200"
  ]
  
  ports {
    internal = 5601
    external = 5601
  }
  
  networks_advanced {
    name         = docker_network.management_network.name
    ipv4_address = "192.168.40.30"
  }
  
  labels {
    label = "archangel.service"
    value = "kibana"
  }
  
  labels {
    label = "archangel.zone"
    value = "management"
  }
  
  depends_on = [docker_container.elasticsearch]
}

# Outputs
output "service_endpoints" {
  value = {
    nginx_lb      = "http://localhost:80"
    kibana        = "http://localhost:5601"
    elasticsearch = "http://localhost:9200"
    mysql         = "localhost:3306"
    postgresql    = "localhost:5432"
  }
}

output "network_info" {
  value = {
    dmz_network        = docker_network.dmz_network.name
    internal_network   = docker_network.internal_network.name
    management_network = docker_network.management_network.name
  }
}