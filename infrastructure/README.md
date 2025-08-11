# Archangel Mock Enterprise Environment Infrastructure

This directory contains the complete infrastructure setup for the Archangel Autonomous AI Evolution mock enterprise environment. The infrastructure simulates a realistic corporate network with intentional vulnerabilities for cybersecurity training and autonomous agent testing.

## 🏗️ Architecture Overview

The mock enterprise environment consists of multiple network zones with realistic services:

### Network Segmentation
- **DMZ Zone (192.168.10.0/24)**: Internet-facing services
- **Internal Zone (192.168.20.0/24)**: Internal corporate services  
- **Management Zone (192.168.40.0/24)**: Monitoring and logging infrastructure

### Services Deployed

#### Frontend Layer (DMZ)
- **Nginx Load Balancer**: Routes traffic to backend services
- **WordPress**: Vulnerable CMS with intentional security flaws
- **OpenCart**: E-commerce platform with misconfigurations

#### Backend Layer (Internal)
- **MySQL Database**: Vulnerable database with weak authentication
- **PostgreSQL Database**: Corporate database with exposed ports
- **SMB File Server**: Network file shares with weak permissions
- **Mail Server**: SMTP/IMAP services for email simulation
- **DVWA**: Damn Vulnerable Web Application for testing

#### Security Layer
- **Suricata IDS**: Network intrusion detection system
- **Firewall Rules**: Network segmentation and traffic filtering

#### Logging Infrastructure (Management)
- **Elasticsearch**: Log storage and search engine
- **Logstash**: Log processing and enrichment
- **Kibana**: Log visualization and dashboards
- **Filebeat**: Log shipping and collection

## 🚀 Quick Start

### Prerequisites
- Docker Engine 20.10+
- Docker Compose 2.0+
- Python 3.8+ (for testing and validation)
- 8GB+ RAM recommended
- 20GB+ disk space

### Deployment

1. **Deploy the environment:**
   ```bash
   cd infrastructure
   ./deploy.sh
   ```

2. **Validate deployment:**
   ```bash
   python3 scripts/validate_deployment.py
   ```

3. **Access services:**
   - Kibana Dashboard: http://localhost:5601
   - WordPress: http://localhost (Host: wordpress.local)
   - OpenCart: http://localhost (Host: shop.local)
   - DVWA: http://localhost:8080
   - MailHog: http://localhost:8025

### Alternative Deployment Methods

#### Using Docker Compose directly:
```bash
docker-compose up -d
```

#### Using Terraform:
```bash
cd terraform
terraform init
terraform plan
terraform apply
```

#### Using Kubernetes:
```bash
kubectl apply -f k8s/
```

## 🔧 Configuration

### Environment Variables
Key configuration options can be set via environment variables:

```bash
# Database passwords
export MYSQL_ROOT_PASSWORD=root123
export POSTGRES_PASSWORD=admin123

# Network configuration
export DMZ_SUBNET=192.168.10.0/24
export INTERNAL_SUBNET=192.168.20.0/24
export MGMT_SUBNET=192.168.40.0/24

# Logging configuration
export ELK_VERSION=7.15.0
export LOG_RETENTION_DAYS=30
```

### Service Configuration Files
- `config/nginx/nginx.conf`: Load balancer configuration
- `config/mysql/vulnerable.cnf`: MySQL vulnerable settings
- `config/postgresql/postgresql.conf`: PostgreSQL configuration
- `config/suricata/suricata.yaml`: IDS configuration
- `config/logstash/pipeline/logstash.conf`: Log processing rules
- `config/filebeat/filebeat.yml`: Log shipping configuration

## 🔒 Security Features

### Intentional Vulnerabilities
The environment includes intentional security flaws for testing:

1. **Database Vulnerabilities:**
   - Exposed database ports (3306, 5432)
   - Weak authentication credentials
   - SQL injection opportunities
   - Excessive user privileges

2. **Web Application Vulnerabilities:**
   - Outdated WordPress with vulnerable plugins
   - Information disclosure endpoints
   - Directory traversal possibilities
   - Weak session management

3. **Network Vulnerabilities:**
   - Misconfigured firewall rules
   - Unencrypted protocols
   - Weak SMB configurations
   - Default credentials

### Security Monitoring
- **Suricata IDS**: Monitors network traffic for suspicious activity
- **ELK Stack**: Centralized logging and security event correlation
- **Container Isolation**: Services run in isolated Docker containers
- **Network Segmentation**: VLANs separate different security zones

## 📊 Monitoring and Logging

### Kibana Dashboards
Pre-configured dashboards for monitoring:
- Security Events Dashboard
- Network Traffic Analysis
- Database Activity Monitoring
- Web Application Logs
- System Performance Metrics

### Log Sources
The system collects logs from:
- Web servers (Nginx, Apache)
- Databases (MySQL, PostgreSQL)
- Network security (Suricata IDS)
- System containers (Docker logs)
- Application services

### Metrics Collection
- Container resource usage
- Network traffic patterns
- Database query performance
- Security event frequency
- Service availability

## 🧪 Testing

### Infrastructure Tests
Run comprehensive infrastructure tests:

```bash
cd tests
python3 test_infrastructure.py
```

Test categories:
- Container deployment validation
- Network connectivity testing
- Service accessibility checks
- Security configuration validation
- Logging infrastructure verification

### Security Testing
Validate security configurations:

```bash
# Test network segmentation
./scripts/test_network_isolation.sh

# Validate vulnerable services
./scripts/test_vulnerabilities.sh

# Check logging functionality
./scripts/test_log_ingestion.sh
```

## 🔧 Management Commands

### Deployment Management
```bash
# Deploy environment
./deploy.sh deploy

# Stop environment
./deploy.sh stop

# Destroy environment (removes all data)
./deploy.sh destroy

# Check status
./deploy.sh status

# View logs
./deploy.sh logs [service_name]
```

### Service Management
```bash
# Restart specific service
docker-compose restart mysql-vulnerable

# Scale service (if supported)
docker-compose up -d --scale wordpress=2

# Update service configuration
docker-compose up -d --force-recreate nginx-lb
```

### Data Management
```bash
# Backup databases
./scripts/backup_databases.sh

# Reset to clean state
./scripts/reset_environment.sh

# Import test data
./scripts/import_test_data.sh
```

## 🐛 Troubleshooting

### Common Issues

1. **Services not starting:**
   ```bash
   # Check container logs
   docker-compose logs [service_name]
   
   # Check resource usage
   docker stats
   
   # Verify network connectivity
   docker network ls
   ```

2. **Database connection issues:**
   ```bash
   # Test MySQL connectivity
   docker exec mysql-vulnerable mysql -u root -proot123 -e "SELECT 1"
   
   # Test PostgreSQL connectivity
   docker exec postgresql-vulnerable psql -U admin -d corporate -c "SELECT 1"
   ```

3. **Logging not working:**
   ```bash
   # Check Elasticsearch health
   curl http://localhost:9200/_cluster/health
   
   # Verify Logstash pipeline
   docker logs logstash
   
   # Check Filebeat status
   docker exec filebeat filebeat test output
   ```

4. **Network isolation issues:**
   ```bash
   # Test network segmentation
   docker exec nginx-loadbalancer ping -c 1 192.168.20.10
   
   # Check iptables rules
   docker exec nginx-loadbalancer iptables -L
   ```

### Performance Tuning

1. **Resource Allocation:**
   - Increase Docker memory limit to 8GB+
   - Allocate sufficient CPU cores (4+ recommended)
   - Use SSD storage for better I/O performance

2. **Service Optimization:**
   - Adjust Elasticsearch heap size based on available RAM
   - Tune MySQL buffer pool size
   - Configure Logstash pipeline workers

3. **Network Performance:**
   - Use host networking for high-throughput scenarios
   - Optimize Docker bridge network settings
   - Consider using overlay networks for multi-host deployment

## 📋 Service Endpoints

| Service | URL | Credentials | Purpose |
|---------|-----|-------------|---------|
| Nginx LB | http://localhost:80 | N/A | Load balancer |
| WordPress | http://localhost (Host: wordpress.local) | admin/admin | Vulnerable CMS |
| OpenCart | http://localhost (Host: shop.local) | admin/admin | E-commerce |
| DVWA | http://localhost:8080 | admin/password | Vulnerable web app |
| MailHog | http://localhost:8025 | N/A | Mail testing |
| Kibana | http://localhost:5601 | N/A | Log visualization |
| Elasticsearch | http://localhost:9200 | N/A | Search engine |
| MySQL | localhost:3306 | root/root123 | Database |
| PostgreSQL | localhost:5432 | admin/admin123 | Database |
| SMB Share | //localhost/public | guest/guest | File sharing |

## ⚠️ Security Warnings

**IMPORTANT**: This environment contains intentional security vulnerabilities and should NEVER be deployed on production networks or exposed to the internet.

- All services use weak default credentials
- Databases are intentionally misconfigured
- Network security is deliberately weakened
- Services contain known vulnerabilities

Use only in isolated, controlled environments for:
- Cybersecurity training
- Penetration testing practice
- Autonomous agent development
- Security research

## 📚 Additional Resources

- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Elasticsearch Guide](https://www.elastic.co/guide/en/elasticsearch/reference/current/index.html)
- [Suricata Documentation](https://suricata.readthedocs.io/)
- [MITRE ATT&CK Framework](https://attack.mitre.org/)
- [OWASP Testing Guide](https://owasp.org/www-project-web-security-testing-guide/)

## 🤝 Contributing

When contributing to the infrastructure:

1. Test all changes in isolated environment
2. Update documentation for new services
3. Add appropriate security warnings
4. Include validation tests for new components
5. Follow infrastructure as code principles

## 📄 License

This infrastructure is part of the Archangel project and is intended for educational and research purposes only.