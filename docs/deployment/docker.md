# Docker Deployment Guide

This guide covers deploying the Archangel Autonomous AI Evolution system using Docker and Docker Compose for single-host environments.

## Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 20.04+ recommended), macOS 10.15+, Windows 10 Pro+
- **CPU**: 8+ cores (16+ recommended for full simulation)
- **RAM**: 16GB minimum (32GB+ recommended)
- **Storage**: 100GB+ available space
- **Network**: Internet access for LLM API calls

### Software Requirements
- Docker Engine 20.10+
- Docker Compose 2.0+
- Git
- Python 3.9+ (for management scripts)

## Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/archangel/autonomous-ai-evolution.git
cd autonomous-ai-evolution
```

### 2. Environment Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit configuration
nano .env
```

### 3. Basic Deployment
```bash
# Start core services
docker-compose up -d

# Verify deployment
docker-compose ps
```

## Detailed Configuration

### Environment Variables

Create and configure your `.env` file:

```bash
# Core Configuration
ARCHANGEL_VERSION=latest
DEPLOYMENT_MODE=development
LOG_LEVEL=INFO

# LLM Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4-turbo
LOCAL_LLM_ENABLED=false
OLLAMA_HOST=http://localhost:11434

# Database Configuration
POSTGRES_DB=archangel
POSTGRES_USER=archangel
POSTGRES_PASSWORD=secure_password_here
REDIS_PASSWORD=redis_password_here

# Vector Database
CHROMADB_HOST=chromadb
CHROMADB_PORT=8000
WEAVIATE_HOST=weaviate
WEAVIATE_PORT=8080

# Network Configuration
NETWORK_SUBNET=192.168.100.0/24
DMZ_SUBNET=192.168.10.0/24
INTERNAL_SUBNET=192.168.20.0/24

# Security Configuration
TLS_ENABLED=true
CERT_PATH=./certs
ENCRYPTION_KEY=your_32_character_encryption_key

# Monitoring
GRAFANA_ADMIN_PASSWORD=admin_password_here
PROMETHEUS_RETENTION=30d
```

### Docker Compose Configuration

The main `docker-compose.yml` includes all necessary services:

```yaml
version: '3.8'

services:
  # Core Orchestration
  coordinator:
    image: archangel/coordinator:${ARCHANGEL_VERSION}
    container_name: archangel-coordinator
    environment:
      - LOG_LEVEL=${LOG_LEVEL}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
      - postgres
      - chromadb
    networks:
      - archangel-network
    volumes:
      - ./config:/app/config
      - ./logs:/app/logs
    restart: unless-stopped

  # Red Team Agents
  red-team-recon:
    image: archangel/red-team:${ARCHANGEL_VERSION}
    container_name: red-team-recon
    environment:
      - AGENT_TYPE=recon
      - AGENT_ID=recon-001
      - TEAM=RED_TEAM
    networks:
      - archangel-network
      - simulation-network
    depends_on:
      - coordinator
    restart: unless-stopped

  red-team-exploit:
    image: archangel/red-team:${ARCHANGEL_VERSION}
    container_name: red-team-exploit
    environment:
      - AGENT_TYPE=exploit
      - AGENT_ID=exploit-001
      - TEAM=RED_TEAM
    networks:
      - archangel-network
      - simulation-network
    depends_on:
      - coordinator
    restart: unless-stopped

  # Blue Team Agents
  blue-team-soc:
    image: archangel/blue-team:${ARCHANGEL_VERSION}
    container_name: blue-team-soc
    environment:
      - AGENT_TYPE=soc_analyst
      - AGENT_ID=soc-001
      - TEAM=BLUE_TEAM
    networks:
      - archangel-network
      - simulation-network
    depends_on:
      - coordinator
    restart: unless-stopped

  # Mock Enterprise Environment
  web-server:
    image: archangel/vulnerable-web:${ARCHANGEL_VERSION}
    container_name: enterprise-web
    networks:
      simulation-dmz:
        ipv4_address: 192.168.10.10
    ports:
      - "8080:80"
      - "8443:443"
    volumes:
      - web-data:/var/www/html
    restart: unless-stopped

  database-server:
    image: archangel/vulnerable-db:${ARCHANGEL_VERSION}
    container_name: enterprise-db
    environment:
      - MYSQL_ROOT_PASSWORD=vulnerable_password
      - MYSQL_DATABASE=enterprise
    networks:
      simulation-internal:
        ipv4_address: 192.168.20.10
    volumes:
      - db-data:/var/lib/mysql
    restart: unless-stopped

  # Infrastructure Services
  redis:
    image: redis:7-alpine
    container_name: archangel-redis
    command: redis-server --requirepass ${REDIS_PASSWORD}
    networks:
      - archangel-network
    volumes:
      - redis-data:/data
    restart: unless-stopped

  postgres:
    image: postgres:15
    container_name: archangel-postgres
    environment:
      - POSTGRES_DB=${POSTGRES_DB}
      - POSTGRES_USER=${POSTGRES_USER}
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    networks:
      - archangel-network
    volumes:
      - postgres-data:/var/lib/postgresql/data
    restart: unless-stopped

  chromadb:
    image: chromadb/chroma:latest
    container_name: archangel-chromadb
    networks:
      - archangel-network
    volumes:
      - chromadb-data:/chroma/chroma
    restart: unless-stopped

  # Monitoring Stack
  grafana:
    image: grafana/grafana:latest
    container_name: archangel-grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_ADMIN_PASSWORD}
    ports:
      - "3000:3000"
    networks:
      - archangel-network
    volumes:
      - grafana-data:/var/lib/grafana
      - ./monitoring/grafana:/etc/grafana/provisioning
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    container_name: archangel-prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.retention.time=${PROMETHEUS_RETENTION}'
    ports:
      - "9090:9090"
    networks:
      - archangel-network
    volumes:
      - prometheus-data:/prometheus
      - ./monitoring/prometheus:/etc/prometheus
    restart: unless-stopped

networks:
  archangel-network:
    driver: bridge
    ipam:
      config:
        - subnet: ${NETWORK_SUBNET}
  
  simulation-dmz:
    driver: bridge
    ipam:
      config:
        - subnet: ${DMZ_SUBNET}
  
  simulation-internal:
    driver: bridge
    ipam:
      config:
        - subnet: ${INTERNAL_SUBNET}

volumes:
  redis-data:
  postgres-data:
  chromadb-data:
  grafana-data:
  prometheus-data:
  web-data:
  db-data:
```

## Deployment Steps

### 1. Pre-deployment Checks

```bash
# Verify Docker installation
docker --version
docker-compose --version

# Check system resources
free -h
df -h

# Verify network connectivity
ping -c 3 api.openai.com
```

### 2. Build Custom Images

```bash
# Build all custom images
docker-compose build

# Or build specific services
docker-compose build coordinator
docker-compose build red-team
docker-compose build blue-team
```

### 3. Initialize Services

```bash
# Start infrastructure services first
docker-compose up -d redis postgres chromadb

# Wait for services to be ready
./scripts/wait-for-services.sh

# Initialize database schema
docker-compose exec postgres psql -U archangel -d archangel -f /docker-entrypoint-initdb.d/init.sql

# Start remaining services
docker-compose up -d
```

### 4. Verify Deployment

```bash
# Check service status
docker-compose ps

# View logs
docker-compose logs -f coordinator

# Test API endpoints
curl http://localhost:8000/health
curl http://localhost:3000  # Grafana dashboard
```

## Service Management

### Starting Services
```bash
# Start all services
docker-compose up -d

# Start specific service
docker-compose up -d coordinator

# Start with logs
docker-compose up coordinator
```

### Stopping Services
```bash
# Stop all services
docker-compose down

# Stop specific service
docker-compose stop coordinator

# Stop and remove volumes (WARNING: Data loss)
docker-compose down -v
```

### Scaling Services
```bash
# Scale red team agents
docker-compose up -d --scale red-team-recon=3

# Scale blue team agents
docker-compose up -d --scale blue-team-soc=2
```

## Configuration Management

### Agent Configuration

Create agent-specific configuration files in `config/agents/`:

```yaml
# config/agents/recon-001.yml
agent_id: recon-001
team: RED_TEAM
type: recon
capabilities:
  - network_scanning
  - service_enumeration
  - vulnerability_assessment
llm_config:
  model: gpt-4-turbo
  temperature: 0.7
  max_tokens: 2000
memory_config:
  vector_store: chromadb
  memory_size: 1000
  clustering_enabled: true
```

### Scenario Configuration

Define scenarios in `config/scenarios/`:

```yaml
# config/scenarios/basic-intrusion.yml
scenario_id: basic-intrusion-001
name: "Basic Network Intrusion"
description: "Red team attempts network reconnaissance and exploitation"
duration: 3600  # 1 hour
phases:
  - name: reconnaissance
    duration: 900  # 15 minutes
    allowed_agents: [recon]
  - name: exploitation
    duration: 1800  # 30 minutes
    allowed_agents: [recon, exploit]
  - name: persistence
    duration: 900  # 15 minutes
    allowed_agents: [exploit, persistence]
objectives:
  red_team:
    - "Discover at least 3 live hosts"
    - "Identify 1 exploitable vulnerability"
    - "Establish persistent access"
  blue_team:
    - "Detect reconnaissance activity"
    - "Block exploitation attempts"
    - "Contain any successful breaches"
```

## Monitoring and Logging

### Log Management

```bash
# View real-time logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f coordinator

# Export logs
docker-compose logs --no-color > archangel-logs.txt
```

### Monitoring Dashboards

Access monitoring interfaces:
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Agent Status**: http://localhost:8000/agents/status

### Health Checks

```bash
# Check service health
curl http://localhost:8000/health

# Check agent status
curl http://localhost:8000/agents/status

# Check database connectivity
docker-compose exec postgres pg_isready -U archangel
```

## Troubleshooting

### Common Issues

#### Services Won't Start
```bash
# Check Docker daemon
sudo systemctl status docker

# Check resource usage
docker system df
docker system prune  # Clean up if needed

# Check port conflicts
netstat -tulpn | grep :8000
```

#### Agent Communication Issues
```bash
# Check network connectivity
docker network ls
docker network inspect archangel_archangel-network

# Test Redis connectivity
docker-compose exec redis redis-cli ping

# Check agent logs
docker-compose logs red-team-recon
```

#### Performance Issues
```bash
# Monitor resource usage
docker stats

# Check system resources
htop
iotop

# Optimize memory usage
echo 'vm.swappiness=10' >> /etc/sysctl.conf
```

### Debug Mode

Enable debug mode for detailed logging:

```bash
# Set debug environment
export LOG_LEVEL=DEBUG
export ARCHANGEL_DEBUG=true

# Restart with debug logging
docker-compose down
docker-compose up -d
```

## Security Considerations

### Network Security
- All inter-service communication uses encrypted channels
- Simulation networks are isolated from host network
- Firewall rules restrict external access

### Data Security
- Sensitive configuration stored in environment variables
- Database passwords are randomly generated
- TLS certificates are automatically managed

### Access Control
- Default passwords should be changed immediately
- API endpoints require authentication
- Role-based access control for different user types

## Backup and Recovery

### Data Backup
```bash
# Backup all volumes
docker run --rm -v archangel_postgres-data:/data -v $(pwd):/backup alpine tar czf /backup/postgres-backup.tar.gz /data

# Backup configuration
tar czf config-backup.tar.gz config/
```

### Recovery
```bash
# Restore from backup
docker run --rm -v archangel_postgres-data:/data -v $(pwd):/backup alpine tar xzf /backup/postgres-backup.tar.gz -C /

# Restart services
docker-compose restart
```

## Performance Tuning

### Resource Limits

Add resource limits to docker-compose.yml:

```yaml
services:
  coordinator:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
```

### Optimization Tips
- Use SSD storage for better I/O performance
- Increase shared memory for PostgreSQL
- Tune Redis memory settings
- Monitor and adjust agent concurrency

---

*Next: [Kubernetes Deployment Guide](kubernetes.md) for production scaling*