# Archangel MCP Integration Architecture

## Overview

The Archangel MCP (Model Context Protocol) integration provides secure, isolated access to external cybersecurity resources and SDKs for autonomous red and blue team agents. This architecture enables elite-level security operations by connecting agents to professional-grade tools, threat intelligence feeds, and vulnerability databases.

## Architecture Components

### 1. MCP Orchestrator (`MCPOrchestrator`)
- **Purpose**: Central management of MCP servers and security
- **Key Features**: 
  - Team isolation enforcement
  - Credential management
  - Token-based authentication
  - Rate limiting and monitoring

### 2. Team-Specific MCP Servers
- **Red Team Server**: Isolated server for offensive tools and attack frameworks
- **Blue Team Server**: Secure server for defensive tools and threat intelligence
- **Isolation**: Complete network and process isolation between teams

### 3. Security Manager (`MCPSecurityManager`)
- **Authentication**: JWT tokens with team-specific permissions
- **Authorization**: Role-based access control (RBAC)
- **Rate Limiting**: Per-agent, per-resource rate limiting
- **Encryption**: Secure credential storage and transmission

## External Resource Integration

### Red Team Resources

#### Attack Frameworks
- **Metasploit Framework**: Full MSF API access for exploit development
- **SQLMap**: Automated SQL injection testing
- **Nuclei**: High-speed vulnerability scanner with templates

#### OSINT Sources
- **Shodan**: Internet-connected device discovery
- **Exploit Database**: Public exploit and PoC repository
- **Custom OSINT Tools**: Reconnaissance and intelligence gathering

### Blue Team Resources

#### Threat Intelligence
- **VirusTotal**: Multi-engine malware analysis
- **MISP**: Threat information sharing platform
- **Custom TI Feeds**: Proprietary threat intelligence sources

#### Defense Platforms
- **Elastic SIEM**: Security event correlation and analysis
- **OSQuery**: Endpoint visibility and live investigation
- **Suricata**: Network intrusion detection system

#### Forensics Tools
- **Volatility3**: Memory forensics and analysis
- **YARA**: Malware pattern matching
- **Custom Forensics**: Specialized analysis tools

## SDK Integration Points

### Container-Based SDKs
```yaml
# Red Team SDKs
- metasploit-framework: Docker container with MSF API
- nuclei: ProjectDiscovery scanner with templates
- sqlmap: Automated SQL injection testing
- nmap: Network discovery and port scanning

# Blue Team SDKs  
- volatility3: Memory analysis framework
- suricata: Network monitoring and IDS
- yara: Malware detection rules
- osquery: Endpoint visibility platform
```

### API-Based Integrations
```python
# External API endpoints
RED_TEAM_APIS = {
    "shodan": "https://api.shodan.io",
    "exploitdb": "https://www.exploit-db.com/api",
    "nuclei": "https://api.nuclei.org"
}

BLUE_TEAM_APIS = {
    "virustotal": "https://www.virustotal.com/vtapi/v2",
    "misp": "https://misp.example.com/api",
    "elastic": "https://elastic.example.com:9200"
}
```

## Security Architecture

### Team Isolation
- **Network Isolation**: Separate Docker networks for red/blue teams
- **Process Isolation**: Dedicated containers with security policies
- **Data Isolation**: Separate credential stores and databases
- **API Isolation**: Team-specific MCP servers with filtered access

### Authentication Flow
```
1. Agent requests access token from MCP Orchestrator
2. Security Manager validates agent identity and team
3. JWT token issued with team-specific permissions
4. Agent uses token for all MCP server interactions
5. MCP server validates token and enforces permissions
```

### Authorization Matrix
| Resource Type | Red Team | Blue Team | Neutral |
|---------------|----------|-----------|---------|
| Attack Frameworks | ✅ | ❌ | ❌ |
| Threat Intelligence | ❌ | ✅ | ❌ |
| Public OSINT | ✅ | ✅ | ✅ |
| Vulnerability DBs | ✅ | ✅ | ❌ |
| Defense Platforms | ❌ | ✅ | ❌ |
| Forensics Tools | ❌ | ✅ | ❌ |

## Configuration Management

### MCP Configuration (`config/mcp_config.json`)
```json
{
  "mcp_architecture": {
    "secret_key": "production_secret",
    "isolation": {
      "enable_team_isolation": true,
      "shared_resources": ["public_osint"],
      "cross_team_communication": false
    },
    "security": {
      "token_ttl_hours": 24,
      "rate_limit_per_hour": 1000,
      "encryption_enabled": true
    }
  }
}
```

### Docker Deployment
- **Compose File**: `docker/mcp-docker-compose.yml`
- **Containers**: 15+ specialized security tool containers
- **Networks**: Isolated networks for team separation
- **Volumes**: Persistent storage for credentials and data

## Usage Examples

### Initialize MCP Architecture
```python
from core.mcp_integration_architecture import create_mcp_orchestrator

# Create orchestrator
orchestrator = create_mcp_orchestrator()

# Initialize MCP architecture
await orchestrator.initialize_mcp_architecture()

# Provision agent access
red_token = orchestrator.provision_agent_access("red_agent_001", TeamType.RED_TEAM)
blue_token = orchestrator.provision_agent_access("blue_agent_001", TeamType.BLUE_TEAM)
```

### Enhanced Agent Operations
```python
from core.archangel_mcp_integration import EnhancedRedTeamAgent

# Create enhanced red team agent
red_agent = EnhancedRedTeamAgent("red_001", mcp_integration)
await red_agent.initialize()

# Advanced vulnerability assessment with MCP tools
vuln_results = await red_agent.advanced_vulnerability_assessment("192.168.1.100")

# Simulated attack chain
attack_results = await red_agent.simulated_attack_chain("test_target")
```

### Blue Team Threat Hunting
```python
from core.archangel_mcp_integration import EnhancedBlueTeamAgent

# Create enhanced blue team agent
blue_agent = EnhancedBlueTeamAgent("blue_001", mcp_integration)
await blue_agent.initialize()

# Comprehensive threat hunting
hunt_results = await blue_agent.comprehensive_threat_hunting("lateral_movement")

# Incident response workflow
ir_results = await blue_agent.incident_response_workflow(incident_data)
```

## Performance Optimization

### Memory Management
- **Container Limits**: 1GB RAM per team server
- **Connection Pooling**: Reuse connections to external APIs
- **Response Caching**: 5-minute TTL for expensive operations
- **Lazy Loading**: Load SDKs only when needed

### Rate Limiting
- **Global Limits**: 1000 requests/hour per agent
- **Per-Resource Limits**: API-specific rate limiting
- **Burst Handling**: Short-term burst allowances
- **Backoff Strategy**: Exponential backoff on rate limit hits

### M2 MacBook Optimization
- **CPU Limits**: 0.5 cores per MCP server
- **Memory Limits**: 1GB per container
- **ARM64 Images**: Native Apple Silicon containers
- **Network Efficiency**: Minimal cross-container communication

## Monitoring and Observability

### Metrics Collection
- **Prometheus**: System and application metrics
- **Response Times**: API call performance tracking
- **Error Rates**: Failed requests and recoveries
- **Rate Limit Usage**: Current usage vs limits

### Logging
- **Structured Logs**: JSON format with correlation IDs
- **Audit Trail**: All MCP operations logged
- **Security Events**: Authentication and authorization events
- **Performance Logs**: Slow queries and bottlenecks

### Health Checks
- **Container Health**: Docker healthcheck endpoints
- **API Availability**: External service monitoring
- **Authentication Status**: Token validation checks
- **Resource Usage**: Memory and CPU monitoring

## Deployment Instructions

### Prerequisites
```bash
# Install Docker and Docker Compose
brew install docker docker-compose

# Clone Archangel repository
git clone <repository_url>
cd Archangel
```

### Configuration Setup
```bash
# Copy configuration templates
cp config/mcp_config.json.example config/mcp_config.json

# Generate MCP secret key
python3 -c "import secrets; print(secrets.token_hex(32))" > .mcp_secret

# Set up credentials (edit with actual API keys)
vim config/mcp_config.json
```

### Docker Deployment
```bash
# Build and start MCP architecture
docker-compose -f docker/mcp-docker-compose.yml up -d

# Verify services are running
docker ps | grep archangel

# Check MCP server health
curl http://localhost:8881/health  # Red team server
curl http://localhost:8882/health  # Blue team server
```

### Agent Integration
```python
# Initialize enhanced orchestrator
from core.archangel_mcp_integration import create_enhanced_archangel_orchestrator

orchestrator = create_enhanced_archangel_orchestrator()
await orchestrator.initialize_enhanced_system()

# Run enhanced operations
results = await orchestrator.run_enhanced_security_exercise()
```

## Troubleshooting

### Common Issues

#### MCP Server Connection Failed
```bash
# Check container status
docker logs archangel-red-mcp
docker logs archangel-blue-mcp

# Verify network connectivity
docker network inspect archangel_red_network
```

#### Authentication Errors
```bash
# Check secret key configuration
cat config/mcp_config.json | jq '.mcp_architecture.secret_key'

# Verify JWT token generation
python3 -c "from core.mcp_integration_architecture import MCPSecurityManager; print('Auth test passed')"
```

#### External API Failures
```bash
# Check API credentials
docker exec archangel-red-mcp cat /opt/archangel/credentials/api_keys.json

# Test external connectivity
docker exec archangel-red-mcp curl -s https://api.shodan.io/api-info
```

### Performance Issues

#### High Memory Usage
- Reduce container memory limits in docker-compose.yml
- Enable swap if necessary (not recommended for production)
- Use smaller SDK container images

#### Slow API Responses
- Check rate limiting settings
- Verify external API performance
- Review caching configuration

## Security Considerations

### Production Deployment
- **Secret Management**: Use external secret management (HashiCorp Vault, AWS Secrets Manager)
- **Network Security**: Deploy behind VPN or private network
- **Certificate Management**: Use proper TLS certificates for all endpoints
- **Access Logging**: Enable comprehensive audit logging

### Threat Model
- **Container Escape**: Mitigated by security policies and user namespaces
- **Network Attacks**: Mitigated by network isolation and firewalls
- **Credential Theft**: Mitigated by encryption and short-lived tokens
- **API Abuse**: Mitigated by rate limiting and monitoring

## Future Enhancements

### Planned Features
- **Multi-Cloud Deployment**: Support for AWS, GCP, Azure
- **Advanced Analytics**: ML-powered threat detection
- **Custom SDK Development**: Framework for adding new tools
- **Federation**: Multi-instance MCP coordination

### Research Areas
- **Zero-Trust Architecture**: Further security hardening
- **Quantum-Safe Cryptography**: Future-proof encryption
- **AI-Powered Resource Selection**: Intelligent tool selection
- **Federated Learning**: Cross-instance knowledge sharing

## Contributing

### Development Setup
```bash
# Create development environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt -r requirements-mcp.txt

# Run tests
pytest tests/test_mcp_integration.py

# Code formatting
black core/mcp_integration_architecture.py
```

### Adding New Resources
1. Define resource in appropriate team configuration
2. Implement SDK integration class
3. Add authentication/authorization rules
4. Update Docker compose configuration
5. Add comprehensive tests

### Security Review Process
1. All MCP changes require security review
2. Penetration testing for new external integrations
3. Code review focusing on authentication/authorization
4. Documentation update for security implications

---

## Quick Reference

### Key Files
- `core/mcp_integration_architecture.py` - Core MCP implementation
- `core/archangel_mcp_integration.py` - Agent integration layer  
- `config/mcp_config.json` - MCP configuration
- `docker/mcp-docker-compose.yml` - Container deployment
- `requirements-mcp.txt` - MCP dependencies

### Key Commands
```bash
# Start MCP architecture
docker-compose -f docker/mcp-docker-compose.yml up -d

# View logs
docker logs archangel-red-mcp -f

# Health check
curl http://localhost:8881/health

# Stop architecture
docker-compose -f docker/mcp-docker-compose.yml down
```

This MCP integration transforms Archangel agents from autonomous security tools into elite-level security professionals with access to the same resources and SDKs used by advanced threat actors and security teams worldwide.