# Archangel Honeypot and Deception Technologies

This directory contains the complete honeypot and deception technology implementation for the Archangel Autonomous AI Evolution project. The system provides multi-tier honeypot infrastructure, honeytoken distribution, decoy services, and comprehensive monitoring capabilities.

## Overview

The honeypot system implements a comprehensive deception layer designed to:

- **Detect and log attack attempts** through realistic honeypot services
- **Mislead attackers** with fake credentials, documents, and services
- **Generate alerts** for Blue Team agents when deception technologies are accessed
- **Provide intelligence** on attack patterns and techniques
- **Enhance security** through early warning and threat detection

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Deception Network (VLAN 50)                 │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │   Cowrie SSH    │ │   Dionaea       │ │   Glastopf      │   │
│  │   Honeypot      │ │   Malware       │ │   Web Honeypot  │   │
│  │   (Port 2222)   │ │   Honeypot      │ │   (Port 8080)   │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │   Decoy         │ │   Honeytoken    │ │   Honeypot      │   │
│  │   Services      │ │   Distributor   │ │   Monitor       │   │
│  │   (Multi-port)  │ │   (Background)  │ │   (Alerting)    │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Cowrie SSH Honeypot

**Purpose**: Captures SSH and Telnet login attempts and command execution

**Features**:
- Emulates SSH-2.0-OpenSSH_7.4 banner
- Logs all login attempts with usernames/passwords
- Records command execution in fake shell environment
- Captures file downloads and uploads
- Provides realistic filesystem simulation

**Configuration**: `cowrie.cfg`
**Ports**: 2222 (SSH), 2323 (Telnet)
**Logs**: JSON format in `/cowrie/log/cowrie.json`

### 2. Dionaea Malware Honeypot

**Purpose**: Captures malware and exploits targeting various network services

**Features**:
- Emulates multiple services (FTP, HTTP, SMB, MySQL, MSSQL, SIP)
- Captures malware binaries and exploits
- Logs connection attempts and attack patterns
- Provides realistic service responses

**Configuration**: `dionaea.cfg`
**Ports**: 21 (FTP), 80 (HTTP), 135 (RPC), 445 (SMB), 1433 (MSSQL), 3306 (MySQL), 5060 (SIP)
**Logs**: JSON format in `/opt/dionaea/var/log/dionaea/dionaea.json`

### 3. Glastopf Web Honeypot

**Purpose**: Captures web application attacks and vulnerabilities

**Features**:
- Emulates vulnerable web applications
- Detects SQL injection, XSS, LFI, RFI, command injection
- Logs attack payloads and techniques
- Provides realistic web application responses

**Configuration**: `glastopf.cfg`
**Ports**: 8080 (HTTP)
**Logs**: Text format in `/opt/glastopf/log/glastopf.log`

### 4. Decoy Services

**Purpose**: Provides additional fake services to expand attack surface

**Features**:
- Fake SSH service with realistic banners
- Fake FTP service with login simulation
- Fake web admin panel for credential harvesting
- Fake database services
- Customizable service responses

**Implementation**: `decoy_services.py`
**Ports**: 2122 (Fake FTP), 2223 (Fake SSH), 8081 (Admin Panel), 3307 (Fake MySQL)

### 5. Honeytoken Distribution System

**Purpose**: Distributes fake credentials and documents throughout the environment

**Features**:
- Generates realistic fake credentials
- Creates fake API keys and certificates
- Distributes fake documents (passwords.txt, config files, etc.)
- Monitors honeytoken access
- Periodic redistribution and rotation

**Implementation**: `honeytokens.py`
**Configuration**: `config.yaml`

### 6. Honeypot Monitor and Alerting

**Purpose**: Monitors all honeypot activities and generates alerts for Blue Team

**Features**:
- Real-time log monitoring and parsing
- Attack pattern detection and correlation
- Threat intelligence integration
- Multi-channel alerting (webhook, syslog, email)
- SQLite database for alert storage
- Blue Team integration via webhooks

**Implementation**: `honeypot_monitor.py`
**Configuration**: `monitor_config.yaml`
**Database**: SQLite at `/var/lib/honeypots/monitor.db`

## Deployment

### Prerequisites

- Docker and Docker Compose
- At least 5GB available disk space
- Network access for pulling container images
- Python 3.x for testing (optional)

### Quick Start

1. **Deploy the complete honeypot infrastructure**:
   ```bash
   cd infrastructure
   ./scripts/deploy_honeypots.sh
   ```

2. **Verify deployment**:
   ```bash
   ./scripts/deploy_honeypots.sh verify
   ```

3. **Run tests**:
   ```bash
   ./scripts/deploy_honeypots.sh test
   ```

### Manual Deployment

1. **Build custom images**:
   ```bash
   docker build -t archangel/decoy-services:latest -f config/honeypots/Dockerfile.decoy config/honeypots/
   docker build -t archangel/honeytoken-distributor:latest -f config/honeypots/Dockerfile.honeytokens config/honeypots/
   docker build -t archangel/honeypot-monitor:latest -f config/honeypots/Dockerfile.monitor config/honeypots/
   ```

2. **Start services**:
   ```bash
   docker-compose up -d cowrie-ssh dionaea-malware glastopf-web decoy-services honeytoken-distributor honeypot-monitor
   ```

3. **Check status**:
   ```bash
   docker-compose ps
   ```

## Configuration

### Honeypot Configuration

Each honeypot can be configured through its respective configuration file:

- **Cowrie**: `config/honeypots/cowrie.cfg`
- **Dionaea**: `config/honeypots/dionaea.cfg`
- **Glastopf**: `config/honeypots/glastopf.cfg`

### Honeytoken Configuration

Configure honeytoken generation in `config/honeypots/config.yaml`:

```yaml
credentials:
  usernames: ["admin", "root", "service"]
  domains: ["corporate.local", "company.com"]
  password_patterns: ["Password123!", "Admin2024"]

documents:
  types: ["passwords.txt", "config.ini", "api_keys.json"]
  locations: ["/tmp", "/var/tmp", "/home/admin"]
```

### Monitoring Configuration

Configure monitoring and alerting in `config/honeypots/monitor_config.yaml`:

```yaml
alerting:
  enabled: true
  channels: ["webhook", "syslog"]
  webhook:
    url: "http://blue-team-coordinator:8080/honeypot-alerts"

pattern_detection:
  enabled: true
  min_events: 3
  confidence_threshold: 0.7
```

## Testing

### Automated Tests

Run the comprehensive test suite:

```bash
python3 infrastructure/tests/test_honeypots.py
```

### Manual Testing

1. **Test SSH honeypot**:
   ```bash
   ssh admin@localhost -p 2222
   # Try password: admin123
   ```

2. **Test web honeypot**:
   ```bash
   curl http://localhost:8080/
   curl "http://localhost:8080/?id=1' OR '1'='1"  # SQL injection test
   ```

3. **Test fake admin panel**:
   ```bash
   curl http://localhost:8081/admin
   curl -X POST -d "username=admin&password=password123" http://localhost:8081/admin/login
   ```

4. **Test FTP honeypot**:
   ```bash
   telnet localhost 21
   # Try: USER admin, PASS admin123
   ```

## Monitoring and Alerts

### Log Locations

- **Cowrie logs**: `logs/honeypots/cowrie/`
- **Dionaea logs**: `logs/honeypots/dionaea/`
- **Glastopf logs**: `logs/honeypots/glastopf/`
- **Monitor logs**: `logs/honeypots/monitor.log`

### Alert Channels

1. **Webhook**: Sends JSON alerts to Blue Team coordinator
2. **Syslog**: Logs alerts to system syslog
3. **Database**: Stores alerts in SQLite database
4. **Email**: Optional email notifications

### Alert Format

```json
{
  "alert_type": "honeypot_activity",
  "alert": {
    "alert_id": "cowrie_1234567890_1234",
    "timestamp": "2024-01-15T10:30:00Z",
    "honeypot_type": "cowrie_ssh",
    "source_ip": "192.168.1.100",
    "target_port": 2222,
    "attack_type": "ssh_brute_force",
    "severity": "medium",
    "details": {
      "username": "admin",
      "password": "admin123",
      "success": false
    }
  }
}
```

## Blue Team Integration

The honeypot system integrates with Blue Team agents through:

1. **Real-time webhooks** to the Blue Team coordinator
2. **SIEM integration** through structured logging
3. **Alert routing** to specific agent types based on attack type
4. **Threat intelligence** sharing and enrichment

### Alert Routing

- **SOC Analyst**: Brute force attempts, successful logins, web exploits
- **Firewall Configurator**: Port scans, distributed attacks
- **SIEM Integrator**: Pattern detection, malware uploads
- **Incident Response**: Successful exploits, critical severity alerts

## Threat Intelligence

The system includes threat intelligence integration:

- **Malicious IP detection** from threat feeds
- **Known scanner identification**
- **Tor exit node detection**
- **Geolocation enrichment**
- **Reputation scoring**

Update threat intelligence files in `config/honeypots/threat_intel/`:
- `malicious_ips.txt`
- `scanners.txt`
- `tor_exits.txt`

## Maintenance

### Regular Tasks

1. **Log rotation**: Logs are automatically rotated to prevent disk space issues
2. **Database cleanup**: Old alerts are automatically purged after 30 days
3. **Honeytoken rotation**: Honeytokens are redistributed every 30 minutes
4. **Threat intel updates**: Update threat intelligence feeds regularly

### Troubleshooting

1. **Check container status**:
   ```bash
   docker-compose ps
   docker-compose logs <service_name>
   ```

2. **Check network connectivity**:
   ```bash
   docker network ls
   docker network inspect archangel_deception_network
   ```

3. **Check disk space**:
   ```bash
   df -h
   docker system df
   ```

4. **Restart services**:
   ```bash
   docker-compose restart <service_name>
   ```

### Performance Tuning

- **Resource limits**: Adjust container resource limits in docker-compose.yml
- **Log retention**: Configure log retention periods in monitor_config.yaml
- **Alert thresholds**: Tune alert thresholds to reduce false positives
- **Database optimization**: Regular database maintenance and indexing

## Security Considerations

1. **Network isolation**: Honeypots run in isolated network segments
2. **Container security**: All services run as non-root users
3. **Log protection**: Logs are stored with appropriate permissions
4. **Alert validation**: All alerts are validated before forwarding
5. **Resource limits**: Containers have resource limits to prevent DoS

## Contributing

When adding new honeypot services or deception techniques:

1. Follow the existing architecture patterns
2. Add comprehensive logging and monitoring
3. Include configuration options
4. Write tests for new functionality
5. Update documentation

## Requirements Mapping

This implementation satisfies the following requirements:

- **Requirement 3.5**: Deception technologies and honeypot systems
- **Requirement 19.1**: Deploy and rotate honeypots to deceive attackers
- **Requirement 19.2**: Create honeytokens in strategic locations
- **Requirement 19.3**: Log Red Team interactions with decoys
- **Requirement 19.4**: Appear realistic to automated reconnaissance

## MITRE ATT&CK Mapping

The honeypot system detects and logs activities mapped to MITRE ATT&CK techniques:

- **T1110**: Brute Force (SSH, FTP login attempts)
- **T1046**: Network Service Scanning (Port scans)
- **T1190**: Exploit Public-Facing Application (Web exploits)
- **T1105**: Ingress Tool Transfer (Malware uploads)
- **T1059**: Command and Scripting Interpreter (Command injection)
- **T1083**: File and Directory Discovery (Directory traversal)

## License

This honeypot implementation is part of the Archangel project and follows the same licensing terms.