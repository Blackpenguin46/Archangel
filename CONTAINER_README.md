# Archangel AI vs AI Containerized System

## Overview

This system implements a fully autonomous AI vs AI cybersecurity environment using Docker containers where:

- 🔴 **Red Team AI** controls a Kali Linux container with real penetration testing tools
- 🔵 **Blue Team AI** controls an Ubuntu SOC container with security monitoring tools  
- 🎯 **Target Environment** provides a realistic enterprise system to attack/defend

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Red Team AI   │    │   Blue Team AI  │    │ Target Enterprise│
│   (Kali Linux)  │    │  (Ubuntu SOC)   │    │   (Nginx Web)   │
│                 │    │                 │    │                 │
│ • nmap scanning │    │ • iptables      │    │ • HTTP Service  │
│ • Port enumeration│  │ • tcpdump       │    │ • Port 8080     │
│ • Service detection│ │ • Process mon   │    │ • Mock data     │
│ • Autonomous AI │    │ • Threat detect │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │ Docker Network  │
                    │ archangel-combat│
                    │ 172.18.0.0/16   │
                    └─────────────────┘
```

## Quick Start

### Prerequisites
- Docker installed and running
- Python 3.x
- ~2GB free disk space for container images

### Launch System

```bash
# Start the complete AI vs AI environment
./start_archangel_containers.sh

# Check system status
python3 container_status_report.py

# Run integration tests
python3 test_container_integration.py
```

## Container Details

### Red Team (Kali Linux)
- **Container**: `archangel-red-team`
- **IP**: `172.18.0.2/16`
- **Tools**: nmap, curl, netcat, Python 3
- **AI Agent**: `/app/red_team_agent.py`
- **Capabilities**:
  - Autonomous network reconnaissance  
  - Port scanning and service enumeration
  - Target vulnerability assessment
  - Real-time attack adaptation

### Blue Team (Ubuntu SOC)
- **Container**: `archangel-blue-team`  
- **IP**: `172.18.0.3/16`
- **Tools**: iptables, tcpdump, ss, Python 3
- **AI Agent**: `/app/blue_team_agent.py`
- **Capabilities**:
  - Network traffic monitoring
  - Threat detection and analysis
  - Automated IP blocking
  - Process and connection monitoring

### Target Enterprise
- **Container**: `target-enterprise`
- **IP**: `172.18.0.4/16`
- **Service**: Nginx web server
- **Access**: http://localhost:8080
- **Purpose**: Realistic attack target

## AI Agent Features

### Autonomous Operation
- Both AI agents operate continuously without human intervention
- Real-time decision making based on current environment state
- Adaptive strategies that evolve during engagement

### Real Tool Execution  
- Red team executes actual nmap, curl, and netcat commands
- Blue team uses real iptables, tcpdump, and monitoring tools
- No simulated attacks - actual penetration testing tools

### Learning and Adaptation
- Agents learn from successful/failed attempts
- Strategy modification based on opponent responses
- Pattern recognition for improved future operations

## Status Monitoring

### Real-time Status Files
```bash
# Check agent status
cat logs/red_team_status.log
cat logs/blue_team_status.log

# View comprehensive system status  
python3 container_status_report.py
```

### Container Logs
```bash
# View red team activity
docker logs archangel-red-team

# View blue team monitoring
docker logs archangel-blue-team

# Access container shells
docker exec -it archangel-red-team /bin/bash
docker exec -it archangel-blue-team /bin/bash
```

## Integration with Main System

The containerized environment integrates with the main Archangel orchestrator:

```bash
# Run with container integration
python3 archangel.py --containers

# BlackHat demo with containers
python3 blackhat_demo.py --demo extended_demo --containers
```

## Commands Reference

### System Management
```bash
# Start system
./start_archangel_containers.sh

# Stop system  
docker stop archangel-red-team archangel-blue-team target-enterprise

# Remove containers
docker rm archangel-red-team archangel-blue-team target-enterprise

# Remove network
docker network rm archangel-combat
```

### Testing and Monitoring
```bash
# Full integration test
python3 test_container_integration.py

# System status report
python3 container_status_report.py

# Network information
docker network inspect archangel-combat

# Container resource usage
docker stats archangel-red-team archangel-blue-team target-enterprise
```

### Manual Testing
```bash
# Red team manual scan
docker exec archangel-red-team nmap -F 172.18.0.3

# Blue team manual monitoring  
docker exec archangel-blue-team ss -tuln

# Target access test
curl -I http://localhost:8080
```

## Log Files

- `logs/red_team_status.log` - Red team agent status and discoveries
- `logs/blue_team_status.log` - Blue team agent status and threats
- Docker logs via `docker logs <container>`

## Security Notes

- Containers run with necessary privileges for realistic security testing
- Network isolation prevents attacks on host system
- All activity contained within Docker environment
- Red team tools are for defensive research only

## Troubleshooting

### Container Start Issues
```bash
# Check Docker daemon
docker info

# Verify network
docker network ls | grep archangel

# Check container status
docker ps -a --filter name=archangel
```

### Agent Communication Issues
```bash
# Verify agent files
docker exec archangel-red-team ls -la /app/
docker exec archangel-blue-team ls -la /app/

# Check Python processes
docker exec archangel-red-team ps aux | grep python
docker exec archangel-blue-team ps aux | grep python
```

### Network Connectivity
```bash
# Test inter-container connectivity
docker exec archangel-red-team ping -c 3 172.18.0.3
docker exec archangel-blue-team ping -c 3 172.18.0.2
```

## Development

### Adding New Capabilities

1. **Red Team Tools**: Edit `container_red_team_agent.py`
2. **Blue Team Tools**: Edit `container_blue_team_agent.py`  
3. **Container Config**: Modify Dockerfiles in `containers/` directory
4. **Integration**: Update `test_container_integration.py`

### Custom Container Images

Build custom images with additional tools:

```bash
# Build custom red team image
docker build -t archangel-red-custom -f containers/red-team/Dockerfile.kali .

# Build custom blue team image  
docker build -t archangel-blue-custom -f containers/blue-team/Dockerfile.ubuntu-soc .
```

## Performance

- **Red Team Container**: ~500MB RAM, 1GB storage
- **Blue Team Container**: ~300MB RAM, 500MB storage  
- **Target Container**: ~50MB RAM, 100MB storage
- **Network Latency**: <1ms between containers

## BlackHat Demonstration Ready

This containerized system provides:
- ✅ Real penetration testing tools execution
- ✅ Autonomous AI decision making  
- ✅ Live security monitoring and response
- ✅ Realistic enterprise attack targets
- ✅ Comprehensive logging and reporting
- ✅ Scalable Docker deployment

Perfect for demonstrating advanced AI vs AI cybersecurity capabilities at conferences and research presentations.