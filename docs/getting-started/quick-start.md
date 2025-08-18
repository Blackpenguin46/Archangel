# Quick Start Guide

Get up and running with Archangel Autonomous AI Evolution in just 15 minutes! This guide will have you deploying your first autonomous cybersecurity agents and running scenarios in no time.

## Prerequisites Check ✅

Before we begin, ensure you have:

- **Docker**: Version 20.10+ ([Install Docker](https://docs.docker.com/get-docker/))
- **Docker Compose**: Version 2.0+ (included with Docker Desktop)
- **Git**: For cloning the repository
- **8GB RAM**: Minimum system memory
- **OpenAI API Key**: For LLM integration ([Get API Key](https://platform.openai.com/api-keys))

### Quick Prerequisites Check
```bash
# Verify Docker installation
docker --version
docker-compose --version

# Check available memory
free -h  # Linux/macOS
# or check Task Manager on Windows

# Test Docker functionality
docker run hello-world
```

## Step 1: Get Archangel (2 minutes)

### Clone the Repository
```bash
# Clone the main repository
git clone https://github.com/archangel/autonomous-ai-evolution.git
cd autonomous-ai-evolution

# Verify you have the latest version
git status
```

### Quick Directory Overview
```
autonomous-ai-evolution/
├── agents/              # Agent implementations
├── scenarios/           # Pre-built scenarios
├── infrastructure/      # Deployment configurations
├── docs/               # Documentation (you're here!)
├── docker-compose.yml  # Main deployment file
└── .env.example       # Environment template
```

## Step 2: Configure Environment (3 minutes)

### Create Environment File
```bash
# Copy the environment template
cp .env.example .env

# Edit the configuration
nano .env  # or use your preferred editor
```

### Essential Configuration
```bash
# .env file - Update these values
OPENAI_API_KEY=sk-your-openai-api-key-here
POSTGRES_PASSWORD=secure_password_123
REDIS_PASSWORD=redis_password_123
ENCRYPTION_KEY=your_32_character_encryption_key_here

# Optional: Adjust resource limits
LOG_LEVEL=INFO
DEPLOYMENT_MODE=development
```

### Generate Encryption Key
```bash
# Generate a secure encryption key
openssl rand -hex 16
# Copy the output to ENCRYPTION_KEY in .env
```

## Step 3: Deploy Archangel (5 minutes)

### Start Core Services
```bash
# Start all services
docker-compose up -d

# This will download images and start:
# - Coordinator (main orchestration)
# - Redis (message bus)
# - PostgreSQL (data storage)
# - ChromaDB (vector memory)
# - Grafana (monitoring)
# - Mock enterprise environment
```

### Monitor Deployment
```bash
# Check service status
docker-compose ps

# Watch logs during startup
docker-compose logs -f coordinator

# Wait for "System ready" message
```

### Verify Installation
```bash
# Test API endpoint
curl http://localhost:8000/health

# Expected response:
# {"status": "healthy", "version": "1.0.0", "agents": 0}

# Check Grafana dashboard
open http://localhost:3000  # macOS
# or visit http://localhost:3000 in your browser
# Login: admin/admin
```

## Step 4: Deploy Your First Agents (3 minutes)

### Start Red Team Agents
```bash
# Deploy reconnaissance agent
docker-compose up -d red-team-recon

# Deploy exploitation agent
docker-compose up -d red-team-exploit

# Verify agents are running
curl http://localhost:8000/agents/status
```

### Start Blue Team Agents
```bash
# Deploy SOC analyst agent
docker-compose up -d blue-team-soc

# Deploy firewall configurator
docker-compose up -d blue-team-firewall

# Check all agents
docker-compose ps | grep team
```

### Agent Registration Check
```bash
# Verify agents registered with coordinator
curl http://localhost:8000/agents/list

# Expected output shows active agents:
# {
#   "red_team": ["recon-001", "exploit-001"],
#   "blue_team": ["soc-001", "firewall-001"]
# }
```

## Step 5: Run Your First Scenario (2 minutes)

### Load Basic Scenario
```bash
# Run the built-in quick start scenario
curl -X POST http://localhost:8000/scenarios/run \
  -H "Content-Type: application/json" \
  -d '{
    "scenario_id": "quick-start-demo",
    "duration": 300
  }'

# This starts a 5-minute basic intrusion scenario
```

### Monitor Live Activity
```bash
# Watch real-time agent activity
curl http://localhost:8000/scenarios/current/status

# View agent logs
docker-compose logs -f red-team-recon
docker-compose logs -f blue-team-soc
```

### Access Monitoring Dashboard
1. Open Grafana: http://localhost:3000
2. Login with admin/admin
3. Navigate to "Archangel Overview" dashboard
4. Watch real-time agent metrics and activities

## What Just Happened? 🎉

Congratulations! You've successfully:

✅ **Deployed Archangel**: Full multi-agent system running  
✅ **Started Agents**: Red and Blue team agents active  
✅ **Ran Scenario**: Autonomous cybersecurity simulation  
✅ **Monitored Activity**: Real-time dashboard access  

### Your System Now Includes:

- **4 Autonomous Agents**: 2 Red Team + 2 Blue Team
- **Mock Enterprise**: Vulnerable web server and database
- **Real-time Monitoring**: Grafana dashboards
- **Vector Memory**: Agent learning and knowledge storage
- **Secure Communication**: Encrypted agent messaging

## Next Steps 🚀

### Immediate Actions (Next 10 minutes)
1. **Explore the Dashboard**: Check out different Grafana panels
2. **View Agent Logs**: See how agents make decisions
3. **Try Different Scenarios**: Load pre-built scenarios
4. **Experiment with Settings**: Modify agent configurations

### Learning Path (Next few hours)
1. **[Agent Development Tutorial](../training/interactive.md)**: Build custom agents
2. **[Scenario Creation Guide](../scenarios/tutorial.md)**: Design your own scenarios
3. **[API Documentation](../api/agents.md)**: Integrate with external systems
4. **[Video Tutorials](../training/videos.md)**: Visual learning resources

## Quick Commands Reference

### System Management
```bash
# Stop all services
docker-compose down

# Restart specific service
docker-compose restart coordinator

# View system status
docker-compose ps

# Clean up (removes all data)
docker-compose down -v
```

### Agent Management
```bash
# List active agents
curl http://localhost:8000/agents/list

# Get agent details
curl http://localhost:8000/agents/recon-001/status

# Send command to agent
curl -X POST http://localhost:8000/agents/recon-001/command \
  -H "Content-Type: application/json" \
  -d '{"action": "scan", "target": "192.168.1.0/24"}'
```

### Scenario Management
```bash
# List available scenarios
curl http://localhost:8000/scenarios/list

# Get scenario status
curl http://localhost:8000/scenarios/current/status

# Stop current scenario
curl -X POST http://localhost:8000/scenarios/stop
```

## Troubleshooting Quick Fixes

### Services Won't Start
```bash
# Check Docker daemon
sudo systemctl status docker

# Check port conflicts
netstat -tulpn | grep :8000

# Restart Docker
sudo systemctl restart docker
docker-compose up -d
```

### Agents Not Responding
```bash
# Check agent logs
docker-compose logs red-team-recon

# Restart agents
docker-compose restart red-team-recon blue-team-soc

# Verify network connectivity
docker-compose exec red-team-recon ping coordinator
```

### API Errors
```bash
# Check OpenAI API key
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
  https://api.openai.com/v1/models

# Verify environment variables
docker-compose exec coordinator env | grep OPENAI
```

## Sample Scenarios to Try

### 1. Basic Web Intrusion (5 minutes)
```bash
curl -X POST http://localhost:8000/scenarios/run \
  -H "Content-Type: application/json" \
  -d '{
    "scenario_id": "basic-web-intrusion",
    "duration": 300,
    "difficulty": "beginner"
  }'
```

### 2. Phishing Campaign (10 minutes)
```bash
curl -X POST http://localhost:8000/scenarios/run \
  -H "Content-Type: application/json" \
  -d '{
    "scenario_id": "phishing-campaign",
    "duration": 600,
    "difficulty": "intermediate"
  }'
```

### 3. Advanced Persistent Threat (15 minutes)
```bash
curl -X POST http://localhost:8000/scenarios/run \
  -H "Content-Type: application/json" \
  -d '{
    "scenario_id": "apt-simulation",
    "duration": 900,
    "difficulty": "advanced"
  }'
```

## Interactive Demo Commands

### Watch Agent Decision Making
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
docker-compose restart coordinator

# Follow agent reasoning
docker-compose logs -f red-team-recon | grep "REASONING"
```

### Real-time Agent Communication
```bash
# Monitor message bus
docker-compose exec redis redis-cli monitor

# Watch agent coordination
docker-compose logs -f coordinator | grep "COORDINATION"
```

### Memory System Exploration
```bash
# Check agent memories
curl http://localhost:8000/agents/recon-001/memory/recent

# Search agent knowledge
curl -X POST http://localhost:8000/agents/recon-001/memory/search \
  -H "Content-Type: application/json" \
  -d '{"query": "web server vulnerabilities"}'
```

## Success Indicators

You'll know everything is working when you see:

### ✅ Healthy System Status
- All services show "Up" in `docker-compose ps`
- Health endpoint returns 200 OK
- Grafana dashboard loads without errors

### ✅ Active Agents
- Agents appear in coordinator logs
- Agent status API returns active agents
- Agent decision logs show reasoning

### ✅ Scenario Execution
- Scenarios start without errors
- Agent activity visible in logs
- Monitoring dashboard shows metrics

### ✅ Learning and Memory
- Agents store experiences
- Memory searches return results
- Knowledge base grows over time

## Getting Help

### Immediate Support
- **Documentation**: Browse the [full documentation](../README.md)
- **Common Issues**: Check [troubleshooting guide](../troubleshooting/common-issues.md)
- **API Reference**: See [API documentation](../api/)

### Community Support
- **Discord**: [Join our community](https://discord.gg/archangel)
- **GitHub Issues**: [Report bugs or request features](https://github.com/archangel/issues)
- **Discussions**: [Community forum](https://github.com/archangel/discussions)

### Professional Support
- **Email**: support@archangel.dev
- **Enterprise**: enterprise@archangel.dev
- **Training**: training@archangel.dev

## What's Next?

Now that you have Archangel running, explore these areas:

### 🤖 **Agent Development**
Learn to create custom agents with specialized capabilities:
- Custom reconnaissance techniques
- Advanced exploitation strategies
- Intelligent defense mechanisms
- Multi-agent coordination patterns

### 🎯 **Scenario Design**
Build engaging training scenarios:
- Educational cybersecurity simulations
- Red team vs Blue team competitions
- Realistic enterprise environments
- Progressive difficulty challenges

### 🏗️ **System Integration**
Connect Archangel to your infrastructure:
- SIEM integration
- Threat intelligence feeds
- Custom monitoring systems
- External security tools

### 📊 **Research and Analytics**
Use Archangel for cybersecurity research:
- Agent behavior analysis
- Attack pattern studies
- Defense effectiveness research
- AI safety in cybersecurity

---

**🎉 Welcome to the future of autonomous cybersecurity!**

You're now ready to explore the full capabilities of Archangel. The system you've deployed represents cutting-edge research in AI-powered cybersecurity, and you're part of a growing community pushing the boundaries of what's possible.

*Continue your journey with our [comprehensive tutorials](../training/interactive.md) or dive deep into [agent development](../api/agents.md).*