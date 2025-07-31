# Archangel Linux Setup Guide

Complete installation and configuration guide for the Archangel autonomous AI security system.

## 📋 Prerequisites

### System Requirements
- **macOS**: 12.0 (Monterey) or later with Apple Silicon (M1/M2/M3)
- **Memory**: 16GB RAM minimum, 32GB recommended
- **Storage**: 50GB free space for models and security tools
- **Network**: Internet connection for model downloads and threat intelligence

### Required Software
- **Python**: 3.9 or later
- **Git**: Latest version
- **Homebrew**: Package manager for macOS
- **Apple Container CLI**: For containerization (will be installed during setup)

### Development Dependencies (Optional)
- **Docker**: For additional containerization options
- **Node.js**: For web interface components
- **VS Code**: Recommended development environment

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Archangel.git
cd Archangel
```

### 2. Run Automated Setup

```bash
chmod +x setup.sh
./setup.sh
```

The setup script will:
- Install Python dependencies
- Setup Apple Container CLI
- Configure security tools
- Initialize AI models
- Create necessary directories
- Set proper permissions

### 3. Manual Setup (Alternative)

If the automated setup fails, follow these manual steps:

#### Install Python Dependencies
```bash
pip3 install -r requirements.txt
```

#### Install Core AI Libraries
```bash
pip3 install torch transformers datasets
pip3 install huggingface_hub accelerate
pip3 install peft bitsandbytes
```

#### Install Security Libraries
```bash
pip3 install requests beautifulsoup4
pip3 install cryptography python-nmap
pip3 install scapy netifaces
```

#### Install Apple Container Dependencies
```bash
# Install Apple Container CLI (if available)
brew install apple-container-cli

# Alternative: Use Docker for containerization
brew install docker
```

## ⚙️ Configuration

### 1. System Configuration

Create the main configuration file:

```bash
cp config/config.example.json config/config.json
```

Edit `config/config.json`:

```json
{
  "system": {
    "log_level": "INFO",
    "data_directory": "./data",
    "model_cache_directory": "./models",
    "container_runtime": "apple-container"
  },
  "ai": {
    "model_name": "deepseek-ai/deepseek-r1-distill-llama-8b",
    "max_tokens": 2048,
    "temperature": 0.7,
    "use_local_models": true
  },
  "security": {
    "enable_red_team": true,
    "enable_blue_team": true,
    "sandbox_mode": true,
    "require_authorization": true
  },
  "containers": {
    "kali_image": "kalilinux/kali-rolling",
    "monitoring_image": "ubuntu:22.04",
    "network_isolation": true,
    "resource_limits": {
      "memory": "4GB",
      "cpu": "2"
    }
  }
}
```

### 2. AI Model Configuration

#### Option A: Use Hugging Face Models (Recommended)
```bash
# Login to Hugging Face (optional for public models)
huggingface-cli login

# Download recommended model
python scripts/download_models.py --model deepseek-ai/deepseek-r1-distill-llama-8b
```

#### Option B: Use Local Models
```bash
# Place your model files in the models directory
mkdir -p models/custom
# Copy your model files to models/custom/
```

### 3. Security Tool Configuration

Configure security tools in `config/tools.json`:

```json
{
  "nmap": {
    "enabled": true,
    "path": "/usr/local/bin/nmap",
    "rate_limit": "1000/minute"
  },
  "metasploit": {
    "enabled": true,
    "path": "/opt/metasploit-framework/msfconsole",
    "sandbox_only": true
  },
  "burpsuite": {
    "enabled": true,
    "path": "/Applications/Burp Suite Professional.app",
    "headless_mode": true
  },
  "wireshark": {
    "enabled": true,
    "path": "/usr/local/bin/tshark"
  }
}
```

### 4. Container Configuration

Setup container environments:

```bash
# Initialize container system
python scripts/apple_container_setup.py --init

# Create Kali Linux container for red team operations
python scripts/apple_container_setup.py --create-kali

# Create monitoring container for blue team operations
python scripts/apple_container_setup.py --create-monitoring
```

## 🧠 AI Model Setup

### 1. Download Pre-trained Models

```bash
# Download and setup the primary reasoning model
python scripts/prepare_training_datasets.py

# Download cybersecurity-specific models
python -c "
from huggingface_hub import snapshot_download
snapshot_download('deepseek-ai/deepseek-r1-distill-llama-8b', local_dir='./models/deepseek')
"
```

### 2. Initialize Training Pipeline

```bash
# Setup training datasets
python scripts/prepare_training_datasets.py

# Initialize model for cybersecurity training
python training/deepseek_training_pipeline.py --init-only
```

### 3. Test AI Integration

```bash
# Test basic AI functionality
python -c "
import asyncio
from core.deepseek_integration import create_deepseek_agent

async def test():
    agent = create_deepseek_agent()
    await agent.initialize()
    print('AI integration successful')

asyncio.run(test())
"
```

## 🛠️ System Initialization

### 1. Initialize Archangel System

```bash
python archangel_autonomous_system.py --init
```

This will:
- Validate all configurations
- Initialize AI agents
- Setup container environments
- Test security tool integrations
- Create necessary databases and logs

### 2. Verify Installation

```bash
# Run system health check
python archangel_autonomous_system.py --health-check

# Test autonomous agents
python archangel_autonomous_system.py --test-agents

# Validate container setup
python scripts/apple_container_setup.py --validate
```

### 3. Initial Security Training

```bash
# Train AI on cybersecurity datasets
python training/deepseek_training_pipeline.py

# Run initial adversarial training
python environments/adversarial_training_environment.py --initial-training
```

## 🔧 Advanced Configuration

### 1. Custom Agent Configuration

Create custom agent profiles in `config/agents/`:

```json
{
  "agent_id": "custom_threat_hunter",
  "agent_type": "ThreatHunterAgent",
  "specialization": "APT_detection",
  "ai_model": "deepseek-enhanced",
  "tools": ["nmap", "wireshark", "splunk"],
  "learning_rate": 0.1,
  "confidence_threshold": 0.8
}
```

### 2. Security Policy Configuration

Edit `config/security_policy.json`:

```json
{
  "authorization": {
    "require_explicit_approval": true,
    "approved_targets": ["127.0.0.1", "192.168.1.0/24"],
    "forbidden_actions": ["destructive_operations", "external_attacks"]
  },
  "logging": {
    "log_all_operations": true,
    "log_level": "INFO",
    "retention_days": 90
  },
  "containment": {
    "isolate_red_team": true,
    "network_restrictions": true,
    "resource_limits": true
  }
}
```

### 3. Performance Optimization

Configure for your hardware in `config/performance.json`:

```json
{
  "ai_inference": {
    "batch_size": 4,
    "max_concurrent_operations": 8,
    "gpu_memory_fraction": 0.8
  },
  "containers": {
    "max_containers": 10,
    "memory_per_container": "2GB",
    "cpu_per_container": 1
  },
  "caching": {
    "enable_model_caching": true,
    "cache_size_gb": 10,
    "threat_intel_cache_hours": 24
  }
}
```

## 🧪 Testing and Validation

### 1. Run System Tests

```bash
# Basic functionality tests
python -m pytest tests/test_basic_functionality.py

# AI integration tests
python -m pytest tests/test_ai_integration.py

# Security tool tests
python -m pytest tests/test_security_tools.py

# Container tests
python -m pytest tests/test_containers.py
```

### 2. Security Validation

```bash
# Validate security controls
python scripts/security_validation.py

# Test sandboxing
python scripts/test_sandbox.py

# Verify authorization controls
python scripts/test_authorization.py
```

### 3. Performance Testing

```bash
# AI performance benchmarks
python scripts/benchmark_ai.py

# System resource monitoring
python scripts/monitor_resources.py

# Container performance tests
python scripts/test_container_performance.py
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Model Download Failures
```bash
# Check internet connection
ping huggingface.co

# Clear model cache
rm -rf ~/.cache/huggingface

# Retry download with verbose logging
HF_HUB_VERBOSITY=debug python scripts/download_models.py
```

#### 2. Container Setup Issues
```bash
# Check Apple Container CLI installation
which apple-container-cli

# Verify containerization permissions
sudo apple-container-cli --version

# Reset container environment
python scripts/apple_container_setup.py --reset
```

#### 3. AI Integration Problems
```bash
# Test basic AI functionality
python -c "import torch; print(torch.__version__)"
python -c "import transformers; print(transformers.__version__)"

# Check model compatibility
python scripts/validate_models.py

# Reset AI configuration
python core/deepseek_integration.py --reset
```

#### 4. Security Tool Integration Issues
```bash
# Verify tool installations
python tools/kali_tool_integration.py --validate

# Check tool permissions
ls -la /usr/local/bin/nmap

# Test individual tools
python -c "
from tools.kali_tool_integration import KaliToolIntegration
tools = KaliToolIntegration()
print(tools.validate_tool_availability())
"
```

### Log Analysis

Check system logs for detailed error information:

```bash
# System logs
tail -f logs/archangel.log

# AI operation logs
tail -f logs/ai_operations.log

# Security operation logs
tail -f logs/security_ops.log

# Container logs
tail -f logs/containers.log
```

### Performance Optimization

If experiencing performance issues:

```bash
# Monitor system resources
python scripts/monitor_system.py

# Optimize AI model settings
python scripts/optimize_ai_settings.py

# Clean up temporary files
python scripts/cleanup.py
```

## 📊 Monitoring and Maintenance

### 1. System Health Monitoring

Setup automated health checks:

```bash
# Add to crontab for regular health checks
echo "0 */6 * * * cd /path/to/Archangel && python archangel_autonomous_system.py --health-check" | crontab -
```

### 2. Log Rotation

Configure log rotation in `config/logging.json`:

```json
{
  "rotation": {
    "max_size": "100MB",
    "backup_count": 5,
    "compress": true
  }
}
```

### 3. Model Updates

Setup automatic model updates:

```bash
# Weekly model update check
echo "0 2 * * 0 cd /path/to/Archangel && python scripts/update_models.py" | crontab -
```

## 🔒 Security Considerations

### 1. Access Control

Ensure proper permissions:

```bash
# Set restrictive permissions on configuration files
chmod 600 config/*.json

# Ensure log files are properly secured
chmod 640 logs/*.log

# Set proper ownership
chown -R $USER:staff .
```

### 2. Network Security

Configure firewall rules for container isolation:

```bash
# Block external access to containers
sudo pfctl -f config/pf.conf

# Monitor network traffic
sudo python scripts/monitor_network.py
```

### 3. Data Protection

Configure data encryption:

```bash
# Encrypt sensitive configuration files
python scripts/encrypt_config.py

# Setup secure key management
python scripts/setup_key_management.py
```

## 🎯 Next Steps

After successful installation:

1. **Complete the [User Guide](docs/user-guide.md)** for operational procedures
2. **Review [Security Operations](docs/security-operations.md)** for best practices
3. **Setup [Training Pipeline](docs/training.md)** for AI enhancement
4. **Configure [Monitoring](docs/monitoring.md)** for ongoing operations

## 📞 Support

If you encounter issues during setup:

1. Check the [Troubleshooting Guide](docs/troubleshooting.md)
2. Review [GitHub Issues](https://github.com/your-username/Archangel/issues)
3. Join the [Community Discussions](https://github.com/your-username/Archangel/discussions)
4. Contact the development team for enterprise support

---

**Setup complete! Your Archangel autonomous AI security system is ready for operation.**