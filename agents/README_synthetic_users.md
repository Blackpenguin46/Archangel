# Synthetic User Simulation System

## Overview

The Synthetic User Simulation System is a comprehensive component of the Archangel Autonomous AI Evolution framework that creates realistic background activity in the mock enterprise environment. It simulates autonomous synthetic users with role-based behavior patterns to provide cover for Red Team activities and test Blue Team detection capabilities.

## Key Features

### 🤖 Autonomous User Agents
- **Comprehensive Behavior Simulation**: Web browsing, file access, email activity, and login/logout patterns
- **Role-Based Differentiation**: Different behavior patterns for Admin, Developer, Sales, HR, Finance, Marketing, and Support roles
- **Realistic Timing**: Work hours awareness, realistic activity frequencies, and natural timing variations
- **Persistent Learning**: Activity logging and pattern evolution over time

### 🎭 Behavior Realism
- **Role-Specific Patterns**: Each user role has unique web browsing, file access, and email patterns
- **Realistic User Agents**: Browser signatures appropriate for each role and department
- **Natural Timing**: Activity durations and intervals that mimic real user behavior
- **Success Rate Modeling**: Realistic failure rates for different types of activities

### 🕵️ Detection Evasion
- **Background Noise Generation**: Creates legitimate traffic to mask attacker activities
- **Temporal Distribution**: Activities spread naturally throughout work hours
- **Network Realism**: Internal IP addresses and realistic traffic patterns
- **Behavioral Diversity**: Multiple users with different activity levels and patterns

### 📊 Comprehensive Logging
- **Activity Tracking**: Complete logs of all synthetic user activities
- **JSONL Format**: Structured logging compatible with SIEM systems
- **Forensic Analysis**: Detailed activity replay and analysis capabilities
- **Performance Metrics**: Success rates, timing analysis, and behavior summaries

## Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                 Synthetic User Manager                          │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │  Configuration  │ │   User Profiles │ │  Activity Log   │   │
│  │    Manager      │ │    Manager      │ │    Manager      │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                 Comprehensive Synthetic Users                   │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │   Web Browsing  │ │  File Access    │ │ Email Activity  │   │
│  │   Simulator     │ │  Simulator      │ │   Simulator     │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                    Activity Simulators                         │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │ Login/Logout    │ │ Background      │ │   Behavioral    │   │
│  │   Patterns      │ │    Noise        │ │   Patterns      │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### User Roles and Behavior Patterns

| Role | Web Patterns | File Patterns | Activity Level | Risk Profile |
|------|-------------|---------------|----------------|--------------|
| **Admin** | /admin, /dashboard, /system | /var/log/, /etc/, /scripts/ | Low (0.5) | Low |
| **Developer** | /api, /docs, /git, /jenkins | /home/dev/, /var/www/, /src/ | High (0.8) | Medium |
| **Sales** | /crm, /leads, /reports | /shared/sales/, /proposals/ | Very High (1.0) | High |
| **HR** | /hr, /employees, /payroll | /shared/hr/, /policies/ | Medium (0.6) | Low |
| **Finance** | /finance, /accounting, /budgets | /shared/finance/, /reports/ | Medium (0.7) | Low |
| **Marketing** | /marketing, /campaigns, /analytics | /shared/marketing/, /content/ | High (0.9) | Medium |
| **Support** | /support, /tickets, /knowledge-base | /shared/support/, /tickets/ | Very High (1.2) | Medium |

## Installation and Setup

### Prerequisites
- Python 3.8+
- asyncio support
- aiohttp (for web simulation)
- aiofiles (for file operations)

### Installation
```bash
# The synthetic user system is part of the Archangel framework
# No additional installation required beyond the main project dependencies
```

### Configuration
```bash
# Create a sample configuration file
python3 agents/synthetic_user_cli.py create-config synthetic_user_config.yaml

# Initialize with configuration
python3 agents/synthetic_user_cli.py init --config synthetic_user_config.yaml

# Validate configuration
python3 agents/synthetic_user_cli.py validate
```

## Usage

### Command Line Interface

#### Basic Operations
```bash
# List configured users
python3 agents/synthetic_user_cli.py list-users

# Show current configuration
python3 agents/synthetic_user_cli.py show-config

# Validate configuration
python3 agents/synthetic_user_cli.py validate
```

#### Running Simulations
```bash
# Run simulation indefinitely
python3 agents/synthetic_user_cli.py run

# Run simulation for specific duration (1 hour)
python3 agents/synthetic_user_cli.py run --duration 3600

# Test specific user for 1 minute
python3 agents/synthetic_user_cli.py test-user dev001 --duration 60
```

#### Log Analysis
```bash
# View recent activity logs
python3 agents/synthetic_user_cli.py logs --lines 100

# View logs in real-time (during simulation)
tail -f synthetic_user_activities.jsonl
```

### Programmatic Usage

#### Basic Setup
```python
from agents.synthetic_users import SyntheticUserManager, UserProfile, UserRole
from agents.synthetic_user_config import SyntheticUserConfigManager

# Create configuration manager
config_manager = SyntheticUserConfigManager()
config_manager.create_default_config()

# Create user manager
environment_config = {
    'web_server_url': 'http://192.168.10.10',
    'user_ip': '192.168.20.100'
}
user_manager = SyntheticUserManager(environment_config)

# Add users from configuration
for profile in config_manager.user_profiles:
    await user_manager.add_user(profile)

# Start simulation
await user_manager.start_all_users()
```

#### Custom User Creation
```python
# Create custom user profile
custom_profile = UserProfile(
    user_id="custom001",
    username="custom_user",
    role=UserRole.DEVELOPER,
    department="Engineering",
    email="custom@company.com",
    work_hours=(9, 17),
    activity_frequency=0.8,
    web_browsing_patterns=["/api", "/docs"],
    file_access_patterns=["/home/dev/"],
    email_patterns={"frequency": "medium"},
    risk_profile="medium"
)

# Add to manager
await user_manager.add_user(custom_profile)
```

## Configuration

### Environment Configuration
```yaml
environment:
  web_server_url: "http://192.168.10.10"
  mail_server_url: "http://192.168.10.20"
  file_server_url: "http://192.168.10.30"
  user_network_range: "192.168.20.0/24"
  default_user_ip: "192.168.20.100"
  simulation_speed: 1.0
  max_concurrent_users: 50
```

### Behavior Configuration
```yaml
behavior:
  min_activity_interval: 30
  max_activity_interval: 3600
  web_browsing_success_rate: 0.95
  file_access_success_rate: 0.90
  email_success_rate: 0.98
  risky_behavior_probability:
    low: 0.05
    medium: 0.15
    high: 0.30
```

### User Profile Configuration
```yaml
user_profiles:
  - user_id: "dev001"
    username: "developer"
    role: "developer"
    department: "Engineering"
    email: "dev@company.com"
    work_hours: [9, 18]
    activity_frequency: 0.8
    web_browsing_patterns:
      - "/api"
      - "/docs"
      - "/git"
    file_access_patterns:
      - "/home/dev/"
      - "/var/www/"
    email_patterns:
      frequency: "medium"
    risk_profile: "medium"
```

## Integration with Archangel Framework

### Red Team Benefits
- **Reconnaissance Cover**: Background traffic masks scanning and enumeration activities
- **Lateral Movement Cover**: Normal file access patterns hide malicious file operations
- **Persistence Cover**: Regular login/logout patterns mask backdoor activities
- **Exfiltration Cover**: Email and web traffic provides cover for data exfiltration

### Blue Team Benefits
- **Detection Testing**: Realistic baseline for anomaly detection systems
- **Alert Tuning**: Helps calibrate SIEM rules and reduce false positives
- **Response Training**: Provides realistic environment for incident response practice
- **Forensic Analysis**: Complex logs for forensic investigation training

### SIEM Integration
- **Structured Logging**: JSONL format compatible with ELK stack, Splunk, and other SIEM systems
- **Metadata Rich**: Comprehensive activity details for correlation and analysis
- **Real-time Streaming**: Continuous log generation during simulation
- **Historical Analysis**: Persistent logs for trend analysis and baseline establishment

## Testing and Validation

### Automated Testing
```bash
# Run comprehensive tests
python3 test_synthetic_users_standalone.py

# Run specific test categories
python3 -c "
from test_synthetic_users_standalone import *
test_realistic_behavior_characteristics()
test_detection_evasion_features()
"
```

### Manual Testing
```bash
# Test individual user for short duration
python3 agents/synthetic_user_cli.py test-user admin001 --duration 30

# Monitor activity logs in real-time
tail -f synthetic_user_activities.jsonl | jq '.'
```

### Validation Metrics
- **Behavior Realism**: Role-appropriate activity patterns
- **Timing Accuracy**: Realistic activity durations and intervals
- **Success Rates**: Appropriate failure rates for different activities
- **Detection Evasion**: Ability to blend with legitimate traffic

## Performance and Scalability

### Performance Characteristics
- **Concurrent Users**: Supports 50+ concurrent synthetic users
- **Activity Rate**: 100+ activities per minute across all users
- **Memory Usage**: ~10MB per active user
- **CPU Usage**: Low overhead with async/await architecture

### Scaling Considerations
- **Horizontal Scaling**: Multiple instances for larger simulations
- **Resource Management**: Configurable activity rates and user limits
- **Log Management**: Automatic log rotation and cleanup
- **Network Impact**: Minimal network overhead with mock services

## Troubleshooting

### Common Issues

#### Configuration Errors
```bash
# Validate configuration
python3 agents/synthetic_user_cli.py validate

# Check for common issues
- Duplicate user IDs or usernames
- Invalid work hours (start >= end)
- Invalid activity frequencies (≤ 0)
- Invalid risk profiles (not low/medium/high)
```

#### Runtime Issues
```bash
# Check user activity
python3 agents/synthetic_user_cli.py logs --lines 50

# Test individual user
python3 agents/synthetic_user_cli.py test-user <user_id> --duration 60

# Monitor system resources
top -p $(pgrep -f synthetic_user)
```

#### Log Analysis
```bash
# Check log file exists and is being written
ls -la synthetic_user_activities.jsonl

# Validate JSON format
tail -n 10 synthetic_user_activities.jsonl | jq '.'

# Count activities by type
cat synthetic_user_activities.jsonl | jq -r '.activity_type' | sort | uniq -c
```

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed logging for troubleshooting
```

## Security Considerations

### Containment
- **Network Isolation**: All activities contained within mock environment
- **No Real Services**: Simulated interactions with mock services only
- **IP Range Restriction**: Activities limited to internal network ranges
- **Service Mocking**: No actual connections to external services

### Data Protection
- **No Real Data**: All simulated data is synthetic and non-sensitive
- **Anonymized Logs**: No personally identifiable information in logs
- **Secure Configuration**: Configuration files contain no real credentials
- **Audit Trail**: Complete logging of all synthetic activities

## Future Enhancements

### Planned Features
- **Machine Learning Integration**: Adaptive behavior patterns based on environment feedback
- **Advanced Deception**: Integration with honeypot and deception technologies
- **Behavioral Evolution**: Learning from Red/Blue team interactions
- **Custom Scenarios**: Scenario-specific behavior pattern loading
- **Real-time Adaptation**: Dynamic behavior adjustment based on detection events

### Integration Roadmap
- **MITRE ATT&CK Mapping**: Activity classification using MITRE framework
- **Threat Intelligence**: Behavior patterns based on real threat actor TTPs
- **Advanced Analytics**: Machine learning for behavior pattern optimization
- **Cloud Integration**: Support for cloud-based mock environments

## Contributing

### Development Setup
```bash
# Clone repository
git clone <repository-url>

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python3 test_synthetic_users_standalone.py
```

### Code Style
- Follow PEP 8 style guidelines
- Use type hints for all function parameters and return values
- Include comprehensive docstrings for all classes and methods
- Write unit tests for all new functionality

### Testing Requirements
- All new features must include unit tests
- Integration tests for complex workflows
- Performance tests for scalability features
- Security tests for containment verification

## License

This synthetic user simulation system is part of the Archangel Autonomous AI Evolution framework and is subject to the same license terms as the main project.

## Support

For questions, issues, or contributions related to the synthetic user simulation system:

1. Check the troubleshooting section above
2. Review existing issues in the project repository
3. Create a new issue with detailed information about the problem
4. Include configuration files, log excerpts, and error messages when reporting issues

---

**Note**: This system is designed for cybersecurity research and training purposes only. All activities are contained within the mock environment and should never be used against real systems or networks without explicit authorization.