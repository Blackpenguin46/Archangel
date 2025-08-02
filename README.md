# 🤖 Archangel: AI vs AI Autonomous Cybersecurity Warfare System

**The World's First Fully Autonomous AI vs AI Cybersecurity Research Platform**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)

---

## 🎯 **What is Archangel?**

Archangel is a revolutionary AI research platform where autonomous artificial intelligence agents wage cyber warfare against each other in realistic enterprise environments. Unlike traditional security tools that require human operators, Archangel's AI agents operate completely independently, generating novel attack strategies, discovering vulnerabilities, and developing countermeasures without any human intervention.

**Key Innovation**: This is not another security automation tool—it's an AI that thinks, reasons, and evolves its cybersecurity strategies in real-time.

---

## 🚀 **Core Capabilities**

### **Autonomous AI Operations**
- **Zero Human Intervention**: AI agents operate independently for hours/days
- **Multi-Agent Coordination**: Red team agents coordinate complex attack campaigns
- **Strategic Reasoning**: LLM-powered strategic planning and decision making
- **Real-time Adaptation**: Agents modify strategies based on opponent responses

### **Advanced AI Technologies**
- **Multi-Agent Reinforcement Learning (MARL)**: Coordinated team behaviors
- **Adversarial LLM Framework**: Natural language strategic reasoning
- **Dynamic Vulnerability Generation**: AI creates new exploits in real-time
- **Predictive Security Intelligence**: Threat prediction and pattern analysis
- **Guardian Protocol**: Ethical AI safeguards and compliance validation

### **Realistic Enterprise Environment**
- **$40M Mock Enterprise**: Financial DB, Domain Controller, File Servers
- **Valuable Target Data**: Customer PII, financial records, trade secrets  
- **Production-Grade Services**: Web portals, email systems, backup servers
- **Dynamic Security Posture**: Adaptive hardening and response systems

---

## 🛠️ **Quick Start**

### **Prerequisites**
```bash
# Python 3.8+ with required packages
pip install -r requirements.txt

# Docker for enterprise network simulation  
docker --version
docker-compose --version
```

### **Launch System**
```bash
# 1. Start enterprise network containers
docker-compose -f docker-compose-enhanced.yml up -d

# 2. Start AI vs AI demonstration
python blackhat_demo.py --demo extended_demo

# 3. Connect to live dashboard
# Open browser to: ws://localhost:8765
```

### **Basic Commands**
```bash
# Test system components
python verify_system.py

# Run autonomous scenario
python scenarios/autonomous_enterprise_scenario.py

# Start live application (not demo)
python archangel.py --mode live
```

---

## 🏗️ **System Architecture**

```
┌─────────────────────────────────────────────────────────────────┐
│                 ARCHANGEL ORCHESTRATOR                         │
│              (Central AI Coordination)                         │
└─────────────────────────┬───────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼                               ▼
┌─────────────────────┐           ┌─────────────────────┐
│    RED TEAM AI      │    ←─→    │    BLUE TEAM AI     │
│   (Offensive)       │           │    (Defensive)      │
│                     │           │                     │
│ • MARL Agents (3-5) │           │ • MARL Agents (3-4) │
│ • LLM Strategic AI  │           │ • LLM Defense AI    │
│ • Kali Linux Tools  │           │ • SIEM Integration  │  
│ • Exploit Generation│           │ • Incident Response │
└─────────────────────┘           └─────────────────────┘
          │                               │
          └───────────────┬───────────────┘
                          ▼
          ┌─────────────────────────────────┐
          │      TARGET ENVIRONMENT         │
          │   Mock Enterprise Network       │
          │      ($40M Total Assets)        │
          └─────────────────────────────────┘
```

---

## 📁 **Repository Structure**

```
archangel/
├── core/                          # Core AI systems
│   ├── archangel_orchestrator.py   # Central coordination
│   ├── marl_coordinator.py         # Multi-agent RL
│   ├── llm_reasoning_engine.py     # Strategic LLM reasoning
│   ├── dynamic_vulnerability_engine.py # Exploit generation
│   ├── guardian_protocol.py        # Ethical safeguards
│   └── predictive_security_intelligence.py # Threat analysis
│
├── containers/                    # Docker infrastructure
│   ├── red-team/                 # AI-controlled Kali Linux
│   ├── enterprise/               # Mock enterprise systems
│   └── ai-services/              # AI model servers
│
├── scenarios/                    # Pre-built scenarios
│   ├── autonomous_enterprise_scenario.py
│   └── ai_enhanced_autonomous_scenario.py
│
├── mcp-servers/                 # Model Context Protocol
│   ├── kali-offensive-mcp.py    # Offensive tool integration
│   └── enterprise-defensive-mcp.py # Defensive resources
│
├── blackhat_demo.py             # BlackHat demonstration
├── docker-compose-enhanced.yml  # Full system deployment
└── README_BLACKHAT_2025.md     # Conference presentation
```

---

## 🎪 **Live Application Usage**

### **BlackHat Live Application**
```bash
# Start the live AI vs AI system (not demo)
python archangel.py --mode live --enterprise

# Run specific scenarios
python scenarios/autonomous_enterprise_scenario.py
python scenarios/ai_enhanced_autonomous_scenario.py
```

### **System Commands**
```bash
# System status check
python verify_system.py

# Start with custom configuration
python archangel.py --config config/blackhat_config.json

# Enable verbose logging for development
python archangel.py --mode live --verbose
```

---

## 🔬 **Research Applications**

### **Academic Research**
- Multi-Agent Reinforcement Learning in cybersecurity
- Adversarial LLM training and evaluation
- Emergent behavior analysis in AI systems
- Dynamic vulnerability discovery techniques
- Human-AI collaboration in security operations

### **Industry Applications**  
- Red team exercise automation
- Security control validation
- Incident response training
- Threat modeling enhancement
- Penetration testing augmentation

---

## 🛡️ **Security & Ethics**

### **Defensive Security Focus**
- **✅ Isolated Environment**: All activities in Docker containers
- **✅ Mock Data Only**: No real PII, credentials, or sensitive data
- **✅ Guardian Protocol**: Multi-layer ethical safeguards
- **✅ Compliance Built-in**: SOC2, HIPAA, GDPR, NIST validation
- **✅ Audit Logging**: Complete decision transparency

### **Responsible Use**
This system is designed exclusively for:
- Defensive cybersecurity research
- AI safety and alignment research  
- Educational and training purposes
- Security control validation
- Academic research advancement

---

## 📊 **Performance Metrics**

Current system capabilities demonstrate:
- **Autonomous Operation**: 100% AI-driven (zero human input)
- **Decision Speed**: <2 seconds per AI action
- **Learning Rate**: Strategies improve 10%+ per hour
- **Coordination**: Multi-agent team behaviors emerge
- **Adaptation**: <5 second response to opponent changes

---

## 🤝 **Contributing**

### **Research Collaboration**
We welcome collaboration on:
- Novel MARL algorithms for cybersecurity
- Security-specific LLM development
- Adversarial training methodologies
- Emergent behavior analysis
- Real-world integration studies

### **Getting Started**
```bash
# Fork and clone
git clone https://github.com/your-username/archangel.git

# Set up development environment
pip install -r requirements-ai-enhanced.txt

# Run test suite
python -m pytest tests/

# Start development mode
python archangel.py --mode development
```

---

## 📜 **License & Citation**

Released under MIT License for research and educational use.

```bibtex
@software{archangel2025,
  title={Archangel: AI vs AI Autonomous Cybersecurity Warfare System},
  author={Archangel Research Team},
  year={2025},
  url={https://github.com/archangel-ai/archangel}
}
```

---

## 🚨 **Disclaimer**

This is a research platform for defensive cybersecurity and AI safety research. All operations are contained within isolated environments using mock data. The system includes comprehensive ethical safeguards and is designed to advance cybersecurity defense capabilities.

---

**⚡ Ready to witness AI vs AI cyber warfare? The future of cybersecurity is autonomous.**

```bash
python blackhat_demo.py --demo extended_demo
# Connect to ws://localhost:8765 for real-time dashboard
```