# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Archangel Linux is an AI security expert system that combines conversational AI with autonomous penetration testing capabilities. This is NOT another security automation tool - it's an AI that thinks, reasons, and teaches security concepts while operating.

**Key Innovation**: An AI that understands security, doesn't just automate it. The system features:
- AI that explains its decision-making process
- Conversational interface for security discussions  
- Autonomous operation with real-time adaptation
- Educational value while performing security operations

## Architecture

This project follows a **hybrid kernel-userspace architecture** designed for defensive security research:

### Planning Phase Structure
- `docs/` - Core planning documents and feasibility analysis
- `tasks/` - Implementation roadmaps and sprint plans
- `BlackHat-requirements/` - Conference submission materials

### Planned Technical Architecture
The feasibility analysis reveals the system should use:
- **Kernel Space**: Lightweight rule-based security filters, pattern matching, decision cache (<1ms response)
- **Userspace**: Complex AI reasoning, LLM planning engines, learning systems (10-1000ms acceptable)
- **Communication Bridge**: High-speed shared memory and lock-free queues

## Development Approach

### Current Status
This repository is in the **planning phase** - contains documentation and specifications for building Archangel Linux. The actual implementation will be done using Claude Code following the AI-centric development plan.

### Implementation Strategy
Based on the documentation, follow this approach:

1. **Week 1: AI Security Expert Brain**
   - AI reasoning and explanation systems
   - Conversational AI interface
   - Autonomous decision making
   - Learning and adaptation capabilities

2. **Week 2: Integration & Demo**
   - AI-driven tool orchestration
   - Demo scenario development
   - Presentation preparation

### Key Design Principles
- **AI-Centric**: All development should prioritize AI reasoning capabilities
- **Educational Focus**: AI must explain its decisions and teach security concepts
- **Hybrid Architecture**: Use kernel space for <1ms decisions, userspace for complex AI
- **Defensive Security Only**: This is a defensive security research project

## Important Constraints

### Security Guidelines
- **DEFENSIVE SECURITY ONLY**: This project is for defensive security research and education
- Never create code that could be used maliciously
- Focus on security analysis, detection, and educational applications
- Maintain ethical boundaries in all AI implementations

### Technical Constraints
- No actual kernel modules exist yet - this is planning phase
- Focus on Python-based AI systems for initial development
- Use local LLM integration (Ollama with CodeLlama recommended)
- Implement hybrid architecture with proper separation of concerns

## Documentation Structure

The key planning documents are:
- `docs/PROJECT_OVERVIEW.md` - Core vision and competitive positioning
- `docs/feasibility_analysis.md` - Technical feasibility assessment of kernel AI architecture
- `docs/REVISED_2_WEEK_AI_CENTRIC_PLAN.md` - AI-focused development roadmap
- `tasks/14_day_blackhat_implementation_plan.md` - Detailed implementation timeline

## Hugging Face Integration Strategy

Hugging Face provides comprehensive AI infrastructure that will be central to Archangel's capabilities:

### Core AI Infrastructure
- **SmolAgents**: Sandboxed autonomous agents with code execution capabilities
- **Transformers Library**: State-of-the-art model deployment and fine-tuning
- **Model Hub**: Access to 500,000+ pre-trained models including security-specific models
- **Datasets**: Security corpora like `zeroshot/cybersecurity-corpus` for training

### Security-Specific Resources
- **Foundation-Sec-8B**: 8B parameter cybersecurity-specialized LLM (Cisco's Foundation AI)
- **Cybersecurity Models**: Specialized models for threat intelligence and incident response
- **Security Datasets**: Curated cybersecurity corpora for model training and fine-tuning
- **SOC Acceleration Tools**: Automated triage, case generation, evidence collection

### Development & Deployment Platform
- **Spaces**: Rapid prototyping and interactive demos for security tools
- **Inference Endpoints**: Dedicated deployment for production security systems
- **Local Deployment**: Privacy-preserving model execution for sensitive security work
- **Gradio Integration**: Quick UI development for security tool interfaces

### Learning & Research Resources
- **Agents Course**: Building and deploying autonomous AI agents
- **LLM Course**: Advanced language model development
- **Research Papers**: Latest security AI research and risk management frameworks
- **AI Cookbook**: Practical implementation examples and best practices

### Security & Compliance
- **SOC2 Type 2 Certified**: Enterprise-grade security standards
- **SafeTensors Format**: Secure model serialization preventing code execution attacks
- **Access Tokens & MFA**: Secure API access and authentication
- **Malware Scanning**: Automated security checks for uploaded models

## Development Commands

Based on the planning documents and Hugging Face integration:

```bash
# Environment setup with Hugging Face
./scripts/ai-centric-setup.sh
pip install transformers datasets gradio spaces-cli
huggingface-cli login

# Hugging Face model deployment
python scripts/deploy-security-models.py
gradio app.py  # Launch interactive demo

# Build system (planned)
make build
make test
make demo

# Kernel module operations (future implementation)
make -C kernel all
sudo make kernel-install

# AI system testing with HF models
./test-ai-demo.sh --model foundation-sec-8b
python test/test_smolagents_integration.py
```

**Note**: These commands combine planning documents with Hugging Face integration. Implement as part of the development process.

## AI Development Guidelines

When implementing the AI systems:

1. **Prioritize Reasoning**: AI must explain every decision it makes
2. **Conversational Interface**: Enable natural language security discussions
3. **Educational Value**: AI should teach security concepts while operating
4. **Adaptive Learning**: System should improve with each operation
5. **Ethical Boundaries**: Maintain defensive security focus throughout

### Hugging Face Integration Best Practices

1. **Use SafeTensors Format**: Always prioritize safetensors over pickle for security
2. **Leverage Foundation-Sec-8B**: Cisco's cybersecurity-specialized model for domain expertise
3. **SmolAgents for Autonomy**: Implement sandboxed autonomous agents for safe tool execution
4. **Gradio for Demos**: Create interactive security tool demonstrations
5. **Security-First Datasets**: Train on curated cybersecurity corpora from HF Datasets
6. **Spaces for Prototyping**: Rapid development and testing of security AI concepts
7. **Local Model Deployment**: Privacy-preserving execution for sensitive security operations

### Additional Hugging Face Advantages

**Complete AI Ecosystem**:
- **500,000+ Models**: Access to the world's largest AI model repository
- **100,000+ Datasets**: Comprehensive data sources including security-specific corpora
- **Enterprise Security**: SOC2 Type 2 certification and malware scanning
- **Multi-Modal Capabilities**: Text, vision, audio for comprehensive security analysis
- **Community Research**: Latest AI security research papers and methodologies
- **Professional Training**: Structured courses for AI agent development
- **Production Infrastructure**: Scalable inference endpoints and deployment tools
- **Open Source Ecosystem**: Transparent, auditable AI development platform

## Project Goals

The ultimate goal is to demonstrate an AI that:
- Thinks like a senior security consultant
- Explains its reasoning process transparently
- Adapts strategies based on discoveries
- Teaches users about security while operating
- Operates autonomously with minimal human intervention

This represents a paradigm shift from "AI that executes commands" to "AI that understands security."