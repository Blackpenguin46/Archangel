# Archangel Linux - 2 Week Sprint MVP

Archangel Linux is an autonomous AI security operating system that executes cybersecurity operations from natural language commands. This is the **2-week sprint MVP** focused on demonstrating core functionality.

## 🎯 Sprint Goal

Build a working demonstration that showcases:
- Natural language command processing
- AI-driven operation planning  
- Basic autonomous penetration testing
- Kernel-userspace coordination
- Live security tool execution

## 🚀 Quick Start (Sprint Setup)

```bash
# 1. Run sprint setup script
chmod +x scripts/sprint-setup.sh
./scripts/sprint-setup.sh

# 2. Activate sprint environment
cd ~/archangel-sprint
source activate-sprint.sh

# 3. Build and test
make build
sudo make kernel-install
make demo
```

## 📋 Sprint Plan

- **Week 1 (Days 1-7)**: Core foundation - kernel communication, AI integration, tool wrapper
- **Week 2 (Days 8-14)**: Demo polish - presentation prep, testing, final demo

See [2_WEEK_SPRINT_PLAN.md](2_WEEK_SPRINT_PLAN.md) for detailed daily breakdown.

## 🎬 Demo Commands

```bash
# Start the demo CLI
make demo

# Try these commands:
archangel> pentest 192.168.1.1
archangel> scan google.com  
archangel> status
archangel> help
```

## 📁 Sprint Structure

```
~/archangel-sprint/
├── core/planner.py          # Simplified LLM planner
├── tools/nmap_wrapper.py    # Basic nmap integration
├── kernel/simple_monitor.c  # Minimal kernel module
├── cli/main.py             # Demo CLI interface
└── tests/demo_test.py      # Functionality tests
```

## ⚡ MVP Features

- ✅ Natural language parsing
- ✅ AI operation planning
- ✅ Nmap tool execution
- ✅ Kernel-userspace communication
- ✅ Interactive CLI
- ✅ Demo-ready system

## 🎯 Success Criteria

- System runs stably for 15-minute demo
- Commands processed in <30 seconds
- AI generates reasonable plans
- Tools execute successfully
- Professional presentation ready

## ⚠️ Sprint Focus

This is a **streamlined MVP** for the 2-week deadline. Focus is on:
- Getting core functionality working
- Stable demo system
- Professional presentation
- **NOT** on perfect architecture or comprehensive features

## 📞 Emergency Protocols

If behind schedule:
- Days 1-7: Focus on core functionality only
- Days 8-11: Simplify demo, ensure stability  
- Days 12-14: Prepare backup presentation materials

Ready to build the future of autonomous security! 🚀