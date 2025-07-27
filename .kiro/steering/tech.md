# Archangel Linux - Technical Stack

## Architecture
- **Hybrid Kernel-Userspace AI**: Custom Linux kernel modules for microsecond decisions, userspace for complex reasoning
- **Base System**: Modified Linux kernel 5.15+ with AI-enhanced syscalls and security hooks
- **Programming Languages**: C (kernel modules), Python (userspace AI), Rust (performance-critical components)

## Core Technologies

### Kernel Components
- **Custom Kernel Modules**: archangel_core, syscall_ai, network_ai, memory_ai
- **Real-time AI Engines**: Decision trees, pattern matchers, anomaly detectors
- **Communication**: Shared memory ring buffers, zero-copy DMA transfers
- **Security**: Guardian Protocol enforcement at kernel level

### Userspace AI Stack
- **LLM Integration**: CodeLlama, Security-BERT, custom security models
- **Frameworks**: PyTorch, TensorFlow Lite (for kernel deployment)
- **Communication**: Kernel-userspace bridge via eventfd and shared memory
- **Tool Orchestration**: Async Python with tool registry and execution engine

### Security Tools Integration
- **Network**: nmap, masscan, zmap, nikto, burpsuite, zaproxy
- **Exploitation**: Metasploit, SQLMap, custom exploit frameworks
- **OSINT**: theHarvester, Shodan API, subfinder, amass
- **Analysis**: Bloodhound, CrackMapExec, Mimikatz integration

## Build System
- **Kernel Modules**: Standard Linux kernel build system with custom Makefiles
- **AI Models**: TensorFlow Lite conversion pipeline for kernel deployment
- **ISO Generation**: Custom debootstrap-based build with squashfs compression
- **Package Management**: Debian-based with custom Archangel repositories

## Common Commands

### Development
```bash
# Build kernel modules
make -C /lib/modules/$(uname -r)/build M=$(PWD) modules

# Install AI models
archangel-models download all

# Build ISO
./build-archangel-iso.sh
```

### Operation
```bash
# Start AI agent
systemctl start archangel-agent

# Monitor kernel AI
archangel-monitor

# Execute operation
archangel-cli "Perform penetration test of target"
```

## Performance Requirements
- **Kernel AI**: <1ms inference time, <10MB memory, <5% CPU
- **Userspace AI**: Complex operations complete within minutes
- **Communication**: >1M messages/sec kernel-userspace throughput