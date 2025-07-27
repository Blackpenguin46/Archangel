# Archangel Linux - Project Structure

## Repository Organization

### Documentation
- `README.md` - Basic project overview
- `ARCHITECHTURE.md` - Detailed technical architecture and implementation
- `OUTLINE.md` - Complete project vision and automation examples
- `CLAUDE.md` - Development workflow and guidelines

### Core Directories (Planned)

#### `/kernel/` - Kernel AI Modules
- `archangel/` - Main kernel module directory
  - `ai_core.c/h` - Core AI engine and coordination
  - `syscall_ai.c/h` - System call filtering and analysis
  - `network_ai.c/h` - Network packet classification
  - `memory_ai.c/h` - Memory pattern analysis
  - `communication.c/h` - Kernel-userspace bridge
  - `security_module.c/h` - Security framework integration
  - `performance.c/h` - Performance optimizations
  - `Makefile` - Kernel module build configuration

#### `/opt/archangel/` - Userspace Components
- `ai/` - AI orchestration and complex reasoning
  - `orchestrator.py` - Main AI coordination engine
  - `complex_operations.py` - Multi-stage operation handlers
  - `models/` - AI model storage and management
- `tools/` - Security tool integration framework
  - `kernel_integration.py` - Kernel-aware tool framework
  - `wrappers/` - Enhanced tool wrappers (nmap, metasploit, etc.)
- `security/` - Guardian Protocol implementation
  - `guardian.py` - Multi-layer security validation
- `gui/` - Mission control interface
  - `mission_control.py` - Main GUI application
- `bin/` - Executable binaries
  - `archangel-cli` - Command-line interface
  - `archangel-agent` - Background AI daemon
  - `archangel-monitor` - System monitoring tool

#### `/build/` - Build System
- `build-archangel-iso.sh` - ISO generation script
- `integrate_kernel_ai.sh` - Kernel integration script
- `compile_models.sh` - AI model compilation

#### `/tests/` - Testing Framework
- `kernel/` - Kernel module tests
- `integration/` - End-to-end integration tests
- `performance/` - Performance benchmarking

#### `/install/` - Installation Scripts
- `install_archangel.sh` - Main installation script

## File Naming Conventions
- **Kernel modules**: `snake_case.c/h`
- **Python modules**: `snake_case.py`
- **Scripts**: `kebab-case.sh`
- **Configuration**: `lowercase.conf`

## Architecture Layers
1. **User Interface Layer** - Natural language shell, mission control GUI, REST API
2. **Userspace AI Orchestration** - LLM planning, complex analysis, learning modules
3. **Kernel-Userspace Bridge** - Shared memory, DMA buffers, event system
4. **Kernel AI Modules** - Syscall filter, network packet AI, memory pattern AI
5. **Security Framework** - Guardian protocol, audit logger, sandboxing
6. **Modified Linux Kernel** - AI-enhanced syscalls, security hooks, performance optimizations

## Development Workflow
- Kernel modules are the foundation - implement first
- Test each kernel module in isolation before integration
- Build userspace components incrementally
- Always maintain system stability during development
- Use hybrid approach: fast decisions in kernel, complex reasoning in userspace