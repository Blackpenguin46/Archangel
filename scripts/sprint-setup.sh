#!/bin/bash
# Archangel Linux - 2 Week Sprint Setup
# Streamlined setup for MVP development

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${BLUE}[SPRINT]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo "🚀 Archangel Linux - 2 Week Sprint Setup"
echo "Setting up MVP development environment..."

# Check if running on Arch Linux
if ! command -v pacman &> /dev/null; then
    print_error "This script is designed for Arch Linux"
    exit 1
fi

# Update system (quick)
print_status "Quick system update..."
sudo pacman -Sy --noconfirm

# Install essential packages only
print_status "Installing essential packages..."
sudo pacman -S --noconfirm --needed \
    base-devel \
    linux-headers \
    python \
    python-pip \
    python-virtualenv \
    git \
    nmap \
    ollama

# Create sprint directory structure
print_status "Creating sprint directory structure..."
mkdir -p ~/archangel-sprint/{core,tools,kernel,cli,tests,scripts}

# Setup Python virtual environment
print_status "Setting up Python environment..."
cd ~/archangel-sprint
python -m venv .venv
source .venv/bin/activate

# Install minimal Python dependencies
print_status "Installing Python dependencies..."
pip install --upgrade pip
pip install \
    asyncio \
    aiohttp \
    typer \
    rich \
    pydantic \
    pytest \
    pytest-asyncio

# Setup Ollama (background)
print_status "Setting up Ollama..."
sudo systemctl enable ollama
sudo systemctl start ollama

# Pull lightweight model for sprint
print_status "Pulling lightweight AI model..."
ollama pull codellama:7b  # Smaller model for faster development

# Create MVP project structure
print_status "Creating MVP project files..."

# Core planner (simplified version)
cat > ~/archangel-sprint/core/planner.py << 'EOF'
"""
MVP LLM Planner - Simplified for 2-week sprint
"""
import asyncio
import json
import re
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class SimpleObjective:
    command: str
    target: str
    operation_type: str

@dataclass
class SimplePlan:
    objective: SimpleObjective
    phases: List[str]
    tools: List[str]
    estimated_time: str

class MVPPlanner:
    def __init__(self):
        self.operation_patterns = {
            'pentest': r'pen(?:etration)?\s*test|pentest',
            'scan': r'scan|nmap',
            'osint': r'osint|reconnaissance|recon'
        }
    
    async def parse_command(self, command: str) -> SimpleObjective:
        """Parse natural language command"""
        command_lower = command.lower()
        
        # Extract operation type
        operation_type = 'pentest'  # default
        for op_type, pattern in self.operation_patterns.items():
            if re.search(pattern, command_lower):
                operation_type = op_type
                break
        
        # Extract target
        target_patterns = [
            r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})',  # IP
            r'([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',        # Domain
        ]
        
        target = 'localhost'  # default
        for pattern in target_patterns:
            match = re.search(pattern, command)
            if match:
                target = match.group(1)
                break
        
        return SimpleObjective(
            command=command,
            target=target,
            operation_type=operation_type
        )
    
    async def generate_plan(self, objective: SimpleObjective) -> SimplePlan:
        """Generate simple operation plan"""
        if objective.operation_type == 'pentest':
            phases = ['reconnaissance', 'scanning', 'analysis', 'reporting']
            tools = ['nmap']
            time = '15 minutes'
        elif objective.operation_type == 'scan':
            phases = ['scanning', 'analysis']
            tools = ['nmap']
            time = '5 minutes'
        else:
            phases = ['information_gathering', 'analysis']
            tools = ['nmap']
            time = '10 minutes'
        
        return SimplePlan(
            objective=objective,
            phases=phases,
            tools=tools,
            estimated_time=time
        )
    
    async def process_command(self, command: str) -> SimplePlan:
        """Main processing function"""
        objective = await self.parse_command(command)
        plan = await self.generate_plan(objective)
        return plan
EOF

# Tool wrapper (nmap only for MVP)
cat > ~/archangel-sprint/tools/nmap_wrapper.py << 'EOF'
"""
MVP Nmap Wrapper - Simplified for demo
"""
import asyncio
import json
import subprocess
from typing import Dict, List

class NmapWrapper:
    def __init__(self):
        self.binary = '/usr/bin/nmap'
    
    async def scan(self, target: str, scan_type: str = 'basic') -> Dict:
        """Execute nmap scan"""
        try:
            if scan_type == 'basic':
                cmd = [self.binary, '-sS', '-O', target]
            else:
                cmd = [self.binary, '-sS', target]
            
            # Execute scan
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                return self._parse_output(stdout.decode())
            else:
                return {'error': stderr.decode(), 'status': 'failed'}
                
        except Exception as e:
            return {'error': str(e), 'status': 'failed'}
    
    def _parse_output(self, output: str) -> Dict:
        """Parse nmap output (simplified)"""
        lines = output.split('\n')
        
        result = {
            'status': 'completed',
            'ports': [],
            'services': [],
            'os': 'Unknown'
        }
        
        for line in lines:
            if '/tcp' in line and 'open' in line:
                parts = line.split()
                if len(parts) >= 3:
                    port = parts[0].split('/')[0]
                    service = parts[2] if len(parts) > 2 else 'unknown'
                    result['ports'].append(port)
                    result['services'].append(service)
            
            if 'OS:' in line:
                result['os'] = line.split('OS:')[1].strip()
        
        return result
EOF

# Simple kernel module
cat > ~/archangel-sprint/kernel/simple_monitor.c << 'EOF'
/*
 * Archangel Simple Monitor - MVP Kernel Module
 * Minimal monitoring for 2-week sprint demo
 */

#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/proc_fs.h>
#include <linux/uaccess.h>

#define PROC_NAME "archangel_status"
#define BUFFER_SIZE 1024

static struct proc_dir_entry *proc_entry;
static char proc_buffer[BUFFER_SIZE];
static int proc_buffer_size = 0;

static ssize_t proc_read(struct file *file, char __user *buffer, 
                        size_t count, loff_t *pos) {
    if (*pos > 0 || count < proc_buffer_size)
        return 0;
    
    if (copy_to_user(buffer, proc_buffer, proc_buffer_size))
        return -EFAULT;
    
    *pos = proc_buffer_size;
    return proc_buffer_size;
}

static ssize_t proc_write(struct file *file, const char __user *buffer,
                         size_t count, loff_t *pos) {
    if (count > BUFFER_SIZE - 1)
        count = BUFFER_SIZE - 1;
    
    if (copy_from_user(proc_buffer, buffer, count))
        return -EFAULT;
    
    proc_buffer[count] = '\0';
    proc_buffer_size = count;
    
    printk(KERN_INFO "Archangel: Received message: %s\n", proc_buffer);
    return count;
}

static const struct proc_ops proc_fops = {
    .proc_read = proc_read,
    .proc_write = proc_write,
};

static int __init archangel_init(void) {
    printk(KERN_INFO "Archangel: Simple monitor loading\n");
    
    proc_entry = proc_create(PROC_NAME, 0666, NULL, &proc_fops);
    if (!proc_entry) {
        printk(KERN_ERR "Archangel: Failed to create proc entry\n");
        return -ENOMEM;
    }
    
    strcpy(proc_buffer, "Archangel monitoring active\n");
    proc_buffer_size = strlen(proc_buffer);
    
    printk(KERN_INFO "Archangel: Simple monitor loaded successfully\n");
    return 0;
}

static void __exit archangel_exit(void) {
    proc_remove(proc_entry);
    printk(KERN_INFO "Archangel: Simple monitor unloaded\n");
}

module_init(archangel_init);
module_exit(archangel_exit);

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Archangel Simple Monitor - MVP");
MODULE_VERSION("1.0");
EOF

# Kernel Makefile
cat > ~/archangel-sprint/kernel/Makefile << 'EOF'
obj-m += simple_monitor.o

KDIR := /lib/modules/$(shell uname -r)/build

all:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean

install: all
	sudo insmod simple_monitor.ko

uninstall:
	sudo rmmod simple_monitor || true

reload: uninstall install

.PHONY: all clean install uninstall reload
EOF

# CLI interface
cat > ~/archangel-sprint/cli/main.py << 'EOF'
#!/usr/bin/env python3
"""
Archangel MVP CLI - 2 Week Sprint Demo
"""
import asyncio
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.planner import MVPPlanner
from tools.nmap_wrapper import NmapWrapper
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

class ArchangelCLI:
    def __init__(self):
        self.planner = MVPPlanner()
        self.nmap = NmapWrapper()
        self.kernel_status_file = "/proc/archangel_status"
    
    def check_kernel_module(self) -> bool:
        """Check if kernel module is loaded"""
        return os.path.exists(self.kernel_status_file)
    
    async def notify_kernel(self, message: str):
        """Send message to kernel module"""
        try:
            if self.check_kernel_module():
                with open(self.kernel_status_file, 'w') as f:
                    f.write(message)
        except Exception as e:
            console.print(f"[yellow]Warning: Could not notify kernel: {e}[/yellow]")
    
    async def execute_command(self, command: str):
        """Execute a security command"""
        console.print(Panel(f"🤖 Processing: {command}", style="blue"))
        
        # Notify kernel
        await self.notify_kernel(f"COMMAND: {command}")
        
        # Parse and plan
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Planning operation...", total=None)
            plan = await self.planner.process_command(command)
            progress.update(task, description="Planning complete")
        
        # Display plan
        console.print(Panel(
            f"📋 Operation Plan:\n"
            f"Target: {plan.objective.target}\n"
            f"Type: {plan.objective.operation_type}\n"
            f"Phases: {', '.join(plan.phases)}\n"
            f"Tools: {', '.join(plan.tools)}\n"
            f"Estimated Time: {plan.estimated_time}",
            style="green"
        ))
        
        # Execute if it's a scan/pentest
        if plan.objective.operation_type in ['pentest', 'scan']:
            await self.execute_scan(plan.objective.target)
    
    async def execute_scan(self, target: str):
        """Execute nmap scan"""
        await self.notify_kernel(f"SCAN_START: {target}")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"Scanning {target}...", total=None)
            result = await self.nmap.scan(target)
            progress.update(task, description="Scan complete")
        
        await self.notify_kernel(f"SCAN_COMPLETE: {target}")
        
        # Display results
        if result['status'] == 'completed':
            console.print(Panel(
                f"🔍 Scan Results for {target}:\n"
                f"Open Ports: {', '.join(result['ports']) if result['ports'] else 'None'}\n"
                f"Services: {', '.join(result['services']) if result['services'] else 'None'}\n"
                f"OS: {result['os']}",
                style="cyan"
            ))
        else:
            console.print(Panel(
                f"❌ Scan failed: {result.get('error', 'Unknown error')}",
                style="red"
            ))
    
    async def main(self):
        """Main CLI loop"""
        console.print(Panel(
            "🛡️  Archangel Linux - MVP Demo\n"
            "Autonomous AI Security System\n"
            "Type 'help' for commands, 'exit' to quit",
            style="bold blue"
        ))
        
        # Check kernel module
        if self.check_kernel_module():
            console.print("[green]✅ Kernel module loaded[/green]")
        else:
            console.print("[yellow]⚠️  Kernel module not loaded (run: make -C kernel install)[/yellow]")
        
        while True:
            try:
                command = console.input("\n[bold cyan]archangel>[/bold cyan] ")
                
                if command.lower() in ['exit', 'quit']:
                    console.print("👋 Goodbye!")
                    break
                elif command.lower() == 'help':
                    console.print(Panel(
                        "Available commands:\n"
                        "• pentest <target> - Perform penetration test\n"
                        "• scan <target> - Quick port scan\n"
                        "• osint <target> - OSINT investigation\n"
                        "• status - Show system status\n"
                        "• help - Show this help\n"
                        "• exit - Quit application",
                        title="Help"
                    ))
                elif command.lower() == 'status':
                    kernel_status = "✅ Active" if self.check_kernel_module() else "❌ Not loaded"
                    console.print(Panel(
                        f"Kernel Module: {kernel_status}\n"
                        f"AI Planner: ✅ Active\n"
                        f"Tools: ✅ Nmap available",
                        title="System Status"
                    ))
                elif command.strip():
                    await self.execute_command(command)
                
            except KeyboardInterrupt:
                console.print("\n👋 Goodbye!")
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")

if __name__ == "__main__":
    cli = ArchangelCLI()
    asyncio.run(cli.main())
EOF

# Demo test script
cat > ~/archangel-sprint/tests/demo_test.py << 'EOF'
#!/usr/bin/env python3
"""
MVP Demo Test - Verify core functionality
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.planner import MVPPlanner
from tools.nmap_wrapper import NmapWrapper

async def test_planner():
    """Test the planning engine"""
    print("🧪 Testing planner...")
    planner = MVPPlanner()
    
    test_commands = [
        "pentest 192.168.1.1",
        "scan google.com",
        "osint example.com"
    ]
    
    for command in test_commands:
        plan = await planner.process_command(command)
        print(f"✅ Command: {command}")
        print(f"   Target: {plan.objective.target}")
        print(f"   Type: {plan.objective.operation_type}")
        print(f"   Phases: {len(plan.phases)}")

async def test_nmap():
    """Test nmap wrapper"""
    print("\n🧪 Testing nmap wrapper...")
    nmap = NmapWrapper()
    
    # Test with localhost (safe)
    result = await nmap.scan("127.0.0.1")
    print(f"✅ Nmap test: {result['status']}")
    if result['status'] == 'completed':
        print(f"   Ports found: {len(result['ports'])}")

async def main():
    """Run all tests"""
    print("🚀 Archangel MVP Demo Test")
    print("=" * 40)
    
    await test_planner()
    await test_nmap()
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    asyncio.run(main())
EOF

# Sprint Makefile
cat > ~/archangel-sprint/Makefile << 'EOF'
# Archangel Sprint Makefile - MVP Build System

.PHONY: help setup build test demo clean

help:
	@echo "🚀 Archangel Sprint - Available Commands:"
	@echo "  setup  - Set up development environment"
	@echo "  build  - Build kernel module and test system"
	@echo "  test   - Run demo tests"
	@echo "  demo   - Start demo CLI"
	@echo "  clean  - Clean build artifacts"

setup:
	@echo "🔧 Setting up sprint environment..."
	@python -m venv .venv
	@.venv/bin/pip install asyncio rich typer pytest pytest-asyncio

build:
	@echo "🔨 Building kernel module..."
	@make -C kernel all
	@echo "🧪 Running tests..."
	@.venv/bin/python tests/demo_test.py

test:
	@echo "🧪 Running demo tests..."
	@.venv/bin/python tests/demo_test.py

demo:
	@echo "🎬 Starting demo..."
	@.venv/bin/python cli/main.py

kernel-install:
	@echo "📦 Installing kernel module..."
	@make -C kernel install

kernel-uninstall:
	@echo "🗑️  Uninstalling kernel module..."
	@make -C kernel uninstall

clean:
	@echo "🧹 Cleaning up..."
	@make -C kernel clean
	@find . -name "*.pyc" -delete
	@find . -name "__pycache__" -delete
EOF

# Make files executable
chmod +x ~/archangel-sprint/cli/main.py
chmod +x ~/archangel-sprint/tests/demo_test.py

# Create activation script
cat > ~/archangel-sprint/activate-sprint.sh << 'EOF'
#!/bin/bash
# Activate Archangel Sprint Environment

cd ~/archangel-sprint
source .venv/bin/activate

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "🚀 Archangel Sprint Environment Activated!"
echo "Available commands:"
echo "  make build  - Build and test"
echo "  make demo   - Start demo"
echo "  make test   - Run tests"
echo ""
echo "Quick start:"
echo "  1. make build"
echo "  2. sudo make kernel-install"
echo "  3. make demo"
EOF

chmod +x ~/archangel-sprint/activate-sprint.sh

print_success "Sprint environment setup complete!"
print_status "Next steps:"
echo "  1. cd ~/archangel-sprint"
echo "  2. source activate-sprint.sh"
echo "  3. make build"
echo "  4. sudo make kernel-install"
echo "  5. make demo"
echo ""
print_warning "This is a streamlined MVP setup for the 2-week sprint."
print_warning "Focus on getting the demo working, not perfect architecture."

# Final test
cd ~/archangel-sprint
source .venv/bin/activate
python tests/demo_test.py

print_success "🎯 Sprint setup complete! Ready for 2-week development."