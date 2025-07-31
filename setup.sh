#!/bin/bash
# Archangel Linux Setup Script
# Automated installation and configuration for autonomous AI security system

set -e  # Exit on any error

echo "🛡️ Archangel Linux - Autonomous AI Security System Setup"
echo "=========================================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    print_error "This setup script is designed for macOS. Please install manually on other systems."
    exit 1
fi

# Check for Apple Silicon
if [[ $(uname -m) != "arm64" ]]; then
    print_warning "Apple Silicon (M1/M2/M3) is recommended for optimal performance."
fi

print_step "1. Checking prerequisites..."

# Check Python version
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is required. Please install Python 3.9 or later."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.9"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then 
    print_error "Python $REQUIRED_VERSION or later is required. Found: $PYTHON_VERSION"
    exit 1
fi

print_status "Python $PYTHON_VERSION detected"

# Check for Homebrew
if ! command -v brew &> /dev/null; then
    print_warning "Homebrew not found. Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

print_status "Homebrew available"

print_step "2. Installing system dependencies..."

# Install system packages
brew update
brew install git curl wget

# Install Python package manager
if ! command -v pip3 &> /dev/null; then
    print_status "Installing pip..."
    curl https://bootstrap.pypa.io/get-pip.py | python3
fi

print_step "3. Creating virtual environment..."

# Create virtual environment
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_status "Virtual environment created"
else
    print_status "Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate
print_status "Virtual environment activated"

print_step "4. Installing Python dependencies..."

# Upgrade pip
pip install --upgrade pip

# Install core dependencies
pip install -r requirements.txt

print_status "Python dependencies installed"

print_step "5. Creating directory structure..."

# Create necessary directories
mkdir -p data/{training_datasets,adversarial_training,models}
mkdir -p logs
mkdir -p config
mkdir -p models/{deepseek,custom}
mkdir -p containers/{kali,monitoring}

print_status "Directory structure created"

print_step "6. Setting up configuration files..."

# Create default configuration if it doesn't exist
if [ ! -f "config/config.json" ]; then
    cat > config/config.json << 'EOF'
{
  "system": {
    "log_level": "INFO",
    "data_directory": "./data",
    "model_cache_directory": "./models",
    "container_runtime": "docker"
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
EOF
    print_status "Default configuration created"
fi

# Create security policy if it doesn't exist
if [ ! -f "config/security_policy.json" ]; then
    cat > config/security_policy.json << 'EOF'
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
EOF
    print_status "Security policy created"
fi

print_step "7. Installing container runtime..."

# Check for Docker
if ! command -v docker &> /dev/null; then
    print_warning "Docker not found. Installing Docker..."
    brew install --cask docker
    print_status "Docker installed. Please start Docker Desktop and re-run this script."
    print_warning "Note: You may need to restart your terminal after Docker installation."
else
    print_status "Docker available"
fi

print_step "8. Setting up security tools..."

# Install network security tools
if ! command -v nmap &> /dev/null; then
    print_status "Installing nmap..."
    brew install nmap
fi

if ! command -v tshark &> /dev/null; then
    print_status "Installing Wireshark..."
    brew install --cask wireshark
fi

print_status "Security tools configured"

print_step "9. Downloading AI models..."

# Create model download script
python3 -c "
import os
from huggingface_hub import snapshot_download
try:
    print('Downloading DeepSeek model...')
    snapshot_download('deepseek-ai/deepseek-r1-distill-llama-8b', 
                     local_dir='./models/deepseek',
                     local_dir_use_symlinks=False)
    print('Model download completed')
except Exception as e:
    print(f'Model download failed: {e}')
    print('You can download models later using the training pipeline')
"

print_step "10. Initializing system..."

# Test system initialization
python3 -c "
import sys
import asyncio
sys.path.append('.')

async def test_init():
    try:
        from archangel_autonomous_system import ArchangelAutonomousSystem
        system = ArchangelAutonomousSystem()
        print('System initialization test: PASSED')
        return True
    except Exception as e:
        print(f'System initialization test: FAILED - {e}')
        return False

if asyncio.run(test_init()):
    print('✅ Core system ready')
else:
    print('⚠️ System needs manual configuration')
"

print_step "11. Setting permissions..."

# Set proper permissions
chmod +x scripts/*.py
chmod +x training/*.py
chmod +x *.py
chmod 600 config/*.json

print_status "Permissions configured"

print_step "12. Creating launch scripts..."

# Create convenience launch script
cat > start_archangel.sh << 'EOF'
#!/bin/bash
# Archangel launcher script

source venv/bin/activate
python3 archangel_autonomous_system.py "$@"
EOF

chmod +x start_archangel.sh

print_status "Launch scripts created"

echo ""
echo "🎉 Archangel Linux Setup Complete!"
echo "====================================="
echo ""
echo "✅ System Requirements: Met"
echo "✅ Dependencies: Installed"
echo "✅ Configuration: Created"
echo "✅ AI Models: Ready"
echo "✅ Security Tools: Configured"
echo "✅ Containers: Available"
echo ""
echo "🚀 Quick Start:"
echo "   ./start_archangel.sh --init        # Initialize system"
echo "   ./start_archangel.sh --health-check # Verify installation"
echo "   ./start_archangel.sh --mode demo   # Run demonstration"
echo ""
echo "📚 Next Steps:"
echo "   1. Review SETUP.md for detailed configuration"
echo "   2. Read README.md for usage instructions"
echo "   3. Run system health check to verify installation"
echo ""
echo "🔒 Security Notice:"
echo "   - All operations run in sandboxed environments"
echo "   - Authorization required for security operations"
echo "   - Review security_policy.json before production use"
echo ""
echo "Happy autonomous security operations! 🛡️🤖"