# Installation Troubleshooting Guide

## Common Installation Issues and Solutions

### 1. Python Dependencies Issues

#### Problem: `asyncio-timeout` or other package version conflicts
```bash
ERROR: Could not find a version that satisfies the requirement asyncio-timeout>=4.0.0
```

**Solution A: Use Minimal Requirements**
```bash
pip install -r requirements-minimal.txt
```

**Solution B: Install Core Packages Individually**
```bash
# Core AI libraries
pip install torch transformers datasets huggingface_hub

# Basic utilities  
pip install requests numpy pandas aiofiles psutil

# Configuration and logging
pip install pydantic click tqdm pyyaml rich
```

**Solution C: Create Clean Virtual Environment**
```bash
# Remove existing environment
rm -rf venv

# Create new environment
python3 -m venv venv
source venv/bin/activate

# Upgrade tools
pip install --upgrade pip setuptools wheel

# Install minimal requirements
pip install -r requirements-minimal.txt
```

### 2. Python Version Compatibility

#### Problem: Python version conflicts
```bash
ERROR: Ignored the following versions that require a different python version
```

**Check Python Version:**
```bash
python3 --version
```

**Requirements:**
- Python 3.9+ recommended
- Python 3.8+ minimum

**Solution: Use Compatible Python Version**
```bash
# Install Python 3.9+ using Homebrew
brew install python@3.9

# Use specific Python version
python3.9 -m venv venv
source venv/bin/activate
pip install -r requirements-minimal.txt
```

### 3. Torch Installation Issues

#### Problem: PyTorch installation fails on Apple Silicon
```bash
ERROR: Could not find a version that satisfies the requirement torch
```

**Solution: Install PyTorch for Apple Silicon**
```bash
# Uninstall existing torch
pip uninstall torch

# Install PyTorch with Apple Silicon support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 4. Transformers/Hugging Face Issues

#### Problem: Transformers library conflicts
```bash
ERROR: transformers has requirement tokenizers>=0.15.0
```

**Solution: Install Compatible Versions**
```bash
pip install transformers==4.30.2
pip install tokenizers==0.13.3
pip install datasets==2.12.0
```

### 5. Network/Download Issues

#### Problem: Cannot download models or packages
```bash
ConnectTimeout: HTTPSConnectionPool host='huggingface.co'
```

**Solution A: Check Network Connection**
```bash
ping huggingface.co
curl -I https://pypi.org
```

**Solution B: Use Alternative Index**
```bash
pip install --index-url https://pypi.org/simple/ -r requirements-minimal.txt
```

**Solution C: Offline Installation**
```bash
# Download packages locally first
pip download -r requirements-minimal.txt -d ./packages/
pip install --find-links ./packages/ -r requirements-minimal.txt
```

### 6. Permission Issues

#### Problem: Permission denied errors
```bash
PermissionError: [Errno 13] Permission denied
```

**Solution A: Use Virtual Environment (Recommended)**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-minimal.txt
```

**Solution B: User Installation**
```bash
pip install --user -r requirements-minimal.txt
```

### 7. macOS Specific Issues

#### Problem: Xcode command line tools missing
```bash
xcrun: error: invalid active developer path
```

**Solution: Install Xcode Command Line Tools**
```bash
xcode-select --install
```

#### Problem: Homebrew not in PATH
```bash
brew: command not found
```

**Solution: Add Homebrew to PATH**
```bash
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
source ~/.zprofile
```

### 8. Memory Issues During Installation

#### Problem: Installation killed due to memory
```bash
Killed: 9
```

**Solution: Increase available memory or install sequentially**
```bash
# Install one package at a time
pip install torch
pip install transformers  
pip install datasets
pip install huggingface_hub
# ... continue with other packages
```

## Quick Recovery Script

If you encounter multiple issues, use this recovery script:

```bash
#!/bin/bash
# recovery_install.sh

echo "🔧 Archangel Recovery Installation"

# Clean slate
rm -rf venv
python3 -m venv venv
source venv/bin/activate

# Upgrade tools
pip install --upgrade pip setuptools wheel

# Install absolutely minimal dependencies
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers datasets huggingface_hub
pip install requests numpy pandas
pip install click tqdm rich

echo "✅ Minimal installation complete"
echo "🚀 Try running: python archangel_autonomous_system.py --health-check"
```

Make it executable and run:
```bash
chmod +x recovery_install.sh
./recovery_install.sh
```

## Test Installation

After resolving issues, test the installation:

```bash
# Test Python imports
python3 -c "
import torch
import transformers  
import datasets
import requests
print('✅ Core libraries imported successfully')
"

# Test Archangel system
python3 archangel_autonomous_system.py --health-check

# Test training pipeline
python3 training/deepseek_training_pipeline.py --test-only
```

## Getting Help

If issues persist:

1. **Check Python version**: `python3 --version` (need 3.9+)
2. **Check pip version**: `pip --version` (need recent version)
3. **Create issue**: Include full error message and system info
4. **Use minimal setup**: Focus on core functionality first

## Alternative: Docker Installation

If local installation continues to fail:

```bash
# Use Docker for consistent environment
docker build -t archangel .
docker run -it archangel python archangel_autonomous_system.py --health-check
```