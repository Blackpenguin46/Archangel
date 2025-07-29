#!/bin/bash
# Archangel AI Security Expert Setup Script

set -e

echo "🛡️  Setting up Archangel AI Security Expert System..."
echo "=================================================="

# Check if we're on Arch Linux
if [ -f /etc/arch-release ]; then
    echo "✅ Detected Arch Linux"
else
    echo "⚠️  This setup script is optimized for Arch Linux"
    echo "   It may work on other systems but YMMV"
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "🐍 Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install core requirements
echo "📚 Installing core Python packages..."
pip install rich typer asyncio

# Install optional security tools (if available)
echo "🔧 Checking for security tools..."

tools_available=0

# Check for nmap
if command -v nmap &> /dev/null; then
    echo "✅ nmap found"
    tools_available=$((tools_available + 1))
else
    echo "❌ nmap not found - install with: sudo pacman -S nmap"
fi

# Check for curl
if command -v curl &> /dev/null; then
    echo "✅ curl found"
    tools_available=$((tools_available + 1))
else
    echo "❌ curl not found - install with: sudo pacman -S curl"
fi

echo "📊 Found $tools_available security tools"

# Make scripts executable
chmod +x scripts/*.sh
chmod +x cli.py
chmod +x demo_archangel.py

echo ""
echo "🎉 Archangel setup complete!"
echo ""
echo "🚀 Quick start:"
echo "   source venv/bin/activate"
echo "   python demo_archangel.py      # Run demo"
echo "   python cli.py                 # Interactive CLI"
echo ""
echo "🎯 What you have:"
echo "   • AI Security Expert Brain that thinks step-by-step"
echo "   • Conversational AI that explains its reasoning"
echo "   • Educational security AI that teaches while working"
echo "   • Tool integration system (works with or without real tools)"
echo ""
echo "⚠️  Remember: Archangel is for defensive security research only!"
echo ""