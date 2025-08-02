#!/bin/bash
# Archangel Live Combat System - M2 MacBook Launcher
# Optimized for Apple Silicon with 16GB RAM

echo "🍎 Archangel Live Combat System - M2 MacBook"
echo "============================================="

# Check system requirements
echo "🔍 Checking M2 MacBook requirements..."

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker Desktop for Mac."
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo "❌ Docker is not running. Please start Docker Desktop."
    exit 1
fi

echo "✅ Docker is ready"

# Check Python
if ! python3 --version &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8+."
    exit 1
fi

echo "✅ Python 3 is ready"

# Check memory
MEMORY_GB=$(sysctl -n hw.memsize | awk '{print int($1/1024/1024/1024)}')
echo "💾 Available RAM: ${MEMORY_GB}GB"

if [ "$MEMORY_GB" -lt 12 ]; then
    echo "⚠️  Warning: Low memory detected. Close other applications for best performance."
fi

# Set M2 optimizations
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "🚀 M2 MacBook optimizations enabled"

# Check virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    
    echo "📥 Installing M2-optimized dependencies..."
    pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install --quiet transformers==4.30.2 datasets==2.12.0 huggingface_hub==0.15.1
    pip install --quiet requests numpy pandas aiofiles psutil
    pip install --quiet pydantic click tqdm rich pyyaml docker
else
    source venv/bin/activate
    # Ensure docker module is available
    pip install --quiet docker
fi

echo "✅ Environment ready"

# Copy M2 MacBook config
if [ -f "config/m2_macbook_config.json" ]; then
    cp config/m2_macbook_config.json config/config.json
    echo "✅ M2 MacBook configuration loaded"
fi

echo ""
echo "🥊 Starting Live Red Team vs Blue Team Combat"
echo "Duration: 5 minutes demonstration"
echo "Press Ctrl+C to stop at any time"
echo ""

sleep 2

# Run the combat system
python3 test_live_combat_system.py

echo ""
echo "🎯 Combat demonstration complete!"
echo ""
echo "Next steps:"
echo "• Run full training: python3 training/deepseek_training_pipeline.py"
echo "• Start autonomous operations: python3 archangel_autonomous_system.py"
echo "• Monitor with GUI: python3 -m streamlit run dashboard/combat_dashboard.py"