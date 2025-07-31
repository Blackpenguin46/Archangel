#!/bin/bash

# Archangel AI Security Expert System - Docker Setup Script
# This script sets up the Docker environment for testing Archangel

set -e

echo "🛡️  Archangel AI Security Expert System - Docker Setup"
echo "======================================================"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "   Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null 2>&1; then
    echo "❌ Docker Compose is not available. Please install Docker Compose."
    echo "   Visit: https://docs.docker.com/compose/install/"
    exit 1
fi

# Determine docker compose command
if command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE="docker-compose"
else
    DOCKER_COMPOSE="docker compose"
fi

echo "✅ Docker and Docker Compose are available"

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file for environment variables..."
    cat > .env << 'EOF'
# Hugging Face Token (replace with your actual token)
# Get one at: https://huggingface.co/settings/tokens
HF_TOKEN=your_hugging_face_token_here

# Optional: Additional environment variables
PYTHONPATH=/app
ARCHANGEL_LOG_LEVEL=INFO
ARCHANGEL_DEBUG=false
EOF
    echo "⚠️  Please edit .env file and add your Hugging Face token"
    echo "   Get a token at: https://huggingface.co/settings/tokens"
fi

# Build the Docker image
echo "🔨 Building Archangel Docker image..."
docker build -t archangel-ai:latest .

echo "✅ Docker image built successfully"

# Function to show usage
show_usage() {
    echo ""
    echo "🚀 Archangel Docker Environment Ready!"
    echo "====================================="
    echo ""
    echo "Quick Start Commands:"
    echo "  $DOCKER_COMPOSE up -d                    # Start in background"
    echo "  $DOCKER_COMPOSE exec archangel bash      # Enter container shell"
    echo "  $DOCKER_COMPOSE logs -f archangel        # View logs"
    echo "  $DOCKER_COMPOSE down                     # Stop containers"
    echo ""
    echo "Testing Archangel AI:"
    echo "  # Enter the container"
    echo "  $DOCKER_COMPOSE exec archangel bash"
    echo ""
    echo "  # Inside container - Test cloud AI (requires HF token)"
    echo "  python3 archangel_lightweight.py analyze google.com"
    echo ""
    echo "  # Test local AI models"
    echo "  python3 archangel_ai.py interactive"
    echo ""
    echo "  # Run AI demos"
    echo "  python3 demo_archangel.py"
    echo "  python3 hybrid_demo.py"
    echo ""
    echo "  # Interactive CLI"
    echo "  python3 cli.py"
    echo ""
    echo "Kernel Module Testing (requires privileged container):"
    echo "  # Build kernel module"
    echo "  cd kernel && make"
    echo ""
    echo "  # Load kernel module (requires root in container)"
    echo "  sudo make load"
    echo ""
    echo "  # Test AI-kernel integration"
    echo "  python3 hybrid_demo.py"
    echo ""
    echo "Advanced Usage:"
    echo "  $DOCKER_COMPOSE --profile with-db up     # Start with database"
    echo "  $DOCKER_COMPOSE --profile with-cache up  # Start with cache"
    echo ""
    echo "Configuration:"
    echo "  - Edit .env file to set your Hugging Face token"
    echo "  - Container runs with privileged mode for kernel module access"
    echo "  - Volumes persist Hugging Face cache and logs"
    echo ""
}

# Check if user wants to start containers immediately
if [ "$1" = "--start" ] || [ "$1" = "-s" ]; then
    echo "🚀 Starting Archangel containers..."
    $DOCKER_COMPOSE up -d
    echo "✅ Containers started successfully"
    echo ""
    echo "Enter the container with:"
    echo "  $DOCKER_COMPOSE exec archangel bash"
else
    show_usage
    echo "💡 Tip: Run '$0 --start' to build and start containers immediately"
fi

echo ""
echo "🛡️  Ready to explore AI that understands security!"