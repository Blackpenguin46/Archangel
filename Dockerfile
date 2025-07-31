# Archangel AI Security Expert System - Docker Testing Environment
FROM ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/.cache/huggingface

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    linux-headers-generic \
    kmod \
    curl \
    wget \
    git \
    vim \
    htop \
    net-tools \
    iputils-ping \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Copy the entire Archangel project
COPY . .

# Create necessary directories
RUN mkdir -p /app/.cache/huggingface \
    && mkdir -p /app/logs \
    && mkdir -p /app/data

# Set permissions
RUN chmod +x scripts/setup.sh 2>/dev/null || true

# Create a non-root user for security
RUN useradd -m -u 1000 archangel && \
    chown -R archangel:archangel /app

# Switch to non-root user for most operations
USER archangel

# Set up Python virtual environment
RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Install Python dependencies in virtual environment
RUN /app/venv/bin/pip install --no-cache-dir -r requirements.txt

# Switch back to root for kernel module operations
USER root

# Expose ports for potential web interfaces
EXPOSE 8000 8888

# Create entrypoint script
RUN echo '#!/bin/bash\n\
echo "🛡️  Archangel AI Security Expert System - Docker Environment"\n\
echo "=========================================="\n\
echo ""\n\
echo "Available commands:"\n\
echo "  python3 archangel_lightweight.py analyze <target>  # Cloud AI analysis"\n\
echo "  python3 archangel_ai.py interactive                # Local AI interactive"\n\
echo "  python3 demo_archangel.py                          # Basic AI demo"\n\
echo "  python3 hybrid_demo.py                             # Full architecture demo"\n\
echo "  python3 cli.py                                     # Interactive CLI"\n\
echo ""\n\
echo "Kernel module commands (requires privileged container):"\n\
echo "  cd kernel && make                                   # Build kernel module"\n\
echo "  cd kernel && make load                              # Load kernel module"\n\
echo "  python3 hybrid_demo.py                             # Test AI-kernel integration"\n\
echo ""\n\
echo "Environment setup:"\n\
echo "  - Set HF_TOKEN environment variable for Hugging Face API access"\n\
echo "  - Run with --privileged flag for kernel module testing"\n\
echo ""\n\
exec "$@"' > /entrypoint.sh && chmod +x /entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Default command
CMD ["/bin/bash"]