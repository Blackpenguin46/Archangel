#!/bin/bash

# Archangel Linux Installation Script
# Installs kernel modules and sets up the system

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
INSTALL_PREFIX="/opt/archangel"
SYSTEMD_DIR="/etc/systemd/system"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root"
        exit 1
    fi
}

# Install kernel modules
install_kernel_modules() {
    log_info "Installing Archangel kernel modules..."
    
    cd "$PROJECT_ROOT/kernel/archangel"
    make install
    
    log_success "Kernel modules installed"
}

# Install userspace components
install_userspace() {
    log_info "Installing userspace components..."
    
    # Create installation directory
    mkdir -p "$INSTALL_PREFIX"
    
    # Copy userspace components
    cp -r "$PROJECT_ROOT/opt/archangel"/* "$INSTALL_PREFIX/"
    
    # Set proper permissions
    chown -R root:root "$INSTALL_PREFIX"
    chmod -R 755 "$INSTALL_PREFIX"
    
    log_success "Userspace components installed to $INSTALL_PREFIX"
}

# Create systemd service
create_systemd_service() {
    log_info "Creating systemd service..."
    
    cat > "$SYSTEMD_DIR/archangel.service" << 'EOF'
[Unit]
Description=Archangel AI Security Operating System
After=network.target
Requires=network.target

[Service]
Type=simple
ExecStart=/opt/archangel/bin/archangel-agent
Restart=always
RestartSec=5
User=root
Group=root

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    
    log_success "Systemd service created"
}

# Load kernel modules
load_modules() {
    log_info "Loading Archangel kernel modules..."
    
    modprobe archangel_core
    
    log_success "Kernel modules loaded"
}

# Main installation
main() {
    log_info "Starting Archangel Linux installation..."
    
    check_root
    install_kernel_modules
    install_userspace
    create_systemd_service
    load_modules
    
    log_success "Archangel Linux installation completed!"
    log_info ""
    log_info "System status:"
    log_info "  Kernel modules: $(lsmod | grep archangel | wc -l) loaded"
    log_info "  Installation directory: $INSTALL_PREFIX"
    log_info "  Service: systemctl status archangel"
    log_info ""
    log_info "To start the service: systemctl start archangel"
    log_info "To enable at boot: systemctl enable archangel"
}

main "$@"