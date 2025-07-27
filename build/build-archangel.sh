#!/bin/bash

# Archangel Linux Build System
# Main build script for kernel modules, userspace components, and distribution

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/build"
KERNEL_DIR="$PROJECT_ROOT/kernel"
USERSPACE_DIR="$PROJECT_ROOT/opt/archangel"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
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
    if [[ $EUID -eq 0 ]]; then
        log_warning "Running as root. Some operations may require non-root privileges."
    fi
}

# Check system requirements
check_requirements() {
    log_info "Checking system requirements..."
    
    # Check for kernel headers
    KERNEL_VERSION=$(uname -r)
    KERNEL_HEADERS="/lib/modules/$KERNEL_VERSION/build"
    
    if [[ ! -d "$KERNEL_HEADERS" ]]; then
        log_error "Kernel headers not found at $KERNEL_HEADERS"
        log_info "Please install kernel headers:"
        log_info "  Ubuntu/Debian: sudo apt install linux-headers-$KERNEL_VERSION"
        log_info "  RHEL/CentOS: sudo yum install kernel-devel-$KERNEL_VERSION"
        log_info "  Arch: sudo pacman -S linux-headers"
        exit 1
    fi
    
    # Check for required tools
    local required_tools=("make" "gcc" "python3")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "Required tool not found: $tool"
            exit 1
        fi
    done
    
    log_success "System requirements check passed"
}

# Build kernel modules
build_kernel_modules() {
    log_info "Building Archangel kernel modules..."
    
    cd "$KERNEL_DIR/archangel"
    
    # Clean previous builds
    make clean
    
    # Compile AI models first
    log_info "Compiling AI models for kernel integration..."
    make compile-models
    
    # Build kernel modules
    log_info "Building kernel modules..."
    make modules
    
    log_success "Kernel modules built successfully"
}

# Build userspace components
build_userspace() {
    log_info "Building userspace components..."
    
    # Create necessary directories
    mkdir -p "$USERSPACE_DIR"/{ai,tools,security,gui,bin}
    
    # For now, just create placeholder files since userspace implementation
    # will be done in later tasks
    log_info "Creating userspace component placeholders..."
    
    # Create basic Python package structure
    cat > "$USERSPACE_DIR/ai/__init__.py" << 'EOF'
"""
Archangel AI Orchestration Package
"""

__version__ = "1.0.0"
__author__ = "Archangel Linux Development Team"
EOF

    cat > "$USERSPACE_DIR/tools/__init__.py" << 'EOF'
"""
Archangel Security Tools Integration Package
"""

__version__ = "1.0.0"
__author__ = "Archangel Linux Development Team"
EOF

    cat > "$USERSPACE_DIR/security/__init__.py" << 'EOF'
"""
Archangel Security Framework Package
"""

__version__ = "1.0.0"
__author__ = "Archangel Linux Development Team"
EOF

    log_success "Userspace component structure created"
}

# Install kernel modules
install_kernel_modules() {
    log_info "Installing Archangel kernel modules..."
    
    cd "$KERNEL_DIR/archangel"
    
    # Install modules
    sudo make install
    
    log_success "Kernel modules installed successfully"
}

# Test kernel modules
test_kernel_modules() {
    log_info "Testing Archangel kernel modules..."
    
    cd "$KERNEL_DIR/archangel"
    
    # Run module tests
    make test
    
    log_success "Kernel module tests passed"
}

# Create development environment
setup_dev_environment() {
    log_info "Setting up development environment..."
    
    # Create development scripts
    cat > "$PROJECT_ROOT/dev-load.sh" << 'EOF'
#!/bin/bash
# Quick development script to load Archangel modules

cd kernel/archangel
make reload
echo "Archangel modules reloaded. Check status with: make info"
EOF

    cat > "$PROJECT_ROOT/dev-status.sh" << 'EOF'
#!/bin/bash
# Quick development script to check Archangel status

cd kernel/archangel
make info
EOF

    chmod +x "$PROJECT_ROOT/dev-load.sh"
    chmod +x "$PROJECT_ROOT/dev-status.sh"
    
    log_success "Development environment set up"
}

# Show usage information
show_usage() {
    echo "Archangel Linux Build System"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  all              Build everything (default)"
    echo "  kernel           Build only kernel modules"
    echo "  userspace        Build only userspace components"
    echo "  install          Install kernel modules"
    echo "  test             Run tests"
    echo "  clean            Clean build artifacts"
    echo "  dev-setup        Set up development environment"
    echo "  help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0               # Build everything"
    echo "  $0 kernel        # Build only kernel modules"
    echo "  $0 install       # Install kernel modules"
    echo "  $0 test          # Run tests"
}

# Clean build artifacts
clean_build() {
    log_info "Cleaning build artifacts..."
    
    # Clean kernel modules
    if [[ -d "$KERNEL_DIR/archangel" ]]; then
        cd "$KERNEL_DIR/archangel"
        make clean
    fi
    
    # Clean other build artifacts
    rm -rf "$BUILD_DIR"/*.log
    rm -rf "$PROJECT_ROOT"/dev-*.sh
    
    log_success "Build artifacts cleaned"
}

# Main build function
build_all() {
    log_info "Starting Archangel Linux build process..."
    
    check_requirements
    build_kernel_modules
    build_userspace
    setup_dev_environment
    
    log_success "Archangel Linux build completed successfully!"
    log_info ""
    log_info "Next steps:"
    log_info "  1. Install kernel modules: $0 install"
    log_info "  2. Test the installation: $0 test"
    log_info "  3. Load modules for development: ./dev-load.sh"
    log_info "  4. Check status: ./dev-status.sh"
}

# Main script logic
main() {
    check_root
    
    case "${1:-all}" in
        "all")
            build_all
            ;;
        "kernel")
            check_requirements
            build_kernel_modules
            ;;
        "userspace")
            build_userspace
            ;;
        "install")
            install_kernel_modules
            ;;
        "test")
            test_kernel_modules
            ;;
        "clean")
            clean_build
            ;;
        "dev-setup")
            setup_dev_environment
            ;;
        "help"|"-h"|"--help")
            show_usage
            ;;
        *)
            log_error "Unknown command: $1"
            show_usage
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"