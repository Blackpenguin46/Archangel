#!/usr/bin/env bash
# Archangel Linux - Cross-Architecture Build System
# Builds compatible .iso and bootable USB files for ARM64 and x86_64

set -e

# Configuration
ARCHANGEL_VERSION="1.0.0"
BUILD_DIR="$(pwd)/build"
OUTPUT_DIR="$BUILD_DIR/output"
TEMP_DIR="$BUILD_DIR/temp"
KERNEL_DIR="$(pwd)/kernel"
USERSPACE_DIR="$(pwd)/opt/archangel"

# Architecture configurations (using functions instead of associative arrays for compatibility)
get_arch_config() {
    local arch=$1
    case $arch in
        x86_64)
            echo "x86_64 amd64 x86_64-linux-gnu"
            ;;
        arm64)
            echo "aarch64 arm64 aarch64-linux-gnu"
            ;;
        *)
            echo ""
            ;;
    esac
}

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

# Check dependencies
check_dependencies() {
    log_info "Checking build dependencies..."
    
    local deps=("debootstrap" "squashfs-tools" "genisoimage" "syslinux" "grub-pc-bin" "grub-efi-amd64-bin" "grub-efi-arm64-bin")
    local missing_deps=()
    
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null && ! dpkg -l | grep -q "$dep"; then
            missing_deps+=("$dep")
        fi
    done
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        log_error "Missing dependencies: ${missing_deps[*]}"
        log_info "Install with: sudo apt-get install ${missing_deps[*]}"
        exit 1
    fi
    
    log_success "All dependencies satisfied"
}

# Setup build environment
setup_build_env() {
    log_info "Setting up build environment..."
    
    # Create directories
    mkdir -p "$OUTPUT_DIR"/{x86_64,arm64}
    mkdir -p "$TEMP_DIR"/{x86_64,arm64}
    
    # Check if running on supported host
    HOST_ARCH=$(uname -m)
    case $HOST_ARCH in
        x86_64|amd64)
            log_info "Host architecture: x86_64"
            ;;
        aarch64|arm64)
            log_info "Host architecture: ARM64"
            ;;
        *)
            log_warning "Host architecture $HOST_ARCH may not be fully supported"
            ;;
    esac
    
    log_success "Build environment ready"
}

# Build kernel modules for specific architecture
build_kernel_modules() {
    local target_arch=$1
    local arch_config=$(get_arch_config "$target_arch")
    local arch_info=($arch_config)
    local kernel_arch=${arch_info[0]}
    local debian_arch=${arch_info[1]}
    local cross_compile=${arch_info[2]}
    
    log_info "Building kernel modules for $target_arch..."
    
    # Set architecture-specific variables
    export ARCH=$kernel_arch
    if [ "$target_arch" != "$(uname -m)" ]; then
        export CROSS_COMPILE=${cross_compile}-
        log_info "Cross-compiling for $target_arch using $CROSS_COMPILE"
    fi
    
    # Build modules
    cd "$KERNEL_DIR/archangel"
    
    # Clean previous builds
    make clean || true
    
    # Build with architecture flags
    make modules ARCH=$kernel_arch
    
    # Copy built modules to output
    mkdir -p "$OUTPUT_DIR/$target_arch/modules"
    cp *.ko "$OUTPUT_DIR/$target_arch/modules/" 2>/dev/null || true
    
    log_success "Kernel modules built for $target_arch"
    
    # Reset environment
    unset ARCH CROSS_COMPILE
    cd - > /dev/null
}

# Create base filesystem for architecture
create_base_filesystem() {
    local target_arch=$1
    local arch_config=$(get_arch_config "$target_arch")
    local arch_info=($arch_config)
    local debian_arch=${arch_info[1]}
    
    log_info "Creating base filesystem for $target_arch..."
    
    local rootfs_dir="$TEMP_DIR/$target_arch/rootfs"
    
    # Create base Debian system
    sudo debootstrap --arch=$debian_arch \
        --include=linux-image-generic,systemd,network-manager,openssh-server,python3,python3-pip \
        bullseye "$rootfs_dir" http://deb.debian.org/debian/
    
    # Configure system
    sudo chroot "$rootfs_dir" /bin/bash -c "
        # Set hostname
        echo 'archangel-linux' > /etc/hostname
        
        # Configure network
        cat > /etc/systemd/network/20-wired.network << EOF
[Match]
Name=en*

[Network]
DHCP=yes
EOF
        
        # Enable services
        systemctl enable systemd-networkd
        systemctl enable systemd-resolved
        systemctl enable ssh
        
        # Set root password (change in production)
        echo 'root:archangel' | chpasswd
        
        # Create archangel user
        useradd -m -s /bin/bash -G sudo archangel
        echo 'archangel:archangel' | chpasswd
    "
    
    # Install Archangel userspace components
    sudo cp -r "$USERSPACE_DIR" "$rootfs_dir/opt/"
    
    # Install kernel modules
    sudo mkdir -p "$rootfs_dir/lib/modules/archangel"
    sudo cp "$OUTPUT_DIR/$target_arch/modules"/*.ko "$rootfs_dir/lib/modules/archangel/" 2>/dev/null || true
    
    # Create module loading script
    sudo tee "$rootfs_dir/etc/systemd/system/archangel.service" > /dev/null << EOF
[Unit]
Description=Archangel AI Security System
After=network.target

[Service]
Type=oneshot
ExecStart=/opt/archangel/bin/archangel-agent --start
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF
    
    sudo chroot "$rootfs_dir" systemctl enable archangel
    
    log_success "Base filesystem created for $target_arch"
}

# Create bootable ISO for architecture
create_iso() {
    local target_arch=$1
    local arch_config=$(get_arch_config "$target_arch")
    local arch_info=($arch_config)
    local kernel_arch=${arch_info[0]}
    
    log_info "Creating ISO for $target_arch..."
    
    local rootfs_dir="$TEMP_DIR/$target_arch/rootfs"
    local iso_dir="$TEMP_DIR/$target_arch/iso"
    local output_iso="$OUTPUT_DIR/$target_arch/archangel-linux-$ARCHANGEL_VERSION-$target_arch.iso"
    
    # Create ISO directory structure
    mkdir -p "$iso_dir"/{boot,live}
    
    # Create squashfs filesystem
    sudo mksquashfs "$rootfs_dir" "$iso_dir/live/filesystem.squashfs" -comp xz
    
    # Copy kernel and initrd
    if [ "$target_arch" = "x86_64" ]; then
        # x86_64 boot setup
        sudo cp "$rootfs_dir/boot/vmlinuz-"* "$iso_dir/boot/vmlinuz"
        sudo cp "$rootfs_dir/boot/initrd.img-"* "$iso_dir/boot/initrd.img"
        
        # Create GRUB configuration
        mkdir -p "$iso_dir/boot/grub"
        cat > "$iso_dir/boot/grub/grub.cfg" << EOF
set timeout=10
set default=0

menuentry "Archangel Linux $ARCHANGEL_VERSION (x86_64)" {
    linux /boot/vmlinuz boot=live live-media-path=/live/ quiet splash
    initrd /boot/initrd.img
}

menuentry "Archangel Linux $ARCHANGEL_VERSION (x86_64) - Safe Mode" {
    linux /boot/vmlinuz boot=live live-media-path=/live/ quiet splash nomodeset
    initrd /boot/initrd.img
}
EOF
        
        # Create ISO with GRUB
        grub-mkrescue -o "$output_iso" "$iso_dir"
        
    elif [ "$target_arch" = "arm64" ]; then
        # ARM64 boot setup
        sudo cp "$rootfs_dir/boot/vmlinuz-"* "$iso_dir/boot/vmlinuz" 2>/dev/null || true
        sudo cp "$rootfs_dir/boot/initrd.img-"* "$iso_dir/boot/initrd.img" 2>/dev/null || true
        
        # Create simple boot configuration for ARM64
        mkdir -p "$iso_dir/boot/grub"
        cat > "$iso_dir/boot/grub/grub.cfg" << EOF
set timeout=10
set default=0

menuentry "Archangel Linux $ARCHANGEL_VERSION (ARM64)" {
    linux /boot/vmlinuz boot=live live-media-path=/live/ quiet splash
    initrd /boot/initrd.img
}
EOF
        
        # Create ISO for ARM64
        genisoimage -r -J -b boot/grub/grub.cfg -c boot/boot.cat \
            -o "$output_iso" "$iso_dir"
    fi
    
    log_success "ISO created: $output_iso"
}

# Create bootable USB image
create_usb_image() {
    local target_arch=$1
    
    log_info "Creating USB image for $target_arch..."
    
    local output_img="$OUTPUT_DIR/$target_arch/archangel-linux-$ARCHANGEL_VERSION-$target_arch.img"
    local iso_file="$OUTPUT_DIR/$target_arch/archangel-linux-$ARCHANGEL_VERSION-$target_arch.iso"
    
    # Create USB image from ISO
    if [ -f "$iso_file" ]; then
        # Create hybrid ISO that can boot from USB
        if command -v isohybrid &> /dev/null; then
            cp "$iso_file" "$output_img"
            isohybrid "$output_img"
            log_success "USB image created: $output_img"
        else
            log_warning "isohybrid not available, copying ISO as USB image"
            cp "$iso_file" "$output_img"
        fi
    else
        log_error "ISO file not found for $target_arch"
        return 1
    fi
}

# Generate checksums
generate_checksums() {
    log_info "Generating checksums..."
    
    for arch in x86_64 arm64; do
        if [ -d "$OUTPUT_DIR/$arch" ]; then
            cd "$OUTPUT_DIR/$arch"
            sha256sum *.iso *.img > "archangel-linux-$ARCHANGEL_VERSION-$arch.sha256" 2>/dev/null || true
            cd - > /dev/null
        fi
    done
    
    log_success "Checksums generated"
}

# Main build function
build_architecture() {
    local target_arch=$1
    
    log_info "Starting build for $target_arch architecture..."
    
    # Check if cross-compilation tools are available
    if [ "$target_arch" != "$(uname -m)" ]; then
        local arch_config=$(get_arch_config "$target_arch")
        local arch_info=($arch_config)
        local cross_compile=${arch_info[2]}
        
        if ! command -v "${cross_compile}-gcc" &> /dev/null; then
            log_warning "Cross-compilation tools for $target_arch not found"
            log_info "Install with: sudo apt-get install gcc-${cross_compile}"
            return 1
        fi
    fi
    
    # Build steps
    build_kernel_modules "$target_arch"
    create_base_filesystem "$target_arch"
    create_iso "$target_arch"
    create_usb_image "$target_arch"
    
    log_success "Build completed for $target_arch"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up temporary files..."
    sudo rm -rf "$TEMP_DIR"
    log_success "Cleanup completed"
}

# Main execution
main() {
    echo "Archangel Linux - Cross-Architecture Build System"
    echo "================================================="
    echo "Version: $ARCHANGEL_VERSION"
    echo "Target Architectures: x86_64, ARM64"
    echo ""
    
    # Parse command line arguments
    ARCHITECTURES=()
    CLEAN_ONLY=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --arch)
                ARCHITECTURES+=("$2")
                shift 2
                ;;
            --all)
                ARCHITECTURES=("x86_64" "arm64")
                shift
                ;;
            --clean)
                CLEAN_ONLY=true
                shift
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  --arch ARCH    Build for specific architecture (x86_64 or arm64)"
                echo "  --all          Build for all supported architectures"
                echo "  --clean        Clean temporary files and exit"
                echo "  --help         Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Default to all architectures if none specified
    if [ ${#ARCHITECTURES[@]} -eq 0 ] && [ "$CLEAN_ONLY" = false ]; then
        ARCHITECTURES=("x86_64" "arm64")
    fi
    
    # Clean and exit if requested
    if [ "$CLEAN_ONLY" = true ]; then
        cleanup
        exit 0
    fi
    
    # Setup trap for cleanup on exit
    trap cleanup EXIT
    
    # Run build process
    check_dependencies
    setup_build_env
    
    # Build for each architecture
    for arch in "${ARCHITECTURES[@]}"; do
        if [[ "$arch" =~ ^(x86_64|arm64)$ ]]; then
            build_architecture "$arch"
        else
            log_error "Unsupported architecture: $arch"
        fi
    done
    
    generate_checksums
    
    echo ""
    log_success "Cross-architecture build completed!"
    echo ""
    echo "Output files:"
    for arch in "${ARCHITECTURES[@]}"; do
        if [ -d "$OUTPUT_DIR/$arch" ]; then
            echo "  $arch:"
            ls -la "$OUTPUT_DIR/$arch/" | grep -E '\.(iso|img|sha256)$' | awk '{print "    " $9 " (" $5 " bytes)"}'
        fi
    done
    echo ""
    echo "To write USB image:"
    echo "  sudo dd if=output/x86_64/archangel-linux-$ARCHANGEL_VERSION-x86_64.img of=/dev/sdX bs=4M status=progress"
    echo "  sudo dd if=output/arm64/archangel-linux-$ARCHANGEL_VERSION-arm64.img of=/dev/sdX bs=4M status=progress"
}

# Run main function with all arguments
main "$@"