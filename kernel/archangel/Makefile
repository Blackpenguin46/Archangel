# Archangel Linux Kernel Module Makefile
# Core AI kernel module build configuration

# Module name and version
MODULE_NAME := archangel_core
MODULE_VERSION := 1.0.0

# Architecture detection and configuration
ARCH ?= $(shell uname -m)
ifeq ($(ARCH),x86_64)
    ARCH_DIR := x86
    ARCH_FLAGS := -DCONFIG_X86_64
else ifeq ($(ARCH),aarch64)
    ARCH_DIR := arm64
    ARCH_FLAGS := -DCONFIG_ARM64
else ifeq ($(ARCH),arm64)
    ARCH_DIR := arm64
    ARCH_FLAGS := -DCONFIG_ARM64
else
    $(error Unsupported architecture: $(ARCH). Supported: x86_64, aarch64, arm64)
endif

# Kernel build directory
KERNEL_DIR ?= /lib/modules/$(shell uname -r)/build

# Source files for the core module
obj-m += $(MODULE_NAME).o
$(MODULE_NAME)-objs := archangel_core.o archangel_comm.o

# Syscall AI module
obj-m += archangel_syscall_ai.o
archangel_syscall_ai-objs := archangel_syscall_ai.o

# Network AI module
obj-m += archangel_network_ai.o
archangel_network_ai-objs := archangel_network_ai.o

# Memory AI module
obj-m += archangel_memory_ai.o
archangel_memory_ai-objs := archangel_memory_ai.o

# Compiler flags
ccflags-y += -DMODULE_VERSION=\"$(MODULE_VERSION)\"
ccflags-y += -Wall -Wextra -Werror
ccflags-y += -O2
ccflags-y += -DARCHANGEL_DEBUG
ccflags-y += $(ARCH_FLAGS)

# AI model integration paths
AI_MODEL_DIR := ../../opt/archangel/ai/models
AI_COMPILED_DIR := ./compiled_models

# Build targets
all: modules

# Build kernel modules
modules:
	@echo "Building Archangel kernel modules..."
	$(MAKE) -C $(KERNEL_DIR) M=$(PWD) modules
	@echo "Kernel modules built successfully"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	$(MAKE) -C $(KERNEL_DIR) M=$(PWD) clean
	rm -rf $(AI_COMPILED_DIR)
	rm -f *.ko.unsigned
	@echo "Clean completed"

# Install kernel modules
install: modules
	@echo "Installing Archangel kernel modules..."
	$(MAKE) -C $(KERNEL_DIR) M=$(PWD) modules_install
	depmod -a
	@echo "Kernel modules installed successfully"

# Load the module
load: modules
	@echo "Loading Archangel core module..."
	sudo insmod $(MODULE_NAME).ko
	@echo "Module loaded successfully"
	@echo "Check /proc/archangel/stats for status"

# Unload the module
unload:
	@echo "Unloading Archangel core module..."
	sudo rmmod $(MODULE_NAME)
	@echo "Module unloaded successfully"

# Reload the module (unload + load)
reload: unload load

# Show module information
info:
	@echo "Module: $(MODULE_NAME)"
	@echo "Version: $(MODULE_VERSION)"
	@echo "Kernel: $(shell uname -r)"
	@echo "Build dir: $(KERNEL_DIR)"
	@if lsmod | grep -q $(MODULE_NAME); then \
		echo "Status: Loaded"; \
		echo "Statistics:"; \
		cat /proc/archangel/stats 2>/dev/null || echo "  Statistics not available"; \
	else \
		echo "Status: Not loaded"; \
	fi

# Compile AI models for kernel integration
compile-models:
	@echo "Compiling AI models for kernel integration..."
	@mkdir -p $(AI_COMPILED_DIR)
	@if [ -d "$(AI_MODEL_DIR)" ]; then \
		echo "Converting TensorFlow models to TensorFlow Lite..."; \
		python3 ../../build/compile_models.py --input $(AI_MODEL_DIR) --output $(AI_COMPILED_DIR); \
	else \
		echo "AI model directory not found: $(AI_MODEL_DIR)"; \
		echo "Skipping model compilation"; \
	fi
	@echo "AI model compilation completed"

# Development helpers
dev-setup:
	@echo "Setting up development environment..."
	@echo "Installing kernel headers if needed..."
	@if [ ! -d "$(KERNEL_DIR)" ]; then \
		echo "Kernel headers not found. Please install:"; \
		echo "  Ubuntu/Debian: sudo apt install linux-headers-$(shell uname -r)"; \
		echo "  RHEL/CentOS: sudo yum install kernel-devel-$(shell uname -r)"; \
		echo "  Arch: sudo pacman -S linux-headers"; \
		exit 1; \
	fi
	@echo "Development environment ready"

# Testing targets
test: modules
	@echo "Running kernel module tests..."
	@echo "Loading module for testing..."
	@sudo insmod $(MODULE_NAME).ko || true
	@sleep 1
	@echo "Checking module status..."
	@lsmod | grep $(MODULE_NAME) || (echo "Module not loaded"; exit 1)
	@echo "Checking proc interface..."
	@cat /proc/archangel/stats || (echo "Proc interface not working"; exit 1)
	@echo "Unloading module..."
	@sudo rmmod $(MODULE_NAME) || true
	@echo "Module tests completed successfully"

# Sign module for secure boot (if needed)
sign: modules
	@echo "Signing kernel module for secure boot..."
	@if [ -f "/usr/src/linux-headers-$(shell uname -r)/scripts/sign-file" ]; then \
		echo "Signing $(MODULE_NAME).ko..."; \
		sudo /usr/src/linux-headers-$(shell uname -r)/scripts/sign-file sha256 \
			/var/lib/shim-signed/mok/MOK.priv \
			/var/lib/shim-signed/mok/MOK.der \
			$(MODULE_NAME).ko; \
	else \
		echo "Module signing tools not found, skipping..."; \
	fi

# Help target
help:
	@echo "Archangel Linux Kernel Module Build System"
	@echo ""
	@echo "Available targets:"
	@echo "  all           - Build all kernel modules (default)"
	@echo "  modules       - Build kernel modules"
	@echo "  clean         - Clean build artifacts"
	@echo "  install       - Install kernel modules to system"
	@echo "  load          - Load the kernel module"
	@echo "  unload        - Unload the kernel module"
	@echo "  reload        - Reload the kernel module"
	@echo "  info          - Show module information and status"
	@echo "  compile-models- Compile AI models for kernel integration"
	@echo "  dev-setup     - Set up development environment"
	@echo "  test          - Run basic module tests"
	@echo "  sign          - Sign module for secure boot"
	@echo "  help          - Show this help message"
	@echo ""
	@echo "Environment variables:"
	@echo "  KERNEL_DIR    - Kernel build directory (default: /lib/modules/$(shell uname -r)/build)"
	@echo ""
	@echo "Examples:"
	@echo "  make                    # Build modules"
	@echo "  make load               # Load module"
	@echo "  make info               # Check status"
	@echo "  make test               # Run tests"
	@echo "  make compile-models     # Compile AI models"

.PHONY: all modules clean install load unload reload info compile-models dev-setup test sign help