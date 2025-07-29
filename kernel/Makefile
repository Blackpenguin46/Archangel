# Archangel Linux Kernel Module Makefile

# Module information
MODULE_NAME = archangel
MODULE_VERSION = 0.1.0

# Object files
obj-m += $(MODULE_NAME).o
$(MODULE_NAME)-objs := src/archangel_main.o src/archangel_syscall.o src/archangel_communication.o

# Kernel build directory
KERNEL_DIR ?= /lib/modules/$(shell uname -r)/build

# Current directory
PWD := $(shell pwd)

# Compiler flags
ccflags-y := -I$(PWD)/include -DMODULE_VERSION=\"$(MODULE_VERSION)\"

# Default target
all: modules

# Build kernel modules
modules:
	@echo "Building Archangel kernel module v$(MODULE_VERSION)..."
	$(MAKE) -C $(KERNEL_DIR) M=$(PWD) modules

# Clean build artifacts
clean:
	@echo "Cleaning Archangel kernel module build..."
	$(MAKE) -C $(KERNEL_DIR) M=$(PWD) clean
	rm -f modules.order Module.symvers

# Install module
install: modules
	@echo "Installing Archangel kernel module..."
	sudo $(MAKE) -C $(KERNEL_DIR) M=$(PWD) modules_install
	sudo depmod -a

# Load module
load:
	@echo "Loading Archangel kernel module..."
	sudo insmod $(MODULE_NAME).ko

# Unload module
unload:
	@echo "Unloading Archangel kernel module..."
	sudo rmmod $(MODULE_NAME) || true

# Reload module (unload + load)
reload: unload load

# Show module info
info:
	@echo "Archangel Kernel Module Information:"
	@echo "  Name: $(MODULE_NAME)"
	@echo "  Version: $(MODULE_VERSION)"
	@echo "  Kernel: $(shell uname -r)"
	@echo "  Build Dir: $(KERNEL_DIR)"
	@if lsmod | grep -q $(MODULE_NAME); then \
		echo "  Status: Loaded"; \
		echo "  Info:"; \
		modinfo $(MODULE_NAME) 2>/dev/null || true; \
	else \
		echo "  Status: Not loaded"; \
	fi

# Show kernel logs
logs:
	@echo "Recent Archangel kernel messages:"
	sudo dmesg | grep -i archangel | tail -20

# Test module functionality
test: load
	@echo "Testing Archangel kernel module..."
	@echo "Checking proc filesystem:"
	@ls -la /proc/archangel/ || echo "Proc filesystem not available"
	@echo ""
	@echo "Module status:"
	@cat /proc/archangel/status 2>/dev/null || echo "Status not available"
	@echo ""
	@echo "Module statistics:"
	@cat /proc/archangel/stats 2>/dev/null || echo "Stats not available"

# Development helpers
dev: clean modules load test

# Create kernel module tarball
package: clean
	@echo "Creating Archangel kernel module package..."
	@tar -czf archangel-kernel-$(MODULE_VERSION).tar.gz \
		Makefile src/ include/ scripts/ README.md
	@echo "Package created: archangel-kernel-$(MODULE_VERSION).tar.gz"

# Check kernel module dependencies
deps:
	@echo "Checking kernel module build dependencies..."
	@echo "Kernel version: $(shell uname -r)"
	@if [ -d "$(KERNEL_DIR)" ]; then \
		echo "Kernel headers: Found at $(KERNEL_DIR)"; \
	else \
		echo "Kernel headers: NOT FOUND"; \
		echo "Install with: sudo pacman -S linux-headers"; \
	fi
	@if which gcc >/dev/null 2>&1; then \
		echo "GCC: $(shell gcc --version | head -1)"; \
	else \
		echo "GCC: NOT FOUND - install with: sudo pacman -S gcc"; \
	fi
	@if which make >/dev/null 2>&1; then \
		echo "Make: $(shell make --version | head -1)"; \
	else \
		echo "Make: NOT FOUND - install with: sudo pacman -S make"; \
	fi

# Help target
help:
	@echo "Archangel Kernel Module Build System"
	@echo ""
	@echo "Available targets:"
	@echo "  all       - Build kernel module (default)"
	@echo "  modules   - Build kernel module"
	@echo "  clean     - Clean build artifacts"
	@echo "  install   - Install module to system"
	@echo "  load      - Load module into kernel"
	@echo "  unload    - Unload module from kernel"
	@echo "  reload    - Unload and reload module"
	@echo "  info      - Show module information"
	@echo "  logs      - Show recent kernel messages"
	@echo "  test      - Load module and run basic tests"
	@echo "  dev       - Development cycle (clean, build, load, test)"
	@echo "  package   - Create source tarball"
	@echo "  deps      - Check build dependencies"
	@echo "  help      - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make dev                 - Quick development cycle"
	@echo "  make load && make test   - Load and test module"
	@echo "  make logs                - View kernel messages"

.PHONY: all modules clean install load unload reload info logs test dev package deps help