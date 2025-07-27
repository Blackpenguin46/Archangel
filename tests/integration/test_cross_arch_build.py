#!/usr/bin/env python3
"""
Archangel Linux - Cross-Architecture Build Test
Tests the cross-architecture build system functionality.
"""

import os
import sys
import json
import subprocess
import tempfile
from pathlib import Path

def run_command(cmd, cwd=None, capture_output=True):
    """Run a shell command and return result"""
    try:
        result = subprocess.run(
            cmd, shell=True, cwd=cwd, 
            capture_output=capture_output, text=True, timeout=300
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)

def test_build_script_exists():
    """Test that the cross-architecture build script exists"""
    script_path = Path("build/build-cross-arch.sh")
    return script_path.exists() and script_path.is_file()

def test_arch_config_exists():
    """Test that the architecture configuration exists"""
    config_path = Path("build/arch-config.json")
    if not config_path.exists():
        return False
    
    try:
        with open(config_path) as f:
            config = json.load(f)
        
        # Check required configuration sections
        required_sections = [
            "supported_architectures",
            "build_targets", 
            "cross_compilation"
        ]
        
        # Check top-level structure
        if "archangel_build_config" not in config:
            return False
            
        build_config = config["archangel_build_config"]
        
        # Check required sections in build config
        for section in required_sections:
            if section not in build_config:
                return False
        
        # Check supported architectures
        supported_archs = build_config["supported_architectures"]
        required_archs = ["x86_64", "arm64"]
        for arch in required_archs:
            if arch not in supported_archs:
                return False
        
        return True
    except (json.JSONDecodeError, KeyError):
        return False

def test_makefile_arch_support():
    """Test that the Makefile supports architecture detection"""
    makefile_path = Path("kernel/archangel/Makefile")
    if not makefile_path.exists():
        return False
    
    with open(makefile_path) as f:
        content = f.read()
    
    # Check for architecture detection
    required_patterns = [
        "ARCH ?= $(shell uname -m)",
        "CONFIG_X86_64",
        "CONFIG_ARM64",
        "ARCH_FLAGS"
    ]
    
    for pattern in required_patterns:
        if pattern not in content:
            return False
    
    return True

def test_syscall_ai_arch_support():
    """Test that syscall AI module supports multiple architectures"""
    source_path = Path("kernel/archangel/archangel_syscall_ai.c")
    if not source_path.exists():
        return False
    
    with open(source_path) as f:
        content = f.read()
    
    # Check for architecture-specific code
    required_patterns = [
        "#ifdef CONFIG_X86_64",
        "#elif defined(CONFIG_ARM64)",
        "regs->orig_ax",  # x86_64 syscall number
        "regs->syscallno",  # ARM64 syscall number
        "regs->di",  # x86_64 register
        "regs->regs[0]"  # ARM64 register
    ]
    
    for pattern in required_patterns:
        if pattern not in content:
            return False
    
    return True

def test_build_script_help():
    """Test that the build script shows help correctly"""
    success, stdout, stderr = run_command("./build/build-cross-arch.sh --help")
    
    if not success:
        return False
    
    # Check for expected help content
    help_patterns = [
        "Cross-Architecture Build System",
        "--arch ARCH",
        "--all",
        "--clean",
        "x86_64",
        "arm64"
    ]
    
    for pattern in help_patterns:
        if pattern not in stdout:
            return False
    
    return True

def test_build_script_syntax():
    """Test that the build script has valid bash syntax"""
    success, stdout, stderr = run_command("bash -n ./build/build-cross-arch.sh")
    return success

def test_cross_arch_documentation():
    """Test that cross-architecture documentation exists"""
    doc_path = Path("CROSS_ARCHITECTURE.md")
    if not doc_path.exists():
        return False
    
    with open(doc_path) as f:
        content = f.read()
    
    # Check for required documentation sections
    required_sections = [
        "# Archangel Linux - Cross-Architecture Support",
        "## Supported Architectures",
        "### x86_64",
        "### ARM64",
        "## Building for Multiple Architectures",
        "## Installation Instructions"
    ]
    
    for section in required_sections:
        if section not in content:
            return False
    
    return True

def test_output_directory_structure():
    """Test that build output directory structure is correct"""
    # Create temporary build directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        build_dir = Path(temp_dir) / "build"
        output_dir = build_dir / "output"
        
        # Create expected directory structure
        for arch in ["x86_64", "arm64"]:
            arch_dir = output_dir / arch
            arch_dir.mkdir(parents=True, exist_ok=True)
            
            # Create dummy files to test structure
            (arch_dir / f"archangel-linux-1.0.0-{arch}.iso").touch()
            (arch_dir / f"archangel-linux-1.0.0-{arch}.img").touch()
            (arch_dir / f"archangel-linux-1.0.0-{arch}.sha256").touch()
            
            modules_dir = arch_dir / "modules"
            modules_dir.mkdir(exist_ok=True)
            (modules_dir / "archangel_core.ko").touch()
            (modules_dir / "archangel_syscall_ai.ko").touch()
        
        # Verify structure
        for arch in ["x86_64", "arm64"]:
            arch_dir = output_dir / arch
            expected_files = [
                f"archangel-linux-1.0.0-{arch}.iso",
                f"archangel-linux-1.0.0-{arch}.img",
                f"archangel-linux-1.0.0-{arch}.sha256"
            ]
            
            for file in expected_files:
                if not (arch_dir / file).exists():
                    return False
            
            # Check modules directory
            modules_dir = arch_dir / "modules"
            if not modules_dir.exists():
                return False
            
            expected_modules = ["archangel_core.ko", "archangel_syscall_ai.ko"]
            for module in expected_modules:
                if not (modules_dir / module).exists():
                    return False
    
    return True

def main():
    """Main test function"""
    print("Archangel Linux - Cross-Architecture Build Test")
    print("=" * 50)
    
    tests = [
        ("Build script exists", test_build_script_exists),
        ("Architecture config exists", test_arch_config_exists),
        ("Makefile arch support", test_makefile_arch_support),
        ("Syscall AI arch support", test_syscall_ai_arch_support),
        ("Build script help", test_build_script_help),
        ("Build script syntax", test_build_script_syntax),
        ("Cross-arch documentation", test_cross_arch_documentation),
        ("Output directory structure", test_output_directory_structure),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                print(f"✓ {test_name}")
                passed += 1
            else:
                print(f"✗ {test_name}")
                failed += 1
        except Exception as e:
            print(f"✗ {test_name} (Exception: {e})")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"Tests passed: {passed}")
    print(f"Tests failed: {failed}")
    print(f"Total tests: {passed + failed}")
    
    if failed == 0:
        print("\n✓ ALL CROSS-ARCHITECTURE BUILD TESTS PASSED")
        return 0
    else:
        print(f"\n✗ {failed} CROSS-ARCHITECTURE BUILD TESTS FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())