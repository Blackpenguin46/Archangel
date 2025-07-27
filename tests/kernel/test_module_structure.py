#!/usr/bin/env python3
"""
Test script to verify kernel module structure and build system
"""

import os
import sys
import subprocess
from pathlib import Path

def test_kernel_module_files():
    """Test that all required kernel module files exist"""
    project_root = Path(__file__).parent.parent.parent
    kernel_dir = project_root / "kernel" / "archangel"
    
    required_files = [
        "archangel_core.h",
        "archangel_core.c",
        "Makefile"
    ]
    
    print("Testing kernel module file structure...")
    
    for file in required_files:
        file_path = kernel_dir / file
        if not file_path.exists():
            print(f"❌ Missing required file: {file}")
            return False
        else:
            print(f"✅ Found: {file}")
    
    return True

def test_makefile_targets():
    """Test that Makefile contains required targets"""
    project_root = Path(__file__).parent.parent.parent
    makefile_path = project_root / "kernel" / "archangel" / "Makefile"
    
    required_targets = [
        "all:",
        "modules:",
        "clean:",
        "install:",
        "load:",
        "unload:",
        "info:",
        "compile-models:",
        "help:"
    ]
    
    print("\nTesting Makefile targets...")
    
    with open(makefile_path, 'r') as f:
        makefile_content = f.read()
    
    for target in required_targets:
        if target in makefile_content:
            print(f"✅ Found target: {target}")
        else:
            print(f"❌ Missing target: {target}")
            return False
    
    return True

def test_header_structure():
    """Test that header file has proper structure"""
    project_root = Path(__file__).parent.parent.parent
    header_path = project_root / "kernel" / "archangel" / "archangel_core.h"
    
    required_elements = [
        "#ifndef _ARCHANGEL_CORE_H",
        "#define _ARCHANGEL_CORE_H",
        "struct archangel_kernel_ai",
        "extern struct archangel_kernel_ai *archangel_ai",
        "int archangel_core_init(void)",
        "void archangel_core_exit(void)"
    ]
    
    print("\nTesting header file structure...")
    
    with open(header_path, 'r') as f:
        header_content = f.read()
    
    for element in required_elements:
        if element in header_content:
            print(f"✅ Found: {element}")
        else:
            print(f"❌ Missing: {element}")
            return False
    
    return True

def test_build_system():
    """Test that build system files exist and are executable"""
    project_root = Path(__file__).parent.parent.parent
    
    build_files = [
        ("build/compile_models.py", True),
        ("build/build-archangel.sh", True),
        ("install/install_archangel.sh", True)
    ]
    
    print("\nTesting build system files...")
    
    for file_path, should_be_executable in build_files:
        full_path = project_root / file_path
        if not full_path.exists():
            print(f"❌ Missing build file: {file_path}")
            return False
        
        if should_be_executable and not os.access(full_path, os.X_OK):
            print(f"❌ File not executable: {file_path}")
            return False
        
        print(f"✅ Found: {file_path}")
    
    return True

def test_directory_structure():
    """Test that all required directories exist"""
    project_root = Path(__file__).parent.parent.parent
    
    required_dirs = [
        "kernel/archangel",
        "opt/archangel/ai",
        "opt/archangel/tools", 
        "opt/archangel/security",
        "opt/archangel/gui",
        "opt/archangel/bin",
        "build",
        "tests/kernel",
        "tests/integration",
        "tests/performance",
        "install"
    ]
    
    print("\nTesting directory structure...")
    
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if not full_path.exists():
            print(f"❌ Missing directory: {dir_path}")
            return False
        else:
            print(f"✅ Found directory: {dir_path}")
    
    return True

def main():
    """Run all tests"""
    print("Archangel Linux Kernel Module Structure Tests")
    print("=" * 50)
    
    tests = [
        test_directory_structure,
        test_kernel_module_files,
        test_makefile_targets,
        test_header_structure,
        test_build_system
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
                print("✅ Test passed\n")
            else:
                print("❌ Test failed\n")
        except Exception as e:
            print(f"❌ Test error: {e}\n")
    
    print("=" * 50)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Kernel module structure is correct.")
        return 0
    else:
        print("❌ Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())