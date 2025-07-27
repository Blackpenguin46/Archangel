#!/usr/bin/env python3
"""
Archangel Linux - Syscall AI Implementation Verification
This script verifies that all required components of the syscall AI module are implemented.
"""

import os
import re
import sys

def check_file_exists(filepath):
    """Check if a file exists"""
    return os.path.exists(filepath)

def check_function_implemented(filepath, function_name):
    """Check if a function is implemented in a file"""
    if not os.path.exists(filepath):
        return False
    
    with open(filepath, 'r') as f:
        content = f.read()
        # Look for function definition
        pattern = rf'{function_name}\s*\([^)]*\)\s*\{{'
        return bool(re.search(pattern, content))

def check_structure_defined(filepath, struct_name):
    """Check if a structure is defined in a file"""
    if not os.path.exists(filepath):
        return False
    
    with open(filepath, 'r') as f:
        content = f.read()
        pattern = rf'struct\s+{struct_name}\s*\{{'
        return bool(re.search(pattern, content))

def check_symbol_exported(filepath, symbol_name):
    """Check if a symbol is exported in a file"""
    if not os.path.exists(filepath):
        return False
    
    with open(filepath, 'r') as f:
        content = f.read()
        pattern = rf'EXPORT_SYMBOL\s*\(\s*{symbol_name}\s*\)'
        return bool(re.search(pattern, content))

def main():
    """Main verification function"""
    print("Archangel Linux - Syscall AI Implementation Verification")
    print("=" * 60)
    
    base_path = "kernel/archangel"
    header_file = f"{base_path}/archangel_syscall_ai.h"
    source_file = f"{base_path}/archangel_syscall_ai.c"
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Check if files exist
    print("\n1. File Existence Tests:")
    if check_file_exists(header_file):
        print("   ✓ Header file exists")
        tests_passed += 1
    else:
        print("   ✗ Header file missing")
        tests_failed += 1
    
    if check_file_exists(source_file):
        print("   ✓ Source file exists")
        tests_passed += 1
    else:
        print("   ✗ Source file missing")
        tests_failed += 1
    
    # Test 2: Check required structures
    print("\n2. Structure Definition Tests:")
    required_structures = [
        "archangel_syscall_ai_engine",
        "archangel_process_profile", 
        "archangel_decision_cache",
        "archangel_syscall_pattern",
        "archangel_decision_tree",
        "archangel_syscall_context"
    ]
    
    for struct in required_structures:
        if check_structure_defined(header_file, struct):
            print(f"   ✓ {struct} structure defined")
            tests_passed += 1
        else:
            print(f"   ✗ {struct} structure missing")
            tests_failed += 1
    
    # Test 3: Check required functions
    print("\n3. Function Implementation Tests:")
    required_functions = [
        "ai_syscall_intercept",
        "archangel_syscall_ai_init",
        "archangel_syscall_ai_cleanup",
        "archangel_syscall_ai_enable",
        "archangel_syscall_ai_disable",
        "archangel_pattern_match",
        "archangel_profile_get_or_create",
        "archangel_profile_update",
        "archangel_decision_cache_lookup",
        "archangel_decision_cache_store",
        "archangel_userspace_defer_request"
    ]
    
    for func in required_functions:
        if check_function_implemented(source_file, func):
            print(f"   ✓ {func} function implemented")
            tests_passed += 1
        else:
            print(f"   ✗ {func} function missing")
            tests_failed += 1
    
    # Test 4: Check exported symbols
    print("\n4. Symbol Export Tests:")
    required_exports = [
        "ai_syscall_intercept",
        "archangel_syscall_ai_enable",
        "archangel_syscall_ai_disable",
        "archangel_syscall_ai_is_enabled"
    ]
    
    for symbol in required_exports:
        if check_symbol_exported(source_file, symbol):
            print(f"   ✓ {symbol} symbol exported")
            tests_passed += 1
        else:
            print(f"   ✗ {symbol} symbol not exported")
            tests_failed += 1
    
    # Test 5: Check task requirements implementation
    print("\n5. Task Requirements Tests:")
    
    # Check for decision trees implementation
    if check_function_implemented(source_file, "archangel_decision_tree_evaluate"):
        print("   ✓ Decision trees implemented")
        tests_passed += 1
    else:
        print("   ✗ Decision trees not implemented")
        tests_failed += 1
    
    # Check for pattern matching implementation
    if check_function_implemented(source_file, "archangel_pattern_add"):
        print("   ✓ Pattern matching implemented")
        tests_passed += 1
    else:
        print("   ✗ Pattern matching not implemented")
        tests_failed += 1
    
    # Check for per-process profiling
    if check_function_implemented(source_file, "archangel_profile_calculate_risk"):
        print("   ✓ Per-process behavioral profiling implemented")
        tests_passed += 1
    else:
        print("   ✗ Per-process behavioral profiling not implemented")
        tests_failed += 1
    
    # Check for decision caching
    if check_function_implemented(source_file, "archangel_decision_cache_init"):
        print("   ✓ Decision caching implemented")
        tests_passed += 1
    else:
        print("   ✗ Decision caching not implemented")
        tests_failed += 1
    
    # Check for userspace deferral
    if check_function_implemented(source_file, "archangel_userspace_defer_wait_response"):
        print("   ✓ Userspace deferral mechanisms implemented")
        tests_passed += 1
    else:
        print("   ✗ Userspace deferral mechanisms not implemented")
        tests_failed += 1
    
    # Print results
    print("\n" + "=" * 60)
    print(f"Tests passed: {tests_passed}")
    print(f"Tests failed: {tests_failed}")
    print(f"Total tests: {tests_passed + tests_failed}")
    
    if tests_failed == 0:
        print("\n✓ ALL SYSCALL AI IMPLEMENTATION REQUIREMENTS MET")
        return 0
    else:
        print(f"\n✗ {tests_failed} IMPLEMENTATION REQUIREMENTS NOT MET")
        return 1

if __name__ == "__main__":
    sys.exit(main())