#!/usr/bin/env python3
"""
Archangel Memory AI Implementation Verification

This script verifies that the memory AI module implementation
is complete and follows the design requirements.
"""

import os
import sys
import re

def check_file_exists(filepath):
    """Check if a file exists and is readable"""
    if not os.path.exists(filepath):
        return False, f"File {filepath} does not exist"
    
    if not os.path.isfile(filepath):
        return False, f"{filepath} is not a file"
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        return True, content
    except Exception as e:
        return False, f"Cannot read {filepath}: {e}"

def verify_header_file():
    """Verify the memory AI header file"""
    print("Verifying memory AI header file...")
    
    header_path = "kernel/archangel/archangel_memory_ai.h"
    exists, content = check_file_exists(header_path)
    
    if not exists:
        print(f"  FAILED: {content}")
        return False
    
    # Check for required includes
    required_includes = [
        "#include <linux/module.h>",
        "#include <linux/mm.h>",
        "#include \"archangel_core.h\""
    ]
    
    for include in required_includes:
        if include not in content:
            print(f"  FAILED: Missing required include: {include}")
            return False
    
    # Check for header guards
    if "#ifndef _ARCHANGEL_MEMORY_AI_H" not in content:
        print("  FAILED: Missing header guard #ifndef")
        return False
    
    if "#define _ARCHANGEL_MEMORY_AI_H" not in content:
        print("  FAILED: Missing header guard #define")
        return False
    
    if "#endif" not in content:
        print("  FAILED: Missing header guard #endif")
        return False
    
    # Check for required structures
    required_structures = [
        "struct archangel_memory_ai_engine",
        "struct archangel_lstm_predictor",
        "struct archangel_exploit_detector",
        "struct archangel_process_profile",
        "struct archangel_mem_access",
        "struct archangel_lstm_cell"
    ]
    
    for struct in required_structures:
        if struct not in content:
            print(f"  FAILED: Missing required structure: {struct}")
            return False
    
    # Check for required enums
    required_enums = [
        "enum archangel_mem_access_type",
        "enum archangel_mem_pattern_type",
        "enum archangel_exploit_type",
        "enum archangel_mem_decision"
    ]
    
    for enum in required_enums:
        if enum not in content:
            print(f"  FAILED: Missing required enum: {enum}")
            return False
    
    # Check for required function declarations
    required_functions = [
        "archangel_memory_ai_init",
        "archangel_memory_ai_cleanup",
        "archangel_ai_handle_mm_fault",
        "archangel_lstm_predict_next_access",
        "archangel_detect_exploit_pattern",
        "archangel_process_profile_lookup"
    ]
    
    for func in required_functions:
        if func not in content:
            print(f"  FAILED: Missing required function declaration: {func}")
            return False
    
    print("  PASSED: Header file verification completed")
    return True

def verify_implementation_file():
    """Verify the memory AI implementation file"""
    print("Verifying memory AI implementation file...")
    
    impl_path = "kernel/archangel/archangel_memory_ai.c"
    exists, content = check_file_exists(impl_path)
    
    if not exists:
        print(f"  FAILED: {content}")
        return False
    
    # Check for required includes
    if "#include \"archangel_memory_ai.h\"" not in content:
        print("  FAILED: Missing header include")
        return False
    
    # Check for module information
    module_info = [
        "MODULE_LICENSE",
        "MODULE_AUTHOR",
        "MODULE_DESCRIPTION"
    ]
    
    for info in module_info:
        if info not in content:
            print(f"  FAILED: Missing module info: {info}")
            return False
    
    # Check for required function implementations
    required_functions = [
        "archangel_memory_ai_init",
        "archangel_memory_ai_cleanup",
        "archangel_ai_handle_mm_fault",
        "archangel_lstm_predict_next_access",
        "archangel_detect_exploit_pattern",
        "archangel_process_profile_lookup",
        "archangel_lstm_predictor_init",
        "archangel_exploit_detector_init"
    ]
    
    for func in required_functions:
        # Look for function definition pattern
        pattern = rf"{func}\s*\([^)]*\)\s*\{{"
        if not re.search(pattern, content):
            print(f"  FAILED: Missing function implementation: {func}")
            return False
    
    # Check for LSTM mathematical functions
    lstm_functions = ["archangel_lstm_sigmoid", "archangel_lstm_tanh"]
    for func in lstm_functions:
        if func not in content:
            print(f"  FAILED: Missing LSTM mathematical function: {func}")
            return False
    
    # Check for exploit signatures
    if "default_exploit_signatures" not in content:
        print("  FAILED: Missing default exploit signatures")
        return False
    
    # Check for global AI instance
    if "archangel_mem_ai" not in content:
        print("  FAILED: Missing global AI instance")
        return False
    
    # Check for module init/exit functions
    if "module_init" not in content or "module_exit" not in content:
        print("  FAILED: Missing module init/exit functions")
        return False
    
    print("  PASSED: Implementation file verification completed")
    return True

def verify_makefile_integration():
    """Verify that the memory AI module is integrated into the Makefile"""
    print("Verifying Makefile integration...")
    
    makefile_path = "kernel/archangel/Makefile"
    exists, content = check_file_exists(makefile_path)
    
    if not exists:
        print(f"  FAILED: {content}")
        return False
    
    # Check for memory AI module entry
    if "archangel_memory_ai.o" not in content:
        print("  FAILED: Memory AI module not found in Makefile")
        return False
    
    # Check for module object definition
    if "archangel_memory_ai-objs" not in content:
        print("  FAILED: Memory AI module objects not defined in Makefile")
        return False
    
    print("  PASSED: Makefile integration verified")
    return True

def verify_test_file():
    """Verify that the test file exists"""
    print("Verifying test file...")
    
    test_path = "tests/kernel/test_archangel_memory_ai_compilation.c"
    exists, content = check_file_exists(test_path)
    
    if not exists:
        print(f"  FAILED: {content}")
        return False
    
    print("  PASSED: Test file exists")
    return True

def check_design_requirements():
    """Check if implementation meets design requirements"""
    print("Checking design requirements compliance...")
    
    impl_path = "kernel/archangel/archangel_memory_ai.c"
    exists, content = check_file_exists(impl_path)
    
    if not exists:
        print(f"  FAILED: Cannot read implementation file")
        return False
    
    # Check for LSTM predictor implementation
    lstm_features = ["lstm", "predictor", "hidden_state", "cell_state"]
    lstm_found = all(feature.lower() in content.lower() for feature in lstm_features)
    
    if not lstm_found:
        print("  FAILED: LSTM predictor features not fully implemented")
        return False
    else:
        print("  PASSED: LSTM predictor features implemented")
    
    # Check for exploit detection implementation
    exploit_features = ["exploit", "pattern", "signature", "detection"]
    exploit_found = all(feature.lower() in content.lower() for feature in exploit_features)
    
    if not exploit_found:
        print("  FAILED: Exploit detection features not fully implemented")
        return False
    else:
        print("  PASSED: Exploit detection features implemented")
    
    # Check for page fault handling
    fault_features = ["handle_mm_fault", "vm_fault", "page_fault"]
    fault_found = any(feature.lower() in content.lower() for feature in fault_features)
    
    if not fault_found:
        print("  FAILED: Page fault handling not implemented")
        return False
    else:
        print("  PASSED: Page fault handling implemented")
    
    # Check for prefetch optimization
    prefetch_features = ["prefetch", "prediction", "optimization"]
    prefetch_found = all(feature.lower() in content.lower() for feature in prefetch_features)
    
    if not prefetch_found:
        print("  FAILED: Prefetch optimization not fully implemented")
        return False
    else:
        print("  PASSED: Prefetch optimization implemented")
    
    # Check for process termination logic
    termination_features = ["terminate", "SIGKILL", "send_sig"]
    termination_found = all(feature in content for feature in termination_features)
    
    if not termination_found:
        print("  FAILED: Process termination logic not implemented")
        return False
    else:
        print("  PASSED: Process termination logic implemented")
    
    # Check for memory access pattern analysis
    pattern_features = ["pattern", "access", "analyze", "memory"]
    pattern_found = all(feature.lower() in content.lower() for feature in pattern_features)
    
    if not pattern_found:
        print("  FAILED: Memory access pattern analysis not implemented")
        return False
    else:
        print("  PASSED: Memory access pattern analysis implemented")
    
    return True

def main():
    """Main verification function"""
    print("=== Archangel Memory AI Implementation Verification ===\n")
    
    tests = [
        verify_header_file,
        verify_implementation_file,
        verify_makefile_integration,
        verify_test_file,
        check_design_requirements
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ERROR: Test failed with exception: {e}")
            failed += 1
        print()
    
    print("=== Verification Results ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print("\nMemory AI implementation verification FAILED")
        return 1
    else:
        print("\nMemory AI implementation verification PASSED")
        return 0

if __name__ == "__main__":
    sys.exit(main())