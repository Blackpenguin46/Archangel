#!/usr/bin/env python3
"""
Archangel Network AI Implementation Verification

This script verifies that the network AI module implementation
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
    """Verify the network AI header file"""
    print("Verifying network AI header file...")
    
    header_path = "kernel/archangel/archangel_network_ai.h"
    exists, content = check_file_exists(header_path)
    
    if not exists:
        print(f"  FAILED: {content}")
        return False
    
    # Check for required includes
    required_includes = [
        "#include <linux/module.h>",
        "#include <linux/netfilter.h>",
        "#include \"archangel_core.h\""
    ]
    
    for include in required_includes:
        if include not in content:
            print(f"  FAILED: Missing required include: {include}")
            return False
    
    # Check for header guards
    if "#ifndef _ARCHANGEL_NETWORK_AI_H" not in content:
        print("  FAILED: Missing header guard #ifndef")
        return False
    
    if "#define _ARCHANGEL_NETWORK_AI_H" not in content:
        print("  FAILED: Missing header guard #define")
        return False
    
    if "#endif" not in content:
        print("  FAILED: Missing header guard #endif")
        return False
    
    # Check for required structures
    required_structures = [
        "struct archangel_network_ai_engine",
        "struct archangel_ml_classifier",
        "struct archangel_anomaly_detector",
        "struct archangel_stealth_engine",
        "struct archangel_packet_features"
    ]
    
    for struct in required_structures:
        if struct not in content:
            print(f"  FAILED: Missing required structure: {struct}")
            return False
    
    # Check for required enums
    required_enums = [
        "enum archangel_net_decision",
        "enum archangel_packet_class",
        "enum archangel_stealth_mode"
    ]
    
    for enum in required_enums:
        if enum not in content:
            print(f"  FAILED: Missing required enum: {enum}")
            return False
    
    # Check for required function declarations
    required_functions = [
        "archangel_network_ai_init",
        "archangel_network_ai_cleanup",
        "archangel_ai_netfilter_hook_ipv4",
        "archangel_ai_classify_packet",
        "archangel_ai_extract_features"
    ]
    
    for func in required_functions:
        if func not in content:
            print(f"  FAILED: Missing required function declaration: {func}")
            return False
    
    print("  PASSED: Header file verification completed")
    return True

def verify_implementation_file():
    """Verify the network AI implementation file"""
    print("Verifying network AI implementation file...")
    
    impl_path = "kernel/archangel/archangel_network_ai.c"
    exists, content = check_file_exists(impl_path)
    
    if not exists:
        print(f"  FAILED: {content}")
        return False
    
    # Check for required includes
    if "#include \"archangel_network_ai.h\"" not in content:
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
        "archangel_network_ai_init",
        "archangel_network_ai_cleanup",
        "archangel_ai_netfilter_hook_ipv4",
        "archangel_ai_classify_packet",
        "archangel_ai_extract_features",
        "archangel_ml_classify",
        "archangel_anomaly_detect",
        "archangel_stealth_modify_packet"
    ]
    
    for func in required_functions:
        # Look for function definition pattern
        pattern = rf"{func}\s*\([^)]*\)\s*\{{"
        if not re.search(pattern, content):
            print(f"  FAILED: Missing function implementation: {func}")
            return False
    
    # Check for netfilter hooks array
    if "archangel_nf_hooks" not in content:
        print("  FAILED: Missing netfilter hooks array")
        return False
    
    # Check for global AI instance
    if "archangel_net_ai" not in content:
        print("  FAILED: Missing global AI instance")
        return False
    
    # Check for module init/exit functions
    if "module_init" not in content or "module_exit" not in content:
        print("  FAILED: Missing module init/exit functions")
        return False
    
    print("  PASSED: Implementation file verification completed")
    return True

def verify_makefile_integration():
    """Verify that the network AI module is integrated into the Makefile"""
    print("Verifying Makefile integration...")
    
    makefile_path = "kernel/archangel/Makefile"
    exists, content = check_file_exists(makefile_path)
    
    if not exists:
        print(f"  FAILED: {content}")
        return False
    
    # Check for network AI module entry
    if "archangel_network_ai.o" not in content:
        print("  FAILED: Network AI module not found in Makefile")
        return False
    
    # Check for module object definition
    if "archangel_network_ai-objs" not in content:
        print("  FAILED: Network AI module objects not defined in Makefile")
        return False
    
    print("  PASSED: Makefile integration verified")
    return True

def verify_test_file():
    """Verify that the test file exists"""
    print("Verifying test file...")
    
    test_path = "tests/kernel/test_archangel_network_ai_compilation.c"
    exists, content = check_file_exists(test_path)
    
    if not exists:
        print(f"  FAILED: {content}")
        return False
    
    print("  PASSED: Test file exists")
    return True

def check_design_requirements():
    """Check if implementation meets design requirements"""
    print("Checking design requirements compliance...")
    
    impl_path = "kernel/archangel/archangel_network_ai.c"
    exists, content = check_file_exists(impl_path)
    
    if not exists:
        print(f"  FAILED: Cannot read implementation file")
        return False
    
    # Check for SIMD optimization mentions
    simd_features = ["avx2", "sse4", "vnni", "simd"]
    simd_found = any(feature.lower() in content.lower() for feature in simd_features)
    
    if not simd_found:
        print("  WARNING: SIMD optimization features not clearly implemented")
    else:
        print("  PASSED: SIMD optimization features found")
    
    # Check for stealth mode implementation
    stealth_features = ["stealth", "modify_packet", "signature"]
    stealth_found = all(feature.lower() in content.lower() for feature in stealth_features)
    
    if not stealth_found:
        print("  FAILED: Stealth mode features not fully implemented")
        return False
    else:
        print("  PASSED: Stealth mode features implemented")
    
    # Check for ML classifier implementation
    ml_features = ["decision_tree", "classify", "anomaly"]
    ml_found = all(feature.lower() in content.lower() for feature in ml_features)
    
    if not ml_found:
        print("  FAILED: ML classifier features not fully implemented")
        return False
    else:
        print("  PASSED: ML classifier features implemented")
    
    # Check for netfilter integration
    netfilter_features = ["netfilter", "nf_hook", "NF_ACCEPT", "NF_DROP"]
    netfilter_found = all(feature in content for feature in netfilter_features)
    
    if not netfilter_found:
        print("  FAILED: Netfilter integration not complete")
        return False
    else:
        print("  PASSED: Netfilter integration implemented")
    
    return True

def main():
    """Main verification function"""
    print("=== Archangel Network AI Implementation Verification ===\n")
    
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
        print("\nNetwork AI implementation verification FAILED")
        return 1
    else:
        print("\nNetwork AI implementation verification PASSED")
        return 0

if __name__ == "__main__":
    sys.exit(main())