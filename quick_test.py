#!/usr/bin/env python3
"""
Archangel Linux - Quick Test Script
Simple test without heavy dependencies
"""

import sys
import os
from pathlib import Path

def test_basic_structure():
    """Test basic project structure"""
    print("🏗️ Testing Project Structure...")
    
    required_files = [
        "archangel_ai.py",
        "archangel_lightweight.py", 
        "cli.py",
        "demo_archangel.py",
        "Dockerfile",
        "docker-compose.yml",
        "requirements.txt"
    ]
    
    required_dirs = [
        "core",
        "tools", 
        "ui",
        "kernel"
    ]
    
    missing_files = []
    missing_dirs = []
    
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
        else:
            print(f"  ✅ {file}")
    
    for dir in required_dirs:
        if not Path(dir).exists():
            missing_dirs.append(dir)
        else:
            print(f"  ✅ {dir}/")
    
    if missing_files:
        print(f"  ❌ Missing files: {missing_files}")
    if missing_dirs:
        print(f"  ❌ Missing directories: {missing_dirs}")
    
    return len(missing_files) == 0 and len(missing_dirs) == 0

def test_core_modules():
    """Test core module structure"""
    print("\n🧠 Testing Core Modules...")
    
    core_files = [
        "core/__init__.py",
        "core/ai_security_expert.py",
        "core/real_ai_security_expert.py",
        "core/conversational_ai.py",
        "core/unified_ai_orchestrator.py"
    ]
    
    existing_files = []
    for file in core_files:
        if Path(file).exists():
            existing_files.append(file)
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file}")
    
    return len(existing_files) >= 3  # At least 3 core files should exist

def test_python_syntax():
    """Test Python syntax of main files"""
    print("\n🐍 Testing Python Syntax...")
    
    main_files = [
        "archangel_ai.py",
        "archangel_lightweight.py",
        "cli.py",
        "demo_archangel.py"
    ]
    
    syntax_ok = 0
    for file in main_files:
        if Path(file).exists():
            try:
                with open(file, 'r') as f:
                    compile(f.read(), file, 'exec')
                print(f"  ✅ {file} - syntax OK")
                syntax_ok += 1
            except SyntaxError as e:
                print(f"  ❌ {file} - syntax error: {e}")
            except Exception as e:
                print(f"  ⚠️ {file} - could not check: {e}")
        else:
            print(f"  ❌ {file} - not found")
    
    return syntax_ok >= 2

def test_docker_setup():
    """Test Docker configuration"""
    print("\n🐳 Testing Docker Setup...")
    
    docker_files = ["Dockerfile", "docker-compose.yml"]
    docker_ok = 0
    
    for file in docker_files:
        if Path(file).exists():
            print(f"  ✅ {file}")
            docker_ok += 1
        else:
            print(f"  ❌ {file}")
    
    # Check if docker-setup.sh is executable
    setup_script = Path("docker-setup.sh")
    if setup_script.exists():
        if os.access(setup_script, os.X_OK):
            print("  ✅ docker-setup.sh (executable)")
        else:
            print("  ⚠️ docker-setup.sh (not executable - run: chmod +x docker-setup.sh)")
        docker_ok += 1
    else:
        print("  ❌ docker-setup.sh")
    
    return docker_ok >= 2

def test_dependencies():
    """Test if key dependencies can be imported"""
    print("\n📦 Testing Dependencies...")
    
    # Test standard library imports
    standard_libs = ['asyncio', 'json', 'pathlib', 'dataclasses', 'enum']
    std_ok = 0
    
    for lib in standard_libs:
        try:
            __import__(lib)
            print(f"  ✅ {lib}")
            std_ok += 1
        except ImportError:
            print(f"  ❌ {lib}")
    
    # Test optional dependencies
    optional_deps = ['rich', 'requests', 'transformers', 'torch']
    opt_ok = 0
    
    for dep in optional_deps:
        try:
            __import__(dep)
            print(f"  ✅ {dep} (optional)")
            opt_ok += 1
        except ImportError:
            print(f"  ⚠️ {dep} (optional - not installed)")
    
    print(f"\n  📊 Standard libraries: {std_ok}/{len(standard_libs)}")
    print(f"  📊 Optional dependencies: {opt_ok}/{len(optional_deps)}")
    
    return std_ok == len(standard_libs)

def show_next_steps(all_passed):
    """Show next steps based on test results"""
    print("\n" + "="*60)
    print("🎯 NEXT STEPS")
    print("="*60)
    
    if all_passed:
        print("🎉 Basic structure looks good!")
        print("\n📋 Recommended testing order:")
        print("  1. Install dependencies:")
        print("     pip install -r requirements.txt")
        print("\n  2. Test basic functionality:")
        print("     python3 demo_archangel.py")
        print("\n  3. Try interactive mode:")
        print("     python3 cli.py")
        print("\n  4. Set up Docker environment:")
        print("     chmod +x docker-setup.sh")
        print("     ./docker-setup.sh --start")
        print("\n  5. Test in Docker:")
        print("     docker-compose exec archangel bash")
        print("     python3 demo_archangel.py")
    else:
        print("🔧 Some issues found. Please fix them first:")
        print("\n  • Check missing files/directories")
        print("  • Fix any syntax errors")
        print("  • Ensure Docker files are present")
        print("\n  Then run this test again!")
    
    print("\n💡 For HuggingFace AI features:")
    print("  • Get a free token at: https://huggingface.co/settings/tokens")
    print("  • Set it with: export HF_TOKEN='your_token_here'")
    print("  • Or add it to .env file")

def main():
    """Run all tests"""
    print("🛡️ ARCHANGEL LINUX - QUICK TEST")
    print("="*50)
    
    tests = [
        ("Project Structure", test_basic_structure),
        ("Core Modules", test_core_modules), 
        ("Python Syntax", test_python_syntax),
        ("Docker Setup", test_docker_setup),
        ("Dependencies", test_dependencies)
    ]
    
    passed = 0
    total = len(tests)
    
    for name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
        except Exception as e:
            print(f"  ❌ Test error: {e}")
    
    print(f"\n📊 SUMMARY: {passed}/{total} tests passed")
    
    show_next_steps(passed == total)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)