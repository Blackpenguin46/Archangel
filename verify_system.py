#!/usr/bin/env python3
"""
System Verification Script - M2 MacBook
Quick checks to ensure all components work correctly
"""

import sys
import subprocess
import importlib
from pathlib import Path

def check_python_version():
    """Check Python version compatibility."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Need Python 3.8+")
        return False

def check_dependencies():
    """Check required Python packages."""
    print("\n📦 Checking dependencies...")
    
    required_packages = [
        'torch',
        'transformers', 
        'datasets',
        'numpy',
        'pandas',
        'aiofiles',
        'psutil',
        'requests',
        'pydantic',
        'docker'
    ]
    
    missing = []
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Missing")
            missing.append(package)
    
    return len(missing) == 0

def check_docker():
    """Check Docker availability."""
    print("\n🐳 Checking Docker...")
    
    try:
        # Check if docker command exists
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ {result.stdout.strip()}")
            
            # Check if Docker daemon is running
            result = subprocess.run(['docker', 'info'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("✅ Docker daemon is running")
                return True
            else:
                print("❌ Docker daemon not running - Start Docker Desktop")
                return False
        else:
            print("❌ Docker command not found")
            return False
            
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ Docker not available")
        return False

def check_memory():
    """Check system memory."""
    print("\n💾 Checking system memory...")
    
    try:
        import psutil
        memory = psutil.virtual_memory()
        total_gb = memory.total / (1024**3)
        available_gb = memory.available / (1024**3)
        
        print(f"📊 Total RAM: {total_gb:.1f} GB")
        print(f"📊 Available RAM: {available_gb:.1f} GB")
        print(f"📊 Memory usage: {memory.percent:.1f}%")
        
        if total_gb >= 16:
            print("✅ Sufficient memory for M2 MacBook operations")
            return True
        else:
            print("⚠️  Limited memory - may need optimizations")
            return True  # Still workable
            
    except ImportError:
        print("❌ Cannot check memory - psutil not installed")
        return False

def check_ai_capabilities():
    """Check AI/ML capabilities."""
    print("\n🧠 Checking AI capabilities...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        # Check Apple Silicon support
        if torch.backends.mps.is_available():
            print("✅ Apple Silicon GPU (MPS) available")
            print("🚀 Hardware acceleration enabled")
        else:
            print("💻 Using CPU (still efficient on M2)")
        
        # Test model loading capability
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
            print("✅ Model loading capability verified")
            return True
        except Exception as e:
            print(f"⚠️  Model loading test failed: {e}")
            return False
            
    except ImportError:
        print("❌ PyTorch not available")
        return False

def check_project_structure():
    """Check project file structure."""
    print("\n📁 Checking project structure...")
    
    required_files = [
        'archangel_autonomous_system.py',
        'core/autonomous_security_agents.py', 
        'training/deepseek_training_pipeline.py',
        'environments/live_adversarial_environment.py',
        'agents/live_combat_agents.py',
        'config/m2_macbook_config.json'
    ]
    
    missing_files = []
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - Missing")
            missing_files.append(file_path)
    
    return len(missing_files) == 0

def main():
    """Run all verification checks."""
    print("🍎 Archangel System Verification - M2 MacBook")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Docker", check_docker), 
        ("Memory", check_memory),
        ("AI Capabilities", check_ai_capabilities),
        ("Project Structure", check_project_structure)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} check failed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Verification Summary")
    print("-" * 25)
    
    passed = 0
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} checks passed")
    
    if passed == len(results):
        print("\n🎉 System ready for live combat operations!")
        print("\nNext steps:")
        print("• Run: ./start_combat.sh")
        print("• Or: python3 test_live_combat_system.py")
    else:
        print("\n⚠️  Some issues detected. Please fix before proceeding.")
        print("\nCommon fixes:")
        print("• Install missing packages: pip install -r requirements.txt")
        print("• Start Docker Desktop application")
        print("• Ensure sufficient free memory")

if __name__ == "__main__":
    main()