#!/usr/bin/env python3
"""
Archangel Linux - Comprehensive Test Script
Tests all major components of the system
"""

import sys
import os
import importlib
import traceback
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

class ArchangelTester:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.results = []
    
    def test(self, name, test_func):
        """Run a test and record results"""
        print(f"\n🧪 Testing: {name}")
        print("-" * 50)
        
        try:
            result = test_func()
            if result:
                print(f"✅ PASSED: {name}")
                self.passed += 1
                self.results.append((name, "PASSED", None))
            else:
                print(f"❌ FAILED: {name}")
                self.failed += 1
                self.results.append((name, "FAILED", "Test returned False"))
        except Exception as e:
            print(f"❌ ERROR: {name} - {str(e)}")
            self.failed += 1
            self.results.append((name, "ERROR", str(e)))
    
    def test_imports(self):
        """Test if core modules can be imported"""
        modules_to_test = [
            'core.ai_security_expert',
            'core.real_ai_security_expert', 
            'core.conversational_ai',
            'core.unified_ai_orchestrator',
            'tools.tool_integration',
            'ui.security_chat_interface'
        ]
        
        success_count = 0
        for module in modules_to_test:
            try:
                importlib.import_module(module)
                print(f"  ✅ {module}")
                success_count += 1
            except Exception as e:
                print(f"  ❌ {module}: {e}")
        
        return success_count == len(modules_to_test)
    
    def test_ai_security_expert(self):
        """Test AI Security Expert functionality"""
        try:
            from core.ai_security_expert import AISecurityExpert
            expert = AISecurityExpert()
            
            # Test basic analysis
            result = expert.analyze_target("example.com")
            print(f"  📊 Analysis result length: {len(result)} characters")
            
            # Test interactive mode setup
            expert.setup_interactive_mode()
            print("  ✅ Interactive mode setup successful")
            
            return len(result) > 100  # Should return substantial analysis
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return False
    
    def test_real_ai_expert(self):
        """Test Real AI Security Expert"""
        try:
            from core.real_ai_security_expert import RealAISecurityExpert
            expert = RealAISecurityExpert()
            
            # Test analysis
            result = expert.analyze_target("google.com")
            print(f"  📊 Real AI analysis length: {len(result)} characters")
            
            # Test reasoning
            reasoning = expert.explain_reasoning("web_application", "reconnaissance")
            print(f"  🧠 Reasoning length: {len(reasoning)} characters")
            
            return len(result) > 50 and len(reasoning) > 50
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return False
    
    def test_conversational_ai(self):
        """Test Conversational AI"""
        try:
            from core.conversational_ai import ConversationalAI
            ai = ConversationalAI()
            
            # Test chat functionality
            response = ai.chat("What is SQL injection?")
            print(f"  💬 Chat response length: {len(response)} characters")
            
            return len(response) > 20
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return False
    
    def test_tool_integration(self):
        """Test tool integration"""
        try:
            from tools.tool_integration import ToolIntegrator
            integrator = ToolIntegrator()
            
            # Test tool availability
            available_tools = integrator.get_available_tools()
            print(f"  🛠️ Available tools: {len(available_tools)}")
            
            # Test tool execution (safe command)
            result = integrator.execute_tool("echo", ["test"])
            print(f"  ⚡ Tool execution result: {result}")
            
            return len(available_tools) > 0
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return False
    
    def test_kernel_interface(self):
        """Test kernel interface (simulation mode)"""
        try:
            from core.kernel_interface import KernelInterface
            ki = KernelInterface()
            
            # Test simulation mode
            result = ki.simulate_security_event("test_event")
            print(f"  🔧 Kernel simulation result: {result}")
            
            # Test status
            status = ki.get_status()
            print(f"  📊 Kernel interface status: {status}")
            
            return result is not None
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return False
    
    def test_cli_functionality(self):
        """Test CLI components"""
        try:
            # Test if CLI script exists and is importable
            cli_path = Path("cli.py")
            if not cli_path.exists():
                print("  ❌ cli.py not found")
                return False
            
            print("  ✅ CLI script exists")
            
            # Test demo scripts
            demo_files = ["demo_archangel.py", "demo_unified_ai.py", "full_demo.py"]
            existing_demos = []
            
            for demo in demo_files:
                if Path(demo).exists():
                    existing_demos.append(demo)
                    print(f"  ✅ {demo} exists")
                else:
                    print(f"  ⚠️ {demo} not found")
            
            return len(existing_demos) > 0
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return False
    
    def test_docker_setup(self):
        """Test Docker configuration"""
        try:
            docker_files = ["Dockerfile", "docker-compose.yml", "docker-setup.sh"]
            existing_files = []
            
            for file in docker_files:
                if Path(file).exists():
                    existing_files.append(file)
                    print(f"  ✅ {file} exists")
                else:
                    print(f"  ❌ {file} missing")
            
            # Check if .env exists or can be created
            env_file = Path(".env")
            if env_file.exists():
                print("  ✅ .env file exists")
            else:
                print("  ⚠️ .env file not found (will be created by setup script)")
            
            return len(existing_files) >= 2  # At least Dockerfile and compose
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return False
    
    def run_all_tests(self):
        """Run all tests"""
        print("🛡️ ARCHANGEL LINUX - COMPREHENSIVE TEST SUITE")
        print("=" * 60)
        
        # Core functionality tests
        self.test("Module Imports", self.test_imports)
        self.test("AI Security Expert", self.test_ai_security_expert)
        self.test("Real AI Expert", self.test_real_ai_expert)
        self.test("Conversational AI", self.test_conversational_ai)
        self.test("Tool Integration", self.test_tool_integration)
        self.test("Kernel Interface", self.test_kernel_interface)
        self.test("CLI Functionality", self.test_cli_functionality)
        self.test("Docker Setup", self.test_docker_setup)
        
        # Print summary
        print("\n" + "=" * 60)
        print("🏁 TEST SUMMARY")
        print("=" * 60)
        
        total = self.passed + self.failed
        success_rate = (self.passed / total * 100) if total > 0 else 0
        
        print(f"✅ Passed: {self.passed}")
        print(f"❌ Failed: {self.failed}")
        print(f"📊 Success Rate: {success_rate:.1f}%")
        
        print("\n📋 DETAILED RESULTS:")
        for name, status, error in self.results:
            if status == "PASSED":
                print(f"  ✅ {name}")
            else:
                print(f"  ❌ {name}: {error}")
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        if self.failed == 0:
            print("  🎉 All tests passed! Your Archangel system is ready.")
            print("  🚀 Try running: python3 demo_archangel.py")
        else:
            print("  🔧 Some tests failed. Check the errors above.")
            print("  📦 Try installing missing dependencies: pip install -r requirements.txt")
            print("  🐳 Consider using Docker for a clean environment: ./docker-setup.sh")
        
        return self.failed == 0

def main():
    """Main test runner"""
    tester = ArchangelTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n🎯 NEXT STEPS:")
        print("  1. python3 demo_archangel.py          # Run basic demo")
        print("  2. python3 archangel_lightweight.py interactive  # Try interactive mode")
        print("  3. python3 cli.py                     # Use the CLI interface")
        print("  4. ./docker-setup.sh --start          # Test in Docker")
        
        sys.exit(0)
    else:
        print("\n🚨 Some tests failed. Please fix the issues above before proceeding.")
        sys.exit(1)

if __name__ == "__main__":
    main()