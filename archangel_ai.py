#!/usr/bin/env python3
"""
Archangel Linux - Real AI Security Expert System
Uses Hugging Face models and SmolAgents for actual AI reasoning
"""

import asyncio
import sys
import os
from core.real_ai_security_expert import RealSecurityExpertAI
from core.kernel_interface import create_kernel_interface
from tools.tool_integration import create_ai_orchestrator

class ArchangelAI:
    """Real Archangel AI Security Expert System"""
    
    def __init__(self, hf_token: str = None):
        self.hf_token = hf_token
        self.ai_expert = None
        self.kernel_interface = None
        self.tool_orchestrator = None
    
    async def initialize(self):
        """Initialize the real AI system"""
        print("🛡️ ARCHANGEL LINUX - Real AI Security Expert System")
        print("=" * 60)
        print("🤖 Initializing with Hugging Face AI models...")
        
        # Initialize Real AI Security Expert
        self.ai_expert = RealSecurityExpertAI(hf_token=self.hf_token)
        await self.ai_expert.initialize()
        
        # Initialize kernel interface
        print("⚡ Connecting to kernel interface...")
        self.kernel_interface = create_kernel_interface()
        self.kernel_interface.init_communication()
        self.ai_expert.set_kernel_interface(self.kernel_interface)
        
        # Initialize tool orchestrator
        print("🛠️ Setting up AI tool orchestration...")
        self.tool_orchestrator = create_ai_orchestrator()
        self.ai_expert.set_tool_orchestrator(self.tool_orchestrator)
        
        print("\n✅ REAL AI SECURITY EXPERT READY!")
        print("🧠 Using actual Hugging Face models for reasoning")
        print("🔧 SmolAgents enabled for autonomous operations")
        print("⚡ Kernel integration active")
        print()
    
    async def analyze_target(self, target: str):
        """Perform real AI security analysis"""
        print(f"🎯 REAL AI ANALYSIS: {target}")
        print("=" * 60)
        print("🧠 AI is thinking with actual neural networks...")
        print()
        
        # Real AI analysis
        analysis = await self.ai_expert.analyze_target(target)
        
        # Display results
        print("📊 REAL AI ANALYSIS COMPLETE")
        print("=" * 40)
        print(f"🎯 Target: {analysis.target}")
        print(f"📝 Type: {analysis.target_type}")
        print(f"📊 AI Confidence: {analysis.confidence.value}")
        print(f"⚠️ Threat Level: {analysis.threat_level.value}")
        
        print(f"\n🧠 REAL AI REASONING:")
        print("-" * 40)
        print(analysis.reasoning)
        
        print(f"\n📋 AI RECOMMENDATIONS:")
        print("-" * 30)
        for i, rec in enumerate(analysis.recommendations, 1):
            print(f"{i}. {rec}")
        
        print(f"\n🎯 AI NEXT ACTIONS:")
        print("-" * 25)
        for i, action in enumerate(analysis.next_actions, 1):
            print(f"{i}. {action}")
        
        # Offer autonomous tool execution
        print(f"\n🤖 Execute with AI autonomous agents? (y/n): ", end="")
        try:
            choice = input().strip().lower()
            if choice == 'y':
                print("\n🚀 Launching AI autonomous execution...")
                print("(Using SmolAgents + Hugging Face models)")
                
                results = await self.ai_expert.execute_strategy_with_tools(target, analysis.strategy)
                
                print(f"\n📊 AI AUTONOMOUS EXECUTION RESULTS:")
                print("=" * 45)
                for i, result in enumerate(results, 1):
                    print(f"\n--- AI Tool {i}: {result.get('tool', 'autonomous_agent')} ---")
                    if 'command' in result:
                        print(f"Command: {result['command']}")
                        print(f"Success: {result['success']}")
                        print(f"Time: {result['execution_time']}")
                    if 'findings' in result and result['findings']:
                        print(f"AI Findings: {result['findings']}")
                    if 'smolagent_result' in result:
                        print(f"SmolAgent Result: {result['smolagent_result']}")
        except (EOFError, KeyboardInterrupt):
            print("\nSkipping autonomous execution.")
        
        return analysis
    
    async def explain_ai_reasoning(self):
        """Get the AI to explain its reasoning process"""
        print("\n🧠 REAL AI REASONING EXPLANATION")
        print("=" * 40)
        explanation = await self.ai_expert.explain_reasoning()
        print(explanation)
    
    async def kernel_ai_integration_demo(self):
        """Demonstrate real AI-kernel integration"""
        print("\n⚡ REAL AI-KERNEL INTEGRATION DEMO")
        print("=" * 40)
        
        # Create test security context
        from core.kernel_interface import SecurityContext
        
        test_context = SecurityContext(
            pid=9999,
            uid=0,  # root
            syscall_nr=59,  # execve
            timestamp=12345678,
            flags=0x0001,
            comm="suspicious_binary"
        )
        
        print("🔍 Kernel sending security context to AI...")
        print(f"  Process: {test_context.comm} (PID {test_context.pid})")
        print(f"  User: {'root' if test_context.uid == 0 else 'user'}")
        print(f"  Syscall: {test_context.syscall_nr}")
        
        print("\n🧠 Real AI analyzing with neural networks...")
        decision = await self.ai_expert.handle_kernel_analysis_request(test_context)
        
        print(f"🎯 Real AI Decision: {decision}")
        print("(This was generated by actual AI reasoning, not rules!)")
    
    async def interactive_session(self):
        """Interactive session with real AI"""
        print("\n🎮 REAL AI INTERACTIVE SESSION")
        print("=" * 40)
        print("Commands:")
        print("  analyze <target>  - Real AI target analysis")
        print("  explain          - AI explains its reasoning")
        print("  kernel           - Demo AI-kernel integration")
        print("  quit             - Exit")
        print()
        
        while True:
            try:
                print("RealAI> ", end="")
                command = input().strip()
                
                if command.lower() in ['quit', 'exit', 'q']:
                    break
                elif command.lower().startswith('analyze '):
                    target = command[8:].strip()
                    if target:
                        await self.analyze_target(target)
                    else:
                        print("❌ Please specify a target")
                elif command.lower() == 'explain':
                    await self.explain_ai_reasoning()
                elif command.lower() == 'kernel':
                    await self.kernel_ai_integration_demo()
                elif command.strip() == '':
                    continue
                else:
                    print(f"❌ Unknown command: {command}")
                
                print()
                
            except (EOFError, KeyboardInterrupt):
                break
        
        print("\n👋 Real AI Security Expert session ended.")

def get_hf_token():
    """Get Hugging Face token from user or environment"""
    # Check environment variable first
    token = os.getenv('HUGGINGFACE_TOKEN') or os.getenv('HF_TOKEN')
    
    if not token:
        print("🔑 Hugging Face Token Setup")
        print("=" * 30)
        print("For best results, provide your Hugging Face token to access:")
        print("• Foundation-Sec-8B (Cisco's cybersecurity model)")
        print("• Advanced security-focused models")
        print("• Private models and datasets")
        print()
        print("You can:")
        print("1. Enter token now")
        print("2. Press Enter to use public models only")
        print("3. Set HF_TOKEN environment variable")
        print()
        
        try:
            token = input("Enter HF token (or press Enter): ").strip()
        except (EOFError, KeyboardInterrupt):
            token = ""
    
    return token if token else None

async def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("🛡️ ARCHANGEL LINUX - Real AI Security Expert")
        print("=" * 50)
        print("Usage:")
        print(f"  {sys.argv[0]} analyze <target>     - AI analyzes target")
        print(f"  {sys.argv[0]} interactive         - Interactive AI session")
        print(f"  {sys.argv[0]} kernel              - Demo AI-kernel integration")
        print()
        print("Examples:")
        print(f"  {sys.argv[0]} analyze google.com")
        print(f"  {sys.argv[0]} analyze 192.168.1.1")
        print(f"  {sys.argv[0]} interactive")
        print()
        print("🤖 This uses REAL AI models from Hugging Face!")
        return
    
    # Get Hugging Face token
    hf_token = get_hf_token()
    
    # Initialize real AI system
    archangel = ArchangelAI(hf_token=hf_token)
    await archangel.initialize()
    
    command = sys.argv[1].lower()
    
    if command == 'analyze' and len(sys.argv) >= 3:
        target = sys.argv[2]
        await archangel.analyze_target(target)
    elif command == 'interactive':
        await archangel.interactive_session()
    elif command == 'kernel':
        await archangel.kernel_ai_integration_demo()
    else:
        print("❌ Invalid command")

if __name__ == "__main__":
    asyncio.run(main())