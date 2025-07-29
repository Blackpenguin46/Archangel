#!/usr/bin/env python3
"""
Archangel Linux - Real AI Security Expert
Direct interface to the AI security expert for actual security work
"""

import asyncio
import sys
from core.ai_security_expert import SecurityExpertAI
from core.kernel_interface import create_kernel_interface
from tools.tool_integration import create_ai_orchestrator

class ArchangelReal:
    """Real Archangel AI Security Expert for actual security work"""
    
    def __init__(self):
        self.ai_expert = None
        self.kernel_interface = None
        self.tool_orchestrator = None
    
    async def initialize(self):
        """Initialize the real Archangel system"""
        print("🛡️ Initializing Archangel AI Security Expert...")
        
        # Initialize AI Security Expert
        self.ai_expert = SecurityExpertAI()
        
        # Initialize kernel interface
        self.kernel_interface = create_kernel_interface()
        self.kernel_interface.init_communication()
        self.ai_expert.set_kernel_interface(self.kernel_interface)
        
        # Initialize tool orchestrator
        self.tool_orchestrator = create_ai_orchestrator()
        self.ai_expert.set_tool_orchestrator(self.tool_orchestrator)
        
        print("✅ Archangel AI Security Expert ready for real security work!")
        print()
    
    async def analyze_target(self, target: str):
        """Perform real security analysis on a target"""
        print(f"🎯 Starting real security analysis of: {target}")
        print("=" * 60)
        
        # AI performs full security analysis
        analysis = await self.ai_expert.analyze_target(target)
        
        # Display results
        print(f"\n📊 ANALYSIS COMPLETE")
        print(f"Target: {analysis.target}")
        print(f"Type: {analysis.target_type}")
        print(f"Confidence: {analysis.confidence.value}")
        print(f"Threat Level: {analysis.threat_level.value}")
        
        print(f"\n🧠 AI REASONING:")
        print(analysis.reasoning)
        
        print(f"\n📋 RECOMMENDATIONS:")
        for i, rec in enumerate(analysis.recommendations, 1):
            print(f"{i}. {rec}")
        
        print(f"\n🎯 NEXT ACTIONS:")
        for i, action in enumerate(analysis.next_actions, 1):
            print(f"{i}. {action}")
        
        # Execute strategy with tools if requested
        print(f"\n🛠️ Execute strategy with tools? (y/n): ", end="")
        try:
            choice = input().strip().lower()
            if choice == 'y':
                print("\n🤖 Executing strategy with AI-driven tools...")
                results = await self.ai_expert.execute_strategy_with_tools(target, analysis.strategy)
                
                print(f"\n📊 TOOL EXECUTION RESULTS:")
                for i, result in enumerate(results, 1):
                    print(f"\n--- Tool {i}: {result['tool']} ---")
                    print(f"Command: {result['command']}")
                    print(f"Success: {result['success']}")
                    print(f"Time: {result['execution_time']}")
                    if result['findings']:
                        print(f"Findings: {result['findings']}")
        except (EOFError, KeyboardInterrupt):
            print("\nSkipping tool execution.")
        
        return analysis
    
    async def explain_reasoning(self):
        """Get AI to explain its reasoning process"""
        explanation = await self.ai_expert.explain_reasoning()
        print("\n🧠 AI REASONING EXPLANATION:")
        print("=" * 60)
        print(explanation)
    
    async def interactive_session(self):
        """Run interactive security analysis session"""
        print("\n🎮 ARCHANGEL INTERACTIVE SECURITY SESSION")
        print("=" * 60)
        print("Commands:")
        print("  analyze <target>  - Analyze a security target")
        print("  explain          - Explain AI reasoning")
        print("  stats            - Show kernel statistics")
        print("  quit             - Exit")
        print()
        
        while True:
            try:
                print("Archangel> ", end="")
                command = input().strip()
                
                if command.lower() in ['quit', 'exit', 'q']:
                    break
                elif command.lower().startswith('analyze '):
                    target = command[8:].strip()
                    if target:
                        await self.analyze_target(target)
                    else:
                        print("❌ Please specify a target to analyze")
                elif command.lower() == 'explain':
                    await self.explain_reasoning()
                elif command.lower() == 'stats':
                    stats = self.kernel_interface.get_kernel_stats()
                    print(f"\n📊 KERNEL STATISTICS:")
                    print(f"Total Decisions: {stats.total_decisions}")
                    print(f"Allow/Deny/Monitor: {stats.allow_decisions}/{stats.deny_decisions}/{stats.monitor_decisions}")
                    print(f"AI Deferrals: {stats.deferred_decisions}")
                    print(f"Uptime: {stats.uptime_seconds} seconds")
                elif command.strip() == '':
                    continue
                else:
                    print(f"❌ Unknown command: {command}")
                    print("Try: analyze <target>, explain, stats, or quit")
                
                print()
                
            except (EOFError, KeyboardInterrupt):
                break
        
        print("\n👋 Archangel AI Security Expert session ended.")

async def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("🛡️ ARCHANGEL LINUX - AI Security Expert")
        print("Usage:")
        print(f"  {sys.argv[0]} analyze <target>     - Analyze a target")
        print(f"  {sys.argv[0]} interactive         - Interactive session")
        print()
        print("Examples:")
        print(f"  {sys.argv[0]} analyze example.com")
        print(f"  {sys.argv[0]} analyze 192.168.1.1")
        print(f"  {sys.argv[0]} interactive")
        return
    
    archangel = ArchangelReal()
    await archangel.initialize()
    
    command = sys.argv[1].lower()
    
    if command == 'analyze' and len(sys.argv) >= 3:
        target = sys.argv[2]
        await archangel.analyze_target(target)
    elif command == 'interactive':
        await archangel.interactive_session()
    else:
        print("❌ Invalid command. Use 'analyze <target>' or 'interactive'")

if __name__ == "__main__":
    asyncio.run(main())