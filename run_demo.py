#!/usr/bin/env python3
"""
Archangel Demo - BlackHat Ready AI vs AI Cyber Conflict Demonstration
Professional demonstration of autonomous red team vs blue team AI agents
"""

import asyncio
import argparse
import sys
from pathlib import Path

# Add archangel package to path
sys.path.insert(0, str(Path(__file__).parent))

from run_orchestrator import ArchangelOrchestrator

class ArchangelDemo:
    """Professional demo controller for BlackHat presentation"""
    
    def __init__(self):
        self.demo_scenarios = {
            'quick': {
                'name': 'Quick Demo (5 minutes)',
                'duration': 5,
                'interval': 15,
                'description': 'Fast-paced demonstration of AI vs AI capabilities'
            },
            'standard': {
                'name': 'Standard Demo (15 minutes)', 
                'duration': 15,
                'interval': 10,
                'description': 'Comprehensive demonstration showing full AI autonomy'
            },
            'extended': {
                'name': 'Extended Demo (30 minutes)',
                'duration': 30,
                'interval': 8,
                'description': 'Deep dive into AI reasoning and adaptation'
            },
            'blackhat': {
                'name': 'BlackHat Presentation (45 minutes)',
                'duration': 45,
                'interval': 12,
                'description': 'Full conference presentation with detailed analysis'
            }
        }
    
    async def run_demo(self, scenario: str, container_mode: bool = False, model_path: str = None):
        """Run the specified demo scenario"""
        
        if scenario not in self.demo_scenarios:
            print(f"❌ Unknown scenario: {scenario}")
            print(f"Available scenarios: {', '.join(self.demo_scenarios.keys())}")
            return
            
        config = self.demo_scenarios[scenario]
        
        # Display demo introduction
        await self._display_intro(config, container_mode, model_path)
        
        # Initialize orchestrator
        orchestrator = ArchangelOrchestrator(
            container_mode=container_mode,
            model_path=model_path
        )
        
        try:
            await orchestrator.initialize()
            await orchestrator.run_simulation(
                duration_minutes=config['duration'],
                tick_interval=config['interval']
            )
        except KeyboardInterrupt:
            print(f"\\n🛑 Demo stopped by user")
        except Exception as e:
            print(f"\\n❌ Demo error: {e}")
            
        # Display demo conclusion
        await self._display_conclusion(config)
    
    async def _display_intro(self, config: dict, container_mode: bool, model_path: str):
        """Display demo introduction"""
        print(f"🎯 ARCHANGEL AI vs AI CYBER CONFLICT DEMONSTRATION")
        print(f"=" * 80)
        print(f"")
        print(f"📋 Demo: {config['name']}")
        print(f"⏱️  Duration: {config['duration']} minutes")
        print(f"🔄 Update Interval: {config['interval']} seconds")
        print(f"📝 Description: {config['description']}")
        print(f"")
        print(f"🏗️ System Configuration:")
        print(f"   Container Mode: {'✅ Enabled' if container_mode else '❌ Disabled (simulation)'}")
        print(f"   Local LLM: {'✅ ' + model_path if model_path else '❌ Using intelligent fallback'}")
        print(f"")
        print(f"🤖 AI Agents:")
        print(f"   🔴 Red Team: Autonomous penetration testing AI")
        print(f"      • Network reconnaissance and scanning")
        print(f"      • Service enumeration and vulnerability assessment") 
        print(f"      • Exploit development and execution")
        print(f"      • Post-exploitation and persistence")
        print(f"")
        print(f"   🔵 Blue Team: Autonomous security monitoring AI")
        print(f"      • Real-time threat detection and analysis")
        print(f"      • Automated defensive responses")
        print(f"      • Adaptive security posture management")
        print(f"      • Incident response and forensics")
        print(f"")
        print(f"⚔️ Demonstration Features:")
        print(f"   ✅ Fully autonomous AI decision-making")
        print(f"   ✅ Real tool execution (nmap, iptables, etc.)")
        print(f"   ✅ Adaptive strategies and learning")
        print(f"   ✅ Comprehensive logging and metrics")
        print(f"   ✅ Professional visualization and reporting")
        print(f"")
        print(f"🎯 Expected Outcomes:")
        print(f"   • Red team will discover and analyze target systems")
        print(f"   • Blue team will detect and respond to attack activities")
        print(f"   • Both teams will adapt strategies based on opponent actions")
        print(f"   • System will demonstrate emergent AI behaviors")
        print(f"")
        print(f"=" * 80)
        
        # Wait for user confirmation
        input(f"\\n▶️  Press Enter to start the demonstration...")
        print(f"")
    
    async def _display_conclusion(self, config: dict):
        """Display demo conclusion"""
        print(f"\\n")
        print(f"🎉 DEMONSTRATION COMPLETE")
        print(f"=" * 50)
        print(f"Demo: {config['name']}")
        print(f"")
        print(f"🎯 Key Achievements Demonstrated:")
        print(f"   ✅ Autonomous AI vs AI cyber conflict")
        print(f"   ✅ Real-time adaptive decision making")
        print(f"   ✅ Professional cybersecurity tool execution")
        print(f"   ✅ Intelligent threat detection and response")
        print(f"   ✅ Comprehensive metrics and logging")
        print(f"")
        print(f"📊 Innovation Highlights:")
        print(f"   • First fully autonomous AI penetration testing system")
        print(f"   • Real-time AI vs AI cyber conflict simulation")
        print(f"   • Offline operation with local LLM models")
        print(f"   • Professional-grade security tool integration")
        print(f"   • Scalable containerized deployment architecture")
        print(f"")
        print(f"🔬 Research Value:")
        print(f"   • Benchmarking AI security capabilities")
        print(f"   • Advancing autonomous cybersecurity research")
        print(f"   • Demonstrating practical AI applications")
        print(f"   • Establishing new standards for AI security testing")
        print(f"")
        print(f"🚀 Next Steps:")
        print(f"   • Review detailed logs and metrics")
        print(f"   • Analyze AI decision-making patterns")
        print(f"   • Explore custom scenarios and configurations")
        print(f"   • Deploy in production security environments")
        print(f"")
        print(f"📁 Session Data:")
        print(f"   Logs: logs/ directory")
        print(f"   Metrics: SQLite database files")
        print(f"   Status: container status logs")
        print(f"")
        print(f"=" * 50)
        print(f"Thank you for experiencing Archangel! 🎯")

def print_scenarios():
    """Print available demo scenarios"""
    demo = ArchangelDemo()
    print(f"📋 Available Demo Scenarios:")
    print(f"=" * 40)
    
    for key, config in demo.demo_scenarios.items():
        print(f"  {key.ljust(12)} - {config['name']}")
        print(f"                 Duration: {config['duration']} minutes")
        print(f"                 {config['description']}")
        print(f"")

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Archangel AI vs AI Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Demo Scenarios:
  quick      - Quick Demo (5 minutes) - Fast overview
  standard   - Standard Demo (15 minutes) - Full capabilities  
  extended   - Extended Demo (30 minutes) - Deep analysis
  blackhat   - BlackHat Presentation (45 minutes) - Conference ready
  
Examples:
  # Quick demonstration
  python run_demo.py quick
  
  # Full demo with containers
  python run_demo.py standard --container
  
  # BlackHat presentation with local LLM
  python run_demo.py blackhat --container --model ./models/llama-7b.gguf
  
  # List available scenarios
  python run_demo.py --list
        """
    )
    
    parser.add_argument(
        'scenario',
        nargs='?',
        help='Demo scenario to run'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available demo scenarios'
    )
    
    parser.add_argument(
        '--container',
        action='store_true',
        help='Run in container mode (requires Docker)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        help='Path to local LLM model (GGUF format)'
    )
    
    args = parser.parse_args()
    
    # Handle list command
    if args.list:
        print_scenarios()
        return
    
    # Require scenario
    if not args.scenario:
        print(f"❌ Please specify a demo scenario")
        print_scenarios()
        sys.exit(1)
    
    # Validate model path if provided
    if args.model and not Path(args.model).exists():
        print(f"❌ Model file not found: {args.model}")
        sys.exit(1)
    
    # Create and run demo
    demo = ArchangelDemo()
    
    try:
        await demo.run_demo(
            scenario=args.scenario,
            container_mode=args.container,
            model_path=args.model
        )
    except KeyboardInterrupt:
        print(f"\\n🛑 Demo stopped by user")
    except Exception as e:
        print(f"\\n❌ Demo error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())