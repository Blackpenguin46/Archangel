#!/usr/bin/env python3
"""
Archangel Red Team Agent - Standalone Entry Point
Autonomous penetration testing AI agent
"""

import asyncio
import argparse
import uuid
import sys
from pathlib import Path

# Add archangel package to path
sys.path.insert(0, str(Path(__file__).parent))

from archangel.agents.red import RedTeamAgent

class RedTeamController:
    """Controller for standalone red team agent"""
    
    def __init__(self, container_mode: bool = False, model_path: str = None):
        self.container_mode = container_mode
        self.model_path = model_path
        self.agent = None
        self.session_id = str(uuid.uuid4())
        
    async def run_autonomous_agent(self, duration_minutes: int = 30, cycle_interval: int = 10):
        """Run autonomous red team agent"""
        print(f"🚀 Starting Archangel Red Team AI Agent")
        print(f"=" * 50)
        print(f"Session ID: {self.session_id}")
        print(f"Container Mode: {self.container_mode}")
        print(f"Duration: {duration_minutes} minutes")
        print(f"Cycle Interval: {cycle_interval} seconds")
        print(f"Model Path: {self.model_path or 'Using intelligent fallback'}")
        print(f"=" * 50)
        
        # Initialize agent
        self.agent = RedTeamAgent(
            container_mode=self.container_mode,
            model_path=self.model_path
        )
        
        await self.agent.initialize(self.session_id)
        
        # Run autonomous cycles
        cycle_count = 0
        max_cycles = (duration_minutes * 60) // cycle_interval
        
        try:
            while cycle_count < max_cycles:
                cycle_count += 1
                print(f"\\n🔴 Red Team Cycle #{cycle_count}")
                print(f"-" * 30)
                
                # Execute autonomous operation
                result = await self.agent.autonomous_operation_cycle()
                
                # Display results
                await self._display_cycle_results(result)
                
                # Get agent status
                status = await self.agent.get_status()
                await self._display_agent_status(status)
                
                # Wait for next cycle
                if cycle_count < max_cycles:
                    print(f"⏳ Waiting {cycle_interval} seconds for next cycle...")
                    await asyncio.sleep(cycle_interval)
                    
        except KeyboardInterrupt:
            print(f"\\n🛑 Red Team Agent stopped by user")
        except Exception as e:
            print(f"\\n❌ Red Team Agent error: {e}")
        
        # Final status
        print(f"\\n🏁 Red Team Agent Session Complete")
        if self.agent:
            final_status = await self.agent.get_status()
            await self._display_final_summary(final_status)
    
    async def _display_cycle_results(self, result: dict):
        """Display results from operation cycle"""
        if 'error' in result:
            print(f"❌ Cycle Error: {result['error']}")
            return
            
        decision = result.get('decision', {})
        execution_result = result.get('result', {})
        
        # Display decision
        action_type = decision.get('action_type', 'unknown')
        target = decision.get('target', 'unknown')
        reasoning = decision.get('reasoning', 'No reasoning provided')
        confidence = decision.get('confidence', 0.0)
        
        print(f"🧠 AI Decision:")
        print(f"   Action: {action_type}")
        print(f"   Target: {target}")
        print(f"   Confidence: {confidence:.2f}")
        print(f"   Reasoning: {reasoning}")
        
        # Display execution results
        success = execution_result.get('success', False)
        print(f"\\n⚡ Execution:")
        print(f"   Success: {'✅' if success else '❌'}")
        
        if success:
            if 'hosts_discovered' in execution_result:
                hosts = execution_result['hosts_discovered']
                print(f"   🎯 Hosts Discovered: {hosts}")
            elif 'open_ports' in execution_result:
                ports = execution_result['open_ports']
                print(f"   🔓 Open Ports: {ports}")
            elif 'services' in execution_result:
                services = execution_result['services']
                print(f"   🔍 Services Found: {len(services)}")
            elif 'vulnerabilities' in execution_result:
                vulns = execution_result['vulnerabilities']
                print(f"   🚨 Vulnerabilities: {len(vulns)}")
            elif 'compromised' in execution_result:
                compromised = execution_result['compromised']
                print(f"   💀 System Compromised: {'Yes' if compromised else 'No'}")
        else:
            error = execution_result.get('error', 'Unknown error')
            print(f"   Error: {error}")
    
    async def _display_agent_status(self, status: dict):
        """Display current agent status"""
        phase = status.get('current_phase', 'unknown')
        discovered = len(status.get('discovered_hosts', []))
        compromised = len(status.get('compromised_hosts', []))
        actions = status.get('actions_taken', 0)
        
        print(f"\\n📊 Agent Status:")
        print(f"   Phase: {phase}")
        print(f"   Hosts Discovered: {discovered}")
        print(f"   Hosts Compromised: {compromised}")
        print(f"   Total Actions: {actions}")
    
    async def _display_final_summary(self, status: dict):
        """Display final operation summary"""
        print(f"=" * 50)
        print(f"📈 FINAL RED TEAM SUMMARY")
        print(f"=" * 50)
        
        discovered_hosts = status.get('discovered_hosts', [])
        compromised_hosts = status.get('compromised_hosts', [])
        total_actions = status.get('actions_taken', 0)
        final_phase = status.get('current_phase', 'unknown')
        
        print(f"Operation Phase: {final_phase}")
        print(f"Total Actions Executed: {total_actions}")
        print(f"Hosts Discovered: {len(discovered_hosts)}")
        for host in discovered_hosts:
            print(f"  • {host}")
        
        print(f"Hosts Compromised: {len(compromised_hosts)}")
        for host in compromised_hosts:
            print(f"  • {host}")
        
        # Calculate success metrics
        discovery_rate = len(discovered_hosts) / max(total_actions, 1)
        compromise_rate = len(compromised_hosts) / max(len(discovered_hosts), 1) if discovered_hosts else 0
        
        print(f"\\nSuccess Metrics:")
        print(f"  Discovery Rate: {discovery_rate:.2f} hosts per action")
        print(f"  Compromise Rate: {compromise_rate:.2f} ({len(compromised_hosts)}/{len(discovered_hosts)})")
        
        print(f"\\nSession ID: {self.session_id}")
        print(f"Logs available in: logs/")

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Archangel Red Team AI Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run in container mode for 30 minutes
  python run_red_agent.py --container --duration 30
  
  # Run with local LLM model
  python run_red_agent.py --model ./models/llama-7b.gguf
  
  # Quick 10 minute test
  python run_red_agent.py --duration 10 --interval 5
        """
    )
    
    parser.add_argument(
        '--container',
        action='store_true',
        help='Run in container mode (uses real tools like nmap)'
    )
    
    parser.add_argument(
        '--duration',
        type=int,
        default=30,
        help='Duration to run in minutes (default: 30)'
    )
    
    parser.add_argument(
        '--interval',
        type=int,
        default=10,
        help='Cycle interval in seconds (default: 10)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        help='Path to local LLM model (GGUF format)'
    )
    
    args = parser.parse_args()
    
    # Validate model path if provided
    if args.model and not Path(args.model).exists():
        print(f"❌ Model file not found: {args.model}")
        sys.exit(1)
    
    # Create and run controller
    controller = RedTeamController(
        container_mode=args.container,
        model_path=args.model
    )
    
    try:
        await controller.run_autonomous_agent(
            duration_minutes=args.duration,
            cycle_interval=args.interval
        )
    except KeyboardInterrupt:
        print(f"\\n🛑 Stopped by user")
    except Exception as e:
        print(f"\\n❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())