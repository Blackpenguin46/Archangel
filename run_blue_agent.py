#!/usr/bin/env python3
"""
Archangel Blue Team Agent - Standalone Entry Point
Autonomous security monitoring and defense AI agent
"""

import asyncio
import argparse
import uuid
import sys
from pathlib import Path

# Add archangel package to path
sys.path.insert(0, str(Path(__file__).parent))

from archangel.agents.blue import BlueTeamAgent

class BlueTeamController:
    """Controller for standalone blue team agent"""
    
    def __init__(self, container_mode: bool = False, model_path: str = None):
        self.container_mode = container_mode
        self.model_path = model_path
        self.agent = None
        self.session_id = str(uuid.uuid4())
        
    async def run_autonomous_agent(self, duration_minutes: int = 30, cycle_interval: int = 8):
        """Run autonomous blue team agent"""
        print(f"🚀 Starting Archangel Blue Team AI Agent")
        print(f"=" * 50)
        print(f"Session ID: {self.session_id}")
        print(f"Container Mode: {self.container_mode}")
        print(f"Duration: {duration_minutes} minutes")
        print(f"Cycle Interval: {cycle_interval} seconds")
        print(f"Model Path: {self.model_path or 'Using intelligent fallback'}")
        print(f"=" * 50)
        
        # Initialize agent
        self.agent = BlueTeamAgent(
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
                print(f"\\n🔵 Blue Team Cycle #{cycle_count}")
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
            print(f"\\n🛑 Blue Team Agent stopped by user")
        except Exception as e:
            print(f"\\n❌ Blue Team Agent error: {e}")
        
        # Final status
        print(f"\\n🏁 Blue Team Agent Session Complete")
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
        threat_analysis = result.get('threat_analysis', {})
        
        # Display threat analysis
        new_threats = threat_analysis.get('new_threats', 0)
        threat_level = threat_analysis.get('threat_level', 'low')
        total_threats = threat_analysis.get('total_threats', 0)
        
        print(f"🔍 Threat Analysis:")
        print(f"   New Threats: {new_threats}")
        print(f"   Threat Level: {threat_level.upper()}")
        print(f"   Total Threats: {total_threats}")
        
        if new_threats > 0:
            categories = threat_analysis.get('threat_categories', {})
            for category, threats in categories.items():
                print(f"   • {category}: {len(threats)} threats")
        
        # Display decision
        action_type = decision.get('action_type', 'unknown')
        target = decision.get('target', 'unknown')
        reasoning = decision.get('reasoning', 'No reasoning provided')
        confidence = decision.get('confidence', 0.0)
        
        print(f"\\n🧠 AI Decision:")
        print(f"   Action: {action_type}")
        print(f"   Target: {target}")
        print(f"   Confidence: {confidence:.2f}")
        print(f"   Reasoning: {reasoning}")
        
        # Display execution results
        success = execution_result.get('success', False)
        print(f"\\n⚡ Execution:")
        print(f"   Success: {'✅' if success else '❌'}")
        
        if success:
            action_taken = execution_result.get('action', 'unknown')
            print(f"   Action Taken: {action_taken}")
            
            if 'blocked_ips_total' in execution_result:
                blocked_total = execution_result['blocked_ips_total']
                print(f"   🚫 Total IPs Blocked: {blocked_total}")
            elif 'new_posture' in execution_result:
                posture = execution_result['new_posture']
                print(f"   🛡️ Defense Posture: {posture}")
            elif 'stats' in execution_result:
                stats = execution_result['stats']
                print(f"   📊 Monitoring Stats: {stats}")
        else:
            error = execution_result.get('error', 'Unknown error')
            print(f"   Error: {error}")
    
    async def _display_agent_status(self, status: dict):
        """Display current agent status"""
        posture = status.get('defense_posture', 'unknown')
        threats = status.get('detected_threats', 0)
        blocked = status.get('blocked_ips', 0)
        connections = status.get('active_connections', 0)
        actions = status.get('actions_taken', 0)
        
        print(f"\\n📊 Agent Status:")
        print(f"   Defense Posture: {posture.upper()}")
        print(f"   Threats Detected: {threats}")
        print(f"   IPs Blocked: {blocked}")
        print(f"   Active Connections: {connections}")
        print(f"   Total Actions: {actions}")
    
    async def _display_final_summary(self, status: dict):
        """Display final operation summary"""
        print(f"=" * 50)
        print(f"📈 FINAL BLUE TEAM SUMMARY")
        print(f"=" * 50)
        
        final_posture = status.get('defense_posture', 'unknown')
        total_threats = status.get('detected_threats', 0)
        total_blocked = status.get('blocked_ips', 0)
        total_connections = status.get('active_connections', 0)
        total_actions = status.get('actions_taken', 0)
        
        print(f"Final Defense Posture: {final_posture.upper()}")
        print(f"Total Actions Executed: {total_actions}")
        print(f"Threats Detected: {total_threats}")
        print(f"IPs Blocked: {total_blocked}")
        print(f"Active Connections Monitored: {total_connections}")
        
        # Calculate defense metrics
        if total_actions > 0:
            threat_detection_rate = total_threats / total_actions
            response_rate = total_blocked / max(total_threats, 1)
            
            print(f"\\nDefense Metrics:")
            print(f"  Threat Detection Rate: {threat_detection_rate:.2f} threats per action")
            print(f"  Response Rate: {response_rate:.2f} ({total_blocked}/{total_threats})")
            
            # Determine effectiveness
            if response_rate > 0.8:
                effectiveness = "EXCELLENT"
            elif response_rate > 0.6:
                effectiveness = "GOOD"
            elif response_rate > 0.4:
                effectiveness = "MODERATE"
            else:
                effectiveness = "NEEDS IMPROVEMENT"
            
            print(f"  Overall Effectiveness: {effectiveness}")
        
        print(f"\\nSession ID: {self.session_id}")
        print(f"Logs available in: logs/")

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Archangel Blue Team AI Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run in container mode for 30 minutes
  python run_blue_agent.py --container --duration 30
  
  # Run with local LLM model
  python run_blue_agent.py --model ./models/llama-7b.gguf
  
  # Quick 10 minute monitoring test
  python run_blue_agent.py --duration 10 --interval 5
        """
    )
    
    parser.add_argument(
        '--container',
        action='store_true',
        help='Run in container mode (uses real tools like iptables)'
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
        default=8,
        help='Cycle interval in seconds (default: 8)'
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
    controller = BlueTeamController(
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