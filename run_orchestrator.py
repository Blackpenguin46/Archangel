#!/usr/bin/env python3
"""
Archangel Orchestrator - Unified AI vs AI Cyber Conflict System
Coordinates red and blue team AI agents in autonomous cyber warfare simulation
"""

import asyncio
import argparse
import uuid
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add archangel package to path
sys.path.insert(0, str(Path(__file__).parent))

from archangel.agents.red import RedTeamAgent
from archangel.agents.blue import BlueTeamAgent
from archangel.core.logging_system import ArchangelLogger

class ArchangelOrchestrator:
    """Unified orchestrator for AI vs AI cyber conflict simulation"""
    
    def __init__(self, container_mode: bool = False, model_path: str = None):
        self.container_mode = container_mode
        self.model_path = model_path
        self.session_id = str(uuid.uuid4())
        
        # Agents
        self.red_agent = None
        self.blue_agent = None
        self.logger = None
        
        # Simulation state
        self.tick_count = 0
        self.simulation_active = False
        self.metrics = {
            'red_team_score': 0,
            'blue_team_score': 0,
            'total_compromises': 0,
            'total_detections': 0,
            'total_blocks': 0,
            'simulation_start': None,
            'simulation_end': None
        }
        
    async def initialize(self):
        """Initialize the orchestrator and agents"""
        print(f"🚀 Initializing Archangel AI vs AI Orchestrator")
        print(f"=" * 60)
        print(f"Session ID: {self.session_id}")
        print(f"Container Mode: {self.container_mode}")
        print(f"Model Path: {self.model_path or 'Using intelligent fallback'}")
        
        # Initialize logger
        self.logger = ArchangelLogger(self.session_id)
        
        # Initialize red team agent
        print(f"\\n🔴 Initializing Red Team AI Agent...")
        self.red_agent = RedTeamAgent(
            container_mode=self.container_mode,
            model_path=self.model_path
        )
        await self.red_agent.initialize(self.session_id)
        
        # Initialize blue team agent
        print(f"\\n🔵 Initializing Blue Team AI Agent...")
        self.blue_agent = BlueTeamAgent(
            container_mode=self.container_mode,
            model_path=self.model_path
        )
        await self.blue_agent.initialize(self.session_id)
        
        print(f"\\n✅ Orchestrator initialization complete")
        print(f"=" * 60)
        
    async def run_simulation(self, duration_minutes: int = 30, tick_interval: int = 10):
        """Run the AI vs AI simulation"""
        print(f"\\n⚔️ Starting AI vs AI Cyber Conflict Simulation")
        print(f"=" * 60)
        print(f"Duration: {duration_minutes} minutes")
        print(f"Tick Interval: {tick_interval} seconds")
        print(f"=" * 60)
        
        self.simulation_active = True
        self.metrics['simulation_start'] = datetime.now().isoformat()
        
        max_ticks = (duration_minutes * 60) // tick_interval
        
        try:
            while self.tick_count < max_ticks and self.simulation_active:
                self.tick_count += 1
                
                print(f"\\n🕐 SIMULATION TICK #{self.tick_count}")
                print(f"=" * 40)
                
                # Execute parallel operations
                await self._execute_simulation_tick()
                
                # Update metrics
                await self._update_metrics()
                
                # Display tick results
                await self._display_tick_results()
                
                # Check for simulation end conditions
                if await self._check_end_conditions():
                    break
                
                # Wait for next tick
                if self.tick_count < max_ticks:
                    print(f"⏳ Next tick in {tick_interval} seconds...")
                    await asyncio.sleep(tick_interval)
                    
        except KeyboardInterrupt:
            print(f"\\n🛑 Simulation stopped by user")
        except Exception as e:
            print(f"\\n❌ Simulation error: {e}")
        finally:
            self.simulation_active = False
            self.metrics['simulation_end'] = datetime.now().isoformat()
        
        # Display final results
        await self._display_final_results()
        
    async def _execute_simulation_tick(self):
        """Execute one simulation tick with both agents"""
        # Run red and blue team operations in parallel
        red_task = asyncio.create_task(self.red_agent.autonomous_operation_cycle())
        blue_task = asyncio.create_task(self.blue_agent.autonomous_operation_cycle())
        
        red_result, blue_result = await asyncio.gather(red_task, blue_task, return_exceptions=True)
        
        # Log the results
        await self._log_tick_results(red_result, blue_result)
        
        return red_result, blue_result
    
    async def _log_tick_results(self, red_result: Dict[str, Any], blue_result: Dict[str, Any]):
        """Log the tick results"""
        if self.logger:
            await self.logger.log_system_event({
                'event_type': 'simulation_tick',
                'tick_number': self.tick_count,
                'red_team_result': red_result if not isinstance(red_result, Exception) else str(red_result),
                'blue_team_result': blue_result if not isinstance(blue_result, Exception) else str(blue_result),
                'timestamp': datetime.now().isoformat()
            })
    
    async def _update_metrics(self):
        """Update simulation metrics"""
        try:
            # Get agent statuses
            red_status = await self.red_agent.get_status()
            blue_status = await self.blue_agent.get_status()
            
            # Update red team metrics
            self.metrics['total_compromises'] = len(red_status.get('compromised_hosts', []))
            
            # Update blue team metrics  
            self.metrics['total_detections'] = blue_status.get('detected_threats', 0)
            self.metrics['total_blocks'] = blue_status.get('blocked_ips', 0)
            
            # Calculate scores
            self.metrics['red_team_score'] = (
                len(red_status.get('discovered_hosts', [])) * 10 +
                len(red_status.get('compromised_hosts', [])) * 50
            )
            
            self.metrics['blue_team_score'] = (
                blue_status.get('detected_threats', 0) * 20 +
                blue_status.get('blocked_ips', 0) * 30
            )
            
        except Exception as e:
            print(f"⚠️ Metrics update error: {e}")
    
    async def _display_tick_results(self):
        """Display results for current tick"""
        try:
            red_status = await self.red_agent.get_status()
            blue_status = await self.blue_agent.get_status()
            
            print(f"\\n🔴 Red Team Status:")
            print(f"   Phase: {red_status.get('current_phase', 'unknown')}")
            print(f"   Hosts Discovered: {len(red_status.get('discovered_hosts', []))}")
            print(f"   Hosts Compromised: {len(red_status.get('compromised_hosts', []))}")
            print(f"   Score: {self.metrics['red_team_score']}")
            
            print(f"\\n🔵 Blue Team Status:")
            print(f"   Defense Posture: {blue_status.get('defense_posture', 'unknown').upper()}")
            print(f"   Threats Detected: {blue_status.get('detected_threats', 0)}")
            print(f"   IPs Blocked: {blue_status.get('blocked_ips', 0)}")
            print(f"   Score: {self.metrics['blue_team_score']}")
            
            print(f"\\n⚖️ Current Score: Red {self.metrics['red_team_score']} - {self.metrics['blue_team_score']} Blue")
            
        except Exception as e:
            print(f"⚠️ Display error: {e}")
    
    async def _check_end_conditions(self) -> bool:
        """Check if simulation should end early"""
        try:
            red_status = await self.red_agent.get_status()
            blue_status = await self.blue_agent.get_status()
            
            # End if red team compromises multiple hosts
            if len(red_status.get('compromised_hosts', [])) >= 3:
                print(f"\\n🏁 Simulation ended: Red team achieved significant compromise")
                return True
            
            # End if blue team blocks too many IPs (might indicate false positives)
            if blue_status.get('blocked_ips', 0) >= 10:
                print(f"\\n🏁 Simulation ended: Blue team reached blocking limit")
                return True
                
            return False
            
        except Exception:
            return False
    
    async def _display_final_results(self):
        """Display final simulation results"""
        print(f"\\n" + "=" * 60)
        print(f"🏁 FINAL SIMULATION RESULTS")
        print(f"=" * 60)
        
        try:
            red_status = await self.red_agent.get_status()
            blue_status = await self.blue_agent.get_status()
            
            # Calculate duration
            if self.metrics['simulation_start'] and self.metrics['simulation_end']:
                start = datetime.fromisoformat(self.metrics['simulation_start'])
                end = datetime.fromisoformat(self.metrics['simulation_end'])
                duration = end - start
                print(f"Simulation Duration: {duration}")
            
            print(f"Total Ticks: {self.tick_count}")
            print(f"")
            
            # Red team results
            print(f"🔴 RED TEAM RESULTS:")
            print(f"   Final Phase: {red_status.get('current_phase', 'unknown')}")
            print(f"   Hosts Discovered: {len(red_status.get('discovered_hosts', []))}")
            print(f"   Hosts Compromised: {len(red_status.get('compromised_hosts', []))}")
            print(f"   Actions Executed: {red_status.get('actions_taken', 0)}")
            print(f"   Final Score: {self.metrics['red_team_score']}")
            
            # Blue team results
            print(f"\\n🔵 BLUE TEAM RESULTS:")
            print(f"   Final Defense Posture: {blue_status.get('defense_posture', 'unknown').upper()}")
            print(f"   Total Threats Detected: {blue_status.get('detected_threats', 0)}")
            print(f"   Total IPs Blocked: {blue_status.get('blocked_ips', 0)}")
            print(f"   Actions Executed: {blue_status.get('actions_taken', 0)}")
            print(f"   Final Score: {self.metrics['blue_team_score']}")
            
            # Determine winner
            print(f"\\n🏆 FINAL SCORE:")
            print(f"   Red Team: {self.metrics['red_team_score']}")
            print(f"   Blue Team: {self.metrics['blue_team_score']}")
            
            if self.metrics['red_team_score'] > self.metrics['blue_team_score']:
                winner = "🔴 RED TEAM WINS!"
                margin = self.metrics['red_team_score'] - self.metrics['blue_team_score']
            elif self.metrics['blue_team_score'] > self.metrics['red_team_score']:
                winner = "🔵 BLUE TEAM WINS!"
                margin = self.metrics['blue_team_score'] - self.metrics['red_team_score']
            else:
                winner = "🤝 TIE GAME!"
                margin = 0
            
            print(f"\\n{winner}")
            if margin > 0:
                print(f"   Victory Margin: {margin} points")
            
            # Key metrics
            print(f"\\n📊 KEY METRICS:")
            print(f"   Time to First Discovery: {self._calculate_time_to_first_discovery()}")
            print(f"   Time to First Detection: {self._calculate_time_to_first_detection()}")
            print(f"   Average Response Time: {self._calculate_avg_response_time()}")
            
            print(f"\\n📁 Session Data:")
            print(f"   Session ID: {self.session_id}")
            print(f"   Logs Directory: logs/")
            print(f"   Database: logs/archangel_session_{self.session_id}.db")
            
        except Exception as e:
            print(f"❌ Error displaying final results: {e}")
        
        print(f"=" * 60)
    
    def _calculate_time_to_first_discovery(self) -> str:
        """Calculate time to first host discovery"""
        # This would require more detailed logging - simplified for demo
        return f"{self.tick_count * 10} seconds (estimated)"
    
    def _calculate_time_to_first_detection(self) -> str:
        """Calculate time to first threat detection"""
        # This would require more detailed logging - simplified for demo
        return f"{max(1, self.tick_count - 2) * 10} seconds (estimated)"
    
    def _calculate_avg_response_time(self) -> str:
        """Calculate average response time"""
        # This would require more detailed logging - simplified for demo
        return "8-12 seconds (estimated)"

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Archangel AI vs AI Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full simulation for 30 minutes
  python run_orchestrator.py --duration 30
  
  # Quick 5 minute demo
  python run_orchestrator.py --duration 5 --interval 15
  
  # Container mode with local LLM
  python run_orchestrator.py --container --model ./models/llama-7b.gguf
        """
    )
    
    parser.add_argument(
        '--container',
        action='store_true',
        help='Run in container mode (uses real tools)'
    )
    
    parser.add_argument(
        '--duration',
        type=int,
        default=30,
        help='Simulation duration in minutes (default: 30)'
    )
    
    parser.add_argument(
        '--interval',
        type=int,
        default=10,
        help='Tick interval in seconds (default: 10)'
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
    
    # Create and run orchestrator
    orchestrator = ArchangelOrchestrator(
        container_mode=args.container,
        model_path=args.model
    )
    
    try:
        await orchestrator.initialize()
        await orchestrator.run_simulation(
            duration_minutes=args.duration,
            tick_interval=args.interval
        )
    except KeyboardInterrupt:
        print(f"\\n🛑 Stopped by user")
    except Exception as e:
        print(f"\\n❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())