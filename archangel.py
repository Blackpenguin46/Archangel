#!/usr/bin/env python3
"""
Archangel AI vs AI Cybersecurity System - Live Application
The world's first fully autonomous AI vs AI cybersecurity research platform

Usage:
    python3 archangel.py --mode live                    # Run live AI vs AI system
    python3 archangel.py --mode live --enterprise       # Enterprise scenario
    python3 archangel.py --mode live --duration 60      # Run for 60 minutes  
    python3 archangel.py --config config.json           # Use custom config
    python3 archangel.py --status                       # Show system status
"""

import asyncio
import argparse
import logging
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

# Add core to path
sys.path.append(str(Path(__file__).parent / 'core'))

# Import the enhanced AI vs AI orchestrator
from archangel_orchestrator import ArchangelOrchestrator, DemonstrationConfig, ArchangelState

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'archangel_live_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

class ArchangelLiveSystem:
    """Live AI vs AI cybersecurity system for production use"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.orchestrator = None
        self.running = False
        
        logger.info("🤖 Archangel Live AI vs AI System initialized")
    
    async def initialize(self):
        """Initialize the live system"""
        logger.info("🚀 Initializing Archangel Live System...")
        
        # Create orchestrator
        self.orchestrator = ArchangelOrchestrator(self.config)
        
        # Initialize all system components
        await self.orchestrator.initialize_system()
        
        logger.info("✅ Archangel Live System ready for operation")
    
    async def run_live_operation(self, duration_minutes: int = 60, scenario: str = 'enterprise'):
        """Run live AI vs AI operation"""
        
        if not self.orchestrator:
            raise RuntimeError("System not initialized. Call initialize() first.")
        
        # Create operation configuration
        operation_config = DemonstrationConfig(
            demo_type='live_operation',
            duration_minutes=duration_minutes,
            red_team_agents=self.config.get('red_team', {}).get('num_agents', 5),
            blue_team_agents=self.config.get('blue_team', {}).get('num_agents', 4),
            target_scenarios=[scenario],
            audience_interaction=False,
            real_time_explanation=True,
            show_ai_reasoning=True,
            highlight_novelty=True
        )
        
        logger.info(f"🎯 Starting live AI vs AI operation - Duration: {duration_minutes} minutes")
        logger.info(f"⚔️ Red Team: {operation_config.red_team_agents} agents vs Blue Team: {operation_config.blue_team_agents} agents")
        
        # Start live operation
        self.running = True
        try:
            await self.orchestrator.start_demonstration(operation_config)
        except KeyboardInterrupt:
            logger.info("🛑 Live operation interrupted by user")
        except Exception as e:
            logger.error(f"❌ Error during live operation: {e}")
        finally:
            self.running = False
            await self.orchestrator.stop_demonstration()
            
        # Get final results
        summary = self.orchestrator.get_demonstration_summary()
        self._display_operation_results(summary)
    
    def _display_operation_results(self, summary: Dict[str, Any]):
        """Display operation results"""
        print("\n" + "="*80)
        print("🎯 ARCHANGEL LIVE OPERATION RESULTS")
        print("="*80)
        
        if not summary:
            print("❌ No results available")
            return
        
        session_info = summary.get('session_info', {})
        final_state = summary.get('final_state', {})
        performance = summary.get('performance_metrics', {})
        
        print(f"📊 Session Duration: {session_info.get('duration_minutes', 0):.1f} minutes")
        print(f"🎯 Final Threat Level: {final_state.get('threat_level', 0):.1%}")
        print(f"💻 Systems Compromised: {final_state.get('compromised_hosts', 0)}")
        print(f"🧠 Emergent Behaviors: {final_state.get('emergent_behaviors', 0)}")
        print(f"🔍 Vulnerabilities Generated: {final_state.get('vulnerabilities_generated', 0)}")
        print(f"💰 Asset Value at Risk: ${final_state.get('compromised_value', 0):,.0f}")
        
        ai_metrics = performance.get('ai_intelligence', {})
        print(f"🤖 AI Autonomy Level: {ai_metrics.get('autonomy_level', 0):.1%}")
        print(f"📈 System Learning Rate: {ai_metrics.get('learning_rate', 0):.1%}")
        print(f"💡 Innovation Index: {ai_metrics.get('innovation_index', 0):.1%}")
        
        print("\n✅ Live operation completed successfully!")
    
    async def show_system_status(self):
        """Show current system status"""
        print("\n🔍 ARCHANGEL SYSTEM STATUS")
        print("="*50)
        
        # Check if orchestrator is available
        if self.orchestrator:
            if hasattr(self.orchestrator, 'current_state') and self.orchestrator.current_state:
                state = self.orchestrator.current_state
                print(f"🤖 System Mode: {state.mode}")
                print(f"⚡ Running: {'Yes' if self.running else 'No'}")
                print(f"🎯 Threat Level: {state.threat_level:.1%}")
                print(f"💻 Compromised Hosts: {len(state.network_state.compromised_hosts)}")
                print(f"🔍 Active Vulnerabilities: {len(state.network_state.active_vulnerabilities)}")
                print(f"🧠 Emergent Behaviors: {len(state.emergent_behaviors)}")
                print(f"💰 Asset Value at Risk: ${state.compromised_value:,.0f}")
            else:
                print("📋 System initialized but not active")
        else:
            print("❌ System not initialized")
        
        print(f"📅 Status checked at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load system configuration"""
    default_config = {
        'red_team': {'num_agents': 5},
        'blue_team': {'num_agents': 4},
        'training': {'max_episodes': 1000},
        'enterprise': {
            'target_value': 40000000,  # $40M enterprise
            'critical_systems': ['financial-db', 'domain-controller', 'hr-system']
        }
    }
    
    if config_path and Path(config_path).exists():
        try:
            with open(config_path, 'r') as f:
                custom_config = json.load(f)
            # Merge with defaults
            default_config.update(custom_config)
            logger.info(f"📁 Configuration loaded from {config_path}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load config from {config_path}: {e}")
            logger.info("📋 Using default configuration")
    else:
        logger.info("📋 Using default configuration")
    
    return default_config

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Archangel AI vs AI Cybersecurity System')
    parser.add_argument('--mode', choices=['live'], default='live',
                       help='Operation mode (default: live)')
    parser.add_argument('--duration', type=int, default=60,
                       help='Operation duration in minutes (default: 60)')
    parser.add_argument('--config', type=str,
                       help='Configuration file path')
    parser.add_argument('--enterprise', action='store_true',
                       help='Use enterprise scenario')
    parser.add_argument('--status', action='store_true',
                       help='Show system status and exit')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    config = load_config(args.config)
    
    # Create and initialize system
    system = ArchangelLiveSystem(config)
    
    if args.status:
        await system.show_system_status()
        return
    
    # Initialize system
    await system.initialize()
    
    # Determine scenario
    scenario = 'enterprise' if args.enterprise else 'financial_apocalypse'
    
    # Show startup banner
    print_startup_banner(args.duration, scenario)
    
    # Run live operation
    if args.mode == 'live':
        await system.run_live_operation(args.duration, scenario)

def print_startup_banner(duration: int, scenario: str):
    """Print startup banner"""
    banner = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║    🤖 ARCHANGEL AI vs AI CYBERSECURITY SYSTEM - LIVE APPLICATION 🤖         ║
║                                                                              ║
║              "The World's First Autonomous AI Cyber Warfare"                ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

🎯 LIVE OPERATION CONFIGURATION:
   • Duration: {duration} minutes
   • Scenario: {scenario}
   • Mode: Fully Autonomous AI vs AI
   • Human Intervention: None Required

🚨 LIVE SYSTEM NOTICE:
   This is the production AI vs AI system. All AI agents will operate
   autonomously without human oversight within isolated environments.

📊 REAL-TIME MONITORING:
   System metrics and AI decisions will be logged in real-time.
   Check the log files for detailed operation analysis.

🚀 STARTING LIVE OPERATION...
"""
    print(banner)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Archangel system shutdown by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)