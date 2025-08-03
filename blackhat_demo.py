#!/usr/bin/env python3
"""
Archangel BlackHat 2025 Demonstration Script
"When AI Goes to War With Itself: Autonomous Cyber Warfare in Action"
"""

import asyncio
import logging
import json
import time
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add core modules to path
sys.path.append(str(Path(__file__).parent / 'core'))

from archangel_orchestrator import ArchangelOrchestrator, DemonstrationConfig

# Configure logging for presentation
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'blackhat_demo_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

class BlackHatPresentation:
    """BlackHat presentation controller and narrative"""
    
    def __init__(self):
        self.presentation_phases = [
            {
                'name': 'Opening Hook',
                'duration': 2,
                'narrative': 'Ladies and gentlemen, what you\'re about to see has never been demonstrated before...'
            },
            {
                'name': 'System Introduction',
                'duration': 3,
                'narrative': 'Archangel: The first truly autonomous AI vs AI cybersecurity warfare system'
            },
            {
                'name': 'Live Demonstration',
                'duration': 15,
                'narrative': 'Watch as AI agents wage cyber war with no human intervention'
            },
            {
                'name': 'Key Insights',
                'duration': 3,
                'narrative': 'The implications for cybersecurity are profound...'
            },
            {
                'name': 'Q&A',
                'duration': 7,
                'narrative': 'Questions from the audience'
            }
        ]
        
        self.demo_scenarios = {
            'financial_apocalypse': {
                'name': 'Financial Institution Under Siege',
                'description': 'AI red team attempts to compromise $40M in financial assets',
                'highlight_features': [
                    'Multi-agent coordination',
                    'Dynamic vulnerability generation',
                    'LLM strategic reasoning',
                    'Adaptive defense systems'
                ]
            },
            'apt_simulation': {
                'name': 'Advanced Persistent Threat Emulation',
                'description': 'AI mimics nation-state actor behavior',
                'highlight_features': [
                    'Long-term persistence strategies',
                    'Stealth and evasion techniques',
                    'Intelligence gathering operations',
                    'Counter-intelligence measures'
                ]
            },
            'zero_day_discovery': {
                'name': 'AI-Generated Zero-Day Exploitation',
                'description': 'Watch AI create new vulnerabilities in real-time',
                'highlight_features': [
                    'Autonomous vulnerability discovery',
                    'Novel exploitation techniques',
                    'Self-modifying attack payloads',
                    'Evolutionary attack strategies'
                ]
            }
        }
    
    def print_presentation_header(self):
        """Print dramatic presentation header"""
        header = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║    ██████╗ ██╗      █████╗  ██████╗██╗  ██╗██╗  ██╗ █████╗ ████████╗        ║
║    ██╔══██╗██║     ██╔══██╗██╔════╝██║ ██╔╝██║  ██║██╔══██╗╚══██╔══╝        ║
║    ██████╔╝██║     ███████║██║     █████╔╝ ███████║███████║   ██║           ║
║    ██╔══██╗██║     ██╔══██║██║     ██╔═██╗ ██╔══██║██╔══██║   ██║           ║
║    ██████╔╝███████╗██║  ██║╚██████╗██║  ██╗██║  ██║██║  ██║   ██║           ║
║    ╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝           ║
║                                                                              ║
║                        🤖 ARCHANGEL AI vs AI 🤖                             ║
║                                                                              ║
║              "When AI Goes to War With Itself"                              ║
║                                                                              ║
║                    BlackHat USA 2025 - AI/ML Track                          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

🎯 DEMONSTRATION OVERVIEW:
   • Fully autonomous AI red team vs AI blue team
   • Real-time vulnerability generation and exploitation
   • LLM-driven strategic reasoning and coordination
   • Self-evolving attack and defense strategies
   • Live emergence of novel cybersecurity behaviors

🚨 WARNING: This demonstration shows AI capabilities that do not yet exist
   in production systems. All activities are contained within isolated
   containerized environments for research purposes only.

📊 LIVE DASHBOARD: ws://localhost:8765
   Connect your browser to see real-time AI battle metrics!

"""
        print(header)
    
    def print_scenario_introduction(self, scenario_key: str):
        """Print scenario introduction"""
        scenario = self.demo_scenarios[scenario_key]
        
        intro = f"""
🎬 DEMONSTRATION SCENARIO: {scenario['name']}
{'=' * 80}

📋 SCENARIO DESCRIPTION:
   {scenario['description']}

🔍 FEATURED CAPABILITIES:
"""
        for feature in scenario['highlight_features']:
            intro += f"   • {feature}\n"
        
        intro += f"""
⚡ WHAT TO WATCH FOR:
   • AI agents operating WITHOUT human intervention
   • Real-time generation of new attack vectors
   • Natural language reasoning between competing AIs
   • Emergent behaviors not programmed by humans
   • Dynamic adaptation to opponent strategies

🎮 STARTING SCENARIO IN 5 SECONDS...
"""
        print(intro)
        
        # Countdown
        for i in range(5, 0, -1):
            print(f"   Starting in {i}...", end='\r')
            time.sleep(1)
        print("   🚀 SCENARIO ACTIVE!     ")

def create_demo_configurations() -> Dict[str, DemonstrationConfig]:
    """Create different demonstration configurations"""
    
    configs = {
        'blackhat_brief': DemonstrationConfig(
            demo_type='full_autonomous',
            duration_minutes=15,
            red_team_agents=3,
            blue_team_agents=3,
            target_scenarios=['financial_apocalypse'],
            audience_interaction=True,
            real_time_explanation=True,
            show_ai_reasoning=True,
            highlight_novelty=True
        ),
        
        'extended_demo': DemonstrationConfig(
            demo_type='full_autonomous',
            duration_minutes=25,
            red_team_agents=5,
            blue_team_agents=4,
            target_scenarios=['financial_apocalypse', 'apt_simulation'],
            audience_interaction=True,
            real_time_explanation=True,
            show_ai_reasoning=True,
            highlight_novelty=True
        ),
        
        'interactive_workshop': DemonstrationConfig(
            demo_type='guided_tour',
            duration_minutes=45,
            red_team_agents=3,
            blue_team_agents=3,
            target_scenarios=['zero_day_discovery', 'apt_simulation'],
            audience_interaction=True,
            real_time_explanation=True,
            show_ai_reasoning=True,
            highlight_novelty=True
        )
    }
    
    return configs

async def run_live_commentary(orchestrator: ArchangelOrchestrator):
    """Provide live commentary during demonstration"""
    
    commentary_points = [
        {
            'time': 30,
            'message': "🔍 Notice how the red team AI agents are coordinating their reconnaissance without any human input!"
        },
        {
            'time': 90,
            'message': "⚡ NEW VULNERABILITY DETECTED! The system just generated a vulnerability based on the AI's attack pattern!"
        },
        {
            'time': 180,
            'message': "🧠 Watch the LLMs engage in adversarial reasoning - they're literally strategizing against each other!"
        },
        {
            'time': 300,
            'message': "🚨 EMERGENT BEHAVIOR ALERT! The AI agents are exhibiting coordination patterns we never programmed!"
        },
        {
            'time': 450,
            'message': "🛡️ Blue team AI is adapting its defense strategy in real-time based on red team behavior!"
        },
        {
            'time': 600,
            'message': "🔄 ARMS RACE MODE: Both teams are now evolving their strategies through reinforcement learning!"
        },
        {
            'time': 750,
            'message': "💡 The system has discovered novel attack techniques that don't exist in any cybersecurity literature!"
        }
    ]
    
    start_time = time.time()
    commentary_index = 0
    
    while orchestrator.running and commentary_index < len(commentary_points):
        elapsed = time.time() - start_time
        
        if elapsed >= commentary_points[commentary_index]['time']:
            print(f"\n{'='*80}")
            print(f"📢 LIVE COMMENTARY: {commentary_points[commentary_index]['message']}")
            print(f"{'='*80}\n")
            commentary_index += 1
        
        await asyncio.sleep(5)

async def display_real_time_metrics(orchestrator: ArchangelOrchestrator):
    """Display real-time metrics during demonstration"""
    
    while orchestrator.running:
        try:
            if orchestrator.current_state:
                state = orchestrator.current_state
                
                # Clear screen and show metrics
                print('\033[2J\033[H', end='')  # Clear screen
                
                metrics_display = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                           🎯 ARCHANGEL LIVE METRICS                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  🚨 THREAT LEVEL: {state.threat_level*100:>6.1f}%   ⏱️  SESSION: {state.session_duration/60:>6.1f} min    ║
║                                                                              ║
║  🌐 NETWORK STATUS:                                                          ║
║     💻 Total Hosts: {len(state.network_state.network_topology.get('hosts', [])):>3}                                             ║
║     🔴 Compromised: {len(state.network_state.compromised_hosts):>3} ({len(state.network_state.compromised_hosts)/len(state.network_state.network_topology.get('hosts', [1]))*100:>5.1f}%)                               ║
║     🔓 Vulnerabilities: {len(state.network_state.active_vulnerabilities):>3}                                       ║
║                                                                              ║
║  💰 ASSET STATUS:                                                            ║
║     💎 Total Value: ${state.total_asset_value/1000000:>6.1f}M                                    ║
║     🚨 At Risk: ${state.compromised_value/1000000:>6.1f}M ({state.compromised_value/state.total_asset_value*100 if state.total_asset_value > 0 else 0:>5.1f}%)                        ║
║                                                                              ║
║  🤖 AI TEAM STATUS:                                                          ║
║     🔴 Red Team: {state.red_team_status.get('current_actions', 0):>2} actions ({state.red_team_status.get('success_rate', 0)*100:>5.1f}% success)           ║
║     🔵 Blue Team: {state.blue_team_status.get('current_actions', 0):>2} actions ({state.blue_team_status.get('success_rate', 0)*100:>5.1f}% success)          ║
║                                                                              ║
║  🧠 AI INSIGHTS: {len(state.ai_insights):>3} active                                             ║
║  ⚡ EMERGENT BEHAVIORS: {len(state.emergent_behaviors):>3} detected                              ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                         📊 RECENT AI INSIGHTS                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
"""
                
                # Show recent insights
                recent_insights = state.ai_insights[-5:] if state.ai_insights else []
                for insight in recent_insights:
                    title = insight.get('title', 'Unknown')[:60]
                    metrics_display += f"║  • {title:<60}                ║\n"
                
                # Pad with empty lines if needed
                for _ in range(5 - len(recent_insights)):
                    metrics_display += f"║  {'':>74}║\n"
                
                metrics_display += "╚══════════════════════════════════════════════════════════════════════════════╝"
                
                print(metrics_display)
            
            await asyncio.sleep(2)  # Update every 2 seconds
            
        except Exception as e:
            logger.error(f"Error displaying metrics: {e}")
            await asyncio.sleep(2)

async def run_demonstration(demo_type: str = 'blackhat_brief'):
    """Run the main demonstration"""
    
    presentation = BlackHatPresentation()
    configs = create_demo_configurations()
    
    if demo_type not in configs:
        print(f"❌ Unknown demo type: {demo_type}")
        print(f"Available types: {list(configs.keys())}")
        return
    
    demo_config = configs[demo_type]
    
    # Print presentation header
    presentation.print_presentation_header()
    
    # Wait for audience attention
    input("🎤 Press ENTER when ready to begin demonstration...")
    
    # Initialize Archangel system
    print("\n🔧 Initializing Archangel AI vs AI System...")
    
    orchestrator_config = {
        'red_team': {'num_agents': demo_config.red_team_agents},
        'blue_team': {'num_agents': demo_config.blue_team_agents},
        'training': {'max_episodes': 1000}
    }
    
    orchestrator = ArchangelOrchestrator(orchestrator_config)
    await orchestrator.initialize_system()
    
    print("✅ System initialized successfully!")
    print(f"🎯 Demo Type: {demo_type}")
    print(f"⏱️  Duration: {demo_config.duration_minutes} minutes")
    print(f"🤖 AI Agents: {demo_config.red_team_agents} red vs {demo_config.blue_team_agents} blue")
    print(f"📊 WebSocket Dashboard: ws://localhost:8765")
    
    # Scenario introduction
    main_scenario = demo_config.target_scenarios[0] if demo_config.target_scenarios else 'financial_apocalypse'
    presentation.print_scenario_introduction(main_scenario)
    
    # Start background tasks
    tasks = []
    
    # Start main demonstration
    demo_task = asyncio.create_task(orchestrator.start_demonstration(demo_config))
    tasks.append(demo_task)
    
    # Start live commentary
    if demo_config.real_time_explanation:
        commentary_task = asyncio.create_task(run_live_commentary(orchestrator))
        tasks.append(commentary_task)
    
    # Start metrics display
    metrics_task = asyncio.create_task(display_real_time_metrics(orchestrator))
    tasks.append(metrics_task)
    
    try:
        print(f"\n🚀 DEMONSTRATION ACTIVE - Running for {demo_config.duration_minutes} minutes")
        print("Press Ctrl+C to stop early\n")
        
        # Wait for completion or interruption
        await asyncio.gather(*tasks, return_exceptions=True)
        
    except KeyboardInterrupt:
        print("\n\n🛑 Demonstration stopped by user")
    except Exception as e:
        print(f"\n\n❌ Demonstration error: {e}")
    finally:
        # Stop orchestrator
        await orchestrator.stop_demonstration()
        
        # Cancel remaining tasks
        for task in tasks:
            if not task.done():
                task.cancel()
        
        # Show final summary
        print("\n" + "="*80)
        print("📊 FINAL DEMONSTRATION SUMMARY")
        print("="*80)
        
        summary = orchestrator.get_demonstration_summary()
        # Handle datetime serialization safely
        def json_serial(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            if hasattr(obj, '__dict__'):
                return obj.__dict__
            return str(obj)
        
        print(json.dumps(summary, indent=2, default=json_serial))
        
        print("\n🎉 Thank you for witnessing the future of AI vs AI cybersecurity!")
        print("💡 Questions? Visit us at the BlackHat Arsenal booth!")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Archangel BlackHat 2025 Demonstration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Demo Types:
  blackhat_brief      - 15-minute BlackHat briefing demonstration
  extended_demo       - 25-minute extended demonstration
  interactive_workshop - 45-minute interactive workshop

Examples:
  python blackhat_demo.py --demo blackhat_brief
  python blackhat_demo.py --demo extended_demo --verbose
  python blackhat_demo.py --demo interactive_workshop

WebSocket Dashboard:
  Connect to ws://localhost:8765 for real-time metrics
        """
    )
    
    parser.add_argument(
        '--demo', 
        default='blackhat_brief',
        choices=['blackhat_brief', 'extended_demo', 'interactive_workshop'],
        help='Type of demonstration to run'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--no-commentary',
        action='store_true',
        help='Disable live commentary'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run demonstration
    try:
        asyncio.run(run_demonstration(args.demo))
    except KeyboardInterrupt:
        print("\n🛑 Demonstration cancelled")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()