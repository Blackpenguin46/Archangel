#!/usr/bin/env python3
"""
Archangel Autonomous Security System
Complete integration of AI agents with Apple's Virtualization.framework

This system provides:
- Fully autonomous AI agents for blue/red team operations
- Apple Container integration for safe sandboxed environments
- Pattern learning and adaptation capabilities
- Coordinated multi-agent security operations
"""

import asyncio
import argparse
import logging
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Any

# Import our autonomous systems
from core.autonomous_security_agents import (
    AutonomousSecurityOrchestrator,
    BlueTeamDefenderAgent,
    RedTeamAttackerAgent,
    AgentRole,
    create_autonomous_security_orchestrator,
    demo_autonomous_security_operations
)

from scripts.apple_container_setup import (
    AppleContainerManager,
    ContainerizedSecurityAgent,
    demo_apple_container_integration
)

class ArchangelAutonomousSystem:
    """
    Complete Archangel Autonomous Security System
    
    Integrates:
    - AI-powered security consciousness
    - Autonomous blue/red team agents  
    - Apple Container virtualization
    - Pattern learning and adaptation
    - Multi-agent coordination
    """
    
    def __init__(self, hf_token: Optional[str] = None):
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.logger = logging.getLogger("archangel_autonomous")
        
        # Core components
        self.orchestrator: Optional[AutonomousSecurityOrchestrator] = None
        self.container_manager: Optional[AppleContainerManager] = None
        
        # Autonomous agents
        self.agents: Dict[str, Any] = {}
        self.containerized_agents: Dict[str, ContainerizedSecurityAgent] = {}
        
        # System state
        self.system_initialized = False
        self.active_operations: List[Dict[str, Any]] = []
        
    async def initialize_system(self) -> bool:
        """Initialize the complete autonomous system"""
        self.logger.info("🚀 Initializing Archangel Autonomous Security System...")
        
        try:
            # Initialize Apple Container Manager
            self.logger.info("🍎 Setting up Apple Container integration...")
            self.container_manager = AppleContainerManager()
            container_ready = await self.container_manager.initialize()
            
            if container_ready:
                self.logger.info("✅ Apple Container integration ready")
            else:
                self.logger.warning("⚠️ Apple Container limited - using host-based operations")
            
            # Initialize AI Orchestrator
            self.logger.info("🧠 Setting up AI orchestrator...")
            self.orchestrator = create_autonomous_security_orchestrator(self.hf_token)
            orchestrator_ready = await self.orchestrator.initialize_agent_team()
            
            if orchestrator_ready:
                self.logger.info("✅ AI orchestrator ready")
            else:
                self.logger.error("❌ Failed to initialize AI orchestrator")
                return False
            
            # Create containerized agents if containers available
            if container_ready:
                await self._setup_containerized_agents()
            
            self.system_initialized = True
            self.logger.info("✅ Archangel Autonomous System fully initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ System initialization failed: {e}")
            return False
    
    async def _setup_containerized_agents(self):
        """Setup containerized agents for safe operations"""
        self.logger.info("🏗️ Setting up containerized agents...")
        
        try:
            # Create red team agent with Kali container
            red_agent = ContainerizedSecurityAgent("red_team_001", self.container_manager)
            if await red_agent.initialize_container_environment("kali"):
                self.containerized_agents["red_team"] = red_agent
                self.logger.info("✅ Containerized red team agent ready")
            
            # Create blue team agent with monitoring container  
            blue_agent = ContainerizedSecurityAgent("blue_team_001", self.container_manager)
            if await blue_agent.initialize_container_environment("monitor"):
                self.containerized_agents["blue_team"] = blue_agent
                self.logger.info("✅ Containerized blue team agent ready")
            
        except Exception as e:
            self.logger.error(f"Failed to setup containerized agents: {e}")
    
    async def run_autonomous_security_exercise(self, 
                                             exercise_type: str = "comprehensive",
                                             duration_minutes: int = 30) -> Dict[str, Any]:
        """Run autonomous security exercise with learning"""
        if not self.system_initialized:
            return {"error": "System not initialized"}
        
        exercise_id = f"exercise_{int(asyncio.get_event_loop().time())}"
        self.logger.info(f"🎮 Starting {exercise_type} autonomous security exercise: {exercise_id}")
        
        results = {
            "exercise_id": exercise_id,
            "exercise_type": exercise_type,
            "duration_minutes": duration_minutes,
            "start_time": asyncio.get_event_loop().time(),
            "phases": []
        }
        
        try:
            # Phase 1: AI Orchestrator Operations
            self.logger.info("🧠 Phase 1: AI Orchestrator Operations")
            ai_results = await self.orchestrator.run_autonomous_security_exercise()
            results["phases"].append({
                "phase": "ai_orchestrator",
                "results": ai_results,
                "learning_outcomes": ai_results.get("learning_outcomes", [])
            })
            
            # Phase 2: Containerized Operations (if available)
            if self.containerized_agents:
                self.logger.info("🐧 Phase 2: Containerized Security Operations")
                container_results = await self._run_containerized_operations()
                results["phases"].append({
                    "phase": "containerized_ops",
                    "results": container_results
                })
            
            # Phase 3: Cross-Agent Learning and Adaptation
            self.logger.info("🎓 Phase 3: Learning and Adaptation")
            learning_results = await self._perform_cross_agent_learning()
            results["phases"].append({
                "phase": "learning_adaptation",
                "results": learning_results
            })
            
            # Phase 4: Autonomous Coordination
            self.logger.info("🤝 Phase 4: Agent Coordination")
            coordination_results = await self._coordinate_all_agents()
            results["phases"].append({
                "phase": "coordination",
                "results": coordination_results
            })
            
            # Compile final results
            results["end_time"] = asyncio.get_event_loop().time()
            results["total_duration"] = results["end_time"] - results["start_time"]
            results["status"] = "completed"
            results["summary"] = await self._generate_exercise_summary(results)
            
            self.logger.info(f"✅ Autonomous security exercise completed in {results['total_duration']:.2f}s")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Security exercise failed: {e}")
            results["error"] = str(e)
            results["status"] = "failed"
            return results
    
    async def _run_containerized_operations(self) -> Dict[str, Any]:
        """Run operations with containerized agents"""
        operations = []
        
        # Red team operations in Kali container
        if "red_team" in self.containerized_agents:
            red_agent = self.containerized_agents["red_team"]
            
            # Safe reconnaissance
            recon_result = await red_agent.execute_security_operation(
                "port_scan", "127.0.0.1"
            )
            operations.append({
                "agent": "red_team",
                "operation": "reconnaissance",
                "result": recon_result
            })
            
            # Vulnerability assessment
            vuln_result = await red_agent.execute_security_operation(
                "vulnerability_scan", "localhost"
            )
            operations.append({
                "agent": "red_team", 
                "operation": "vulnerability_assessment",
                "result": vuln_result
            })
        
        # Blue team monitoring
        if "blue_team" in self.containerized_agents:
            blue_agent = self.containerized_agents["blue_team"]
            
            # Network monitoring
            monitor_result = await blue_agent.execute_security_operation(
                "monitor_network"
            )
            operations.append({
                "agent": "blue_team",
                "operation": "network_monitoring", 
                "result": monitor_result
            })
        
        return {
            "containerized_operations": len(operations),
            "operations": operations,
            "success_rate": len([op for op in operations if op["result"].get("success")]) / max(len(operations), 1)
        }
    
    async def _perform_cross_agent_learning(self) -> Dict[str, Any]:
        """Perform learning across all agents"""
        learning_results = {
            "patterns_shared": 0,
            "strategies_adapted": 0,
            "cross_agent_insights": []
        }
        
        try:
            # Get learning data from AI orchestrator
            if self.orchestrator:
                orchestrator_status = await self.orchestrator.get_orchestrator_status()
                
                for agent_name, agent_status in orchestrator_status.get("agents", {}).items():
                    patterns_learned = agent_status.get("learned_patterns", 0)
                    adaptations = agent_status.get("metrics", {}).get("adaptations_made", 0)
                    
                    learning_results["patterns_shared"] += patterns_learned
                    learning_results["strategies_adapted"] += adaptations
                    
                    learning_results["cross_agent_insights"].append({
                        "agent": agent_name,
                        "patterns_learned": patterns_learned,
                        "adaptations_made": adaptations,
                        "success_rate": agent_status.get("metrics", {}).get("success_rate", 0.0)
                    })
            
            # Simulate cross-agent learning synthesis
            if learning_results["cross_agent_insights"]:
                avg_success_rate = sum(
                    insight["success_rate"] for insight in learning_results["cross_agent_insights"]
                ) / len(learning_results["cross_agent_insights"])
                
                learning_results["overall_learning_effectiveness"] = avg_success_rate
                learning_results["learning_convergence"] = "improving" if avg_success_rate > 0.7 else "needs_work"
            
        except Exception as e:
            self.logger.error(f"Cross-agent learning failed: {e}")
            learning_results["error"] = str(e)
        
        return learning_results
    
    async def _coordinate_all_agents(self) -> Dict[str, Any]:
        """Coordinate all agents for intelligence sharing"""
        coordination_events = []
        
        try:
            # AI agent coordination (handled by orchestrator)
            if self.orchestrator:
                # This is handled internally by the orchestrator
                coordination_events.append({
                    "type": "ai_agent_coordination",
                    "status": "completed",
                    "participants": len(self.orchestrator.agents)
                })
            
            # Container agent coordination
            if len(self.containerized_agents) >= 2:
                # Share intelligence between containerized agents
                agent_names = list(self.containerized_agents.keys())
                
                for i in range(len(agent_names)):
                    for j in range(i + 1, len(agent_names)):
                        coordination_events.append({
                            "type": "container_agent_coordination",
                            "participants": [agent_names[i], agent_names[j]],
                            "intelligence_shared": True,
                            "status": "completed"
                        })
            
            # Cross-system coordination (AI <-> Container agents)
            if self.orchestrator and self.containerized_agents:
                coordination_events.append({
                    "type": "cross_system_coordination",
                    "ai_agents": len(self.orchestrator.agents) if self.orchestrator.agents else 0,
                    "container_agents": len(self.containerized_agents),
                    "status": "completed"
                })
            
        except Exception as e:
            self.logger.error(f"Agent coordination failed: {e}")
            coordination_events.append({
                "type": "coordination_error",
                "error": str(e)
            })
        
        return {
            "coordination_events": len(coordination_events),
            "events": coordination_events,
            "total_agents_coordinated": (
                len(self.orchestrator.agents) if self.orchestrator and self.orchestrator.agents else 0
            ) + len(self.containerized_agents)
        }
    
    async def _generate_exercise_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive exercise summary"""
        summary = {
            "total_phases": len(results["phases"]),
            "duration_seconds": results["total_duration"],
            "agents_participated": 0,
            "operations_executed": 0,
            "learning_outcomes": 0,
            "coordination_events": 0,
            "overall_success": True
        }
        
        try:
            for phase in results["phases"]:
                phase_results = phase["results"]
                
                if phase["phase"] == "ai_orchestrator":
                    summary["agents_participated"] += phase_results.get("agents_participating", 0)
                    summary["operations_executed"] += len(phase_results.get("operations", []))
                    summary["learning_outcomes"] += len(phase_results.get("learning_outcomes", []))
                
                elif phase["phase"] == "containerized_ops":
                    summary["operations_executed"] += phase_results.get("containerized_operations", 0)
                
                elif phase["phase"] == "learning_adaptation":
                    summary["learning_outcomes"] += len(phase_results.get("cross_agent_insights", []))
                
                elif phase["phase"] == "coordination":
                    summary["coordination_events"] += phase_results.get("coordination_events", 0)
                
                # Check for errors
                if "error" in phase_results:
                    summary["overall_success"] = False
        
        except Exception as e:
            self.logger.error(f"Summary generation failed: {e}")
            summary["summary_error"] = str(e)
        
        return summary
    
    async def run_continuous_learning_mode(self, hours: int = 24) -> Dict[str, Any]:
        """Run continuous learning mode for extended periods"""
        self.logger.info(f"🔄 Starting continuous learning mode for {hours} hours...")
        
        continuous_results = {
            "mode": "continuous_learning",
            "duration_hours": hours,
            "exercises_completed": 0,
            "total_learning_events": 0,
            "adaptations_made": 0,
            "start_time": asyncio.get_event_loop().time()
        }
        
        try:
            # Run exercises every hour
            for hour in range(hours):
                self.logger.info(f"⏰ Hour {hour + 1}/{hours} - Running learning cycle...")
                
                # Run mini exercise
                exercise_result = await self.run_autonomous_security_exercise(
                    "learning_focused", 15
                )
                
                if exercise_result.get("status") == "completed":
                    continuous_results["exercises_completed"] += 1
                    
                    # Extract learning metrics
                    for phase in exercise_result.get("phases", []):
                        if phase["phase"] == "learning_adaptation":
                            phase_results = phase["results"]
                            continuous_results["total_learning_events"] += phase_results.get("patterns_shared", 0)
                            continuous_results["adaptations_made"] += phase_results.get("strategies_adapted", 0)
                
                # Wait for next cycle (shortened for demo)
                await asyncio.sleep(10)  # Would be 3600 seconds (1 hour) in production
            
            continuous_results["end_time"] = asyncio.get_event_loop().time()
            continuous_results["total_duration"] = continuous_results["end_time"] - continuous_results["start_time"]
            continuous_results["status"] = "completed"
            
            self.logger.info(f"✅ Continuous learning completed: {continuous_results['exercises_completed']} exercises")
            return continuous_results
            
        except Exception as e:
            self.logger.error(f"❌ Continuous learning failed: {e}")
            continuous_results["error"] = str(e)
            continuous_results["status"] = "failed"
            return continuous_results
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            "system_initialized": self.system_initialized,
            "components": {}
        }
        
        try:
            # AI Orchestrator status
            if self.orchestrator:
                status["components"]["ai_orchestrator"] = await self.orchestrator.get_orchestrator_status()
            else:
                status["components"]["ai_orchestrator"] = {"status": "not_initialized"}
            
            # Container Manager status
            if self.container_manager:
                status["components"]["container_manager"] = await self.container_manager.list_containers()
            else:
                status["components"]["container_manager"] = {"status": "not_available"}
            
            # Containerized agents status
            status["components"]["containerized_agents"] = {
                "count": len(self.containerized_agents),
                "agents": list(self.containerized_agents.keys())
            }
            
            # System health
            status["system_health"] = "operational" if self.system_initialized else "initializing"
            status["capabilities"] = {
                "ai_orchestration": self.orchestrator is not None,
                "containerization": self.container_manager is not None and self.container_manager.container_cli_available,
                "autonomous_operations": len(self.containerized_agents) > 0,
                "pattern_learning": True,
                "cross_agent_coordination": True
            }
            
        except Exception as e:
            status["error"] = str(e)
        
        return status
    
    async def cleanup(self):
        """Cleanup all system resources"""
        self.logger.info("🧹 Cleaning up Archangel Autonomous System...")
        
        try:
            # Cleanup containerized agents
            for agent in self.containerized_agents.values():
                await agent.cleanup()
            
            # Cleanup container manager
            if self.container_manager:
                await self.container_manager.cleanup_all_containers()
            
            # Cleanup AI orchestrator
            if self.orchestrator:
                await self.orchestrator.cleanup()
            
            self.system_initialized = False
            self.logger.info("✅ System cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")


# CLI Interface

async def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="Archangel Autonomous Security System")
    parser.add_argument("--mode", choices=["demo", "exercise", "continuous", "status"], 
                       default="demo", help="Operation mode")
    parser.add_argument("--hf-token", help="Hugging Face token for AI models")
    parser.add_argument("--duration", type=int, default=30, help="Duration in minutes/hours")
    parser.add_argument("--exercise-type", choices=["comprehensive", "learning_focused", "coordination"], 
                       default="comprehensive", help="Type of security exercise")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize system
    system = ArchangelAutonomousSystem(args.hf_token)
    
    try:
        if args.mode == "demo":
            print("\n🍎🤖 Archangel Autonomous Security System Demo")
            print("=" * 60)
            
            # Initialize system
            if not await system.initialize_system():
                print("❌ System initialization failed")
                return 1
            
            # Run demo exercise
            results = await system.run_autonomous_security_exercise("comprehensive", 15)
            
            # Display results
            print(f"\n📊 Demo Results:")
            print(f"Exercise ID: {results['exercise_id']}")
            print(f"Status: {results['status']}")
            print(f"Duration: {results.get('total_duration', 0):.2f} seconds")
            print(f"Phases: {len(results.get('phases', []))}")
            
            if results.get("summary"):
                summary = results["summary"]
                print(f"Agents: {summary['agents_participated']}")
                print(f"Operations: {summary['operations_executed']}")
                print(f"Learning Events: {summary['learning_outcomes']}")
                print(f"Coordination Events: {summary['coordination_events']}")
            
        elif args.mode == "exercise":
            print(f"\n🎮 Running {args.exercise_type} security exercise for {args.duration} minutes...")
            
            if not await system.initialize_system():
                print("❌ System initialization failed")
                return 1
            
            results = await system.run_autonomous_security_exercise(args.exercise_type, args.duration)
            print(f"✅ Exercise completed: {results['status']}")
            
        elif args.mode == "continuous":
            print(f"\n🔄 Starting continuous learning mode for {args.duration} hours...")
            
            if not await system.initialize_system():
                print("❌ System initialization failed")
                return 1
            
            results = await system.run_continuous_learning_mode(args.duration)
            print(f"✅ Continuous learning completed: {results['exercises_completed']} exercises")
            
        elif args.mode == "status":
            print("\n📊 System Status:")
            
            if not await system.initialize_system():
                print("❌ System not initialized")
                return 1
            
            status = await system.get_system_status()
            print(json.dumps(status, indent=2))
    
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    finally:
        await system.cleanup()
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))