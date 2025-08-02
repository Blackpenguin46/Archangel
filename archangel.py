#!/usr/bin/env python3
"""
Archangel Autonomous Security System - Main Entry Point
Fully autonomous AI agents for red team vs blue team cybersecurity operations

Usage:
    python3 archangel.py                    # Run live combat demo
    python3 archangel.py --duration 30      # Run for 30 minutes  
    python3 archangel.py --training          # Train AI models first
    python3 archangel.py --status           # Show system status
"""

import asyncio
import argparse
import logging
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Any

# Import live combat system
from environments.live_adversarial_environment import LiveAdversarialEnvironment
from agents.live_combat_agents import LiveCombatOrchestrator, LiveRedTeamAgent, LiveBlueTeamAgent

# Import autonomous enterprise scenario
try:
    from scenarios.autonomous_enterprise_scenario import AutonomousScenarioOrchestrator
    AUTONOMOUS_SCENARIO_AVAILABLE = True
except ImportError:
    AUTONOMOUS_SCENARIO_AVAILABLE = False
    AutonomousScenarioOrchestrator = None

# Import AI-enhanced autonomous scenario
try:
    from scenarios.ai_enhanced_autonomous_scenario import AIEnhancedScenarioOrchestrator
    AI_ENHANCED_SCENARIO_AVAILABLE = True
except ImportError:
    AI_ENHANCED_SCENARIO_AVAILABLE = False
    AIEnhancedScenarioOrchestrator = None

# Import MCP integration system
try:
    from core.archangel_mcp_integration import EnhancedArchangelSystem, MCPEnhancedAgent
    from core.mcp_guardian_protocol import GuardianProtocol
    from core.mcp_security_orchestrator import MCPSecurityOrchestrator
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    EnhancedArchangelSystem = None

# Import training system  
try:
    from training.deepseek_training_pipeline import main as train_main
    TRAINING_AVAILABLE = True
except ImportError:
    TRAINING_AVAILABLE = False
    train_main = None

class ArchangelSystem:
    """
    Archangel Autonomous Security System
    
    Fully autonomous AI agents that:
    - Think and reason about security decisions
    - Operate red team attacks and blue team defenses autonomously
    - Learn and adapt strategies in real-time
    - Run in isolated Docker containers for safety
    """
    
    def __init__(self, enable_mcp=True):
        self.logger = self._setup_logging()
        self.combat_env = LiveAdversarialEnvironment()
        self.orchestrator = LiveCombatOrchestrator()
        self.system_ready = False
        
        # MCP Integration
        self.mcp_enabled = enable_mcp and MCP_AVAILABLE
        self.enhanced_system = None
        self.guardian_protocol = None
        self.security_orchestrator = None
        
        # Autonomous Enterprise Scenario
        self.autonomous_scenario_orchestrator = None
        self.ai_enhanced_scenario_orchestrator = None
        
        if self.mcp_enabled:
            self.logger.info("🚀 MCP integration enabled - Elite capabilities activated")
        else:
            self.logger.info("📋 Standard mode - Basic autonomous operations")
        
        self.logger.info("🛡️ Archangel Autonomous Security System initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('archangel.log')
            ]
        )
        return logging.getLogger('archangel')
    
    async def initialize_system(self) -> bool:
        """Initialize the complete system"""
        try:
            self.logger.info("🚀 Initializing Archangel system...")
            
            # Check Docker availability
            if not await self._check_docker():
                return False
            
            # Initialize MCP enhanced system if available
            if self.mcp_enabled:
                try:
                    self.logger.info("🔧 Initializing MCP enhanced capabilities...")
                    self.enhanced_system = EnhancedArchangelSystem()
                    self.guardian_protocol = GuardianProtocol()
                    self.security_orchestrator = MCPSecurityOrchestrator()
                    
                    # Initialize Guardian Protocol
                    await self.guardian_protocol.initialize()
                    self.logger.info("✅ Guardian Protocol active - Security governance enabled")
                    
                    # Initialize Security Orchestrator
                    await self.security_orchestrator.initialize()
                    self.logger.info("✅ Security Orchestrator active - Elite tools available")
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ MCP initialization failed: {e}")
                    self.logger.info("Falling back to standard mode...")
                    self.mcp_enabled = False
            
            # Setup combat environment
            if not await self.combat_env.setup_combat_environment():
                self.logger.error("❌ Failed to setup combat environment")
                return False
            
            # Initialize orchestrator
            if not await self.orchestrator.setup_live_combat_simulation():
                self.logger.error("❌ Failed to initialize combat orchestrator")
                return False
            
            # Initialize autonomous enterprise scenario orchestrator if available
            if AUTONOMOUS_SCENARIO_AVAILABLE:
                try:
                    # Get container names from combat environment - will be set after containers are created
                    red_container = getattr(self.combat_env, 'red_team_container', None) or self.combat_env.red_team_config.get("name", "archangel-red-autonomous")
                    blue_container = getattr(self.combat_env, 'blue_team_container', None) or self.combat_env.blue_team_config.get("name", "archangel-blue-autonomous")
                    
                    self.autonomous_scenario_orchestrator = AutonomousScenarioOrchestrator(
                        red_container, blue_container
                    )
                    self.logger.info("✅ Autonomous enterprise scenario orchestrator initialized")
                except Exception as e:
                    self.logger.warning(f"⚠️ Autonomous scenario initialization failed: {e}")
                    self.autonomous_scenario_orchestrator = None
            
            # Initialize AI-enhanced scenario orchestrator if available
            if AI_ENHANCED_SCENARIO_AVAILABLE:
                try:
                    red_container = getattr(self.combat_env, 'red_team_container', None) or self.combat_env.red_team_config.get("name", "archangel-red-ai-enhanced")
                    blue_container = getattr(self.combat_env, 'blue_team_container', None) or self.combat_env.blue_team_config.get("name", "archangel-blue-ai-enhanced")
                    
                    self.ai_enhanced_scenario_orchestrator = AIEnhancedScenarioOrchestrator(
                        red_container, blue_container
                    )
                    self.logger.info("✅ AI-Enhanced scenario orchestrator initialized")
                except Exception as e:
                    self.logger.warning(f"⚠️ AI-Enhanced scenario initialization failed: {e}")
                    self.ai_enhanced_scenario_orchestrator = None
            
            self.system_ready = True
            
            if self.mcp_enabled:
                self.logger.info("✅ Archangel system ready for ELITE autonomous operations")
                self.logger.info("🎯 External resources: Shodan, VirusTotal, Metasploit, MISP, and more...")
            else:
                self.logger.info("✅ Archangel system ready for basic autonomous operations")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ System initialization failed: {e}")
            return False
    
    async def _check_docker(self) -> bool:
        """Check if Docker is available and running"""
        try:
            import docker
            client = docker.from_env()
            client.ping()
            self.logger.info("✅ Docker is available and running")
            return True
        except Exception as e:
            self.logger.error(f"❌ Docker check failed: {e}")
            self.logger.error("Please ensure Docker Desktop is running")
            return False
    
    async def run_live_combat(self, duration_minutes: int = 10) -> Dict[str, Any]:
        """Run live autonomous red vs blue team combat"""
        if not self.system_ready:
            self.logger.error("❌ System not initialized")
            return {"error": "System not ready"}
        
        self.logger.info(f"🥊 Starting live combat for {duration_minutes} minutes")
        
        try:
            # Use MCP-enhanced combat if available
            if self.mcp_enabled and self.enhanced_system:
                self.logger.info("🚀 Running ELITE combat with external resources...")
                results = await self.enhanced_system.run_enhanced_adversarial_exercise(
                    scenario_name="Elite Autonomous Combat",
                    duration_minutes=duration_minutes,
                    use_external_resources=True
                )
            else:
                # Run standard combat
                results = await self.orchestrator.execute_live_adversarial_exercise(
                    scenario_name="Autonomous Combat",
                    duration_minutes=duration_minutes
                )
            
            # Display results
            self._display_combat_results(results)
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Combat failed: {e}")
            return {"error": str(e)}
    
    async def run_security_operation(self, operation_description: str) -> Dict[str, Any]:
        """Run a natural language security operation"""
        if not self.system_ready:
            return {"error": "System not initialized"}
        
        if not self.mcp_enabled:
            return {"error": "Elite operations require MCP integration"}
        
        self.logger.info(f"🎯 Processing security operation: {operation_description}")
        
        try:
            # Use enhanced system for natural language operations
            results = await self.enhanced_system.process_natural_language_operation(
                operation_description
            )
            
            # Display operation results
            self._display_operation_results(results)
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Security operation failed: {e}")
            return {"error": str(e)}
    
    async def run_threat_hunt(self, target_description: str) -> Dict[str, Any]:
        """Run comprehensive threat hunting operation"""
        if not self.mcp_enabled:
            return {"error": "Threat hunting requires MCP integration"}
        
        self.logger.info(f"🔍 Starting threat hunt: {target_description}")
        
        try:
            results = await self.enhanced_system.comprehensive_threat_hunt(
                target_description
            )
            
            self._display_threat_hunt_results(results)
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Threat hunt failed: {e}")
            return {"error": str(e)}
    
    async def run_vulnerability_assessment(self, target: str) -> Dict[str, Any]:
        """Run elite vulnerability assessment with external tools"""
        if not self.mcp_enabled:
            return {"error": "Elite assessments require MCP integration"}
        
        self.logger.info(f"🎯 Starting vulnerability assessment: {target}")
        
        try:
            results = await self.enhanced_system.elite_vulnerability_assessment(
                target
            )
            
            self._display_vuln_assessment_results(results)
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Vulnerability assessment failed: {e}")
            return {"error": str(e)}
    
    def _display_combat_results(self, results: Dict[str, Any]):
        """Display combat results in a readable format"""
        print("\n" + "=" * 60)
        print("🎯 AUTONOMOUS COMBAT RESULTS")
        print("=" * 60)
        
        if "error" in results:
            print(f"❌ Combat failed: {results['error']}")
            return
        
        # Basic stats
        print(f"🕐 Duration: {results.get('duration_minutes', 0)} minutes")
        print(f"🔴 Red Team Agent: Autonomous attacker")
        print(f"🔵 Blue Team Agent: Autonomous defender")
        
        # Combat metrics
        red_results = results.get('red_team_results', {})
        blue_results = results.get('blue_team_results', {})
        
        print(f"\n📊 Combat Statistics:")
        print(f"  Red Team Attacks: {red_results.get('total_attacks', 0)}")
        print(f"  Blue Team Defenses: {blue_results.get('total_defenses', 0)}")
        print(f"  Successful Attacks: {red_results.get('successful_attacks', 0)}")
        print(f"  Blocked Attacks: {blue_results.get('blocked_attacks', 0)}")
        
        # Learning metrics
        learning = results.get('learning_analysis', {})
        print(f"\n🧠 AI Learning:")
        print(f"  Red Team Adaptations: {learning.get('red_adaptations', 0)}")
        print(f"  Blue Team Improvements: {learning.get('blue_improvements', 0)}")
        print(f"  Strategic Evolution: {learning.get('evolution_score', 0):.1f}%")
        
        # Key insights
        insights = results.get('insights', [])
        if insights:
            print(f"\n💡 Key Insights:")
            for insight in insights[:3]:  # Show top 3
                print(f"  • {insight}")
        
        print("\n✅ Autonomous combat completed successfully!")
    
    def _display_operation_results(self, results: Dict[str, Any]):
        """Display security operation results"""
        print("\n" + "=" * 60)
        print("🎯 SECURITY OPERATION RESULTS")
        print("=" * 60)
        
        if "error" in results:
            print(f"❌ Operation failed: {results['error']}")
            return
        
        print(f"📋 Operation: {results.get('operation_type', 'Unknown')}")
        print(f"🎯 Target: {results.get('target', 'N/A')}")
        print(f"⏱️ Duration: {results.get('duration_seconds', 0):.1f}s")
        
        # Tool usage
        tools_used = results.get('tools_used', [])
        if tools_used:
            print(f"\n🛠️ Tools Used: {', '.join(tools_used)}")
        
        # Key findings
        findings = results.get('findings', [])
        if findings:
            print(f"\n🔍 Key Findings:")
            for finding in findings[:5]:  # Show top 5
                print(f"  • {finding}")
        
        # Security status
        status = results.get('security_status', 'unknown')
        print(f"\n🛡️ Security Status: {status}")
    
    def _display_threat_hunt_results(self, results: Dict[str, Any]):
        """Display threat hunting results"""
        print("\n" + "=" * 60)
        print("🔍 THREAT HUNTING RESULTS")
        print("=" * 60)
        
        if "error" in results:
            print(f"❌ Threat hunt failed: {results['error']}")
            return
        
        print(f"🎯 Hunt Target: {results.get('target', 'Unknown')}")
        print(f"⏱️ Hunt Duration: {results.get('duration_minutes', 0)} minutes")
        
        # Threats detected
        threats = results.get('threats_detected', [])
        if threats:
            print(f"\n🚨 Threats Detected: {len(threats)}")
            for threat in threats[:3]:  # Show top 3
                print(f"  • {threat.get('name', 'Unknown')} - {threat.get('severity', 'Unknown')}")
        else:
            print("\n✅ No active threats detected")
        
        # IOCs found
        iocs = results.get('iocs_found', [])
        if iocs:
            print(f"\n🔍 Indicators of Compromise: {len(iocs)}")
        
        # Recommendations
        recommendations = results.get('recommendations', [])
        if recommendations:
            print(f"\n💡 Recommendations:")
            for rec in recommendations[:3]:
                print(f"  • {rec}")
    
    def _display_vuln_assessment_results(self, results: Dict[str, Any]):
        """Display vulnerability assessment results"""
        print("\n" + "=" * 60)
        print("🎯 VULNERABILITY ASSESSMENT RESULTS")  
        print("=" * 60)
        
        if "error" in results:
            print(f"❌ Assessment failed: {results['error']}")
            return
        
        print(f"🎯 Target: {results.get('target', 'Unknown')}")
        print(f"⏱️ Scan Duration: {results.get('duration_minutes', 0)} minutes")
        
        # Vulnerability summary
        vulns = results.get('vulnerabilities', [])
        if vulns:
            critical = len([v for v in vulns if v.get('severity') == 'critical'])
            high = len([v for v in vulns if v.get('severity') == 'high'])
            medium = len([v for v in vulns if v.get('severity') == 'medium'])
            low = len([v for v in vulns if v.get('severity') == 'low'])
            
            print(f"\n🚨 Vulnerabilities Found: {len(vulns)}")
            print(f"  • Critical: {critical}")
            print(f"  • High: {high}")
            print(f"  • Medium: {medium}")
            print(f"  • Low: {low}")
            
            # Show top critical/high vulns
            critical_high = [v for v in vulns if v.get('severity') in ['critical', 'high']]
            if critical_high:
                print(f"\n🔥 Priority Vulnerabilities:")
                for vuln in critical_high[:3]:
                    print(f"  • {vuln.get('name', 'Unknown')} ({vuln.get('severity', 'Unknown')})")
        else:
            print("\n✅ No vulnerabilities found")
        
        # Exploits available
        exploits = results.get('exploits_available', [])
        if exploits:
            print(f"\n💣 Exploits Available: {len(exploits)}")
            
        print("\n✅ Vulnerability assessment completed successfully!")
    
    async def run_autonomous_enterprise_scenario(self, duration_minutes: int = 30) -> Dict[str, Any]:
        """Run fully autonomous enterprise attack/defense scenario"""
        if not self.system_ready:
            return {"error": "System not initialized"}
        
        if not AUTONOMOUS_SCENARIO_AVAILABLE:
            return {"error": "Autonomous enterprise scenario not available"}
        
        if not self.autonomous_scenario_orchestrator:
            return {"error": "Autonomous scenario orchestrator not initialized"}
        
        self.logger.info(f"🎭 Starting autonomous enterprise scenario for {duration_minutes} minutes")
        
        try:
            # Run the autonomous enterprise scenario
            results = await self.autonomous_scenario_orchestrator.run_autonomous_scenario(
                duration_minutes=duration_minutes
            )
            
            # Display results are handled by the orchestrator
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Autonomous enterprise scenario failed: {e}")
            return {"error": str(e)}
    
    async def run_ai_enhanced_enterprise_scenario(self, duration_minutes: int = 30) -> Dict[str, Any]:
        """Run AI-enhanced autonomous enterprise attack/defense scenario"""
        if not self.system_ready:
            return {"error": "System not initialized"}
        
        if not AI_ENHANCED_SCENARIO_AVAILABLE:
            return {"error": "AI-enhanced enterprise scenario not available"}
        
        if not self.ai_enhanced_scenario_orchestrator:
            return {"error": "AI-enhanced scenario orchestrator not initialized"}
        
        self.logger.info(f"🧠 Starting AI-enhanced autonomous enterprise scenario for {duration_minutes} minutes")
        
        try:
            # Run the AI-enhanced autonomous enterprise scenario
            results = await self.ai_enhanced_scenario_orchestrator.run_ai_enhanced_scenario(
                duration_minutes=duration_minutes
            )
            
            # Display results are handled by the orchestrator
            return results
            
        except Exception as e:
            self.logger.error(f"❌ AI-enhanced enterprise scenario failed: {e}")
            return {"error": str(e)}
    
    async def train_ai_models(self) -> bool:
        """Train the AI models for better performance"""
        if not TRAINING_AVAILABLE:
            self.logger.warning("⚠️ Training pipeline not available")
            print("⚠️ Training pipeline not available, continuing with existing models")
            return True
            
        self.logger.info("🧠 Starting AI model training...")
        
        try:
            # Run the training pipeline
            print("🧠 Running cybersecurity AI training pipeline...")
            await train_main()
            
            self.logger.info("✅ AI training completed successfully")
            print("🎓 AI models trained and ready for combat")
            return True
                
        except Exception as e:
            self.logger.error(f"❌ Training failed: {e}")
            print(f"⚠️ Training failed: {e}")
            print("Continuing with existing models...")
            return False
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            "system_ready": self.system_ready,
            "timestamp": asyncio.get_event_loop().time(),
        }
        
        try:
            # Docker status
            if await self._check_docker():
                status["docker"] = "available"
            else:
                status["docker"] = "unavailable"
            
            # Combat environment status
            status["combat_environment"] = {
                "initialized": hasattr(self.combat_env, 'red_team_config'),
                "red_team_ip": getattr(self.combat_env.red_team_config, 'ip', 'unknown') if hasattr(self.combat_env, 'red_team_config') else 'unknown',
                "blue_team_ip": getattr(self.combat_env.blue_team_config, 'ip', 'unknown') if hasattr(self.combat_env, 'blue_team_config') else 'unknown'
            }
            
            # AI agents status
            status["ai_agents"] = {
                "orchestrator_ready": hasattr(self.orchestrator, 'red_team_agent'),
                "red_team_agent": "initialized" if hasattr(self.orchestrator, 'red_team_agent') else "pending",
                "blue_team_agent": "initialized" if hasattr(self.orchestrator, 'blue_team_agent') else "pending"
            }
            
        except Exception as e:
            status["error"] = str(e)
        
        return status
    
    async def cleanup(self):
        """Clean up system resources"""
        self.logger.info("🧹 Cleaning up Archangel system...")
        
        try:
            # Cleanup orchestrator
            if hasattr(self.orchestrator, 'cleanup_combat_simulation'):
                await self.orchestrator.cleanup_combat_simulation()
            
            # Cleanup combat environment
            if hasattr(self.combat_env, 'cleanup'):
                await self.combat_env.cleanup()
            
            self.logger.info("✅ System cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")

async def run_ai_capabilities_demo():
    """Quick demonstration of AI-enhanced capabilities"""
    from core.ai_enhanced_agents import (
        AdvancedReasoningEngine, SmartTeamCoordinator, PredictiveThreatAnalyzer
    )
    from datetime import datetime
    
    # 1. Advanced AI Reasoning Demo
    print("\n🧠 1. ADVANCED AI REASONING ENGINE")
    print("-" * 40)
    
    reasoning_engine = AdvancedReasoningEngine()
    
    security_event = {
        'event_type': 'advanced_persistent_threat',
        'source_ip': '203.45.67.89',
        'target_ip': '10.0.1.100',
        'description': 'Multi-vector attack with lateral movement, credential harvesting, and data exfiltration patterns',
        'timestamp': '2025-08-01T11:30:00Z',
        'indicators': ['powershell_obfuscation', 'wmi_persistence', 'dns_tunneling'],
        'business_context': 'financial_services_enterprise'
    }
    
    decision = await reasoning_engine.analyze_security_event(security_event)
    
    print(f"🎯 AI Decision: {decision.action}")
    print(f"🧠 Confidence: {decision.confidence:.2f} (with {decision.uncertainty:.2f} uncertainty)")
    print(f"💡 AI Reasoning: {decision.reasoning}")
    print(f"🔮 Predicted Outcome: {decision.predicted_outcome}")
    print(f"📋 Recommended Actions: {', '.join(decision.recommended_followup[:2])}")
    
    # 2. Smart Team Coordination Demo
    print("\n🤖 2. MULTI-AGENT REINFORCEMENT LEARNING COORDINATION")
    print("-" * 55)
    
    # Red Team Coordination
    red_coordinator = SmartTeamCoordinator("red_team")
    red_coordinator.add_agent("ai_reconnaissance", ["network_scanning", "intelligence_gathering", "stealth_operations"])
    red_coordinator.add_agent("ai_exploitation", ["vulnerability_analysis", "exploit_development", "privilege_escalation"])
    red_coordinator.add_agent("ai_persistence", ["backdoor_creation", "lateral_movement", "data_exfiltration"])
    
    red_coordination = await red_coordinator.coordinate_team_action(
        "Execute AI-enhanced multi-stage enterprise compromise with maximum stealth",
        {"target": "enterprise_financial_network", "stealth_level": "maximum", "business_value": "critical"}
    )
    
    print(f"🔴 Red Team AI Coordination:")
    print(f"   Strategy: {red_coordination.team_strategy}")
    print(f"   Coordination Score: {red_coordination.coordination_score:.2f}")
    print(f"   Predicted Success: {red_coordination.predicted_success:.2f}")
    
    # Blue Team Coordination  
    blue_coordinator = SmartTeamCoordinator("blue_team")
    blue_coordinator.add_agent("ai_threat_hunter", ["anomaly_detection", "behavioral_analysis", "threat_intelligence"])
    blue_coordinator.add_agent("ai_incident_responder", ["forensics", "containment", "recovery"])
    blue_coordinator.add_agent("ai_security_analyst", ["predictive_analysis", "risk_assessment", "business_protection"])
    
    blue_coordination = await blue_coordinator.coordinate_team_action(
        "Deploy AI-enhanced predictive defense with real-time threat intelligence",
        {"threat_level": "advanced_persistent", "protection_priority": "critical_assets", "response_mode": "proactive"}
    )
    
    print(f"🔵 Blue Team AI Coordination:")
    print(f"   Strategy: {blue_coordination.team_strategy}")
    print(f"   Coordination Score: {blue_coordination.coordination_score:.2f}")
    print(f"   Predicted Success: {blue_coordination.predicted_success:.2f}")
    
    # 3. Predictive Threat Intelligence Demo
    print("\n🔮 3. PREDICTIVE THREAT INTELLIGENCE WITH UNCERTAINTY")
    print("-" * 50)
    
    threat_analyzer = PredictiveThreatAnalyzer()
    
    current_threats = [
        {'type': 'reconnaissance', 'stage': 'active_scanning', 'network_exposure': True, 'confidence': 0.87},
        {'type': 'initial_access', 'stage': 'credential_attack', 'unpatched_systems': True, 'confidence': 0.92},
        {'type': 'privilege_escalation', 'stage': 'exploitation', 'social_engineering': True, 'confidence': 0.78}
    ]
    
    threat_predictions = await threat_analyzer.predict_threat_evolution(current_threats, "24h")
    
    print(f"🎯 Generated {len(threat_predictions)} AI threat predictions:")
    for i, prediction in enumerate(threat_predictions[:3], 1):
        print(f"   {i}. {prediction.threat_type}")
        print(f"      Probability: {prediction.attack_probability:.2f}")
        print(f"      Business Impact: {prediction.business_impact}")
        print(f"      AI Countermeasures: {', '.join(prediction.countermeasures[:2])}")
    
    # 4. AI Learning and Adaptation Demo
    print("\n⚡ 4. REAL-TIME AI LEARNING AND ADAPTATION")
    print("-" * 45)
    
    novel_attack_patterns = [
        {'attack_type': 'ai_evasion_technique', 'sophistication': 0.9, 'success_rate': 0.7},
        {'attack_type': 'quantum_resistant_crypto_attack', 'sophistication': 0.95, 'success_rate': 0.6},
        {'attack_type': 'llm_prompt_injection_lateral', 'sophistication': 0.8, 'success_rate': 0.8}
    ]
    
    adaptation_result = await threat_analyzer.adaptive_threat_modeling(novel_attack_patterns)
    
    print(f"🧠 AI Adaptation Results:")
    print(f"   Novel Patterns Learned: {adaptation_result['patterns_learned']}")
    print(f"   Model Updated: {adaptation_result['model_updated']}")
    print(f"   Performance Improvement: +{adaptation_result['performance_improvement']:.1%}")
    print(f"   New Baseline Accuracy: {adaptation_result.get('new_accuracy', 0.85):.2f}")
    
    # 5. Uncertainty Quantification Demo
    print("\n🤔 5. UNCERTAINTY QUANTIFICATION AND CONFIDENCE INTERVALS")
    print("-" * 60)
    
    uncertainty_analysis = threat_analyzer.analyze_prediction_uncertainty(threat_predictions)
    
    print(f"📊 Uncertainty Analysis:")
    print(f"   Average Uncertainty: {uncertainty_analysis['average_uncertainty']:.2f}")
    print(f"   Max Uncertainty: {uncertainty_analysis['max_uncertainty']:.2f}")
    print(f"   Confidence Distribution:")
    for level, count in uncertainty_analysis['uncertainty_distribution'].items():
        print(f"      {level.capitalize()}: {count} predictions")
    print(f"   Confidence Calibration: {uncertainty_analysis['confidence_calibration']}")
    print(f"   AI Recommendations: {', '.join(uncertainty_analysis['recommendations'][:2])}")
    
    # Final Summary
    print("\n" + "=" * 70)
    print("🎉 AI-ENHANCED CAPABILITIES DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\n🏆 REVOLUTIONARY AI FEATURES DEMONSTRATED:")
    print("• 🧠 Advanced AI Reasoning with semantic understanding")
    print("• 🤖 Multi-Agent Reinforcement Learning coordination")
    print("• 🔮 Predictive threat intelligence with uncertainty quantification")
    print("• ⚡ Real-time learning and adaptation to novel attacks")
    print("• 🎯 Business-aware decision making with risk assessment")
    print("• 🤔 Uncertainty quantification and confidence intervals")
    print("• 💡 Explainable AI with human-readable reasoning")
    print("• 📊 Meta-learning for rapid adaptation to new threats")
    print(f"\n🚀 This represents the world's most advanced AI-driven cybersecurity platform!")
    print("🎯 Ready for Black Hat conference demonstration")
    print("🧠 Combining 8 cutting-edge AI techniques in practical implementation")

async def run_enhanced_docker_orchestration():
    """Run enhanced multi-container Docker orchestration"""
    import subprocess
    import yaml
    
    print("🐳 Starting enhanced Docker orchestration...")
    print("📊 This will launch:")
    print("• 🔴 AI-Controlled Kali Linux containers (full penetration testing suite)")
    print("• 🔵 Specialized Blue Team containers (SOC, IR, threat hunting)")
    print("• 🤖 Hugging Face model serving infrastructure")
    print("• 🏢 Realistic enterprise target environment")
    print("• 📈 Monitoring and observability stack")
    print("")
    print("🔴 RED TEAM: Autonomous AI agents controlling real Kali Linux tools:")
    print("   • nmap, masscan, nikto, gobuster, sqlmap, hydra, metasploit")
    print("   • AI decides which tools to use and how to use them")
    print("   • Real penetration testing methodology executed by AI")
    print("")
    
    try:
        # Use enhanced Docker Compose configuration
        cmd = [
            "docker-compose",
            "-f", "docker-compose-enhanced.yml",
            "up", "-d",
            "--scale", "red-team-reconnaissance=2",
            "--scale", "blue-team-threat-hunter=2"
        ]
        
        print(f"🚀 Executing: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Enhanced Docker orchestration started successfully!")
            print("\n📊 Services available:")
            print("• Archangel Orchestrator: http://localhost:8000")
            print("• Model Server: http://localhost:8080")
            print("• Grafana Dashboard: http://localhost:3000")
            print("• Prometheus Metrics: http://localhost:9090")
            print("• Vector Database: http://localhost:6333")
            
            print("\n🔄 To monitor containers: docker-compose -f docker-compose-enhanced.yml logs -f")
            print("🛑 To stop: docker-compose -f docker-compose-enhanced.yml down")
        else:
            print(f"❌ Failed to start enhanced orchestration: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Docker orchestration failed: {e}")
        print("💡 Make sure Docker and docker-compose are installed and running")

async def run_model_training_pipeline():
    """Run specialized model training pipeline"""
    try:
        from training.specialized_cybersec_trainer import train_cybersecurity_models
        
        print("🧠 Starting comprehensive cybersecurity model training...")
        print("📊 This will train:")
        print("• 🔴 Red Team Models: reconnaissance, exploitation, persistence, social engineering")
        print("• 🔵 Blue Team Models: threat detection, incident response, threat hunting, vulnerability assessment")
        print("• 🤖 Advanced AI techniques: LoRA fine-tuning, specialized datasets, performance optimization")
        
        # Run the training pipeline
        await train_cybersecurity_models()
        
        print("✅ Model training pipeline completed!")
        print("\n📁 Trained models available in:")
        print("• ./trained_models/red_team_*/")
        print("• ./trained_models/blue_team_*/")
        
    except ImportError as e:
        print(f"❌ Training modules not available: {e}")
        print("💡 Install training dependencies: pip install -r requirements-ai-enhanced.txt")
    except Exception as e:
        print(f"❌ Model training failed: {e}")

async def run_model_server():
    """Run Hugging Face model serving infrastructure"""
    try:
        from core.enhanced_hf_model_manager import get_model_manager
        
        print("🤖 Starting Hugging Face model server...")
        
        # Initialize model manager
        manager = await get_model_manager()
        status = manager.get_model_ecosystem_status()
        
        print("📊 Model Ecosystem Status:")
        print(f"• Total Models: {status['total_models']}")
        print(f"• Model Categories: {status['model_categories']}")
        print(f"• Specializations: {list(status['specializations'].keys())}")
        
        # Start model serving (simplified for demo)
        print("🚀 Model server would start here with FastAPI/Gradio interface")
        print("📍 Server endpoints would be available at:")
        print("• http://localhost:8080/models/")
        print("• http://localhost:8080/predict/")
        print("• http://localhost:8080/docs/")
        
    except ImportError as e:
        print(f"❌ Model server modules not available: {e}")
    except Exception as e:
        print(f"❌ Model server failed: {e}")

async def run_orchestrator_mode():
    """Run in Docker orchestrator mode (internal use)"""
    print("🎛️ Archangel Orchestrator starting...")
    print("🐳 Managing container lifecycle and AI coordination")
    
    # This would be the main orchestrator process running inside the container
    # For now, just demonstrate the concept
    import asyncio
    
    try:
        # Simulate orchestrator operations
        while True:
            print("🔄 Orchestrator heartbeat - managing containers and AI models")
            await asyncio.sleep(30)
            
    except KeyboardInterrupt:
        print("🛑 Orchestrator shutting down")

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Archangel Autonomous Security System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 archangel.py                    # Run 10-minute combat demo
  python3 archangel.py --duration 30      # Run 30-minute combat session
  python3 archangel.py --training          # Train AI models first
  python3 archangel.py --status           # Show system status
        """
    )
    
    parser.add_argument("--duration", type=int, default=10, 
                       help="Combat duration in minutes (default: 10)")
    parser.add_argument("--training", action="store_true",
                       help="Train AI models before combat")
    parser.add_argument("--status", action="store_true",
                       help="Show system status and exit")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    # Elite MCP operations
    parser.add_argument("--operation", type=str,
                       help="Natural language security operation (requires MCP)")
    parser.add_argument("--hunt", type=str,
                       help="Threat hunting target description (requires MCP)")
    parser.add_argument("--assess", type=str,
                       help="Vulnerability assessment target (requires MCP)")
    parser.add_argument("--disable-mcp", action="store_true",
                       help="Disable MCP integration (standard mode only)")
    parser.add_argument("--elite", action="store_true",
                       help="Enable all elite capabilities (MCP + external resources)")
    parser.add_argument("--enterprise", action="store_true",
                       help="Run autonomous enterprise attack/defense scenario")
    parser.add_argument("--ai-enhanced", action="store_true",
                       help="Run AI-enhanced autonomous enterprise scenario with advanced reasoning")
    parser.add_argument("--demo-ai", action="store_true",
                       help="Quick demo of AI-enhanced capabilities (no containers)")
    parser.add_argument("--enhanced-docker", action="store_true",
                       help="Use enhanced multi-container Docker orchestration")
    parser.add_argument("--train-models", action="store_true",
                       help="Train specialized red/blue team cybersecurity models")
    parser.add_argument("--orchestrator-mode", action="store_true",
                       help="Run in Docker orchestrator mode (internal use)")
    parser.add_argument("--model-server", action="store_true",
                       help="Start Hugging Face model serving infrastructure")
    
    args = parser.parse_args()
    
    # Setup verbose logging if requested
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize system with MCP settings
    enable_mcp = not args.disable_mcp
    if args.elite:
        enable_mcp = True
        
    system = ArchangelSystem(enable_mcp=enable_mcp)
    
    try:
        # Handle status request
        if args.status:
            print("📊 Archangel System Status")
            print("-" * 30)
            
            # Quick status without full initialization
            quick_status = {
                "archangel_main": "available",
                "docker_available": await system._check_docker(),
                "training_available": TRAINING_AVAILABLE,
                "mcp_integration": MCP_AVAILABLE,
                "elite_mode": system.mcp_enabled,
                "components": {
                    "live_combat_environment": "available",
                    "combat_orchestrator": "available", 
                    "ai_agents": "available",
                    "external_resources": "available" if MCP_AVAILABLE else "disabled"
                }
            }
            print(json.dumps(quick_status, indent=2))
            return 0
        
        # Handle elite operations first
        if args.operation or args.hunt or args.assess:
            if not MCP_AVAILABLE:
                print("❌ Elite operations require MCP integration")
                print("Please install MCP dependencies: pip install -r requirements-mcp.txt")
                return 1
                
            print("\n🚀 ARCHANGEL ELITE SECURITY OPERATIONS")
            print("🍎 M2 MacBook + External Resources + AI")
            print("=" * 55)
            
            # Initialize system
            print("🔧 Initializing elite AI security system...")
            if not await system.initialize_system():
                print("❌ System initialization failed")
                return 1
            
            # Handle specific operation
            if args.operation:
                print(f"\n🎯 Processing: {args.operation}")
                results = await system.run_security_operation(args.operation)
            elif args.hunt:
                print(f"\n🔍 Threat Hunting: {args.hunt}")
                results = await system.run_threat_hunt(args.hunt)
            elif args.assess:
                print(f"\n🎯 Vulnerability Assessment: {args.assess}")
                results = await system.run_vulnerability_assessment(args.assess)
            
            if "error" not in results:
                print("\n🎉 Elite operation completed successfully!")
            
            return 0
        
        # Handle autonomous enterprise scenario
        if args.enterprise:
            print("\n🎭 ARCHANGEL AUTONOMOUS ENTERPRISE SCENARIO")
            print("🔴 Elite Red Team vs 🔵 Enterprise Blue Team")
            print("🏢 Realistic Enterprise with Valuable Data")
            print("=" * 60)
            
            if not AUTONOMOUS_SCENARIO_AVAILABLE:
                print("❌ Autonomous enterprise scenario not available")
                print("Please ensure the scenarios module is properly installed")
                return 1
            
            # Initialize system
            print("🔧 Initializing autonomous enterprise environment...")
            if not await system.initialize_system():
                print("❌ System initialization failed")
                return 1
            
            # Run autonomous enterprise scenario
            print(f"\n🎭 Starting fully autonomous enterprise scenario...")
            print(f"Duration: {args.duration} minutes")
            print("Press Ctrl+C to stop early")
            
            results = await system.run_autonomous_enterprise_scenario(args.duration)
            
            if "error" not in results:
                print("\n🎉 Autonomous enterprise scenario completed successfully!")
                print("\nWhat you just witnessed:")
                print("• 🔴 Elite hacking group autonomously attacking enterprise")
                print("• 🔵 Enterprise security team autonomously defending data")
                print("• 🏢 Realistic enterprise with valuable financial/HR/IP data")
                print("• 🎯 Red team attempting to discover and exfiltrate data")
                print("• 🛡️ Blue team detecting threats and protecting assets")
                print("• 🧠 Fully autonomous AI agents with no human commands")
                print("\nThis demonstrates elite-level autonomous cybersecurity AI!")
            
            return 0
        
        # Handle AI demo
        if args.demo_ai:
            print("\n🧠🎭 ARCHANGEL AI-ENHANCED CAPABILITIES DEMONSTRATION")
            print("=" * 70)
            print("🚀 World's Most Advanced AI-Driven Cybersecurity Platform")
            print("=" * 70)
            
            try:
                from core.ai_enhanced_agents import (
                    AdvancedReasoningEngine, SmartTeamCoordinator, PredictiveThreatAnalyzer
                )
                
                # Quick AI capabilities demo
                await run_ai_capabilities_demo()
                return 0
                
            except ImportError as e:
                print(f"❌ AI enhancement modules not available: {e}")
                return 1
        
        # Handle enhanced Docker orchestration
        if args.enhanced_docker:
            print("\n🐳 ARCHANGEL ENHANCED DOCKER ORCHESTRATION")
            print("🔴🔵 AI-Controlled Kali Linux vs Advanced Blue Team")
            print("🤖 Full Kali Linux + Specialized HF Models + Service Mesh")
            print("=" * 65)
            
            await run_enhanced_docker_orchestration()
            return 0
        
        # Handle model training
        if args.train_models:
            print("\n🧠 ARCHANGEL SPECIALIZED MODEL TRAINING")
            print("🔴 Red Team Offensive Models + 🔵 Blue Team Defensive Models")
            print("🤖 Hugging Face Integration + Advanced Cybersecurity Training")
            print("=" * 70)
            
            await run_model_training_pipeline()
            return 0
        
        # Handle model server
        if args.model_server:
            print("\n🤖 ARCHANGEL HUGGING FACE MODEL SERVER")
            print("🚀 Specialized Cybersecurity Model Serving Infrastructure")
            print("=" * 60)
            
            await run_model_server()
            return 0
        
        # Handle orchestrator mode
        if args.orchestrator_mode:
            print("\n🎛️ ARCHANGEL ORCHESTRATOR MODE")
            print("🐳 Docker Container Command & Control")
            print("=" * 40)
            
            await run_orchestrator_mode()
            return 0
        
        # Handle AI-enhanced autonomous enterprise scenario
        if args.ai_enhanced:
            print("\n🧠🎭 ARCHANGEL AI-ENHANCED AUTONOMOUS ENTERPRISE SCENARIO")
            print("🔴🧠 AI-Enhanced Elite Red Team vs 🔵🧠 AI-Enhanced Enterprise Blue Team")
            print("🏢🤖 Realistic Enterprise with AI-Protected Data + Advanced Reasoning")
            print("🧠 AI Features: Reasoning, Prediction, Adaptation, Learning, Uncertainty")
            print("=" * 75)
            
            if not AI_ENHANCED_SCENARIO_AVAILABLE:
                print("❌ AI-enhanced enterprise scenario not available")
                print("Please ensure the AI enhancement modules are properly installed")
                return 1
            
            # Initialize system
            print("🔧🧠 Initializing AI-enhanced autonomous enterprise environment...")
            if not await system.initialize_system():
                print("❌ System initialization failed")
                return 1
            
            # Run AI-enhanced autonomous enterprise scenario
            print(f"\n🎭🧠 Starting fully AI-enhanced autonomous enterprise scenario...")
            print(f"Duration: {args.duration} minutes")
            print("Press Ctrl+C to stop early")
            
            results = await system.run_ai_enhanced_enterprise_scenario(args.duration)
            
            if "error" not in results:
                print("\n🎉🧠 AI-Enhanced autonomous enterprise scenario completed successfully!")
                print("\nWhat you just witnessed:")
                print("• 🔴🧠 Elite AI-enhanced hacking group with advanced reasoning capabilities")
                print("• 🔵🧠 AI-enhanced enterprise security team with predictive intelligence") 
                print("• 🏢🤖 Realistic enterprise with AI-protected valuable data")
                print("• 🎯🧠 Red team using AI reasoning to discover and exfiltrate data")
                print("• 🛡️🧠 Blue team using predictive AI to detect threats and protect assets")
                print("• 🤖⚡ Advanced AI coordination, adaptation, and real-time learning")
                print("• 🔮📊 Uncertainty quantification and predictive threat analysis")
                print("• 🧠💡 Meta-learning and strategy evolution during combat")
                print("\nThis demonstrates the world's most advanced AI-driven cybersecurity platform!")
            
            return 0
        
        # Main application flow
        print("\n🛡️ ARCHANGEL AUTONOMOUS SECURITY SYSTEM")
        if system.mcp_enabled:
            print("🚀 ELITE MODE: External Resources + AI")
        else:
            print("📋 STANDARD MODE: Basic Autonomous Operations")
        print("🍎 Optimized for M2 MacBook (16GB RAM)")
        print("=" * 55)
        
        # Initialize system
        init_msg = "elite AI security system" if system.mcp_enabled else "autonomous AI security system"
        print(f"🚀 Initializing {init_msg}...")
        if not await system.initialize_system():
            print("❌ System initialization failed")
            print("\nTroubleshooting:")
            print("• Ensure Docker Desktop is running")
            print("• Check available memory (need ~4GB free)")
            print("• Run: ./cleanup_combat.sh")
            if system.mcp_enabled:
                print("• For elite mode: pip install -r requirements-mcp.txt")
            return 1
        
        # Handle training request
        if args.training:
            print("🧠 Training AI models for better performance...")
            if not await system.train_ai_models():
                print("⚠️ Training failed, continuing with existing models") 
        
        # Run live combat
        print(f"\n🥊 Starting autonomous red vs blue team combat...")
        print(f"Duration: {args.duration} minutes")
        print("Press Ctrl+C to stop early")
        
        results = await system.run_live_combat(args.duration)
        
        if "error" not in results:
            if system.mcp_enabled:
                print("\n🎉 Elite autonomous combat session completed successfully!")
                print("\nWhat you just witnessed:")
                print("• 🤖 AI agents with external security tool access")
                print("• 🌐 Real-world threat intelligence integration")
                print("• 🛠️ Professional-grade security tool automation")
                print("• 🧠 Elite-level reasoning with external resources")
                print("• 🔄 Strategic evolution using live threat data")
                print("\nThis represents elite-level AI-driven cybersecurity!")
            else:
                print("\n🎉 Autonomous combat session completed successfully!")
                print("\nWhat you just witnessed:")
                print("• 🤖 AI agents thinking and reasoning about security")
                print("• 🎯 Autonomous attack and defense strategies")
                print("• 🧠 Real-time learning and adaptation")
                print("• 🔄 Strategic evolution over time")
                print("\nThis represents a breakthrough in AI-driven cybersecurity!")
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Combat stopped by user")
    except Exception as e:
        print(f"❌ System error: {e}")
        return 1
    finally:
        await system.cleanup()
    
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))