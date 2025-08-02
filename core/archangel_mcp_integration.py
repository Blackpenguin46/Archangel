"""
Archangel MCP Integration Layer
Connects existing Archangel agents with the MCP architecture
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import json

from .mcp_integration_architecture import (
    MCPOrchestrator, ArchangelMCPIntegration, TeamType,
    create_mcp_orchestrator
)
from .autonomous_security_agents import (
    AutonomousSecurityAgent, BlueTeamDefenderAgent, RedTeamAttackerAgent,
    AutonomousSecurityOrchestrator, AgentRole
)

class EnhancedArchangelAgent(AutonomousSecurityAgent):
    """Enhanced Archangel agent with MCP capabilities"""
    
    def __init__(self, agent_id: str, role: AgentRole, 
                 mcp_integration: ArchangelMCPIntegration,
                 hf_token: Optional[str] = None, use_container: bool = True):
        super().__init__(agent_id, role, hf_token, use_container)
        
        self.mcp_integration = mcp_integration
        self.mcp_token: Optional[str] = None
        self.mcp_capabilities: List[str] = []
        
        # Determine team type from role
        if role in [AgentRole.RED_TEAM_ATTACKER]:
            self.team_type = TeamType.RED_TEAM
        elif role in [AgentRole.BLUE_TEAM_DEFENDER, AgentRole.THREAT_HUNTER, 
                     AgentRole.INCIDENT_RESPONDER]:
            self.team_type = TeamType.BLUE_TEAM
        else:
            self.team_type = TeamType.NEUTRAL
    
    async def initialize(self) -> bool:
        """Initialize agent with MCP capabilities"""
        # Initialize base agent
        base_initialized = await super().initialize()
        if not base_initialized:
            return False
        
        try:
            # Register with MCP system
            self.mcp_token = await self.mcp_integration.register_agent(
                self.agent_id, self.team_type
            )
            
            # Get available MCP capabilities
            capabilities_response = await self.mcp_integration.mcp_orchestrator.get_agent_capabilities(
                self.mcp_token
            )
            
            if "available_tools" in capabilities_response:
                self.mcp_capabilities = [
                    tool["name"] for tool in capabilities_response["available_tools"]
                ]
            
            self.logger.info(f"🔗 MCP integration complete: {len(self.mcp_capabilities)} tools available")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ MCP integration failed: {e}")
            return False
    
    async def execute_mcp_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute MCP tool through integration layer"""
        if not self.mcp_token:
            return {"error": "MCP not initialized"}
        
        try:
            result = await self.mcp_integration.mcp_orchestrator.execute_agent_tool(
                self.mcp_token, tool_name, parameters
            )
            
            self.logger.info(f"🔧 MCP tool executed: {tool_name}")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ MCP tool execution failed: {tool_name} - {e}")
            return {"error": f"Tool execution failed: {e}"}
    
    async def enhanced_reconnaissance(self, target: str) -> Dict[str, Any]:
        """Enhanced reconnaissance using MCP tools"""
        results = {
            "target": target,
            "reconnaissance_type": "enhanced_mcp",
            "findings": [],
            "external_intelligence": []
        }
        
        # Use base reconnaissance
        base_recon = await self.execute_autonomous_operation(
            f"Perform reconnaissance on {target}",
            {"target": target, "environment": "test"}
        )
        results["base_reconnaissance"] = base_recon.results
        
        # Enhanced with MCP tools
        if "host_search" in self.mcp_capabilities:
            shodan_result = await self.execute_mcp_tool("host_search", {"query": target})
            if shodan_result.get("success"):
                results["external_intelligence"].append({
                    "source": "shodan",
                    "data": shodan_result["result"]
                })
        
        if "exploit_search" in self.mcp_capabilities:
            exploit_result = await self.execute_mcp_tool("exploit_search", {"query": target})
            if exploit_result.get("success"):
                results["external_intelligence"].append({
                    "source": "exploit_database",
                    "data": exploit_result["result"]
                })
        
        return results
    
    async def enhanced_threat_analysis(self, indicators: List[str]) -> Dict[str, Any]:
        """Enhanced threat analysis using MCP tools"""
        results = {
            "indicators": indicators,
            "analysis_type": "enhanced_mcp",
            "threat_intelligence": [],
            "risk_assessment": {}
        }
        
        # Base threat analysis
        base_analysis = await self.execute_autonomous_operation(
            "Analyze threat indicators",
            {"indicators": indicators}
        )
        results["base_analysis"] = base_analysis.results
        
        # Enhanced with MCP tools
        for indicator in indicators:
            # File hash analysis
            if len(indicator) in [32, 40, 64] and all(c in '0123456789abcdefABCDEF' for c in indicator):
                if "file_analysis" in self.mcp_capabilities:
                    vt_result = await self.execute_mcp_tool("file_analysis", {"hash": indicator})
                    if vt_result.get("success"):
                        results["threat_intelligence"].append({
                            "indicator": indicator,
                            "type": "file_hash",
                            "source": "virustotal",
                            "analysis": vt_result["result"]
                        })
            
            # IOC search
            if "ioc_search" in self.mcp_capabilities:
                ioc_result = await self.execute_mcp_tool("ioc_search", {"terms": [indicator]})
                if ioc_result.get("success"):
                    results["threat_intelligence"].append({
                        "indicator": indicator,
                        "type": "ioc",
                        "source": "threat_intelligence",
                        "analysis": ioc_result["result"]
                    })
        
        # Calculate enhanced risk score
        risk_score = len([ti for ti in results["threat_intelligence"] 
                         if ti.get("analysis", {}).get("malicious", False)])
        results["risk_assessment"] = {
            "score": min(risk_score * 2.5, 10.0),
            "level": "high" if risk_score >= 3 else "medium" if risk_score >= 1 else "low",
            "confidence": 0.8 if results["threat_intelligence"] else 0.3
        }
        
        return results

class EnhancedRedTeamAgent(EnhancedArchangelAgent):
    """Enhanced Red Team agent with offensive MCP tools"""
    
    def __init__(self, agent_id: str, mcp_integration: ArchangelMCPIntegration,
                 hf_token: Optional[str] = None, use_container: bool = True):
        super().__init__(agent_id, AgentRole.RED_TEAM_ATTACKER, mcp_integration, hf_token, use_container)
    
    async def advanced_vulnerability_assessment(self, target: str) -> Dict[str, Any]:
        """Advanced vulnerability assessment using multiple MCP tools"""
        results = {
            "target": target,
            "assessment_type": "advanced_mcp",
            "vulnerabilities": [],
            "exploitation_potential": []
        }
        
        # Nuclei vulnerability scanning
        if "vulnerability_scanning" in self.mcp_capabilities:
            nuclei_result = await self.execute_mcp_tool(
                "vulnerability_scanning", 
                {"target": target, "templates": "all"}
            )
            if nuclei_result.get("success"):
                results["vulnerabilities"].extend(nuclei_result["result"].get("vulnerabilities", []))
        
        # Search for exploits for found vulnerabilities
        for vuln in results["vulnerabilities"]:
            cve = vuln.get("cve")
            if cve and "exploit_search" in self.mcp_capabilities:
                exploit_result = await self.execute_mcp_tool("exploit_search", {"query": cve})
                if exploit_result.get("success"):
                    results["exploitation_potential"].append({
                        "vulnerability": cve,
                        "exploits": exploit_result["result"].get("exploits_found", [])
                    })
        
        return results
    
    async def simulated_attack_chain(self, target: str) -> Dict[str, Any]:
        """Simulate complete attack chain (in safe environment only)"""
        if not self._is_test_environment({"target": target, "sandbox": True}):
            return {"error": "Attack simulation only allowed in test environment"}
        
        attack_chain = {
            "target": target,
            "chain_type": "simulated_attack",
            "phases": [],
            "timeline": []
        }
        
        # Phase 1: Reconnaissance
        recon_result = await self.enhanced_reconnaissance(target)
        attack_chain["phases"].append({
            "phase": "reconnaissance",
            "result": recon_result,
            "success": bool(recon_result.get("external_intelligence"))
        })
        
        # Phase 2: Vulnerability Assessment
        vuln_result = await self.advanced_vulnerability_assessment(target)
        attack_chain["phases"].append({
            "phase": "vulnerability_assessment",
            "result": vuln_result,
            "success": bool(vuln_result.get("vulnerabilities"))
        })
        
        # Phase 3: Simulated Exploitation (safe)
        if vuln_result.get("vulnerabilities"):
            exploit_simulation = {
                "simulated": True,
                "message": "Safe exploitation simulation would occur here",
                "vulnerabilities_tested": len(vuln_result["vulnerabilities"]),
                "exploitation_success_probability": 0.7
            }
            attack_chain["phases"].append({
                "phase": "exploitation_simulation",
                "result": exploit_simulation,
                "success": True
            })
        
        return attack_chain

class EnhancedBlueTeamAgent(EnhancedArchangelAgent):
    """Enhanced Blue Team agent with defensive MCP tools"""
    
    def __init__(self, agent_id: str, mcp_integration: ArchangelMCPIntegration,
                 hf_token: Optional[str] = None, use_container: bool = True):
        super().__init__(agent_id, AgentRole.BLUE_TEAM_DEFENDER, mcp_integration, hf_token, use_container)
    
    async def comprehensive_threat_hunting(self, hunt_hypothesis: str) -> Dict[str, Any]:
        """Comprehensive threat hunting using multiple MCP tools"""
        hunt_results = {
            "hypothesis": hunt_hypothesis,
            "hunt_type": "comprehensive_mcp",
            "evidence": [],
            "indicators_found": [],
            "threat_assessment": {}
        }
        
        # Threat hunting with SIEM tools
        if "threat_hunting" in self.mcp_capabilities:
            siem_result = await self.execute_mcp_tool(
                "threat_hunting",
                {"query": hunt_hypothesis, "time_range": "24h"}
            )
            if siem_result.get("success"):
                hunt_results["evidence"].append({
                    "source": "siem",
                    "findings": siem_result["result"].get("threats_found", [])
                })
        
        # Memory analysis for artifacts
        if "endpoint_query" in self.mcp_capabilities:
            osquery_result = await self.execute_mcp_tool(
                "endpoint_query",
                {"query": f"SELECT * FROM processes WHERE name LIKE '%{hunt_hypothesis}%'"}
            )
            if osquery_result.get("success"):
                hunt_results["evidence"].append({
                    "source": "osquery",
                    "findings": osquery_result["result"]
                })
        
        # YARA rule matching
        if "rule_matching" in self.mcp_capabilities:
            yara_result = await self.execute_mcp_tool(
                "rule_matching",
                {"rules": f"rule hunt_{hunt_hypothesis} {{ strings: $a = \"{hunt_hypothesis}\" condition: $a }}"}
            )
            if yara_result.get("success"):
                hunt_results["evidence"].append({
                    "source": "yara",
                    "findings": yara_result["result"]
                })
        
        # Compile threat assessment
        total_evidence = sum(len(e.get("findings", [])) for e in hunt_results["evidence"])
        hunt_results["threat_assessment"] = {
            "evidence_count": total_evidence,
            "confidence": min(total_evidence * 0.2, 1.0),
            "threat_level": "high" if total_evidence >= 5 else "medium" if total_evidence >= 2 else "low"
        }
        
        return hunt_results
    
    async def incident_response_workflow(self, incident_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute comprehensive incident response workflow"""
        ir_workflow = {
            "incident_id": incident_data.get("incident_id", "unknown"),
            "workflow_type": "enhanced_mcp",
            "phases": [],
            "timeline": [],
            "containment_actions": []
        }
        
        # Phase 1: Initial Assessment
        threat_indicators = incident_data.get("indicators", [])
        assessment_result = await self.enhanced_threat_analysis(threat_indicators)
        ir_workflow["phases"].append({
            "phase": "initial_assessment",
            "result": assessment_result,
            "timestamp": asyncio.get_event_loop().time()
        })
        
        # Phase 2: Threat Hunting
        if incident_data.get("hypothesis"):
            hunt_result = await self.comprehensive_threat_hunting(incident_data["hypothesis"])
            ir_workflow["phases"].append({
                "phase": "threat_hunting",
                "result": hunt_result,
                "timestamp": asyncio.get_event_loop().time()
            })
        
        # Phase 3: Containment Recommendations
        risk_level = assessment_result.get("risk_assessment", {}).get("level", "low")
        if risk_level in ["high", "medium"]:
            containment_actions = [
                "Isolate affected systems",
                "Block malicious indicators",
                "Reset compromised credentials",
                "Enable enhanced monitoring"
            ]
            ir_workflow["containment_actions"] = containment_actions
        
        return ir_workflow

class EnhancedArchangelOrchestrator:
    """Enhanced orchestrator managing MCP-enabled agents"""
    
    def __init__(self, hf_token: Optional[str] = None):
        self.hf_token = hf_token
        
        # Initialize MCP orchestrator
        self.mcp_orchestrator = create_mcp_orchestrator()
        self.mcp_integration = ArchangelMCPIntegration(self.mcp_orchestrator)
        
        # Enhanced agents
        self.enhanced_agents: Dict[str, EnhancedArchangelAgent] = {}
        
        self.logger = logging.getLogger("enhanced_orchestrator")
    
    async def initialize_enhanced_system(self) -> bool:
        """Initialize the enhanced MCP-enabled system"""
        self.logger.info("🚀 Initializing Enhanced Archangel System with MCP")
        
        try:
            # Initialize MCP architecture
            if not await self.mcp_orchestrator.initialize_mcp_architecture():
                return False
            
            # Create enhanced agents
            enhanced_red_agent = EnhancedRedTeamAgent(
                "enhanced_red_001", self.mcp_integration, self.hf_token
            )
            enhanced_blue_agent = EnhancedBlueTeamAgent(
                "enhanced_blue_001", self.mcp_integration, self.hf_token
            )
            
            # Initialize enhanced agents
            agents_to_init = [
                ("red_team", enhanced_red_agent),
                ("blue_team", enhanced_blue_agent)
            ]
            
            for name, agent in agents_to_init:
                if await agent.initialize():
                    self.enhanced_agents[name] = agent
                    self.logger.info(f"✅ Enhanced {name} agent ready with MCP")
                else:
                    self.logger.error(f"❌ Failed to initialize enhanced {name} agent")
            
            return len(self.enhanced_agents) > 0
            
        except Exception as e:
            self.logger.error(f"Enhanced system initialization failed: {e}")
            return False
    
    async def run_enhanced_security_exercise(self) -> Dict[str, Any]:
        """Run enhanced security exercise with MCP capabilities"""
        exercise_id = f"enhanced_exercise_{int(asyncio.get_event_loop().time())}"
        
        self.logger.info(f"🎮 Starting enhanced security exercise: {exercise_id}")
        
        results = {
            "exercise_id": exercise_id,
            "exercise_type": "enhanced_mcp_enabled",
            "red_team_operations": [],
            "blue_team_operations": [],
            "mcp_tool_usage": {},
            "learning_outcomes": []
        }
        
        try:
            # Red team enhanced operations
            if "red_team" in self.enhanced_agents:
                red_agent = self.enhanced_agents["red_team"]
                
                # Enhanced reconnaissance
                recon_result = await red_agent.enhanced_reconnaissance("test_target_192.168.1.100")
                results["red_team_operations"].append({
                    "operation": "enhanced_reconnaissance",
                    "result": recon_result
                })
                
                # Simulated attack chain
                attack_result = await red_agent.simulated_attack_chain("test_target_192.168.1.100")
                results["red_team_operations"].append({
                    "operation": "simulated_attack_chain",
                    "result": attack_result
                })
            
            # Blue team enhanced operations
            if "blue_team" in self.enhanced_agents:
                blue_agent = self.enhanced_agents["blue_team"]
                
                # Comprehensive threat hunting
                hunt_result = await blue_agent.comprehensive_threat_hunting("suspicious_process")
                results["blue_team_operations"].append({
                    "operation": "comprehensive_threat_hunting",
                    "result": hunt_result
                })
                
                # Incident response workflow
                incident_data = {
                    "incident_id": "INC-001",
                    "indicators": ["192.168.1.100", "malicious.exe", "suspicious_domain.com"],
                    "hypothesis": "lateral_movement"
                }
                ir_result = await blue_agent.incident_response_workflow(incident_data)
                results["blue_team_operations"].append({
                    "operation": "incident_response_workflow",
                    "result": ir_result
                })
            
            # Get MCP system status
            mcp_status = await self.mcp_orchestrator.get_system_status()
            results["mcp_system_status"] = mcp_status
            
            results["status"] = "completed"
            self.logger.info(f"✅ Enhanced security exercise completed")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Enhanced security exercise failed: {e}")
            results["error"] = str(e)
            results["status"] = "failed"
            return results
    
    async def cleanup(self):
        """Cleanup enhanced system"""
        self.logger.info("🧹 Cleaning up enhanced system...")
        
        # Cleanup enhanced agents
        for agent in self.enhanced_agents.values():
            await agent.cleanup()
        
        # Cleanup MCP orchestrator
        await self.mcp_orchestrator.shutdown_mcp_architecture()
        
        self.enhanced_agents.clear()
        self.logger.info("✅ Enhanced system cleanup completed")

# Factory functions

def create_enhanced_archangel_orchestrator(hf_token: Optional[str] = None) -> EnhancedArchangelOrchestrator:
    """Create enhanced Archangel orchestrator with MCP capabilities"""
    return EnhancedArchangelOrchestrator(hf_token)

async def demo_enhanced_archangel_system(hf_token: Optional[str] = None) -> Dict[str, Any]:
    """Demonstrate enhanced Archangel system with MCP integration"""
    orchestrator = create_enhanced_archangel_orchestrator(hf_token)
    
    # Initialize enhanced system
    if not await orchestrator.initialize_enhanced_system():
        return {"error": "Failed to initialize enhanced system"}
    
    # Run enhanced security exercise
    exercise_results = await orchestrator.run_enhanced_security_exercise()
    
    # Cleanup
    await orchestrator.cleanup()
    
    return {
        "demo_type": "enhanced_archangel_with_mcp",
        "results": exercise_results,
        "demo_completed": True
    }

if __name__ == "__main__":
    # Run enhanced demo
    async def main():
        result = await demo_enhanced_archangel_system()
        print(json.dumps(result, indent=2, default=str))
    
    asyncio.run(main())