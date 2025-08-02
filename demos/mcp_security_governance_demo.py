#!/usr/bin/env python3
"""
Archangel MCP Security Governance Demonstration
Shows comprehensive security governance for AI agents with external resource access

This demo showcases:
1. Guardian Protocol multi-layer validation
2. Authorization scope enforcement
3. Legal compliance checking
4. Damage prevention safeguards
5. Real-time monitoring and alerting
6. Audit logging and compliance reporting

This is a demonstration of defensive security research capabilities.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.mcp_security_orchestrator import (
    MCPSecurityOrchestrator, MCPOperationRequest, MCPOperationStatus,
    MCPTeamType, MCPToolCategory, DataClassification, LegalAuthorization
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MCPSecurityGovernanceDemo:
    """Demonstration of MCP Security Governance capabilities"""
    
    def __init__(self):
        self.orchestrator = MCPSecurityOrchestrator()
        self.demo_results = []
        
        logger.info("🛡️  MCP Security Governance Demo initialized")
    
    async def run_comprehensive_demo(self) -> Dict[str, Any]:
        """Run comprehensive security governance demonstration"""
        logger.info("🚀 Starting MCP Security Governance Demo")
        
        demo_results = {
            "demo_title": "Archangel MCP Security Governance Demonstration",
            "demo_description": "Comprehensive security governance for AI agents with external resource access",
            "timestamp": datetime.now().isoformat(),
            "scenarios": []
        }
        
        # Scenario 1: Authorized Red Team Operation
        logger.info("📋 Scenario 1: Authorized Red Team Vulnerability Scanning")
        scenario1_result = await self._demo_authorized_red_team_operation()
        demo_results["scenarios"].append(scenario1_result)
        
        # Scenario 2: Blue Team Incident Response 
        logger.info("📋 Scenario 2: Blue Team Incident Response Operation")
        scenario2_result = await self._demo_blue_team_incident_response()
        demo_results["scenarios"].append(scenario2_result)
        
        # Scenario 3: Unauthorized Production Access (Blocked)
        logger.info("📋 Scenario 3: Unauthorized Production Access (Should be Blocked)")
        scenario3_result = await self._demo_unauthorized_production_access()
        demo_results["scenarios"].append(scenario3_result)
        
        # Scenario 4: High-Risk Exploitation (Requires Approval)
        logger.info("📋 Scenario 4: High-Risk Exploitation Tools (Should Require Approval)")
        scenario4_result = await self._demo_high_risk_exploitation()
        demo_results["scenarios"].append(scenario4_result)
        
        # Scenario 5: Compliance Violation Detection
        logger.info("📋 Scenario 5: Compliance Violation Detection")
        scenario5_result = await self._demo_compliance_violation()
        demo_results["scenarios"].append(scenario5_result)
        
        # Scenario 6: Emergency Stop Trigger
        logger.info("📋 Scenario 6: Emergency Stop and Recovery")
        scenario6_result = await self._demo_emergency_stop()
        demo_results["scenarios"].append(scenario6_result)
        
        # Generate security dashboard
        logger.info("📊 Generating Security Dashboard")
        dashboard = self.orchestrator.get_security_dashboard()
        demo_results["security_dashboard"] = dashboard
        
        # Generate compliance report
        logger.info("📋 Generating Compliance Report")
        start_date = datetime.now() - timedelta(days=1)
        end_date = datetime.now()
        compliance_report = self.orchestrator.generate_security_report(start_date, end_date)
        demo_results["compliance_report"] = compliance_report
        
        logger.info("✅ MCP Security Governance Demo completed successfully")
        return demo_results
    
    async def _demo_authorized_red_team_operation(self) -> Dict[str, Any]:
        """Demo authorized red team vulnerability scanning operation"""
        request = MCPOperationRequest(
            request_id="DEMO_RED_001",
            agent_id="red_team_agent_001",
            team_type=MCPTeamType.RED_TEAM,
            tool_name="nuclei",
            tool_category=MCPToolCategory.VULNERABILITY_SCANNING,
            action_description="Perform authorized vulnerability scanning on test environment",
            target_systems=["test-web-app.lab.archangel.local", "staging-db.lab.archangel.local"],
            external_apis=["nuclei", "shodan"],
            parameters={
                "scan_type": "comprehensive",
                "templates": "web-applications",
                "rate_limit": "moderate",
                "test_environment": True
            },
            legal_authorization=LegalAuthorization.AUTHORIZED_TESTING,
            data_classification=DataClassification.INTERNAL,
            business_justification="Scheduled vulnerability assessment for test environment",
            user_id="security_analyst_001",
            session_id="session_001",
            source_ip="192.168.100.10",
            timestamp=datetime.now(),
            penetration_test_id="PENTEST_2024_001"
        )
        
        result = await self.orchestrator.process_operation_request(request)
        
        return {
            "scenario": "Authorized Red Team Vulnerability Scanning",
            "expected_outcome": "AUTHORIZED - Low risk, proper authorization",
            "actual_outcome": result.status.value.upper(),
            "success": result.authorized,
            "risk_score": result.risk_score,
            "validation_errors": result.validation_errors,
            "authorization_token": result.authorization_token is not None,
            "compliance_status": result.compliance_status,
            "processing_time_ms": result.processing_time_ms,
            "guardian_decision": result.guardian_decision.value,
            "details": {
                "team_type": request.team_type.value,
                "tool_category": request.tool_category.value,
                "legal_authorization": request.legal_authorization.value,
                "data_classification": request.data_classification.value,
                "target_systems": request.target_systems,
                "external_apis": request.external_apis
            }
        }
    
    async def _demo_blue_team_incident_response(self) -> Dict[str, Any]:
        """Demo blue team incident response operation"""
        request = MCPOperationRequest(
            request_id="DEMO_BLUE_001",
            agent_id="blue_team_agent_001", 
            team_type=MCPTeamType.BLUE_TEAM,
            tool_name="volatility",
            tool_category=MCPToolCategory.MEMORY_ANALYSIS,
            action_description="Analyze memory dump for incident INC-2024-001",
            target_systems=["incident-workstation-001.archangel.local"],
            external_apis=["volatility", "yara", "virustotal"],
            parameters={
                "memory_dump_path": "/forensics/dumps/workstation-001.mem",
                "analysis_profile": "comprehensive",
                "malware_detection": True,
                "network_artifacts": True
            },
            legal_authorization=LegalAuthorization.EMERGENCY_RESPONSE,
            data_classification=DataClassification.CONFIDENTIAL,
            business_justification="Critical incident response - suspected APT activity",
            user_id="incident_responder_001",
            session_id="session_002",
            source_ip="192.168.100.20",
            timestamp=datetime.now(),
            incident_id="INC-2024-001"
        )
        
        result = await self.orchestrator.process_operation_request(request)
        
        return {
            "scenario": "Blue Team Incident Response",
            "expected_outcome": "AUTHORIZED - Emergency response authorization",
            "actual_outcome": result.status.value.upper(),
            "success": result.authorized,
            "risk_score": result.risk_score,
            "validation_errors": result.validation_errors,
            "authorization_token": result.authorization_token is not None,
            "compliance_status": result.compliance_status,
            "processing_time_ms": result.processing_time_ms,
            "guardian_decision": result.guardian_decision.value,
            "details": {
                "team_type": request.team_type.value,
                "tool_category": request.tool_category.value,
                "legal_authorization": request.legal_authorization.value,
                "data_classification": request.data_classification.value,
                "incident_id": request.incident_id
            }
        }
    
    async def _demo_unauthorized_production_access(self) -> Dict[str, Any]:
        """Demo unauthorized production access attempt (should be blocked)"""
        request = MCPOperationRequest(
            request_id="DEMO_BLOCK_001",
            agent_id="rogue_agent_001",
            team_type=MCPTeamType.RED_TEAM,
            tool_name="metasploit",
            tool_category=MCPToolCategory.EXPLOITATION,
            action_description="Attempt exploitation of production web server",
            target_systems=["prod-web-01.archangel.com", "prod-db-01.archangel.com"],
            external_apis=["metasploit", "exploit_db"],
            parameters={
                "exploit_module": "web/apache/struts_rce",
                "payload": "reverse_shell",
                "target_port": 8080,
                "production_target": True
            },
            legal_authorization=LegalAuthorization.UNAUTHORIZED,
            data_classification=DataClassification.RESTRICTED,
            business_justification="Testing production systems",  # Insufficient justification
            user_id="guest_user_001",
            session_id="session_003",
            source_ip="203.0.113.100",  # External IP
            timestamp=datetime.now()
        )
        
        result = await self.orchestrator.process_operation_request(request)
        
        return {
            "scenario": "Unauthorized Production Access Attempt",
            "expected_outcome": "DENIED - Production access without authorization",
            "actual_outcome": result.status.value.upper(),
            "success": not result.authorized,  # Success means it was properly blocked
            "risk_score": result.risk_score,
            "validation_errors": result.validation_errors,
            "authorization_token": result.authorization_token is not None,
            "compliance_status": result.compliance_status,
            "processing_time_ms": result.processing_time_ms,
            "guardian_decision": result.guardian_decision.value,
            "security_alerts_triggered": len([
                alert for alert in self.orchestrator.security_alerts
                if "prod" in alert.message.lower()
            ]),
            "details": {
                "team_type": request.team_type.value,
                "tool_category": request.tool_category.value,
                "legal_authorization": request.legal_authorization.value,
                "target_systems": request.target_systems,
                "blocked_reasons": result.validation_errors
            }
        }
    
    async def _demo_high_risk_exploitation(self) -> Dict[str, Any]:
        """Demo high-risk exploitation requiring approval"""
        request = MCPOperationRequest(
            request_id="DEMO_APPROVAL_001",
            agent_id="red_team_agent_002",
            team_type=MCPTeamType.RED_TEAM,
            tool_name="metasploit",
            tool_category=MCPToolCategory.EXPLOITATION,
            action_description="Advanced persistent threat simulation using exploit chains",
            target_systems=["test-domain-controller.lab.archangel.local"],
            external_apis=["metasploit", "exploit_db"],
            parameters={
                "exploit_chain": ["zerologon", "golden_ticket", "dcsync"],
                "persistence_mechanisms": ["scheduled_tasks", "registry_keys"],
                "lateral_movement": True,
                "data_exfiltration_simulation": True
            },
            legal_authorization=LegalAuthorization.AUTHORIZED_TESTING,
            data_classification=DataClassification.CONFIDENTIAL,
            business_justification="APT simulation for blue team training",
            user_id="senior_pentester_001",
            session_id="session_004",
            source_ip="192.168.100.30",
            timestamp=datetime.now(),
            penetration_test_id="PENTEST_2024_002"
        )
        
        result = await self.orchestrator.process_operation_request(request)
        
        return {
            "scenario": "High-Risk Exploitation Tools",
            "expected_outcome": "PENDING - Requires management approval",
            "actual_outcome": result.status.value.upper(),
            "success": result.approval_required,
            "risk_score": result.risk_score,
            "validation_errors": result.validation_errors,
            "approval_chain": result.approval_chain,
            "authorization_token": result.authorization_token is not None,
            "compliance_status": result.compliance_status,
            "processing_time_ms": result.processing_time_ms,
            "guardian_decision": result.guardian_decision.value,
            "details": {
                "team_type": request.team_type.value,
                "tool_category": request.tool_category.value,
                "legal_authorization": request.legal_authorization.value,
                "approval_required": result.approval_required,
                "approval_chain": result.approval_chain
            }
        }
    
    async def _demo_compliance_violation(self) -> Dict[str, Any]:
        """Demo compliance violation detection"""
        request = MCPOperationRequest(
            request_id="DEMO_COMPLIANCE_001",
            agent_id="blue_team_agent_002",
            team_type=MCPTeamType.BLUE_TEAM,
            tool_name="osquery",
            tool_category=MCPToolCategory.NETWORK_MONITORING,
            action_description="Query personal data from employee workstations without GDPR compliance",
            target_systems=["employee-workstation-*.archangel.com"],
            external_apis=["osquery"],
            parameters={
                "query": "SELECT * FROM users, processes, browser_history WHERE personal_data = true",
                "include_personal_data": True,
                "gdpr_consent": False,  # Missing GDPR consent
                "data_minimization": False
            },
            legal_authorization=LegalAuthorization.RESEARCH_ONLY,
            data_classification=DataClassification.RESTRICTED,  # Personal data
            business_justification="Employee behavior monitoring",
            user_id="hr_analyst_001",
            session_id="session_005",
            source_ip="192.168.100.40",
            timestamp=datetime.now()
        )
        
        result = await self.orchestrator.process_operation_request(request)
        
        return {
            "scenario": "GDPR Compliance Violation Detection", 
            "expected_outcome": "DENIED - GDPR compliance violation",
            "actual_outcome": result.status.value.upper(),
            "success": not result.authorized,  # Should be blocked
            "risk_score": result.risk_score,
            "validation_errors": result.validation_errors,
            "compliance_status": result.compliance_status,
            "compliance_violations": [
                fw for fw, compliant in result.compliance_status.items() 
                if not compliant
            ],
            "processing_time_ms": result.processing_time_ms,
            "guardian_decision": result.guardian_decision.value,
            "details": {
                "data_classification": request.data_classification.value,
                "gdpr_consent": request.parameters.get("gdpr_consent", False),
                "personal_data_access": request.parameters.get("include_personal_data", False),
                "compliance_violations": result.validation_errors
            }
        }
    
    async def _demo_emergency_stop(self) -> Dict[str, Any]:
        """Demo emergency stop and recovery procedures"""
        # First, trigger an emergency stop
        self.orchestrator.trigger_emergency_stop(
            "Demo emergency stop - suspicious activity detected",
            "security_manager_001"
        )
        
        # Try to execute operation during emergency stop
        request = MCPOperationRequest(
            request_id="DEMO_EMERGENCY_001",
            agent_id="any_agent_001",
            team_type=MCPTeamType.NEUTRAL,
            tool_name="osquery",
            tool_category=MCPToolCategory.OSINT,
            action_description="Basic OSINT research during emergency stop",
            target_systems=["public-website.example.com"],
            external_apis=["shodan"],
            parameters={"search_type": "basic"},
            legal_authorization=LegalAuthorization.RESEARCH_ONLY,
            data_classification=DataClassification.PUBLIC,
            business_justification="Basic research activity",
            user_id="analyst_001",
            session_id="session_006",
            source_ip="192.168.100.50",
            timestamp=datetime.now()
        )
        
        # This should be blocked due to emergency stop
        blocked_result = await self.orchestrator.process_operation_request(request)
        
        # Now clear the emergency stop
        self.orchestrator.clear_emergency_stop("security_manager_001", "MCP_GUARDIAN_OVERRIDE")
        
        # Try the same operation again - should now work
        request.request_id = "DEMO_RECOVERY_001"
        recovery_result = await self.orchestrator.process_operation_request(request)
        
        return {
            "scenario": "Emergency Stop and Recovery",
            "emergency_stop_phase": {
                "expected_outcome": "EMERGENCY_STOPPED - All operations blocked",
                "actual_outcome": blocked_result.status.value.upper(),
                "success": blocked_result.status == MCPOperationStatus.EMERGENCY_STOPPED,
                "validation_errors": blocked_result.validation_errors
            },
            "recovery_phase": {
                "expected_outcome": "AUTHORIZED - Normal operations resumed",
                "actual_outcome": recovery_result.status.value.upper(), 
                "success": recovery_result.authorized,
                "validation_errors": recovery_result.validation_errors
            },
            "emergency_metrics": {
                "emergency_stops_triggered": self.orchestrator.metrics["emergency_stops"],
                "security_alerts_count": len(self.orchestrator.security_alerts),
                "system_recovery_time_ms": recovery_result.processing_time_ms
            },
            "details": {
                "emergency_stop_reason": "Demo emergency stop - suspicious activity detected",
                "recovery_authorization": "MCP_GUARDIAN_OVERRIDE",
                "system_resilience": "Successfully recovered after emergency stop"
            }
        }

async def main():
    """Main demo function"""
    print("🛡️  Archangel MCP Security Governance Demonstration")
    print("=" * 60)
    print("This demonstration showcases comprehensive security governance")
    print("for AI agents with external resource access capabilities.")
    print("=" * 60)
    
    demo = MCPSecurityGovernanceDemo()
    
    try:
        # Run comprehensive demo
        results = await demo.run_comprehensive_demo()
        
        # Display results
        print("\n📊 DEMONSTRATION RESULTS")
        print("=" * 60)
        
        for i, scenario in enumerate(results["scenarios"], 1):
            print(f"\n{i}. {scenario['scenario']}")
            print(f"   Expected: {scenario['expected_outcome']}")
            print(f"   Actual:   {scenario['actual_outcome']}")
            print(f"   Success:  {'✅' if scenario['success'] else '❌'}")
            
            if 'risk_score' in scenario:
                print(f"   Risk Score: {scenario['risk_score']:.1f}/10.0")
            
            if scenario.get('validation_errors'):
                print(f"   Validation Errors: {len(scenario['validation_errors'])}")
                for error in scenario['validation_errors'][:3]:  # Show first 3
                    print(f"     - {error}")
        
        # Security Dashboard Summary
        dashboard = results["security_dashboard"]
        print(f"\n📈 SECURITY DASHBOARD SUMMARY")
        print("=" * 60)
        print(f"Total Operations: {dashboard['metrics']['total_requests']}")
        print(f"Authorized: {dashboard['metrics']['authorized_requests']}")
        print(f"Denied: {dashboard['metrics']['denied_requests']}")
        print(f"Emergency Stops: {dashboard['metrics']['emergency_stops']}")
        print(f"Avg Processing Time: {dashboard['metrics']['avg_processing_time_ms']:.1f}ms")
        print(f"Active Alerts: {dashboard['alerts']['recent_count']}")
        
        # Compliance Summary
        compliance = results["compliance_report"]
        print(f"\n📋 COMPLIANCE SUMMARY")
        print("=" * 60)
        print(f"Total Operations: {compliance['executive_summary']['total_operations']}")
        print(f"Authorization Rate: {(compliance['executive_summary']['authorized_operations'] / max(compliance['executive_summary']['total_operations'], 1) * 100):.1f}%")
        print(f"Average Risk Score: {compliance['executive_summary']['avg_risk_score']:.1f}")
        
        print(f"\n✅ MCP Security Governance Demo completed successfully!")
        print(f"Timestamp: {results['timestamp']}")
        
        # Optional: Save detailed results to file
        output_file = f"mcp_security_demo_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"📄 Detailed results saved to: {output_file}")
        
        return results
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"❌ Demo failed: {e}")
        raise

if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(main())