"""
Archangel MCP Security Orchestrator
Comprehensive security orchestration layer for MCP-enabled AI agents

This module provides centralized security governance, integrating Guardian Protocol,
authorization validation, and real-time monitoring for secure AI operations.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
from pathlib import Path

from .mcp_guardian_protocol import (
    MCPGuardianProtocol, MCPSecurityContext, MCPTeamType, MCPToolCategory,
    DataClassification, LegalAuthorization, ValidationResult as GuardianValidationResult
)
from .authorization_validator import (
    AuthorizationValidator, MCPAuthorizationContext, AuthorizationLevel,
    PermissionScope, ValidationResult as AuthValidationResult
)
from .guardian_protocol import GuardianDecision, RiskLevel, ComplianceFramework, ActionCategory, SecurityContext

logger = logging.getLogger(__name__)

class MCPOperationStatus(Enum):
    """Status of MCP operations"""
    PENDING = "pending"
    AUTHORIZED = "authorized"
    DENIED = "denied"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    EMERGENCY_STOPPED = "emergency_stopped"

class MCPAlertLevel(Enum):
    """Alert levels for MCP monitoring"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class MCPOperationRequest:
    """Request for MCP operation"""
    request_id: str
    agent_id: str
    team_type: MCPTeamType
    tool_name: str
    tool_category: MCPToolCategory
    action_description: str
    target_systems: List[str]
    external_apis: List[str]
    parameters: Dict[str, Any]
    legal_authorization: LegalAuthorization
    data_classification: DataClassification
    business_justification: str
    user_id: str
    session_id: str
    source_ip: str
    timestamp: datetime
    penetration_test_id: Optional[str] = None
    incident_id: Optional[str] = None

@dataclass
class MCPOperationResult:
    """Result of MCP operation processing"""
    request_id: str
    status: MCPOperationStatus
    authorized: bool
    guardian_decision: GuardianDecision
    authorization_token: Optional[str]
    approval_required: bool
    approval_chain: List[str]
    risk_score: float
    compliance_status: Dict[str, bool]
    validation_errors: List[str]
    scope_limitations: List[str]
    recommendations: List[str]
    processing_time_ms: float
    audit_trail_id: str
    expires_at: Optional[datetime]

@dataclass
class MCPSecurityAlert:
    """Security alert for MCP operations"""
    alert_id: str
    level: MCPAlertLevel
    agent_id: str
    team_type: MCPTeamType
    alert_type: str
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    resolved: bool = False
    resolution_notes: Optional[str] = None

class MCPSecurityOrchestrator:
    """
    Comprehensive MCP Security Orchestrator
    
    Provides centralized security governance for MCP-enabled AI agents:
    1. Multi-layer validation (Guardian Protocol + Authorization)
    2. Real-time monitoring and alerting
    3. Compliance reporting and audit trails
    4. Emergency response and intervention
    5. Legal and ethical boundary enforcement
    6. Data protection and classification handling
    """
    
    def __init__(self, 
                 guardian_config_path: Optional[str] = None,
                 auth_config_path: Optional[str] = None,
                 mcp_config_path: Optional[str] = None):
        
        # Initialize core security components
        self.guardian_protocol = MCPGuardianProtocol(
            config_path=guardian_config_path,
            mcp_config_path=mcp_config_path
        )
        self.authorization_validator = AuthorizationValidator(config_path=auth_config_path)
        
        # Operation tracking
        self.active_operations: Dict[str, MCPOperationRequest] = {}
        self.operation_results: Dict[str, MCPOperationResult] = {}
        self.operation_history: List[MCPOperationResult] = []
        
        # Security monitoring
        self.security_alerts: List[MCPSecurityAlert] = []
        self.threat_indicators: Dict[str, List[Dict[str, Any]]] = {}
        self.rate_limit_violations: Dict[str, List[datetime]] = {}
        
        # Emergency controls
        self.emergency_stop_active = False
        self.blocked_agents: Set[str] = set()
        self.quarantined_operations: Set[str] = set()
        
        # Performance metrics
        self.metrics = {
            "total_requests": 0,
            "authorized_requests": 0,
            "denied_requests": 0,
            "emergency_stops": 0,
            "avg_processing_time_ms": 0.0,
            "compliance_violations": 0
        }
        
        logger.info("MCP Security Orchestrator initialized with comprehensive governance")
    
    async def process_operation_request(self, request: MCPOperationRequest) -> MCPOperationResult:
        """
        Process MCP operation request with comprehensive security validation
        
        Args:
            request: MCP operation request
            
        Returns:
            MCPOperationResult with validation outcome and any required actions
        """
        start_time = time.time()
        self.metrics["total_requests"] += 1
        
        try:
            # Step 1: Pre-validation checks
            precheck_result = await self._perform_prevalidation_checks(request)
            if precheck_result.status == MCPOperationStatus.DENIED:
                return precheck_result
            
            # Step 2: Create security contexts
            guardian_context = self._create_guardian_context(request)
            auth_context = self._create_authorization_context(request)
            
            # Step 3: Guardian Protocol validation
            guardian_result = await self.guardian_protocol.validate_mcp_action(
                request.action_description,
                guardian_context,
                request.parameters
            )
            
            # Step 4: Authorization validation
            auth_result = await self.authorization_validator.validate_mcp_authorization(
                auth_context,
                request.action_description,
                ",".join(request.target_systems)
            )
            
            # Step 5: Combine validation results
            final_result = await self._combine_validation_results(
                request, guardian_result, auth_result, start_time
            )
            
            # Step 6: Post-processing
            await self._post_process_operation(request, final_result)
            
            # Update metrics
            if final_result.authorized:
                self.metrics["authorized_requests"] += 1
            else:
                self.metrics["denied_requests"] += 1
            
            processing_time = (time.time() - start_time) * 1000
            self.metrics["avg_processing_time_ms"] = (
                (self.metrics["avg_processing_time_ms"] * (self.metrics["total_requests"] - 1) + processing_time) /
                self.metrics["total_requests"]
            )
            
            logger.info(f"MCP operation processed: {request.request_id} -> {final_result.status.value} "
                       f"(time: {processing_time:.2f}ms)")
            
            return final_result
            
        except Exception as e:
            logger.error(f"MCP operation processing failed: {e}")
            
            # Create emergency result
            emergency_result = MCPOperationResult(
                request_id=request.request_id,
                status=MCPOperationStatus.EMERGENCY_STOPPED,
                authorized=False,
                guardian_decision=GuardianDecision.EMERGENCY_STOP,
                authorization_token=None,
                approval_required=True,
                approval_chain=[],
                risk_score=10.0,
                compliance_status={},
                validation_errors=[f"System error: {str(e)}"],
                scope_limitations=[],
                recommendations=["Contact security team immediately"],
                processing_time_ms=(time.time() - start_time) * 1000,
                audit_trail_id="",
                expires_at=None
            )
            
            await self._create_security_alert(
                request.agent_id,
                request.team_type,
                MCPAlertLevel.EMERGENCY,
                "System Error",
                f"Critical error processing operation: {str(e)}",
                {"request_id": request.request_id, "error": str(e)}
            )
            
            return emergency_result
    
    async def _perform_prevalidation_checks(self, request: MCPOperationRequest) -> MCPOperationResult:
        """Perform pre-validation security checks"""
        errors = []
        
        # Emergency stop check
        if self.emergency_stop_active:
            return MCPOperationResult(
                request_id=request.request_id,
                status=MCPOperationStatus.EMERGENCY_STOPPED,
                authorized=False,
                guardian_decision=GuardianDecision.EMERGENCY_STOP,
                authorization_token=None,
                approval_required=True,
                approval_chain=[],
                risk_score=10.0,
                compliance_status={},
                validation_errors=["System emergency stop is active"],
                scope_limitations=[],
                recommendations=["Contact security team to resolve emergency stop"],
                processing_time_ms=0.0,
                audit_trail_id="",
                expires_at=None
            )
        
        # Blocked agent check
        if request.agent_id in self.blocked_agents:
            errors.append(f"Agent {request.agent_id} is blocked")
        
        # Rate limiting check
        if self._check_rate_limiting_violation(request):
            errors.append("Rate limiting violation detected")
        
        # Suspicious activity check
        if self._detect_suspicious_activity(request):
            errors.append("Suspicious activity pattern detected")
            await self._create_security_alert(
                request.agent_id,
                request.team_type,
                MCPAlertLevel.WARNING,
                "Suspicious Activity",
                "Potentially suspicious operation pattern detected",
                {"request_id": request.request_id, "patterns": "multiple_rapid_requests"}
            )
        
        # Basic validation
        if not request.business_justification:
            errors.append("Business justification required")
        
        if not request.target_systems:
            errors.append("Target systems must be specified")
        
        if errors:
            return MCPOperationResult(
                request_id=request.request_id,
                status=MCPOperationStatus.DENIED,
                authorized=False,
                guardian_decision=GuardianDecision.DENY,
                authorization_token=None,
                approval_required=True,
                approval_chain=[],
                risk_score=5.0,
                compliance_status={},
                validation_errors=errors,
                scope_limitations=[],
                recommendations=[],
                processing_time_ms=0.0,
                audit_trail_id="",
                expires_at=None
            )
        
        # Prevalidation passed
        return MCPOperationResult(
            request_id=request.request_id,
            status=MCPOperationStatus.PENDING,
            authorized=False,
            guardian_decision=GuardianDecision.ALLOW,
            authorization_token=None,
            approval_required=False,
            approval_chain=[],
            risk_score=0.0,
            compliance_status={},
            validation_errors=[],
            scope_limitations=[],
            recommendations=[],
            processing_time_ms=0.0,
            audit_trail_id="",
            expires_at=None
        )
    
    def _create_guardian_context(self, request: MCPOperationRequest) -> MCPSecurityContext:
        """Create Guardian Protocol security context"""
        base_context = SecurityContext(
            user_id=request.user_id,
            session_id=request.session_id,
            source_ip=request.source_ip,
            target_system=",".join(request.target_systems),
            action_category=self._map_tool_to_action_category(request.tool_category),
            risk_level=self._calculate_risk_level(request),
            timestamp=request.timestamp,
            authorization_level="security_admin",  # Will be properly determined
            business_justification=request.business_justification,
            compliance_requirements=self._determine_compliance_requirements(request)
        )
        
        return MCPSecurityContext(
            base_context=base_context,
            mcp_agent_id=request.agent_id,
            team_type=request.team_type,
            tool_name=request.tool_name,
            tool_category=request.tool_category,
            target_systems=request.target_systems,
            external_apis=request.external_apis,
            data_classification=request.data_classification,
            legal_authorization=request.legal_authorization,
            penetration_test_scope=None,  # Would be retrieved from registry
            business_approval=request.business_justification,
            technical_supervisor=None  # Would be determined from request
        )
    
    def _create_authorization_context(self, request: MCPOperationRequest) -> MCPAuthorizationContext:
        """Create authorization context"""
        return MCPAuthorizationContext(
            user_id=request.user_id,
            session_id=request.session_id,
            authorization_level=self._determine_authorization_level(request),
            roles=self._determine_user_roles(request),
            permissions=self._determine_permissions(request),
            source_ip=request.source_ip,
            user_agent="MCP-Agent",
            timestamp=request.timestamp,
            mfa_verified=True,  # Would be properly validated
            risk_score=self._calculate_operation_risk_score(request),
            mcp_agent_id=request.agent_id,
            team_type=request.team_type,
            tool_category=request.tool_category,
            target_systems=request.target_systems,
            external_apis=request.external_apis,
            data_classification=request.data_classification,
            legal_authorization=request.legal_authorization,
            penetration_test_id=request.penetration_test_id,
            incident_id=request.incident_id
        )
    
    async def _combine_validation_results(self,
                                        request: MCPOperationRequest,
                                        guardian_result: GuardianValidationResult,
                                        auth_result: AuthValidationResult,
                                        start_time: float) -> MCPOperationResult:
        """Combine Guardian and Authorization validation results"""
        
        # Determine overall authorization
        authorized = guardian_result.decision == GuardianDecision.ALLOW and auth_result.authorized
        
        # Determine status
        if guardian_result.emergency_stop_triggered:
            status = MCPOperationStatus.EMERGENCY_STOPPED
        elif not authorized:
            status = MCPOperationStatus.DENIED
        elif guardian_result.decision == GuardianDecision.REQUIRE_APPROVAL or auth_result.requires_approval:
            status = MCPOperationStatus.PENDING  # Requires approval
        else:
            status = MCPOperationStatus.AUTHORIZED
        
        # Combine errors
        all_errors = guardian_result.validation_errors + auth_result.validation_errors
        
        # Combine scope limitations
        all_limitations = guardian_result.recommendations + auth_result.scope_limitations
        
        # Combine recommendations
        all_recommendations = guardian_result.recommendations
        
        # Calculate final risk score
        final_risk_score = max(guardian_result.risk_score, 
                              self._calculate_operation_risk_score(request))
        
        # Determine approval chain
        approval_chain = []
        if guardian_result.required_approvals:
            approval_chain.extend(guardian_result.required_approvals)
        if auth_result.requires_approval:
            approval_chain.append("security_manager")
        
        # Create result
        result = MCPOperationResult(
            request_id=request.request_id,
            status=status,
            authorized=authorized,
            guardian_decision=guardian_result.decision,
            authorization_token=auth_result.token_id,
            approval_required=len(approval_chain) > 0,
            approval_chain=list(set(approval_chain)),
            risk_score=final_risk_score,
            compliance_status=guardian_result.compliance_status,
            validation_errors=all_errors,
            scope_limitations=all_limitations,
            recommendations=all_recommendations,
            processing_time_ms=(time.time() - start_time) * 1000,
            audit_trail_id=guardian_result.audit_entry_id,
            expires_at=auth_result.expires_at
        )
        
        # Store operation tracking
        self.active_operations[request.request_id] = request
        self.operation_results[request.request_id] = result
        
        return result
    
    async def _post_process_operation(self, request: MCPOperationRequest, result: MCPOperationResult):
        """Post-process operation result"""
        
        # Create alerts for high-risk operations
        if result.risk_score >= 8.0:
            await self._create_security_alert(
                request.agent_id,
                request.team_type,
                MCPAlertLevel.ERROR if result.authorized else MCPAlertLevel.CRITICAL,
                "High-Risk Operation",
                f"High-risk operation {result.status.value}: {request.action_description}",
                {
                    "request_id": request.request_id,
                    "risk_score": result.risk_score,
                    "authorized": result.authorized
                }
            )
        
        # Monitor for compliance violations
        compliance_violations = [fw for fw, compliant in result.compliance_status.items() if not compliant]
        if compliance_violations:
            self.metrics["compliance_violations"] += len(compliance_violations)
            await self._create_security_alert(
                request.agent_id,
                request.team_type,
                MCPAlertLevel.WARNING,
                "Compliance Violation",
                f"Compliance frameworks violated: {compliance_violations}",
                {
                    "request_id": request.request_id,
                    "violations": compliance_violations
                }
            )
        
        # Track emergency stops
        if result.status == MCPOperationStatus.EMERGENCY_STOPPED:
            self.metrics["emergency_stops"] += 1
            await self._create_security_alert(
                request.agent_id,
                request.team_type,
                MCPAlertLevel.EMERGENCY,
                "Emergency Stop",
                f"Operation emergency stopped: {request.action_description}",
                {
                    "request_id": request.request_id,
                    "errors": result.validation_errors
                }
            )
        
        # Update threat indicators
        if result.risk_score >= 6.0:
            self._update_threat_indicators(request, result)
    
    async def _create_security_alert(self,
                                   agent_id: str,
                                   team_type: MCPTeamType,
                                   level: MCPAlertLevel,
                                   alert_type: str,
                                   message: str,
                                   details: Dict[str, Any]):
        """Create security alert"""
        alert = MCPSecurityAlert(
            alert_id=str(uuid.uuid4()),
            level=level,
            agent_id=agent_id,
            team_type=team_type,
            alert_type=alert_type,
            message=message,
            details=details,
            timestamp=datetime.now()
        )
        
        self.security_alerts.append(alert)
        
        # Log alert
        log_level = {
            MCPAlertLevel.INFO: logging.INFO,
            MCPAlertLevel.WARNING: logging.WARNING,
            MCPAlertLevel.ERROR: logging.ERROR,
            MCPAlertLevel.CRITICAL: logging.CRITICAL,
            MCPAlertLevel.EMERGENCY: logging.CRITICAL
        }.get(level, logging.INFO)
        
        logger.log(log_level, f"MCP Security Alert [{level.value}]: {alert_type} - {message}")
    
    def _update_threat_indicators(self, request: MCPOperationRequest, result: MCPOperationResult):
        """Update threat indicators based on operation"""
        if request.agent_id not in self.threat_indicators:
            self.threat_indicators[request.agent_id] = []
        
        indicator = {
            "timestamp": datetime.now(),
            "request_id": request.request_id,
            "risk_score": result.risk_score,
            "status": result.status.value,
            "tool_category": request.tool_category.value,
            "target_systems": request.target_systems,
            "errors": result.validation_errors
        }
        
        self.threat_indicators[request.agent_id].append(indicator)
        
        # Keep only recent indicators (last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.threat_indicators[request.agent_id] = [
            ind for ind in self.threat_indicators[request.agent_id]
            if ind["timestamp"] > cutoff_time
        ]
    
    # Helper methods for context creation and validation
    
    def _map_tool_to_action_category(self, tool_category: MCPToolCategory) -> ActionCategory:
        """Map MCP tool category to Guardian action category"""
        mapping = {
            MCPToolCategory.RECONNAISSANCE: ActionCategory.RECONNAISSANCE,
            MCPToolCategory.VULNERABILITY_SCANNING: ActionCategory.VULNERABILITY_SCANNING,
            MCPToolCategory.EXPLOITATION: ActionCategory.OFFENSIVE_SECURITY,
            MCPToolCategory.THREAT_INTELLIGENCE: ActionCategory.THREAT_HUNTING,
            MCPToolCategory.INCIDENT_RESPONSE: ActionCategory.INCIDENT_RESPONSE,
            MCPToolCategory.MEMORY_ANALYSIS: ActionCategory.LOG_ANALYSIS,
            MCPToolCategory.NETWORK_MONITORING: ActionCategory.NETWORK_MONITORING,
            MCPToolCategory.MALWARE_ANALYSIS: ActionCategory.LOG_ANALYSIS,
            MCPToolCategory.OSINT: ActionCategory.RECONNAISSANCE,
            MCPToolCategory.INFRASTRUCTURE: ActionCategory.SYSTEM_MODIFICATION
        }
        return mapping.get(tool_category, ActionCategory.LOG_ANALYSIS)
    
    def _calculate_risk_level(self, request: MCPOperationRequest) -> RiskLevel:
        """Calculate risk level for request"""
        risk_score = self._calculate_operation_risk_score(request)
        
        if risk_score >= 9.0:
            return RiskLevel.CATASTROPHIC
        elif risk_score >= 7.0:
            return RiskLevel.CRITICAL
        elif risk_score >= 5.0:
            return RiskLevel.HIGH
        elif risk_score >= 3.0:
            return RiskLevel.MEDIUM
        elif risk_score >= 1.0:
            return RiskLevel.LOW
        else:
            return RiskLevel.MINIMAL
    
    def _calculate_operation_risk_score(self, request: MCPOperationRequest) -> float:
        """Calculate comprehensive risk score for operation"""
        risk_score = 1.0
        
        # Team type risk
        if request.team_type == MCPTeamType.RED_TEAM:
            risk_score += 2.0
        
        # Tool category risk
        high_risk_tools = [MCPToolCategory.EXPLOITATION, MCPToolCategory.VULNERABILITY_SCANNING]
        if request.tool_category in high_risk_tools:
            risk_score += 3.0
        
        # Target system risk
        for target in request.target_systems:
            if any(keyword in target.lower() for keyword in ["prod", "production", "live"]):
                risk_score += 3.0
            elif any(keyword in target.lower() for keyword in ["admin", "root", "system"]):
                risk_score += 2.0
        
        # Legal authorization risk
        if request.legal_authorization == LegalAuthorization.UNAUTHORIZED:
            risk_score += 4.0
        elif request.legal_authorization == LegalAuthorization.RESEARCH_ONLY:
            risk_score += 2.0
        
        # Data classification risk
        classification_risk = {
            DataClassification.PUBLIC: 0.0,
            DataClassification.INTERNAL: 0.5,
            DataClassification.CONFIDENTIAL: 1.5,
            DataClassification.RESTRICTED: 2.5,
            DataClassification.TOP_SECRET: 4.0
        }
        risk_score += classification_risk.get(request.data_classification, 0.0)
        
        # External API risk
        risky_apis = ["metasploit", "exploit_db", "nuclei"]
        risk_score += sum(1.0 for api in request.external_apis if api in risky_apis)
        
        return min(risk_score, 10.0)
    
    def _determine_compliance_requirements(self, request: MCPOperationRequest) -> List[ComplianceFramework]:
        """Determine compliance requirements for request"""
        requirements = [ComplianceFramework.SOC2]  # Base requirement
        
        if request.data_classification in [DataClassification.RESTRICTED, DataClassification.TOP_SECRET]:
            requirements.append(ComplianceFramework.NIST)
        
        if request.team_type == MCPTeamType.RED_TEAM:
            requirements.append(ComplianceFramework.NIST)
        
        # Add other compliance requirements based on context
        return requirements
    
    def _determine_authorization_level(self, request: MCPOperationRequest) -> AuthorizationLevel:
        """Determine authorization level for request"""
        if request.legal_authorization == LegalAuthorization.PRODUCTION_APPROVED:
            return AuthorizationLevel.PRODUCTION_AUTHORIZED
        elif request.team_type == MCPTeamType.RED_TEAM:
            return AuthorizationLevel.PENETRATION_TESTER
        elif request.tool_category == MCPToolCategory.INCIDENT_RESPONSE:
            return AuthorizationLevel.INCIDENT_RESPONDER
        else:
            return AuthorizationLevel.SECURITY_ANALYST
    
    def _determine_user_roles(self, request: MCPOperationRequest) -> List[str]:
        """Determine user roles based on request"""
        roles = ["security_analyst"]
        
        if request.team_type == MCPTeamType.RED_TEAM:
            roles.append("penetration_tester")
        elif request.team_type == MCPTeamType.BLUE_TEAM:
            roles.append("incident_responder")
        
        return roles
    
    def _determine_permissions(self, request: MCPOperationRequest) -> Set[PermissionScope]:
        """Determine permissions based on request"""
        permissions = {PermissionScope.READ_ONLY}
        
        if request.tool_category == MCPToolCategory.RECONNAISSANCE:
            permissions.add(PermissionScope.NETWORK_SCAN)
        elif request.tool_category == MCPToolCategory.VULNERABILITY_SCANNING:
            permissions.add(PermissionScope.VULNERABILITY_SCAN)
        elif request.tool_category == MCPToolCategory.EXPLOITATION:
            permissions.add(PermissionScope.PENETRATION_TESTING)
        elif request.tool_category == MCPToolCategory.INCIDENT_RESPONSE:
            permissions.add(PermissionScope.INCIDENT_INVESTIGATION)
        
        return permissions
    
    def _check_rate_limiting_violation(self, request: MCPOperationRequest) -> bool:
        """Check for rate limiting violations"""
        now = datetime.now()
        
        # Check agent-specific rate limits
        if request.agent_id not in self.rate_limit_violations:
            self.rate_limit_violations[request.agent_id] = []
        
        # Clean old violations (last hour)
        cutoff_time = now - timedelta(hours=1)
        self.rate_limit_violations[request.agent_id] = [
            violation for violation in self.rate_limit_violations[request.agent_id]
            if violation > cutoff_time
        ]
        
        # Check if too many requests in last hour
        violations_count = len(self.rate_limit_violations[request.agent_id])
        rate_limit = 100  # 100 requests per hour per agent
        
        if violations_count >= rate_limit:
            return True
        
        # Record this request
        self.rate_limit_violations[request.agent_id].append(now)
        return False
    
    def _detect_suspicious_activity(self, request: MCPOperationRequest) -> bool:
        """Detect suspicious activity patterns"""
        if request.agent_id not in self.threat_indicators:
            return False
        
        recent_indicators = self.threat_indicators[request.agent_id]
        
        # Check for rapid-fire high-risk operations
        high_risk_operations = [
            ind for ind in recent_indicators
            if ind["risk_score"] >= 6.0 and 
            ind["timestamp"] > datetime.now() - timedelta(minutes=10)
        ]
        
        if len(high_risk_operations) >= 5:
            return True
        
        # Check for repeated failed operations
        failed_operations = [
            ind for ind in recent_indicators
            if ind["status"] in ["denied", "emergency_stopped"] and
            ind["timestamp"] > datetime.now() - timedelta(minutes=30)
        ]
        
        if len(failed_operations) >= 10:
            return True
        
        return False
    
    # Public API methods for security management
    
    def trigger_emergency_stop(self, reason: str, user_id: str):
        """Trigger system-wide emergency stop"""
        self.emergency_stop_active = True
        self.guardian_protocol.trigger_emergency_stop(reason, user_id)
        
        logger.critical(f"MCP SYSTEM EMERGENCY STOP: {reason} (triggered by {user_id})")
        
        # Create emergency alert
        asyncio.create_task(self._create_security_alert(
            "system",
            MCPTeamType.ADMIN,
            MCPAlertLevel.EMERGENCY,
            "Emergency Stop",
            f"System emergency stop activated: {reason}",
            {"reason": reason, "triggered_by": user_id}
        ))
    
    def clear_emergency_stop(self, user_id: str, authorization_code: str):
        """Clear emergency stop"""
        if authorization_code == "MCP_GUARDIAN_OVERRIDE":
            self.emergency_stop_active = False
            self.guardian_protocol.clear_emergency_stop(user_id, "GUARDIAN_OVERRIDE")
            logger.warning(f"MCP emergency stop cleared by {user_id}")
        else:
            logger.error(f"Invalid emergency stop clear attempt by {user_id}")
    
    def block_agent(self, agent_id: str, reason: str, user_id: str):
        """Block specific agent"""
        self.blocked_agents.add(agent_id)
        logger.warning(f"Agent blocked: {agent_id} by {user_id} - {reason}")
        
        asyncio.create_task(self._create_security_alert(
            agent_id,
            MCPTeamType.ADMIN,
            MCPAlertLevel.WARNING,
            "Agent Blocked",
            f"Agent {agent_id} has been blocked: {reason}",
            {"agent_id": agent_id, "reason": reason, "blocked_by": user_id}
        ))
    
    def unblock_agent(self, agent_id: str, user_id: str):
        """Unblock specific agent"""
        if agent_id in self.blocked_agents:
            self.blocked_agents.remove(agent_id)
            logger.info(f"Agent unblocked: {agent_id} by {user_id}")
        
    def get_security_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive security dashboard"""
        now = datetime.now()
        
        # Recent alerts (last 24 hours)
        recent_alerts = [
            alert for alert in self.security_alerts
            if (now - alert.timestamp).days == 0
        ]
        
        # Alert breakdown
        alert_breakdown = {}
        for level in MCPAlertLevel:
            alert_breakdown[level.value] = len([
                alert for alert in recent_alerts
                if alert.level == level
            ])
        
        # Active operations
        active_ops = len(self.active_operations)
        
        # Threat indicators summary
        threat_summary = {}
        for agent_id, indicators in self.threat_indicators.items():
            if indicators:
                avg_risk = sum(ind["risk_score"] for ind in indicators) / len(indicators)
                threat_summary[agent_id] = {
                    "indicator_count": len(indicators),
                    "avg_risk_score": round(avg_risk, 2),
                    "high_risk_operations": len([ind for ind in indicators if ind["risk_score"] >= 7.0])
                }
        
        return {
            "system_status": {
                "emergency_stop_active": self.emergency_stop_active,
                "blocked_agents": len(self.blocked_agents),
                "active_operations": active_ops,
                "total_alerts": len(recent_alerts)
            },
            "metrics": self.metrics,
            "alerts": {
                "recent_count": len(recent_alerts),
                "breakdown": alert_breakdown,
                "unresolved": len([alert for alert in recent_alerts if not alert.resolved])
            },
            "threat_indicators": threat_summary,
            "top_risks": self._get_top_risk_operations(),
            "compliance_status": self._get_compliance_summary(),
            "last_updated": now.isoformat()
        }
    
    def _get_top_risk_operations(self) -> List[Dict[str, Any]]:
        """Get top risk operations from recent history"""
        recent_ops = [
            op for op in self.operation_history
            if (datetime.now() - datetime.fromisoformat(op.audit_trail_id.split('_')[0] if op.audit_trail_id else '2024-01-01')).days <= 7
        ]
        
        # Sort by risk score
        top_risks = sorted(recent_ops, key=lambda x: x.risk_score, reverse=True)[:10]
        
        return [
            {
                "request_id": op.request_id,
                "risk_score": op.risk_score,
                "status": op.status.value,
                "guardian_decision": op.guardian_decision.value,
                "errors": op.validation_errors
            }
            for op in top_risks
        ]
    
    def _get_compliance_summary(self) -> Dict[str, Any]:
        """Get compliance status summary"""
        total_ops = len(self.operation_history)
        if total_ops == 0:
            return {"total_operations": 0, "compliance_rate": 100.0}
        
        compliant_ops = len([
            op for op in self.operation_history
            if all(op.compliance_status.values()) or not op.compliance_status
        ])
        
        compliance_rate = (compliant_ops / total_ops) * 100.0
        
        return {
            "total_operations": total_ops,
            "compliant_operations": compliant_ops,
            "compliance_rate": round(compliance_rate, 2),
            "violations": self.metrics["compliance_violations"]
        }
    
    def generate_security_report(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate comprehensive security report"""
        # Filter operations by date range
        period_operations = [
            op for op in self.operation_history
            if start_date <= datetime.fromisoformat(op.audit_trail_id.split('_')[0] if op.audit_trail_id else '2024-01-01') <= end_date
        ]
        
        # Generate Guardian Protocol compliance report
        guardian_report = self.guardian_protocol.generate_mcp_compliance_report(
            ComplianceFramework.SOC2, start_date, end_date
        )
        
        return {
            "report_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "executive_summary": {
                "total_operations": len(period_operations),
                "authorized_operations": len([op for op in period_operations if op.authorized]),
                "denied_operations": len([op for op in period_operations if not op.authorized]),
                "emergency_stops": len([op for op in period_operations if op.status == MCPOperationStatus.EMERGENCY_STOPPED]),
                "avg_risk_score": sum(op.risk_score for op in period_operations) / len(period_operations) if period_operations else 0.0
            },
            "guardian_compliance": guardian_report,
            "team_breakdown": self._generate_team_breakdown(period_operations),
            "tool_usage_analysis": self._generate_tool_usage_analysis(period_operations),
            "risk_analysis": self._generate_risk_analysis(period_operations),
            "recommendations": self._generate_security_recommendations(period_operations)
        }
    
    def _generate_team_breakdown(self, operations: List[MCPOperationResult]) -> Dict[str, Any]:
        """Generate team-specific breakdown"""
        # This would analyze operations by team type
        # Simplified implementation for now
        return {
            "red_team": {"operations": 0, "success_rate": 0.0},
            "blue_team": {"operations": 0, "success_rate": 0.0},
            "neutral": {"operations": 0, "success_rate": 0.0}
        }
    
    def _generate_tool_usage_analysis(self, operations: List[MCPOperationResult]) -> Dict[str, Any]:
        """Generate tool usage analysis"""
        # This would analyze tool usage patterns
        return {"tools": [], "trends": []}
    
    def _generate_risk_analysis(self, operations: List[MCPOperationResult]) -> Dict[str, Any]:
        """Generate risk analysis"""
        if not operations:
            return {"risk_distribution": {}, "trends": []}
        
        # Risk distribution
        risk_buckets = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        for op in operations:
            if op.risk_score >= 8.0:
                risk_buckets["critical"] += 1
            elif op.risk_score >= 6.0:
                risk_buckets["high"] += 1
            elif op.risk_score >= 3.0:
                risk_buckets["medium"] += 1
            else:
                risk_buckets["low"] += 1
        
        return {
            "risk_distribution": risk_buckets,
            "avg_risk_score": sum(op.risk_score for op in operations) / len(operations),
            "trends": []
        }
    
    def _generate_security_recommendations(self, operations: List[MCPOperationResult]) -> List[str]:
        """Generate security recommendations"""
        recommendations = []
        
        # Analyze patterns and generate recommendations
        high_risk_ops = [op for op in operations if op.risk_score >= 7.0]
        if len(high_risk_ops) > len(operations) * 0.1:  # More than 10% high-risk
            recommendations.append("Consider implementing additional approval workflows for high-risk operations")
        
        emergency_stops = [op for op in operations if op.status == MCPOperationStatus.EMERGENCY_STOPPED]
        if emergency_stops:
            recommendations.append("Review emergency stop triggers and consider additional preventive measures")
        
        if not recommendations:
            recommendations.append("Security posture appears adequate based on current analysis")
        
        return recommendations