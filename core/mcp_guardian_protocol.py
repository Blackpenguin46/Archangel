"""
Archangel MCP Guardian Protocol
Enhanced security governance layer for Model Context Protocol integration

This module extends the Guardian Protocol to provide comprehensive security
governance for MCP-enabled AI agents with external resource access.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import uuid
from pathlib import Path
import re
import ipaddress

from .guardian_protocol import (
    GuardianProtocol, GuardianDecision, RiskLevel, ComplianceFramework,
    ActionCategory, SecurityContext, GuardianRule, AuditEntry, ValidationResult
)

logger = logging.getLogger(__name__)

class MCPTeamType(Enum):
    """MCP team classifications"""
    RED_TEAM = "red_team"
    BLUE_TEAM = "blue_team"
    NEUTRAL = "neutral"
    ADMIN = "admin"

class MCPToolCategory(Enum):
    """Categories of MCP tools"""
    RECONNAISSANCE = "reconnaissance"
    VULNERABILITY_SCANNING = "vulnerability_scanning"
    EXPLOITATION = "exploitation"
    THREAT_INTELLIGENCE = "threat_intelligence"
    INCIDENT_RESPONSE = "incident_response"
    MEMORY_ANALYSIS = "memory_analysis"
    NETWORK_MONITORING = "network_monitoring"
    OSINT = "osint"
    MALWARE_ANALYSIS = "malware_analysis"
    INFRASTRUCTURE = "infrastructure"

class DataClassification(Enum):
    """Data classification levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"

class LegalAuthorization(Enum):
    """Legal authorization levels"""
    UNAUTHORIZED = "unauthorized"
    RESEARCH_ONLY = "research_only"
    AUTHORIZED_TESTING = "authorized_testing"
    PRODUCTION_APPROVED = "production_approved"
    EMERGENCY_RESPONSE = "emergency_response"

@dataclass
class MCPSecurityContext:
    """Extended security context for MCP operations"""
    base_context: SecurityContext
    mcp_agent_id: str
    team_type: MCPTeamType
    tool_name: str
    tool_category: MCPToolCategory
    target_systems: List[str]
    external_apis: List[str]
    data_classification: DataClassification
    legal_authorization: LegalAuthorization
    penetration_test_scope: Optional[str]
    business_approval: Optional[str]
    technical_supervisor: Optional[str]

@dataclass
class ExternalAPIPolicy:
    """Policy for external API usage"""
    api_name: str
    allowed_teams: List[MCPTeamType]
    rate_limits: Dict[str, int]
    data_retention_days: int
    requires_approval: bool
    compliance_frameworks: List[ComplianceFramework]
    terms_of_service_url: str
    data_sharing_restrictions: List[str]
    geographic_restrictions: List[str]

@dataclass
class MCPAuditEntry(AuditEntry):
    """Extended audit entry for MCP operations"""
    mcp_agent_id: str
    team_type: MCPTeamType
    tool_name: str
    tool_category: MCPToolCategory
    external_api_calls: List[str]
    data_accessed: List[str]
    legal_authorization: LegalAuthorization
    supervisor_approval: Optional[str]
    damage_prevention_triggered: bool

class MCPGuardianProtocol(GuardianProtocol):
    """
    Enhanced Guardian Protocol for MCP operations
    
    Provides comprehensive security governance for AI agents with external resource access:
    1. Ethical AI boundaries for autonomous operations
    2. Legal compliance validation for external API usage
    3. Authorization scope enforcement by team type
    4. Damage prevention for offensive security tools
    5. Real-time monitoring and intervention
    6. Data classification and protection
    """
    
    def __init__(self, config_path: Optional[str] = None, mcp_config_path: Optional[str] = None):
        super().__init__(config_path)
        
        self.mcp_config_path = mcp_config_path or "/Users/samoakes/Desktop/Archangel/config/mcp_guardian_config.json"
        
        # MCP-specific governance
        self.api_policies: Dict[str, ExternalAPIPolicy] = {}
        self.team_permissions: Dict[MCPTeamType, Set[MCPToolCategory]] = {}
        self.active_penetration_tests: Dict[str, Dict[str, Any]] = {}
        self.blocked_targets: Set[str] = set()
        self.emergency_contacts: Dict[str, str] = {}
        
        # Real-time monitoring
        self.mcp_audit_trail: List[MCPAuditEntry] = []
        self.rate_limit_tracking: Dict[str, Dict[str, Any]] = {}
        self.suspicious_activities: List[Dict[str, Any]] = []
        
        # Load MCP-specific configuration
        self._load_mcp_configuration()
        
        logger.info("MCP Guardian Protocol initialized with enhanced governance")
    
    def _load_mcp_configuration(self):
        """Load MCP-specific Guardian configuration"""
        try:
            config_file = Path(self.mcp_config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    mcp_config = json.load(f)
                self._parse_mcp_configuration(mcp_config)
            else:
                logger.warning(f"MCP configuration file not found: {self.mcp_config_path}")
                self._load_default_mcp_configuration()
        except Exception as e:
            logger.error(f"Failed to load MCP Guardian configuration: {e}")
            self._load_default_mcp_configuration()
    
    def _load_default_mcp_configuration(self):
        """Load default MCP Guardian configuration"""
        # Team permissions
        self.team_permissions = {
            MCPTeamType.RED_TEAM: {
                MCPToolCategory.RECONNAISSANCE,
                MCPToolCategory.VULNERABILITY_SCANNING,
                MCPToolCategory.EXPLOITATION,
                MCPToolCategory.OSINT
            },
            MCPTeamType.BLUE_TEAM: {
                MCPToolCategory.THREAT_INTELLIGENCE,
                MCPToolCategory.INCIDENT_RESPONSE,
                MCPToolCategory.MEMORY_ANALYSIS,
                MCPToolCategory.NETWORK_MONITORING,
                MCPToolCategory.MALWARE_ANALYSIS
            },
            MCPTeamType.NEUTRAL: {
                MCPToolCategory.OSINT,
                MCPToolCategory.INFRASTRUCTURE
            },
            MCPTeamType.ADMIN: set(MCPToolCategory)  # Admin has access to all tools
        }
        
        # External API policies
        self.api_policies = {
            "shodan": ExternalAPIPolicy(
                api_name="shodan",
                allowed_teams=[MCPTeamType.RED_TEAM, MCPTeamType.BLUE_TEAM],
                rate_limits={"requests_per_hour": 1000, "requests_per_minute": 20},
                data_retention_days=30,
                requires_approval=False,
                compliance_frameworks=[ComplianceFramework.SOC2],
                terms_of_service_url="https://www.shodan.io/legal",
                data_sharing_restrictions=["no_resale", "research_only"],
                geographic_restrictions=[]
            ),
            "virustotal": ExternalAPIPolicy(
                api_name="virustotal",
                allowed_teams=[MCPTeamType.BLUE_TEAM],
                rate_limits={"requests_per_minute": 4, "requests_per_day": 1000},
                data_retention_days=90,
                requires_approval=False,
                compliance_frameworks=[ComplianceFramework.SOC2, ComplianceFramework.GDPR],
                terms_of_service_url="https://www.virustotal.com/gui/terms-of-service",
                data_sharing_restrictions=["confidential_handling"],
                geographic_restrictions=["eu_gdpr_compliant"]
            ),
            "exploit_db": ExternalAPIPolicy(
                api_name="exploit_db",
                allowed_teams=[MCPTeamType.RED_TEAM],
                rate_limits={"requests_per_hour": 500},
                data_retention_days=7,
                requires_approval=True,
                compliance_frameworks=[ComplianceFramework.NIST],
                terms_of_service_url="https://www.exploit-db.com/",
                data_sharing_restrictions=["authorized_testing_only"],
                geographic_restrictions=[]
            ),
            "metasploit": ExternalAPIPolicy(
                api_name="metasploit",
                allowed_teams=[MCPTeamType.RED_TEAM],
                rate_limits={"requests_per_hour": 100},
                data_retention_days=1,
                requires_approval=True,
                compliance_frameworks=[ComplianceFramework.NIST, ComplianceFramework.SOC2],
                terms_of_service_url="https://www.rapid7.com/legal/terms/",
                data_sharing_restrictions=["authorized_testing_only", "no_production"],
                geographic_restrictions=[]
            )
        }
        
        # Blocked targets (production systems, sensitive networks)
        self.blocked_targets = {
            "192.168.1.0/24",
            "10.0.0.0/8",
            "172.16.0.0/12",
            "production.*",
            "*.gov",
            "*.mil",
            "*.edu",
            "critical-infrastructure.*"
        }
        
        logger.info("Loaded default MCP Guardian configuration")
    
    def _parse_mcp_configuration(self, config: Dict[str, Any]):
        """Parse MCP configuration from JSON"""
        # Implementation for parsing complex MCP configuration
        if "team_permissions" in config:
            self.team_permissions = config["team_permissions"]
        
        if "api_policies" in config:
            self.api_policies = config["api_policies"]
        
        if "blocked_targets" in config:
            self.blocked_targets = set(config["blocked_targets"])
    
    async def validate_mcp_action(self, 
                                 action: str,
                                 mcp_context: MCPSecurityContext,
                                 tool_parameters: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Comprehensive validation of MCP security action
        
        Args:
            action: Description of the MCP action
            mcp_context: MCP-specific security context
            tool_parameters: Parameters for the MCP tool
            
        Returns:
            ValidationResult with MCP-specific validation
        """
        start_time = time.time()
        
        try:
            # Layer 1: Base Guardian validation
            base_result = await self.validate_action(action, mcp_context.base_context)
            if base_result.decision == GuardianDecision.EMERGENCY_STOP:
                return base_result
            
            # Layer 2: MCP ethical boundaries
            ethical_result = await self._mcp_ethical_validation(action, mcp_context, tool_parameters)
            if ethical_result.decision == GuardianDecision.EMERGENCY_STOP:
                return ethical_result
            
            # Layer 3: Legal compliance validation
            legal_result = await self._mcp_legal_compliance(action, mcp_context, tool_parameters)
            if legal_result.decision == GuardianDecision.EMERGENCY_STOP:
                return legal_result
            
            # Layer 4: Authorization scope validation
            auth_result = await self._mcp_authorization_validation(action, mcp_context, tool_parameters)
            if auth_result.decision == GuardianDecision.EMERGENCY_STOP:
                return auth_result
            
            # Layer 5: Damage prevention
            damage_result = await self._mcp_damage_prevention(action, mcp_context, tool_parameters)
            if damage_result.decision == GuardianDecision.EMERGENCY_STOP:
                return damage_result
            
            # Layer 6: Data protection validation
            data_result = await self._mcp_data_protection(action, mcp_context, tool_parameters)
            
            # Combine all results
            all_results = [base_result, ethical_result, legal_result, auth_result, damage_result, data_result]
            final_result = self._combine_validation_results(all_results)
            
            # Create MCP audit entry
            mcp_audit_entry = await self._create_mcp_audit_entry(action, mcp_context, final_result)
            final_result.audit_entry_id = mcp_audit_entry.entry_id
            
            # Real-time monitoring
            await self._update_mcp_monitoring(mcp_context, final_result)
            
            validation_time = (time.time() - start_time) * 1000
            logger.info(f"MCP Guardian validation complete: {final_result.decision.value} "
                       f"(risk: {final_result.risk_score:.2f}, time: {validation_time:.2f}ms)")
            
            return final_result
            
        except Exception as e:
            logger.error(f"MCP Guardian validation failed: {e}")
            return ValidationResult(
                decision=GuardianDecision.EMERGENCY_STOP,
                risk_score=10.0,
                compliance_status={},
                validation_errors=[f"MCP validation system error: {str(e)}"],
                required_approvals=[],
                audit_entry_id="",
                emergency_stop_triggered=True,
                recommendations=["Contact MCP security team immediately"]
            )
    
    async def _mcp_ethical_validation(self, action: str, mcp_context: MCPSecurityContext,
                                    tool_parameters: Optional[Dict[str, Any]]) -> ValidationResult:
        """MCP ethical boundaries validation"""
        errors = []
        risk_score = 0.0
        recommendations = []
        emergency_stop = False
        
        # Ethical boundary checks for autonomous AI agents
        ethical_violations = self._check_mcp_ethical_boundaries(action, mcp_context, tool_parameters)
        if ethical_violations:
            errors.extend(ethical_violations)
            risk_score += len(ethical_violations) * 2.0
        
        # Autonomous operation limits
        if self._exceeds_autonomous_limits(mcp_context, tool_parameters):
            errors.append("Operation exceeds autonomous AI limits")
            risk_score += 3.0
            recommendations.append("Require human supervision for this operation")
        
        # Red team ethical constraints
        if mcp_context.team_type == MCPTeamType.RED_TEAM:
            red_team_violations = self._check_red_team_ethics(action, mcp_context, tool_parameters)
            if red_team_violations:
                errors.extend(red_team_violations)
                risk_score += len(red_team_violations) * 1.5
        
        # Unauthorized target detection
        if self._targets_unauthorized_systems(mcp_context.target_systems):
            emergency_stop = True
            errors.append("Targeting unauthorized systems detected")
            risk_score = 10.0
        
        # Determine decision
        decision = GuardianDecision.ALLOW
        if emergency_stop or risk_score >= 8.0:
            decision = GuardianDecision.EMERGENCY_STOP
        elif risk_score >= 5.0:
            decision = GuardianDecision.DENY
        elif risk_score >= 3.0:
            decision = GuardianDecision.REQUIRE_APPROVAL
        elif risk_score >= 1.0:
            decision = GuardianDecision.MONITOR
        
        return ValidationResult(
            decision=decision,
            risk_score=risk_score,
            compliance_status={},
            validation_errors=errors,
            required_approvals=[],
            audit_entry_id="",
            emergency_stop_triggered=emergency_stop,
            recommendations=recommendations
        )
    
    async def _mcp_legal_compliance(self, action: str, mcp_context: MCPSecurityContext,
                                  tool_parameters: Optional[Dict[str, Any]]) -> ValidationResult:
        """Legal compliance validation for external API usage"""
        errors = []
        risk_score = 0.0
        compliance_status = {}
        required_approvals = []
        
        # Check legal authorization level
        if mcp_context.legal_authorization == LegalAuthorization.UNAUTHORIZED:
            errors.append("No legal authorization for this operation")
            risk_score += 5.0
        elif mcp_context.legal_authorization == LegalAuthorization.RESEARCH_ONLY:
            if mcp_context.team_type == MCPTeamType.RED_TEAM and "exploit" in action.lower():
                errors.append("Research-only authorization insufficient for exploitation")
                risk_score += 3.0
        
        # External API compliance
        for api_name in mcp_context.external_apis:
            if api_name in self.api_policies:
                policy = self.api_policies[api_name]
                
                # Team authorization check
                if mcp_context.team_type not in policy.allowed_teams:
                    errors.append(f"Team {mcp_context.team_type.value} not authorized for {api_name}")
                    risk_score += 2.0
                
                # Rate limiting compliance
                if not self._check_api_rate_limits(mcp_context.mcp_agent_id, api_name, policy):
                    errors.append(f"Rate limit exceeded for {api_name}")
                    risk_score += 1.0
                
                # Approval requirements
                if policy.requires_approval and not mcp_context.business_approval:
                    required_approvals.append(f"business_approval_for_{api_name}")
                    risk_score += 1.0
                
                # Compliance framework validation
                for framework in policy.compliance_frameworks:
                    framework_valid = await self._validate_api_compliance(api_name, framework, mcp_context)
                    compliance_status[framework] = framework_valid
                    if not framework_valid:
                        errors.append(f"{api_name} fails {framework.value} compliance")
                        risk_score += 1.5
        
        # Penetration testing scope validation
        if (mcp_context.team_type == MCPTeamType.RED_TEAM and 
            mcp_context.tool_category in [MCPToolCategory.VULNERABILITY_SCANNING, MCPToolCategory.EXPLOITATION]):
            
            if not mcp_context.penetration_test_scope:
                errors.append("Penetration testing requires documented scope")
                risk_score += 2.0
                required_approvals.append("penetration_test_authorization")
        
        # Geographic compliance
        if self._violates_geographic_restrictions(mcp_context):
            errors.append("Operation violates geographic data restrictions")
            risk_score += 2.0
        
        # Determine decision
        decision = GuardianDecision.ALLOW
        if risk_score >= 7.0:
            decision = GuardianDecision.DENY
        elif required_approvals or risk_score >= 4.0:
            decision = GuardianDecision.REQUIRE_APPROVAL
        elif risk_score >= 2.0:
            decision = GuardianDecision.MONITOR
        
        return ValidationResult(
            decision=decision,
            risk_score=risk_score,
            compliance_status=compliance_status,
            validation_errors=errors,
            required_approvals=required_approvals,
            audit_entry_id="",
            emergency_stop_triggered=False,
            recommendations=[]
        )
    
    async def _mcp_authorization_validation(self, action: str, mcp_context: MCPSecurityContext,
                                          tool_parameters: Optional[Dict[str, Any]]) -> ValidationResult:
        """Authorization scope validation for MCP operations"""
        errors = []
        risk_score = 0.0
        
        # Team permission validation
        if mcp_context.tool_category not in self.team_permissions.get(mcp_context.team_type, set()):
            errors.append(f"Team {mcp_context.team_type.value} not authorized for {mcp_context.tool_category.value}")
            risk_score += 3.0
        
        # Supervisor approval for high-risk operations
        if (mcp_context.base_context.risk_level.value >= 4 and 
            not mcp_context.technical_supervisor):
            errors.append("High-risk operation requires technical supervisor approval")
            risk_score += 2.0
        
        # Cross-team operation restrictions
        if self._involves_cross_team_operation(mcp_context, tool_parameters):
            errors.append("Cross-team operation detected - requires additional authorization")
            risk_score += 1.5
        
        # Time-based restrictions
        if self._violates_time_restrictions(mcp_context):
            errors.append("Operation attempted outside authorized time windows")
            risk_score += 1.0
        
        # Resource quota validation
        if self._exceeds_resource_quotas(mcp_context):
            errors.append("Operation exceeds authorized resource quotas")
            risk_score += 1.0
        
        # Determine decision
        decision = GuardianDecision.ALLOW
        if risk_score >= 6.0:
            decision = GuardianDecision.DENY
        elif risk_score >= 3.0:
            decision = GuardianDecision.REQUIRE_APPROVAL
        elif risk_score >= 1.0:
            decision = GuardianDecision.MONITOR
        
        return ValidationResult(
            decision=decision,
            risk_score=risk_score,
            compliance_status={},
            validation_errors=errors,
            required_approvals=[],
            audit_entry_id="",
            emergency_stop_triggered=False,
            recommendations=[]
        )
    
    async def _mcp_damage_prevention(self, action: str, mcp_context: MCPSecurityContext,
                                   tool_parameters: Optional[Dict[str, Any]]) -> ValidationResult:
        """Damage prevention for MCP tool execution"""
        errors = []
        risk_score = 0.0
        emergency_stop = False
        recommendations = []
        
        # Production system protection
        production_targets = self._identify_production_targets(mcp_context.target_systems)
        if production_targets:
            emergency_stop = True
            errors.append(f"Production systems targeted: {production_targets}")
            risk_score = 10.0
        
        # Destructive operation detection
        if self._is_destructive_operation(action, tool_parameters):
            if mcp_context.legal_authorization != LegalAuthorization.PRODUCTION_APPROVED:
                emergency_stop = True
                errors.append("Destructive operation without production approval")
                risk_score = 10.0
        
        # Network range validation
        dangerous_ranges = self._check_dangerous_network_ranges(mcp_context.target_systems)
        if dangerous_ranges:
            errors.extend([f"Dangerous network range targeted: {range_}" for range_ in dangerous_ranges])
            risk_score += len(dangerous_ranges) * 2.0
        
        # Tool-specific safety checks
        tool_risks = self._assess_tool_specific_risks(mcp_context.tool_name, tool_parameters)
        if tool_risks:
            errors.extend(tool_risks)
            risk_score += len(tool_risks) * 1.0
        
        # Rate limiting for aggressive operations
        if self._is_aggressive_operation(action, mcp_context):
            if not self._check_aggressive_operation_limits(mcp_context):
                errors.append("Aggressive operation rate limit exceeded")
                risk_score += 2.0
        
        # Sandbox validation
        if not self._is_sandboxed_environment(mcp_context):
            if mcp_context.team_type == MCPTeamType.RED_TEAM:
                errors.append("Red team operations must be sandboxed")
                risk_score += 3.0
                recommendations.append("Ensure operations are contained within test environment")
        
        # Determine decision
        decision = GuardianDecision.ALLOW
        if emergency_stop or risk_score >= 8.0:
            decision = GuardianDecision.EMERGENCY_STOP
        elif risk_score >= 5.0:
            decision = GuardianDecision.DENY
        elif risk_score >= 3.0:
            decision = GuardianDecision.REQUIRE_APPROVAL
        elif risk_score >= 1.0:
            decision = GuardianDecision.MONITOR
        
        return ValidationResult(
            decision=decision,
            risk_score=risk_score,
            compliance_status={},
            validation_errors=errors,
            required_approvals=[],
            audit_entry_id="",
            emergency_stop_triggered=emergency_stop,
            recommendations=recommendations
        )
    
    async def _mcp_data_protection(self, action: str, mcp_context: MCPSecurityContext,
                                 tool_parameters: Optional[Dict[str, Any]]) -> ValidationResult:
        """Data classification and protection validation"""
        errors = []
        risk_score = 0.0
        
        # Data classification validation
        if mcp_context.data_classification in [DataClassification.RESTRICTED, DataClassification.TOP_SECRET]:
            if mcp_context.base_context.authorization_level not in ["security_clearance", "admin"]:
                errors.append(f"Insufficient clearance for {mcp_context.data_classification.value} data")
                risk_score += 4.0
        
        # PII/PHI protection
        if self._involves_sensitive_data(action, tool_parameters):
            if ComplianceFramework.GDPR not in mcp_context.base_context.compliance_requirements:
                errors.append("Sensitive data operation requires GDPR compliance")
                risk_score += 2.0
        
        # Data retention validation
        for api_name in mcp_context.external_apis:
            if api_name in self.api_policies:
                policy = self.api_policies[api_name]
                if policy.data_retention_days < self._get_required_retention_days(mcp_context):
                    errors.append(f"Data retention policy violation for {api_name}")
                    risk_score += 1.5
        
        # Cross-border data transfer
        if self._involves_cross_border_transfer(mcp_context):
            errors.append("Cross-border data transfer requires additional compliance checks")
            risk_score += 1.0
        
        # Data sharing restrictions
        if self._violates_data_sharing_restrictions(mcp_context):
            errors.append("Operation violates data sharing restrictions")
            risk_score += 2.0
        
        # Determine decision
        decision = GuardianDecision.ALLOW
        if risk_score >= 6.0:
            decision = GuardianDecision.DENY
        elif risk_score >= 3.0:
            decision = GuardianDecision.REQUIRE_APPROVAL
        elif risk_score >= 1.0:
            decision = GuardianDecision.MONITOR
        
        return ValidationResult(
            decision=decision,
            risk_score=risk_score,
            compliance_status={},
            validation_errors=errors,
            required_approvals=[],
            audit_entry_id="",
            emergency_stop_triggered=False,
            recommendations=[]
        )
    
    async def _create_mcp_audit_entry(self, action: str, mcp_context: MCPSecurityContext,
                                    result: ValidationResult) -> MCPAuditEntry:
        """Create MCP-specific audit entry"""
        entry_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        # Create MCP audit data
        mcp_audit_data = {
            "entry_id": entry_id,
            "timestamp": timestamp.isoformat(),
            "mcp_agent_id": mcp_context.mcp_agent_id,
            "team_type": mcp_context.team_type.value,
            "tool_name": mcp_context.tool_name,
            "tool_category": mcp_context.tool_category.value,
            "action": action,
            "decision": result.decision.value,
            "risk_score": result.risk_score,
            "external_apis": mcp_context.external_apis,
            "target_systems": mcp_context.target_systems,
            "legal_authorization": mcp_context.legal_authorization.value,
            "data_classification": mcp_context.data_classification.value
        }
        
        # Generate tamper-proof checksum
        checksum_data = json.dumps(mcp_audit_data, sort_keys=True)
        checksum = hashlib.sha256(checksum_data.encode()).hexdigest()
        
        mcp_audit_entry = MCPAuditEntry(
            entry_id=entry_id,
            timestamp=timestamp,
            user_id=mcp_context.base_context.user_id,
            action=action,
            decision=result.decision,
            risk_level=mcp_context.base_context.risk_level,
            security_context=asdict(mcp_context.base_context),
            justification=mcp_context.base_context.business_justification,
            approval_chain=[mcp_context.technical_supervisor] if mcp_context.technical_supervisor else [],
            compliance_tags=[fw.value for fw in mcp_context.base_context.compliance_requirements],
            checksum=checksum,
            mcp_agent_id=mcp_context.mcp_agent_id,
            team_type=mcp_context.team_type,
            tool_name=mcp_context.tool_name,
            tool_category=mcp_context.tool_category,
            external_api_calls=mcp_context.external_apis,
            data_accessed=[],  # Will be populated by monitoring
            legal_authorization=mcp_context.legal_authorization,
            supervisor_approval=mcp_context.technical_supervisor,
            damage_prevention_triggered=result.emergency_stop_triggered
        )
        
        self.mcp_audit_trail.append(mcp_audit_entry)
        
        # Persist MCP audit entry
        await self._persist_mcp_audit_entry(mcp_audit_entry)
        
        logger.info(f"MCP audit entry created: {entry_id}")
        return mcp_audit_entry
    
    async def _update_mcp_monitoring(self, mcp_context: MCPSecurityContext, result: ValidationResult):
        """Update real-time MCP monitoring"""
        # Update rate limiting tracking
        for api_name in mcp_context.external_apis:
            if api_name not in self.rate_limit_tracking:
                self.rate_limit_tracking[api_name] = {"requests": [], "violations": 0}
            
            self.rate_limit_tracking[api_name]["requests"].append({
                "timestamp": datetime.now(),
                "agent_id": mcp_context.mcp_agent_id,
                "team_type": mcp_context.team_type.value
            })
        
        # Track suspicious activities
        if result.risk_score >= 7.0:
            self.suspicious_activities.append({
                "timestamp": datetime.now(),
                "agent_id": mcp_context.mcp_agent_id,
                "team_type": mcp_context.team_type.value,
                "tool_name": mcp_context.tool_name,
                "risk_score": result.risk_score,
                "decision": result.decision.value,
                "errors": result.validation_errors
            })
    
    # Helper methods for MCP-specific validation
    
    def _check_mcp_ethical_boundaries(self, action: str, mcp_context: MCPSecurityContext,
                                    tool_parameters: Optional[Dict[str, Any]]) -> List[str]:
        """Check MCP-specific ethical boundaries"""
        violations = []
        
        # Autonomous AI constraints
        if "autonomous" in action.lower() and mcp_context.base_context.authorization_level == "guest":
            violations.append("Autonomous operations not authorized for guest users")
        
        # Red team ethical constraints
        if mcp_context.team_type == MCPTeamType.RED_TEAM:
            if "production" in action.lower() and mcp_context.legal_authorization != LegalAuthorization.PRODUCTION_APPROVED:
                violations.append("Red team production access without proper authorization")
            
            if any(keyword in action.lower() for keyword in ["steal", "exfiltrate", "destroy"]):
                violations.append("Red team operation contains unethical objectives")
        
        # Blue team constraints
        if mcp_context.team_type == MCPTeamType.BLUE_TEAM:
            if mcp_context.tool_category == MCPToolCategory.EXPLOITATION:
                violations.append("Blue team should not use exploitation tools")
        
        return violations
    
    def _exceeds_autonomous_limits(self, mcp_context: MCPSecurityContext,
                                 tool_parameters: Optional[Dict[str, Any]]) -> bool:
        """Check if operation exceeds autonomous AI limits"""
        # High-risk operations require human oversight
        if mcp_context.base_context.risk_level.value >= 4:
            return not mcp_context.technical_supervisor
        
        # Exploitation tools require approval
        if mcp_context.tool_category == MCPToolCategory.EXPLOITATION:
            return not mcp_context.business_approval
        
        return False
    
    def _check_red_team_ethics(self, action: str, mcp_context: MCPSecurityContext,
                             tool_parameters: Optional[Dict[str, Any]]) -> List[str]:
        """Check red team specific ethical constraints"""
        violations = []
        
        # Require penetration test documentation
        if mcp_context.tool_category in [MCPToolCategory.VULNERABILITY_SCANNING, MCPToolCategory.EXPLOITATION]:
            if not mcp_context.penetration_test_scope:
                violations.append("Red team operation requires penetration test scope documentation")
        
        # Prevent real attacks
        if tool_parameters and tool_parameters.get("live_target", False):
            if mcp_context.legal_authorization != LegalAuthorization.PRODUCTION_APPROVED:
                violations.append("Live target attacks require production approval")
        
        return violations
    
    def _targets_unauthorized_systems(self, target_systems: List[str]) -> bool:
        """Check if targeting unauthorized systems"""
        for target in target_systems:
            for blocked_pattern in self.blocked_targets:
                if self._matches_target_pattern(target, blocked_pattern):
                    return True
        return False
    
    def _matches_target_pattern(self, target: str, pattern: str) -> bool:
        """Check if target matches blocked pattern"""
        try:
            # IP network matching
            if "/" in pattern:
                return ipaddress.ip_address(target) in ipaddress.ip_network(pattern, strict=False)
            
            # Domain pattern matching
            if "*" in pattern:
                regex_pattern = pattern.replace("*", ".*")
                return bool(re.match(regex_pattern, target))
            
            # Exact match
            return target == pattern
            
        except (ValueError, ipaddress.AddressValueError):
            # Fallback to string matching
            return pattern.lower() in target.lower()
    
    def _check_api_rate_limits(self, agent_id: str, api_name: str, policy: ExternalAPIPolicy) -> bool:
        """Check API rate limits"""
        if api_name not in self.rate_limit_tracking:
            return True
        
        now = datetime.now()
        requests = self.rate_limit_tracking[api_name]["requests"]
        
        # Check hourly limit
        hour_ago = now - timedelta(hours=1)
        recent_requests = [r for r in requests if r["timestamp"] > hour_ago and r["agent_id"] == agent_id]
        
        if len(recent_requests) >= policy.rate_limits.get("requests_per_hour", float('inf')):
            return False
        
        # Check minute limit
        if "requests_per_minute" in policy.rate_limits:
            minute_ago = now - timedelta(minutes=1)
            recent_minute_requests = [r for r in requests if r["timestamp"] > minute_ago and r["agent_id"] == agent_id]
            
            if len(recent_minute_requests) >= policy.rate_limits["requests_per_minute"]:
                return False
        
        return True
    
    def _identify_production_targets(self, target_systems: List[str]) -> List[str]:
        """Identify production systems in target list"""
        production_indicators = ["prod", "production", "live", "prd"]
        production_targets = []
        
        for target in target_systems:
            if any(indicator in target.lower() for indicator in production_indicators):
                production_targets.append(target)
        
        return production_targets
    
    def _is_destructive_operation(self, action: str, tool_parameters: Optional[Dict[str, Any]]) -> bool:
        """Check if operation is destructive"""
        destructive_keywords = [
            "delete", "destroy", "wipe", "format", "drop", "truncate",
            "shutdown", "kill", "terminate", "exploit", "attack", "compromise"
        ]
        
        action_lower = action.lower()
        destructive_count = sum(1 for keyword in destructive_keywords if keyword in action_lower)
        
        # Also check tool parameters
        if tool_parameters:
            param_str = json.dumps(tool_parameters).lower()
            destructive_count += sum(1 for keyword in destructive_keywords if keyword in param_str)
        
        return destructive_count >= 2  # Threshold for destructive operations
    
    def _check_dangerous_network_ranges(self, target_systems: List[str]) -> List[str]:
        """Check for dangerous network ranges"""
        dangerous_ranges = []
        sensitive_networks = [
            "192.168.0.0/16",  # Private networks
            "10.0.0.0/8",
            "172.16.0.0/12",
            "127.0.0.0/8",     # Localhost
            "169.254.0.0/16"   # Link-local
        ]
        
        for target in target_systems:
            try:
                target_ip = ipaddress.ip_address(target)
                for network in sensitive_networks:
                    if target_ip in ipaddress.ip_network(network):
                        dangerous_ranges.append(f"{target} in {network}")
            except ValueError:
                # Not an IP address, skip network checks
                continue
        
        return dangerous_ranges
    
    def _assess_tool_specific_risks(self, tool_name: str, tool_parameters: Optional[Dict[str, Any]]) -> List[str]:
        """Assess risks specific to the tool being used"""
        risks = []
        
        # Metasploit risks
        if "metasploit" in tool_name.lower():
            if tool_parameters and tool_parameters.get("payload"):
                risks.append("Metasploit payload execution detected - high risk")
        
        # SQLMap risks
        if "sqlmap" in tool_name.lower():
            if tool_parameters and tool_parameters.get("risk", 1) > 2:
                risks.append("SQLMap high-risk injection testing")
        
        # Nuclei risks
        if "nuclei" in tool_name.lower():
            if tool_parameters and "all" in str(tool_parameters.get("templates", "")):
                risks.append("Nuclei running all templates - potential for aggressive scanning")
        
        return risks
    
    def _is_aggressive_operation(self, action: str, mcp_context: MCPSecurityContext) -> bool:
        """Check if operation is aggressive (high volume/intensive)"""
        aggressive_indicators = [
            "scan", "brute", "fuzz", "enumerate", "crawl", "spider"
        ]
        
        return any(indicator in action.lower() for indicator in aggressive_indicators)
    
    def _check_aggressive_operation_limits(self, mcp_context: MCPSecurityContext) -> bool:
        """Check limits for aggressive operations"""
        # Simplified implementation - in production would check detailed rate limits
        return True  # Allow for now, but log for monitoring
    
    def _is_sandboxed_environment(self, mcp_context: MCPSecurityContext) -> bool:
        """Check if operation is in sandboxed environment"""
        # Check if targets are in test/sandbox ranges
        test_indicators = ["test", "sandbox", "lab", "dev", "staging"]
        
        for target in mcp_context.target_systems:
            if any(indicator in target.lower() for indicator in test_indicators):
                return True
        
        # Check for sandbox IP ranges
        sandbox_ranges = ["192.168.100.0/24", "10.0.100.0/24"]
        for target in mcp_context.target_systems:
            try:
                target_ip = ipaddress.ip_address(target)
                for network in sandbox_ranges:
                    if target_ip in ipaddress.ip_network(network):
                        return True
            except ValueError:
                continue
        
        return False
    
    async def _validate_api_compliance(self, api_name: str, framework: ComplianceFramework,
                                     mcp_context: MCPSecurityContext) -> bool:
        """Validate API usage against compliance framework"""
        # Simplified compliance validation
        if framework == ComplianceFramework.GDPR:
            return mcp_context.data_classification != DataClassification.TOP_SECRET
        
        if framework == ComplianceFramework.SOC2:
            return mcp_context.legal_authorization != LegalAuthorization.UNAUTHORIZED
        
        return True
    
    def _violates_geographic_restrictions(self, mcp_context: MCPSecurityContext) -> bool:
        """Check for geographic restriction violations"""
        # Simplified geographic compliance
        for api_name in mcp_context.external_apis:
            if api_name in self.api_policies:
                policy = self.api_policies[api_name]
                if "eu_gdpr_compliant" in policy.geographic_restrictions:
                    # Would check actual geographic location in production
                    pass
        
        return False
    
    def _involves_cross_team_operation(self, mcp_context: MCPSecurityContext,
                                     tool_parameters: Optional[Dict[str, Any]]) -> bool:
        """Check if operation involves cross-team resources"""
        # Check if accessing tools/data from other teams
        if tool_parameters and "cross_team" in str(tool_parameters):
            return True
        
        return False
    
    def _violates_time_restrictions(self, mcp_context: MCPSecurityContext) -> bool:
        """Check time-based operation restrictions"""
        current_hour = datetime.now().hour
        
        # High-risk operations only during business hours
        if mcp_context.base_context.risk_level.value >= 4:
            return current_hour < 8 or current_hour > 18
        
        # Red team operations have time restrictions
        if mcp_context.team_type == MCPTeamType.RED_TEAM:
            if mcp_context.tool_category == MCPToolCategory.EXPLOITATION:
                return current_hour < 9 or current_hour > 17
        
        return False
    
    def _exceeds_resource_quotas(self, mcp_context: MCPSecurityContext) -> bool:
        """Check resource quota violations"""
        # Simplified quota checking
        # In production, would check against detailed resource usage metrics
        return False
    
    def _involves_sensitive_data(self, action: str, tool_parameters: Optional[Dict[str, Any]]) -> bool:
        """Check if operation involves sensitive data"""
        sensitive_keywords = ["personal", "private", "confidential", "pii", "phi", "ssn", "credit"]
        
        action_lower = action.lower()
        if any(keyword in action_lower for keyword in sensitive_keywords):
            return True
        
        if tool_parameters:
            param_str = json.dumps(tool_parameters).lower()
            if any(keyword in param_str for keyword in sensitive_keywords):
                return True
        
        return False
    
    def _get_required_retention_days(self, mcp_context: MCPSecurityContext) -> int:
        """Get required data retention days based on classification"""
        retention_requirements = {
            DataClassification.PUBLIC: 30,
            DataClassification.INTERNAL: 90,
            DataClassification.CONFIDENTIAL: 365,
            DataClassification.RESTRICTED: 2555,  # 7 years
            DataClassification.TOP_SECRET: 3650   # 10 years
        }
        
        return retention_requirements.get(mcp_context.data_classification, 90)
    
    def _involves_cross_border_transfer(self, mcp_context: MCPSecurityContext) -> bool:
        """Check if operation involves cross-border data transfer"""
        # Simplified cross-border detection
        international_apis = ["shodan", "virustotal", "exploit_db"]
        
        return any(api in international_apis for api in mcp_context.external_apis)
    
    def _violates_data_sharing_restrictions(self, mcp_context: MCPSecurityContext) -> bool:
        """Check data sharing restriction violations"""
        for api_name in mcp_context.external_apis:
            if api_name in self.api_policies:
                policy = self.api_policies[api_name]
                
                # Check restrictions
                if "no_resale" in policy.data_sharing_restrictions:
                    # Would validate against actual usage patterns
                    pass
                
                if "authorized_testing_only" in policy.data_sharing_restrictions:
                    if mcp_context.legal_authorization not in [
                        LegalAuthorization.AUTHORIZED_TESTING,
                        LegalAuthorization.PRODUCTION_APPROVED
                    ]:
                        return True
        
        return False
    
    async def _persist_mcp_audit_entry(self, audit_entry: MCPAuditEntry):
        """Persist MCP audit entry to secure storage"""
        # In production, this would write to secure, tamper-proof storage
        logger.debug(f"Persisting MCP audit entry: {audit_entry.entry_id}")
    
    # Public API extensions for MCP
    
    def register_penetration_test(self, test_id: str, scope: str, authorization: str,
                                supervisor: str, duration_days: int) -> bool:
        """Register authorized penetration test"""
        try:
            self.active_penetration_tests[test_id] = {
                "scope": scope,
                "authorization": authorization,
                "supervisor": supervisor,
                "start_date": datetime.now(),
                "end_date": datetime.now() + timedelta(days=duration_days),
                "active": True
            }
            
            logger.info(f"Penetration test registered: {test_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register penetration test: {e}")
            return False
    
    def revoke_penetration_test(self, test_id: str, user_id: str) -> bool:
        """Revoke active penetration test"""
        try:
            if test_id in self.active_penetration_tests:
                self.active_penetration_tests[test_id]["active"] = False
                self.active_penetration_tests[test_id]["revoked_by"] = user_id
                self.active_penetration_tests[test_id]["revoked_at"] = datetime.now()
                
                logger.warning(f"Penetration test revoked: {test_id} by {user_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to revoke penetration test: {e}")
            return False
    
    def add_blocked_target(self, target: str, reason: str, user_id: str) -> bool:
        """Add target to blocked list"""
        try:
            self.blocked_targets.add(target)
            
            # Create audit entry for blocking
            block_entry = {
                "target": target,
                "reason": reason,
                "blocked_by": user_id,
                "blocked_at": datetime.now()
            }
            
            logger.warning(f"Target blocked: {target} by {user_id} - {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to block target: {e}")
            return False
    
    def remove_blocked_target(self, target: str, authorization_code: str, user_id: str) -> bool:
        """Remove target from blocked list (requires authorization)"""
        try:
            if authorization_code == "MCP_GUARDIAN_UNBLOCK":
                if target in self.blocked_targets:
                    self.blocked_targets.remove(target)
                    logger.warning(f"Target unblocked: {target} by {user_id}")
                    return True
            else:
                logger.error(f"Invalid unblock authorization by {user_id}")
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to unblock target: {e}")
            return False
    
    def get_mcp_system_status(self) -> Dict[str, Any]:
        """Get MCP Guardian system status"""
        return {
            "mcp_guardian_version": "1.0.0",
            "base_guardian_status": self.get_system_status(),
            "active_penetration_tests": len([t for t in self.active_penetration_tests.values() if t["active"]]),
            "blocked_targets": len(self.blocked_targets),
            "mcp_audit_entries": len(self.mcp_audit_trail),
            "api_policies": len(self.api_policies),
            "team_permissions": {team.value: len(perms) for team, perms in self.team_permissions.items()},
            "suspicious_activities": len(self.suspicious_activities),
            "rate_limit_tracking": len(self.rate_limit_tracking),
            "last_updated": datetime.now().isoformat()
        }
    
    def generate_mcp_compliance_report(self, framework: ComplianceFramework,
                                     start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate MCP-specific compliance report"""
        base_report = self.generate_compliance_report(framework, start_date, end_date)
        
        # Add MCP-specific compliance data
        relevant_mcp_entries = [
            entry for entry in self.mcp_audit_trail
            if start_date <= entry.timestamp <= end_date
            and framework.value in entry.compliance_tags
        ]
        
        mcp_report = {
            "base_compliance": base_report,
            "mcp_specific": {
                "total_mcp_events": len(relevant_mcp_entries),
                "team_breakdown": {},
                "tool_usage": {},
                "external_api_usage": {},
                "legal_authorizations": {},
                "damage_prevention_triggers": len([e for e in relevant_mcp_entries if e.damage_prevention_triggered]),
                "high_risk_operations": len([e for e in relevant_mcp_entries if e.risk_level.value >= 4])
            }
        }
        
        # Team breakdown
        for entry in relevant_mcp_entries:
            team = entry.team_type.value
            mcp_report["mcp_specific"]["team_breakdown"][team] = \
                mcp_report["mcp_specific"]["team_breakdown"].get(team, 0) + 1
        
        # Tool usage breakdown
        for entry in relevant_mcp_entries:
            tool = entry.tool_name
            mcp_report["mcp_specific"]["tool_usage"][tool] = \
                mcp_report["mcp_specific"]["tool_usage"].get(tool, 0) + 1
        
        # External API usage
        for entry in relevant_mcp_entries:
            for api in entry.external_api_calls:
                mcp_report["mcp_specific"]["external_api_usage"][api] = \
                    mcp_report["mcp_specific"]["external_api_usage"].get(api, 0) + 1
        
        # Legal authorization breakdown
        for entry in relevant_mcp_entries:
            auth = entry.legal_authorization.value
            mcp_report["mcp_specific"]["legal_authorizations"][auth] = \
                mcp_report["mcp_specific"]["legal_authorizations"].get(auth, 0) + 1
        
        return mcp_report