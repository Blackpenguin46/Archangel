"""
Archangel Guardian Protocol
Multi-layer ethical AI governance and security compliance framework

This module implements the Guardian Protocol - a comprehensive system that ensures
all AI security decisions are ethical, legal, and safe.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

class GuardianDecision(Enum):
    """Guardian Protocol decisions"""
    ALLOW = "allow"
    DENY = "deny" 
    REQUIRE_APPROVAL = "require_approval"
    MONITOR = "monitor"
    DEFER = "defer"
    EMERGENCY_STOP = "emergency_stop"

class RiskLevel(Enum):
    """Risk assessment levels"""
    MINIMAL = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    CRITICAL = 5
    CATASTROPHIC = 6

class ComplianceFramework(Enum):
    """Supported compliance frameworks"""
    SOC2 = "soc2"
    HIPAA = "hipaa"
    GDPR = "gdpr"
    PCI_DSS = "pci_dss"
    NIST = "nist"
    ISO27001 = "iso27001"
    FISMA = "fisma"
    SOX = "sox"

class ActionCategory(Enum):
    """Categories of security actions"""
    RECONNAISSANCE = "reconnaissance"
    PENETRATION_TESTING = "penetration_testing"
    VULNERABILITY_SCANNING = "vulnerability_scanning"
    INCIDENT_RESPONSE = "incident_response"
    THREAT_HUNTING = "threat_hunting"
    LOG_ANALYSIS = "log_analysis"
    NETWORK_MONITORING = "network_monitoring"
    SYSTEM_MODIFICATION = "system_modification"
    DATA_ACCESS = "data_access"
    OFFENSIVE_SECURITY = "offensive_security"

@dataclass
class SecurityContext:
    """Security context for Guardian validation"""
    user_id: str
    session_id: str
    source_ip: str
    target_system: Optional[str]
    action_category: ActionCategory
    risk_level: RiskLevel
    timestamp: datetime
    authorization_level: str
    business_justification: str
    compliance_requirements: List[ComplianceFramework]

@dataclass
class GuardianRule:
    """Guardian Protocol rule definition"""
    rule_id: str
    name: str
    description: str
    category: ActionCategory
    compliance_frameworks: List[ComplianceFramework]
    conditions: Dict[str, Any]
    decision: GuardianDecision
    requires_approval: bool
    emergency_stop: bool
    metadata: Dict[str, Any]

@dataclass
class AuditEntry:
    """Audit trail entry"""
    entry_id: str
    timestamp: datetime
    user_id: str
    action: str
    decision: GuardianDecision
    risk_level: RiskLevel
    security_context: Dict[str, Any]
    justification: str
    approval_chain: List[str]
    compliance_tags: List[str]
    checksum: str

@dataclass 
class ValidationResult:
    """Result of Guardian Protocol validation"""
    decision: GuardianDecision
    risk_score: float
    compliance_status: Dict[ComplianceFramework, bool]
    validation_errors: List[str]
    required_approvals: List[str]
    audit_entry_id: str
    emergency_stop_triggered: bool
    recommendations: List[str]

class GuardianProtocol:
    """
    Core Guardian Protocol implementation
    
    Provides multi-layer validation for AI security decisions:
    1. Pre-check validation (basic safety)
    2. Policy validation (rules and compliance)
    3. AI safety check (ethical boundaries)
    4. Damage prevention (fail-safe mechanisms)
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "/Users/samoakes/Desktop/Archangel/config/guardian_config.json"
        self.rules: List[GuardianRule] = []
        self.audit_trail: List[AuditEntry] = []
        self.emergency_stop_active = False
        self.pending_approvals: Dict[str, Dict[str, Any]] = {}
        
        # Load configuration
        self._load_configuration()
        
        # Initialize compliance frameworks
        self.compliance_handlers = {
            ComplianceFramework.SOC2: self._validate_soc2,
            ComplianceFramework.HIPAA: self._validate_hipaa,
            ComplianceFramework.GDPR: self._validate_gdpr,
            ComplianceFramework.PCI_DSS: self._validate_pci_dss,
            ComplianceFramework.NIST: self._validate_nist,
            ComplianceFramework.ISO27001: self._validate_iso27001,
            ComplianceFramework.FISMA: self._validate_fisma,
            ComplianceFramework.SOX: self._validate_sox
        }
        
        logger.info("Guardian Protocol initialized with multi-layer validation")
    
    def _load_configuration(self):
        """Load Guardian Protocol configuration"""
        try:
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                self._parse_configuration(config)
            else:
                logger.warning(f"Configuration file not found: {self.config_path}")
                self._load_default_configuration()
        except Exception as e:
            logger.error(f"Failed to load Guardian configuration: {e}")
            self._load_default_configuration()
    
    def _load_default_configuration(self):
        """Load default Guardian Protocol rules"""
        default_rules = [
            GuardianRule(
                rule_id="PROD_ACCESS_DENY",
                name="Production Access Prohibition",
                description="Deny all unauthorized production system access",
                category=ActionCategory.SYSTEM_MODIFICATION,
                compliance_frameworks=[ComplianceFramework.SOC2, ComplianceFramework.SOX],
                conditions={"target_environment": "production", "authorized": False},
                decision=GuardianDecision.DENY,
                requires_approval=False,
                emergency_stop=True,
                metadata={"severity": "critical", "auto_block": True}
            ),
            GuardianRule(
                rule_id="PENTEST_APPROVAL",
                name="Penetration Testing Authorization",
                description="Require approval for penetration testing activities",
                category=ActionCategory.PENETRATION_TESTING,
                compliance_frameworks=[ComplianceFramework.SOC2, ComplianceFramework.NIST],
                conditions={"requires_documentation": True},
                decision=GuardianDecision.REQUIRE_APPROVAL,
                requires_approval=True,
                emergency_stop=False,
                metadata={"approval_level": "security_manager", "documentation_required": True}
            ),
            GuardianRule(
                rule_id="PHI_ACCESS_RESTRICT",
                name="PHI Access Restriction",
                description="Strict controls for PHI data access",
                category=ActionCategory.DATA_ACCESS,
                compliance_frameworks=[ComplianceFramework.HIPAA],
                conditions={"data_type": "phi", "access_logged": True},
                decision=GuardianDecision.REQUIRE_APPROVAL,
                requires_approval=True,
                emergency_stop=False,
                metadata={"approval_level": "privacy_officer", "audit_required": True}
            ),
            GuardianRule(
                rule_id="OFFENSIVE_OPS_CONTROL",
                name="Offensive Operations Control",
                description="Strict control over offensive security operations",
                category=ActionCategory.OFFENSIVE_SECURITY,
                compliance_frameworks=[ComplianceFramework.NIST, ComplianceFramework.ISO27001],
                conditions={"legal_authorization": True, "scope_documented": True},
                decision=GuardianDecision.REQUIRE_APPROVAL,
                requires_approval=True,
                emergency_stop=False,
                metadata={"approval_level": "ciso", "legal_review": True}
            )
        ]
        
        self.rules = default_rules
        logger.info(f"Loaded {len(default_rules)} default Guardian rules")
    
    def _parse_configuration(self, config: Dict[str, Any]):
        """Parse configuration from JSON"""
        # Implementation for parsing complex configuration
        # This would convert JSON config to GuardianRule objects
        pass
    
    async def validate_action(self, 
                            action: str,
                            context: SecurityContext,
                            additional_data: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Multi-layer validation of security action
        
        Args:
            action: Description of the security action
            context: Security context with user, target, etc.
            additional_data: Additional validation data
            
        Returns:
            ValidationResult with decision and compliance status
        """
        start_time = time.time()
        
        try:
            # Layer 1: Pre-check validation
            precheck_result = await self._precheck_validation(action, context)
            if precheck_result.decision == GuardianDecision.EMERGENCY_STOP:
                return precheck_result
            
            # Layer 2: Policy validation
            policy_result = await self._policy_validation(action, context, additional_data)
            if policy_result.decision == GuardianDecision.EMERGENCY_STOP:
                return policy_result
            
            # Layer 3: AI safety check
            safety_result = await self._ai_safety_validation(action, context, additional_data)
            if safety_result.decision == GuardianDecision.EMERGENCY_STOP:
                return safety_result
            
            # Layer 4: Damage prevention
            damage_result = await self._damage_prevention_check(action, context, additional_data)
            
            # Combine results
            final_result = self._combine_validation_results(
                [precheck_result, policy_result, safety_result, damage_result]
            )
            
            # Create audit entry
            audit_entry = await self._create_audit_entry(action, context, final_result)
            final_result.audit_entry_id = audit_entry.entry_id
            
            # Log validation
            validation_time = (time.time() - start_time) * 1000
            logger.info(f"Guardian validation complete: {final_result.decision.value} "
                       f"(risk: {final_result.risk_score:.2f}, time: {validation_time:.2f}ms)")
            
            return final_result
            
        except Exception as e:
            logger.error(f"Guardian validation failed: {e}")
            return ValidationResult(
                decision=GuardianDecision.EMERGENCY_STOP,
                risk_score=10.0,
                compliance_status={},
                validation_errors=[f"Validation system error: {str(e)}"],
                required_approvals=[],
                audit_entry_id="",
                emergency_stop_triggered=True,
                recommendations=["Contact security team immediately"]
            )
    
    async def _precheck_validation(self, action: str, context: SecurityContext) -> ValidationResult:
        """Layer 1: Basic safety pre-checks"""
        errors = []
        risk_score = 0.0
        
        # Check emergency stop status
        if self.emergency_stop_active:
            return ValidationResult(
                decision=GuardianDecision.EMERGENCY_STOP,
                risk_score=10.0,
                compliance_status={},
                validation_errors=["Emergency stop is active"],
                required_approvals=[],
                audit_entry_id="",
                emergency_stop_triggered=True,
                recommendations=["Contact security team to resolve emergency stop"]
            )
        
        # Check basic context validity
        if not context.user_id or not context.session_id:
            errors.append("Invalid user or session context")
            risk_score += 2.0
        
        # Check authorization level
        if context.risk_level == RiskLevel.CATASTROPHIC:
            return ValidationResult(
                decision=GuardianDecision.EMERGENCY_STOP,
                risk_score=10.0,
                compliance_status={},
                validation_errors=["Catastrophic risk level detected"],
                required_approvals=[],
                audit_entry_id="",
                emergency_stop_triggered=True,
                recommendations=["Action blocked due to catastrophic risk"]
            )
        
        # Risk scoring based on context
        risk_score += context.risk_level.value * 0.5
        
        if context.action_category == ActionCategory.OFFENSIVE_SECURITY:
            risk_score += 2.0
        
        if context.target_system and "production" in context.target_system.lower():
            risk_score += 3.0
        
        # Determine decision
        decision = GuardianDecision.ALLOW
        if risk_score >= 7.0:
            decision = GuardianDecision.DENY
        elif risk_score >= 4.0:
            decision = GuardianDecision.REQUIRE_APPROVAL
        elif risk_score >= 2.0:
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
    
    async def _policy_validation(self, action: str, context: SecurityContext, 
                                additional_data: Optional[Dict[str, Any]]) -> ValidationResult:
        """Layer 2: Policy and rule validation"""
        errors = []
        risk_score = 0.0
        required_approvals = []
        compliance_status = {}
        emergency_stop = False
        
        # Check applicable rules
        applicable_rules = [rule for rule in self.rules 
                          if self._rule_matches_context(rule, context)]
        
        for rule in applicable_rules:
            # Check rule conditions
            if self._evaluate_rule_conditions(rule, context, additional_data):
                risk_score += 1.0  # Each matching rule increases risk
                
                if rule.requires_approval:
                    required_approvals.append(rule.metadata.get("approval_level", "manager"))
                
                if rule.emergency_stop:
                    emergency_stop = True
                    break
                
                # Check compliance for this rule
                for framework in rule.compliance_frameworks:
                    if framework not in compliance_status:
                        compliance_status[framework] = True
                    
                    # Validate against specific compliance framework
                    framework_valid = await self.compliance_handlers[framework](
                        action, context, rule
                    )
                    compliance_status[framework] &= framework_valid
        
        # Overall compliance check
        all_compliant = all(compliance_status.values()) if compliance_status else True
        if not all_compliant:
            risk_score += 2.0
            errors.append("Compliance validation failed")
        
        # Determine decision
        decision = GuardianDecision.ALLOW
        if emergency_stop:
            decision = GuardianDecision.EMERGENCY_STOP
        elif required_approvals:
            decision = GuardianDecision.REQUIRE_APPROVAL
        elif risk_score >= 5.0:
            decision = GuardianDecision.DENY
        elif risk_score >= 2.0:
            decision = GuardianDecision.MONITOR
        
        return ValidationResult(
            decision=decision,
            risk_score=risk_score,
            compliance_status=compliance_status,
            validation_errors=errors,
            required_approvals=list(set(required_approvals)),
            audit_entry_id="",
            emergency_stop_triggered=emergency_stop,
            recommendations=[]
        )
    
    async def _ai_safety_validation(self, action: str, context: SecurityContext,
                                  additional_data: Optional[Dict[str, Any]]) -> ValidationResult:
        """Layer 3: AI ethical and safety validation"""
        errors = []
        risk_score = 0.0
        recommendations = []
        
        # Ethical boundary checks
        ethical_violations = self._check_ethical_boundaries(action, context)
        if ethical_violations:
            errors.extend(ethical_violations)
            risk_score += len(ethical_violations) * 1.5
        
        # Intent analysis
        malicious_intent = self._analyze_intent(action, context)
        if malicious_intent:
            errors.append("Potentially malicious intent detected")
            risk_score += 3.0
        
        # Scope validation
        scope_violations = self._validate_scope(action, context)
        if scope_violations:
            errors.extend(scope_violations)
            risk_score += len(scope_violations) * 1.0
        
        # Generate safety recommendations
        if context.action_category == ActionCategory.PENETRATION_TESTING:
            recommendations.append("Ensure penetration testing scope is clearly defined")
            recommendations.append("Verify written authorization is obtained")
        
        if context.risk_level.value >= 4:
            recommendations.append("Consider additional oversight for high-risk operations")
        
        # Determine decision
        decision = GuardianDecision.ALLOW
        if risk_score >= 8.0:
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
            emergency_stop_triggered=(decision == GuardianDecision.EMERGENCY_STOP),
            recommendations=recommendations
        )
    
    async def _damage_prevention_check(self, action: str, context: SecurityContext,
                                     additional_data: Optional[Dict[str, Any]]) -> ValidationResult:
        """Layer 4: Damage prevention and fail-safe mechanisms"""
        errors = []
        risk_score = 0.0
        emergency_stop = False
        
        # Check for destructive operations
        destructive_keywords = [
            "delete", "remove", "destroy", "wipe", "format", "drop", "truncate",
            "shutdown", "reboot", "kill", "terminate", "exploit", "attack"
        ]
        
        action_lower = action.lower()
        destructive_count = sum(1 for keyword in destructive_keywords 
                              if keyword in action_lower)
        
        if destructive_count > 0:
            risk_score += destructive_count * 2.0
            errors.append(f"Potentially destructive operation detected ({destructive_count} indicators)")
        
        # Production system protection
        if (context.target_system and 
            any(keyword in context.target_system.lower() 
                for keyword in ["prod", "production", "live"])):
            
            # Production systems require extra validation
            if context.authorization_level != "production_authorized":
                emergency_stop = True
                errors.append("Unauthorized production system access attempt")
                risk_score = 10.0
        
        # Rate limiting check
        if self._check_rate_limiting(context.user_id, context.action_category):
            errors.append("Rate limit exceeded for this action type")
            risk_score += 2.0
        
        # Time-based restrictions
        if self._check_time_restrictions(context):
            errors.append("Action attempted outside allowed time window")
            risk_score += 1.0
        
        # IP geolocation check
        if self._check_suspicious_location(context.source_ip):
            errors.append("Action originated from suspicious location")
            risk_score += 1.5
        
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
            recommendations=[]
        )
    
    def _combine_validation_results(self, results: List[ValidationResult]) -> ValidationResult:
        """Combine results from all validation layers"""
        # Use most restrictive decision
        decisions = [result.decision for result in results]
        
        if GuardianDecision.EMERGENCY_STOP in decisions:
            final_decision = GuardianDecision.EMERGENCY_STOP
        elif GuardianDecision.DENY in decisions:
            final_decision = GuardianDecision.DENY
        elif GuardianDecision.REQUIRE_APPROVAL in decisions:
            final_decision = GuardianDecision.REQUIRE_APPROVAL
        elif GuardianDecision.MONITOR in decisions:
            final_decision = GuardianDecision.MONITOR
        else:
            final_decision = GuardianDecision.ALLOW
        
        # Combine risk scores (weighted average)
        total_risk = sum(result.risk_score for result in results) / len(results)
        
        # Combine compliance status
        combined_compliance = {}
        for result in results:
            combined_compliance.update(result.compliance_status)
        
        # Combine errors and approvals
        all_errors = []
        all_approvals = []
        all_recommendations = []
        
        for result in results:
            all_errors.extend(result.validation_errors)
            all_approvals.extend(result.required_approvals)
            all_recommendations.extend(result.recommendations)
        
        # Check if any layer triggered emergency stop
        emergency_triggered = any(result.emergency_stop_triggered for result in results)
        
        return ValidationResult(
            decision=final_decision,
            risk_score=min(total_risk, 10.0),  # Cap at 10.0
            compliance_status=combined_compliance,
            validation_errors=list(set(all_errors)),  # Remove duplicates
            required_approvals=list(set(all_approvals)),
            audit_entry_id="",
            emergency_stop_triggered=emergency_triggered,
            recommendations=list(set(all_recommendations))
        )
    
    async def _create_audit_entry(self, action: str, context: SecurityContext, 
                                result: ValidationResult) -> AuditEntry:
        """Create audit trail entry for the validation"""
        entry_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        # Create data for checksum
        audit_data = {
            "entry_id": entry_id,
            "timestamp": timestamp.isoformat(),
            "user_id": context.user_id,
            "action": action,
            "decision": result.decision.value,
            "risk_score": result.risk_score,
            "context": asdict(context)
        }
        
        # Generate tamper-proof checksum
        checksum_data = json.dumps(audit_data, sort_keys=True)
        checksum = hashlib.sha256(checksum_data.encode()).hexdigest()
        
        audit_entry = AuditEntry(
            entry_id=entry_id,
            timestamp=timestamp,
            user_id=context.user_id,
            action=action,
            decision=result.decision,
            risk_level=context.risk_level,
            security_context=asdict(context),
            justification=context.business_justification,
            approval_chain=[],
            compliance_tags=[fw.value for fw in context.compliance_requirements],
            checksum=checksum
        )
        
        self.audit_trail.append(audit_entry)
        
        # Persist audit entry (in production, this would go to secure storage)
        await self._persist_audit_entry(audit_entry)
        
        logger.info(f"Audit entry created: {entry_id}")
        return audit_entry
    
    # Helper methods for validation logic
    def _rule_matches_context(self, rule: GuardianRule, context: SecurityContext) -> bool:
        """Check if a rule applies to the given context"""
        return rule.category == context.action_category
    
    def _evaluate_rule_conditions(self, rule: GuardianRule, context: SecurityContext,
                                additional_data: Optional[Dict[str, Any]]) -> bool:
        """Evaluate rule conditions against context"""
        # Simplified condition evaluation
        # In production, this would be a more sophisticated rule engine
        return True
    
    def _check_ethical_boundaries(self, action: str, context: SecurityContext) -> List[str]:
        """Check for ethical boundary violations"""
        violations = []
        
        # Check for unauthorized access attempts
        if "unauthorized" in action.lower() and context.authorization_level == "guest":
            violations.append("Unauthorized access attempt by guest user")
        
        # Check for potential privacy violations
        if any(keyword in action.lower() for keyword in ["personal", "private", "confidential"]):
            if ComplianceFramework.GDPR not in context.compliance_requirements:
                violations.append("Potential privacy violation without GDPR compliance")
        
        return violations
    
    def _analyze_intent(self, action: str, context: SecurityContext) -> bool:
        """Analyze potential malicious intent"""
        # Simplified intent analysis
        malicious_indicators = [
            "bypass", "circumvent", "unauthorized", "exploit", "attack", 
            "hack", "breach", "compromise", "steal", "extract"
        ]
        
        action_lower = action.lower()
        malicious_count = sum(1 for indicator in malicious_indicators 
                            if indicator in action_lower)
        
        return malicious_count > 2  # Threshold for malicious intent
    
    def _validate_scope(self, action: str, context: SecurityContext) -> List[str]:
        """Validate action scope"""
        violations = []
        
        # Check if action exceeds authorized scope
        if context.action_category == ActionCategory.PENETRATION_TESTING:
            if not context.business_justification:
                violations.append("Penetration testing without business justification")
        
        return violations
    
    def _check_rate_limiting(self, user_id: str, action_category: ActionCategory) -> bool:
        """Check rate limiting for user and action type"""
        # Simplified rate limiting check
        # In production, this would check against a time-series database
        return False
    
    def _check_time_restrictions(self, context: SecurityContext) -> bool:
        """Check time-based access restrictions"""
        # Check if action is during allowed hours
        current_hour = datetime.now().hour
        
        # High-risk operations only during business hours
        if context.risk_level.value >= 4:
            return current_hour < 8 or current_hour > 18
        
        return False
    
    def _check_suspicious_location(self, source_ip: str) -> bool:
        """Check for suspicious IP locations"""
        # Simplified location check
        # In production, this would use geolocation services
        suspicious_ranges = ["10.0.0.", "192.168.", "127.0.0."]
        return not any(source_ip.startswith(range_) for range_ in suspicious_ranges)
    
    async def _persist_audit_entry(self, audit_entry: AuditEntry):
        """Persist audit entry to secure storage"""
        # In production, this would write to secure, tamper-proof storage
        logger.debug(f"Persisting audit entry: {audit_entry.entry_id}")
    
    # Compliance validation methods
    async def _validate_soc2(self, action: str, context: SecurityContext, rule: GuardianRule) -> bool:
        """Validate SOC2 compliance requirements"""
        # SOC2 focuses on security, availability, processing integrity, confidentiality, privacy
        if context.action_category in [ActionCategory.DATA_ACCESS, ActionCategory.SYSTEM_MODIFICATION]:
            return bool(context.business_justification and context.authorization_level != "guest")
        return True
    
    async def _validate_hipaa(self, action: str, context: SecurityContext, rule: GuardianRule) -> bool:
        """Validate HIPAA compliance requirements"""
        # HIPAA requires strict controls for PHI
        if "phi" in action.lower() or "patient" in action.lower():
            return context.authorization_level in ["privacy_officer", "authorized_personnel"]
        return True
    
    async def _validate_gdpr(self, action: str, context: SecurityContext, rule: GuardianRule) -> bool:
        """Validate GDPR compliance requirements"""
        # GDPR requires consent and legitimate interest for personal data
        if any(keyword in action.lower() for keyword in ["personal", "gdpr", "consent"]):
            return bool(context.business_justification)
        return True
    
    async def _validate_pci_dss(self, action: str, context: SecurityContext, rule: GuardianRule) -> bool:
        """Validate PCI-DSS compliance requirements"""
        # PCI-DSS requires strict controls for cardholder data
        if any(keyword in action.lower() for keyword in ["card", "payment", "pci"]):
            return context.authorization_level in ["pci_authorized", "security_admin"]
        return True
    
    async def _validate_nist(self, action: str, context: SecurityContext, rule: GuardianRule) -> bool:
        """Validate NIST framework compliance"""
        # NIST framework focuses on identify, protect, detect, respond, recover
        return context.risk_level.value <= 4  # Allow up to HIGH risk
    
    async def _validate_iso27001(self, action: str, context: SecurityContext, rule: GuardianRule) -> bool:
        """Validate ISO27001 compliance requirements"""
        # ISO27001 requires information security management
        return bool(context.business_justification and context.authorization_level != "guest")
    
    async def _validate_fisma(self, action: str, context: SecurityContext, rule: GuardianRule) -> bool:
        """Validate FISMA compliance requirements"""
        # FISMA requires federal information security management
        return context.authorization_level in ["federal_authorized", "security_admin"]
    
    async def _validate_sox(self, action: str, context: SecurityContext, rule: GuardianRule) -> bool:
        """Validate SOX compliance requirements"""
        # SOX requires financial reporting controls
        if context.target_system and "financial" in context.target_system.lower():
            return context.authorization_level in ["financial_authorized", "audit_approved"]
        return True
    
    # Public API methods
    def trigger_emergency_stop(self, reason: str, user_id: str):
        """Trigger emergency stop for all operations"""
        self.emergency_stop_active = True
        logger.critical(f"EMERGENCY STOP activated by {user_id}: {reason}")
        
        # Create emergency audit entry
        emergency_entry = AuditEntry(
            entry_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            user_id=user_id,
            action="EMERGENCY_STOP",
            decision=GuardianDecision.EMERGENCY_STOP,
            risk_level=RiskLevel.CATASTROPHIC,
            security_context={"reason": reason},
            justification="Emergency stop activated",
            approval_chain=[user_id],
            compliance_tags=["emergency"],
            checksum=hashlib.sha256(f"emergency_stop_{time.time()}".encode()).hexdigest()
        )
        
        self.audit_trail.append(emergency_entry)
    
    def clear_emergency_stop(self, user_id: str, authorization_code: str):
        """Clear emergency stop (requires high-level authorization)"""
        # In production, this would require multi-factor authentication
        if authorization_code == "GUARDIAN_OVERRIDE":
            self.emergency_stop_active = False
            logger.warning(f"Emergency stop cleared by {user_id}")
        else:
            logger.error(f"Invalid emergency stop clear attempt by {user_id}")
    
    def get_audit_trail(self, user_id: str, start_time: Optional[datetime] = None,
                       end_time: Optional[datetime] = None) -> List[AuditEntry]:
        """Retrieve audit trail entries"""
        entries = self.audit_trail
        
        if start_time:
            entries = [e for e in entries if e.timestamp >= start_time]
        
        if end_time:
            entries = [e for e in entries if e.timestamp <= end_time]
        
        return entries
    
    def generate_compliance_report(self, framework: ComplianceFramework,
                                 start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate compliance report for specific framework"""
        relevant_entries = [
            entry for entry in self.audit_trail
            if start_date <= entry.timestamp <= end_date
            and framework.value in entry.compliance_tags
        ]
        
        report = {
            "framework": framework.value,
            "reporting_period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "total_events": len(relevant_entries),
            "decisions": {
                "allow": len([e for e in relevant_entries if e.decision == GuardianDecision.ALLOW]),
                "deny": len([e for e in relevant_entries if e.decision == GuardianDecision.DENY]),
                "require_approval": len([e for e in relevant_entries if e.decision == GuardianDecision.REQUIRE_APPROVAL]),
                "emergency_stop": len([e for e in relevant_entries if e.decision == GuardianDecision.EMERGENCY_STOP])
            },
            "risk_distribution": {},
            "compliance_violations": [],
            "recommendations": []
        }
        
        # Add risk distribution
        for entry in relevant_entries:
            risk_level = entry.risk_level.name
            report["risk_distribution"][risk_level] = report["risk_distribution"].get(risk_level, 0) + 1
        
        return report
    
    def add_rule(self, rule: GuardianRule) -> bool:
        """Add a new Guardian rule"""
        try:
            # Validate rule
            if not rule.rule_id or not rule.name:
                return False
            
            # Check for duplicate rule IDs
            if any(r.rule_id == rule.rule_id for r in self.rules):
                return False
            
            self.rules.append(rule)
            logger.info(f"Guardian rule added: {rule.rule_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to add Guardian rule: {e}")
            return False
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove a Guardian rule"""
        try:
            self.rules = [r for r in self.rules if r.rule_id != rule_id]
            logger.info(f"Guardian rule removed: {rule_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to remove Guardian rule: {e}")
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get Guardian Protocol system status"""
        return {
            "guardian_protocol_version": "1.0.0",
            "emergency_stop_active": self.emergency_stop_active,
            "total_rules": len(self.rules),
            "audit_entries": len(self.audit_trail),
            "pending_approvals": len(self.pending_approvals),
            "supported_frameworks": [fw.value for fw in ComplianceFramework],
            "uptime": datetime.now().isoformat()
        }