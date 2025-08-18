"""
Ethics overseer with real-time action validation for autonomous agents.
"""

import time
import threading
import logging
import hashlib
from typing import Dict, List, Optional, Callable, Any, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import re

logger = logging.getLogger(__name__)

class EthicalPrinciple(Enum):
    """Core ethical principles for AI behavior."""
    NO_HARM = "no_harm"                    # Do not cause harm to real systems or people
    SIMULATION_CONTAINMENT = "containment"  # Stay within simulation boundaries
    PROPORTIONATE_RESPONSE = "proportionate" # Responses should match the scenario
    TRANSPARENCY = "transparency"           # Actions should be auditable and explainable  
    CONSENT = "consent"                     # Respect user consent and permissions
    PRIVACY = "privacy"                     # Protect sensitive information
    FAIRNESS = "fairness"                   # Treat all entities fairly
    ACCOUNTABILITY = "accountability"       # Actions must be traceable and justified

class ActionCategory(Enum):
    """Categories of agent actions for ethical evaluation."""
    RECONNAISSANCE = "reconnaissance"       # Information gathering
    EXPLOITATION = "exploitation"          # Vulnerability exploitation  
    PERSISTENCE = "persistence"            # Maintaining access
    LATERAL_MOVEMENT = "lateral_movement"  # Moving through network
    DATA_ACCESS = "data_access"            # Accessing data/files
    SYSTEM_MODIFICATION = "system_modification"  # Changing system state
    COMMUNICATION = "communication"        # External communications
    ANALYSIS = "analysis"                  # Data analysis and processing

class EthicalJudgment(Enum):
    """Ethical evaluation outcomes."""
    APPROVED = "approved"                  # Action is ethically acceptable
    APPROVED_WITH_CONDITIONS = "approved_conditional"  # Acceptable with restrictions
    DENIED = "denied"                      # Action violates ethical principles
    REQUIRES_HUMAN_REVIEW = "human_review"  # Needs human oversight
    EMERGENCY_STOP = "emergency_stop"      # Requires immediate system halt

@dataclass
class EthicalDecision:
    """Record of an ethical decision made by the overseer."""
    decision_id: str
    agent_id: str
    action_type: ActionCategory
    action_description: str
    judgment: EthicalJudgment
    violated_principles: List[EthicalPrinciple] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)
    reasoning: str = ""
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EthicalViolation:
    """Record of an ethical violation."""
    violation_id: str
    agent_id: str
    principle: EthicalPrinciple
    severity: str  # "low", "medium", "high", "critical"
    description: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False
    resolution_notes: str = ""

class EthicsOverseer:
    """
    Real-time ethics overseer that validates agent actions against ethical principles.
    
    This system provides:
    - Real-time action validation before execution
    - Ethical principle enforcement
    - Violation detection and reporting
    - Human oversight integration
    - Audit trail of all ethical decisions
    - Learning and adaptation of ethical rules
    """
    
    def __init__(self):
        """Initialize the ethics overseer."""
        self.ethical_rules: Dict[str, Callable[[Dict[str, Any]], EthicalJudgment]] = {}
        self.decisions: List[EthicalDecision] = []
        self.violations: List[EthicalViolation] = []
        self.agent_permissions: Dict[str, Set[ActionCategory]] = {}
        
        self._lock = threading.RLock()
        self.max_decision_history = 10000
        self.max_violation_history = 1000
        
        # Callbacks for ethical events
        self._decision_callbacks: List[Callable[[EthicalDecision], None]] = []
        self._violation_callbacks: List[Callable[[EthicalViolation], None]] = []
        
        # Initialize default ethical rules
        self._initialize_ethical_rules()
        
    def validate_action(self, agent_id: str, action_type: ActionCategory,
                       action_description: str, context: Dict[str, Any]) -> EthicalDecision:
        """Validate an agent action against ethical principles.
        
        Args:
            agent_id: ID of the agent requesting the action
            action_type: Category of action being requested
            action_description: Detailed description of the action
            context: Additional context about the action
            
        Returns:
            EthicalDecision with judgment and reasoning
        """
        decision_id = self._generate_decision_id(agent_id, action_type, action_description)
        
        try:
            # Check agent permissions
            if not self._check_agent_permissions(agent_id, action_type):
                return self._create_denial_decision(
                    decision_id, agent_id, action_type, action_description,
                    "Agent does not have permission for this action type",
                    [EthicalPrinciple.CONSENT]
                )
                
            # Evaluate against ethical rules
            judgment = self._evaluate_ethical_rules(action_type, action_description, context)
            
            # Check for high-risk patterns
            risk_assessment = self._assess_action_risk(agent_id, action_type, action_description, context)
            
            # Make final decision
            final_judgment, violated_principles, conditions, reasoning = self._make_final_decision(
                judgment, risk_assessment, context
            )
            
            decision = EthicalDecision(
                decision_id=decision_id,
                agent_id=agent_id,
                action_type=action_type,
                action_description=action_description,
                judgment=final_judgment,
                violated_principles=violated_principles,
                conditions=conditions,
                reasoning=reasoning,
                confidence=self._calculate_confidence(final_judgment, risk_assessment),
                metadata=context.copy()
            )
            
            # Record decision
            with self._lock:
                self.decisions.append(decision)
                self._trim_decision_history()
                
            # Notify callbacks
            for callback in self._decision_callbacks:
                try:
                    callback(decision)
                except Exception as e:
                    logger.error(f"Error in ethics decision callback: {e}")
                    
            # Record violations if any
            if violated_principles:
                self._record_violations(agent_id, violated_principles, action_description, context)
                
            logger.info(f"Ethics decision for {agent_id}: {final_judgment.value} - {reasoning}")
            return decision
            
        except Exception as e:
            logger.error(f"Error in ethical validation: {e}")
            # Fail safe - deny on error
            return self._create_denial_decision(
                decision_id, agent_id, action_type, action_description,
                f"Ethics validation error: {str(e)}",
                []
            )
            
    def register_agent(self, agent_id: str, permissions: Set[ActionCategory]):
        """Register an agent with specific permissions.
        
        Args:
            agent_id: ID of the agent to register
            permissions: Set of action categories the agent is allowed to perform
        """
        with self._lock:
            self.agent_permissions[agent_id] = permissions.copy()
            logger.info(f"Registered agent {agent_id} with permissions: {permissions}")
            
    def revoke_agent_permissions(self, agent_id: str, revoked_permissions: Set[ActionCategory]):
        """Revoke specific permissions from an agent.
        
        Args:
            agent_id: ID of the agent
            revoked_permissions: Permissions to revoke
        """
        with self._lock:
            if agent_id in self.agent_permissions:
                self.agent_permissions[agent_id] -= revoked_permissions
                logger.warning(f"Revoked permissions for {agent_id}: {revoked_permissions}")
                
    def add_ethical_rule(self, rule_name: str, rule_function: Callable[[Dict[str, Any]], EthicalJudgment]):
        """Add a custom ethical rule.
        
        Args:
            rule_name: Name of the ethical rule
            rule_function: Function that evaluates actions and returns judgment
        """
        with self._lock:
            self.ethical_rules[rule_name] = rule_function
            logger.info(f"Added ethical rule: {rule_name}")
            
    def get_agent_violations(self, agent_id: str, severity: Optional[str] = None) -> List[EthicalViolation]:
        """Get violations for a specific agent.
        
        Args:
            agent_id: ID of the agent
            severity: Optional filter by severity level
            
        Returns:
            List of ethical violations
        """
        with self._lock:
            violations = [v for v in self.violations if v.agent_id == agent_id]
            if severity:
                violations = [v for v in violations if v.severity == severity]
            return violations
            
    def resolve_violation(self, violation_id: str, resolution_notes: str):
        """Mark a violation as resolved.
        
        Args:
            violation_id: ID of the violation to resolve
            resolution_notes: Notes about how the violation was resolved
        """
        with self._lock:
            for violation in self.violations:
                if violation.violation_id == violation_id:
                    violation.resolved = True
                    violation.resolution_notes = resolution_notes
                    logger.info(f"Resolved ethical violation {violation_id}: {resolution_notes}")
                    break
                    
    def get_ethics_summary(self) -> Dict[str, Any]:
        """Get summary of ethical oversight activity.
        
        Returns:
            Dictionary with ethics statistics
        """
        with self._lock:
            total_decisions = len(self.decisions)
            if total_decisions == 0:
                return {
                    "total_decisions": 0,
                    "approved": 0,
                    "denied": 0,
                    "approval_rate": 0.0,
                    "total_violations": 0,
                    "unresolved_violations": 0
                }
                
            approved = sum(1 for d in self.decisions if d.judgment == EthicalJudgment.APPROVED)
            denied = sum(1 for d in self.decisions if d.judgment == EthicalJudgment.DENIED)
            unresolved_violations = sum(1 for v in self.violations if not v.resolved)
            
            return {
                "total_decisions": total_decisions,
                "approved": approved,
                "denied": denied,
                "approved_conditional": sum(1 for d in self.decisions if d.judgment == EthicalJudgment.APPROVED_WITH_CONDITIONS),
                "human_review": sum(1 for d in self.decisions if d.judgment == EthicalJudgment.REQUIRES_HUMAN_REVIEW),
                "emergency_stops": sum(1 for d in self.decisions if d.judgment == EthicalJudgment.EMERGENCY_STOP),
                "approval_rate": approved / total_decisions,
                "total_violations": len(self.violations),
                "unresolved_violations": unresolved_violations,
                "registered_agents": len(self.agent_permissions)
            }
            
    def add_decision_callback(self, callback: Callable[[EthicalDecision], None]):
        """Add callback for ethical decisions.
        
        Args:
            callback: Function to call when decisions are made
        """
        self._decision_callbacks.append(callback)
        
    def add_violation_callback(self, callback: Callable[[EthicalViolation], None]):
        """Add callback for ethical violations.
        
        Args:
            callback: Function to call when violations are detected
        """
        self._violation_callbacks.append(callback)
        
    def _initialize_ethical_rules(self):
        """Initialize default ethical rules."""
        
        def no_harm_rule(context: Dict[str, Any]) -> EthicalJudgment:
            """Rule to prevent actions that could cause real harm."""
            action_desc = context.get("action_description", "").lower()
            
            # Dangerous keywords that indicate potential harm
            harmful_keywords = [
                "delete production", "rm -rf /", "format drive",
                "drop database", "shutdown server", "kill process",
                "modify firewall", "disable security", "install backdoor",
                "exfiltrate data", "corrupt files", "crash system"
            ]
            
            if any(keyword in action_desc for keyword in harmful_keywords):
                return EthicalJudgment.DENIED
                
            return EthicalJudgment.APPROVED
            
        def simulation_containment_rule(context: Dict[str, Any]) -> EthicalJudgment:
            """Rule to ensure actions stay within simulation boundaries."""
            target = context.get("target", "").lower()
            
            # Real system indicators
            real_system_indicators = [
                "192.168.1.", "10.0.0.", "production",
                "prod", "live", "external", "internet",
                "google.com", "github.com", "real"
            ]
            
            if any(indicator in target for indicator in real_system_indicators):
                return EthicalJudgment.DENIED
                
            return EthicalJudgment.APPROVED
            
        def proportionate_response_rule(context: Dict[str, Any]) -> EthicalJudgment:
            """Rule to ensure responses are proportionate to the scenario."""
            severity = context.get("severity", "medium")
            action_type = context.get("action_type")
            
            # High-impact actions require high-severity scenarios
            if action_type in [ActionCategory.SYSTEM_MODIFICATION, ActionCategory.DATA_ACCESS]:
                if severity in ["low", "minimal"]:
                    return EthicalJudgment.APPROVED_WITH_CONDITIONS
                    
            return EthicalJudgment.APPROVED
            
        # Register default rules
        self.add_ethical_rule("no_harm", no_harm_rule)
        self.add_ethical_rule("simulation_containment", simulation_containment_rule)
        self.add_ethical_rule("proportionate_response", proportionate_response_rule)
        
    def _check_agent_permissions(self, agent_id: str, action_type: ActionCategory) -> bool:
        """Check if agent has permission for the action type.
        
        Args:
            agent_id: ID of the agent
            action_type: Type of action being requested
            
        Returns:
            True if agent has permission
        """
        with self._lock:
            permissions = self.agent_permissions.get(agent_id, set())
            return action_type in permissions
            
    def _evaluate_ethical_rules(self, action_type: ActionCategory, 
                               action_description: str, context: Dict[str, Any]) -> EthicalJudgment:
        """Evaluate action against all ethical rules.
        
        Args:
            action_type: Type of action
            action_description: Description of action
            context: Action context
            
        Returns:
            Combined ethical judgment
        """
        rule_context = {
            "action_type": action_type,
            "action_description": action_description,
            **context
        }
        
        judgments = []
        for rule_name, rule_function in self.ethical_rules.items():
            try:
                judgment = rule_function(rule_context)
                judgments.append(judgment)
                
                # Immediate denial or emergency stop
                if judgment in [EthicalJudgment.DENIED, EthicalJudgment.EMERGENCY_STOP]:
                    return judgment
                    
            except Exception as e:
                logger.error(f"Error in ethical rule {rule_name}: {e}")
                
        # Combine judgments - most restrictive wins
        if EthicalJudgment.REQUIRES_HUMAN_REVIEW in judgments:
            return EthicalJudgment.REQUIRES_HUMAN_REVIEW
        elif EthicalJudgment.APPROVED_WITH_CONDITIONS in judgments:
            return EthicalJudgment.APPROVED_WITH_CONDITIONS
        else:
            return EthicalJudgment.APPROVED
            
    def _assess_action_risk(self, agent_id: str, action_type: ActionCategory,
                          action_description: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the risk level of an action.
        
        Args:
            agent_id: ID of the agent
            action_type: Type of action
            action_description: Description of action
            context: Action context
            
        Returns:
            Risk assessment data
        """
        risk_score = 0.0
        risk_factors = []
        
        # High-risk action types
        if action_type in [ActionCategory.SYSTEM_MODIFICATION, ActionCategory.PERSISTENCE]:
            risk_score += 0.3
            risk_factors.append("high_risk_action_type")
            
        # Check for suspicious patterns in description
        suspicious_patterns = [
            r"bypass.*security", r"disable.*protection", r"escalate.*privilege",
            r"admin.*access", r"root.*shell", r"backdoor",
            r"lateral.*movement", r"persist.*access"
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, action_description, re.IGNORECASE):
                risk_score += 0.2
                risk_factors.append(f"suspicious_pattern: {pattern}")
                
        # Check agent history
        recent_decisions = [d for d in self.decisions if d.agent_id == agent_id and 
                          time.time() - d.timestamp < 300]  # Last 5 minutes
        
        if len(recent_decisions) > 10:
            risk_score += 0.1
            risk_factors.append("high_activity_rate")
            
        # Check for recent violations
        recent_violations = [v for v in self.violations if v.agent_id == agent_id and
                           time.time() - v.timestamp < 3600 and not v.resolved]  # Last hour
        
        if recent_violations:
            risk_score += len(recent_violations) * 0.15
            risk_factors.append(f"recent_violations: {len(recent_violations)}")
            
        return {
            "risk_score": min(risk_score, 1.0),  # Cap at 1.0
            "risk_factors": risk_factors,
            "assessment_time": time.time()
        }
        
    def _make_final_decision(self, initial_judgment: EthicalJudgment, 
                           risk_assessment: Dict[str, Any], 
                           context: Dict[str, Any]) -> tuple:
        """Make final ethical decision based on all factors.
        
        Args:
            initial_judgment: Initial judgment from rules
            risk_assessment: Risk assessment data
            context: Action context
            
        Returns:
            Tuple of (judgment, violated_principles, conditions, reasoning)
        """
        violated_principles = []
        conditions = []
        reasoning = f"Initial judgment: {initial_judgment.value}"
        
        risk_score = risk_assessment.get("risk_score", 0.0)
        
        # High risk requires additional scrutiny
        if risk_score > 0.7:
            if initial_judgment == EthicalJudgment.APPROVED:
                initial_judgment = EthicalJudgment.REQUIRES_HUMAN_REVIEW
                reasoning += f", elevated to human review due to high risk score: {risk_score:.2f}"
                
        # Add conditions for approved actions
        if initial_judgment in [EthicalJudgment.APPROVED, EthicalJudgment.APPROVED_WITH_CONDITIONS]:
            if risk_score > 0.4:
                conditions.append("Enhanced monitoring required")
                conditions.append("Action must be logged with full context")
                
            if context.get("target_criticality") == "high":
                conditions.append("Action must be reversible")
                conditions.append("Backup must be created before action")
                
        # Check for principle violations
        if risk_score > 0.5:
            violated_principles.append(EthicalPrinciple.NO_HARM)
            
        if "external" in context.get("target", "").lower():
            violated_principles.append(EthicalPrinciple.SIMULATION_CONTAINMENT)
            
        return initial_judgment, violated_principles, conditions, reasoning
        
    def _calculate_confidence(self, judgment: EthicalJudgment, risk_assessment: Dict[str, Any]) -> float:
        """Calculate confidence in the ethical decision.
        
        Args:
            judgment: The ethical judgment made
            risk_assessment: Risk assessment data
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        base_confidence = 0.8
        
        # Higher confidence for clear approvals/denials
        if judgment in [EthicalJudgment.APPROVED, EthicalJudgment.DENIED]:
            base_confidence = 0.9
        elif judgment == EthicalJudgment.REQUIRES_HUMAN_REVIEW:
            base_confidence = 0.5
            
        # Lower confidence for high-risk actions
        risk_score = risk_assessment.get("risk_score", 0.0)
        confidence_penalty = risk_score * 0.3
        
        return max(0.1, base_confidence - confidence_penalty)
        
    def _generate_decision_id(self, agent_id: str, action_type: ActionCategory, 
                            action_description: str) -> str:
        """Generate unique decision ID.
        
        Args:
            agent_id: ID of the agent
            action_type: Type of action
            action_description: Description of action
            
        Returns:
            Unique decision ID
        """
        timestamp = str(time.time())
        content = f"{agent_id}_{action_type.value}_{action_description}_{timestamp}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
        
    def _create_denial_decision(self, decision_id: str, agent_id: str,
                              action_type: ActionCategory, action_description: str,
                              reason: str, violated_principles: List[EthicalPrinciple]) -> EthicalDecision:
        """Create a denial decision.
        
        Args:
            decision_id: Decision ID
            agent_id: Agent ID
            action_type: Action type
            action_description: Action description
            reason: Reason for denial
            violated_principles: Violated ethical principles
            
        Returns:
            EthicalDecision with denial judgment
        """
        return EthicalDecision(
            decision_id=decision_id,
            agent_id=agent_id,
            action_type=action_type,
            action_description=action_description,
            judgment=EthicalJudgment.DENIED,
            violated_principles=violated_principles,
            reasoning=reason,
            confidence=0.95
        )
        
    def _record_violations(self, agent_id: str, principles: List[EthicalPrinciple],
                         action_description: str, context: Dict[str, Any]):
        """Record ethical violations.
        
        Args:
            agent_id: ID of the agent
            principles: Violated principles
            action_description: Action that violated principles
            context: Action context
        """
        for principle in principles:
            violation_id = hashlib.md5(f"{agent_id}_{principle.value}_{time.time()}".encode()).hexdigest()[:12]
            
            # Determine severity based on principle and context
            severity = "medium"
            if principle == EthicalPrinciple.NO_HARM:
                severity = "high"
            elif principle == EthicalPrinciple.SIMULATION_CONTAINMENT:
                severity = "critical"
                
            violation = EthicalViolation(
                violation_id=violation_id,
                agent_id=agent_id,
                principle=principle,
                severity=severity,
                description=f"Violation of {principle.value} in action: {action_description}",
                evidence=context.copy()
            )
            
            with self._lock:
                self.violations.append(violation)
                self._trim_violation_history()
                
            # Notify callbacks
            for callback in self._violation_callbacks:
                try:
                    callback(violation)
                except Exception as e:
                    logger.error(f"Error in violation callback: {e}")
                    
            logger.warning(f"Recorded ethical violation: {violation_id} for agent {agent_id}")
            
    def _trim_decision_history(self):
        """Trim decision history to maximum size."""
        if len(self.decisions) > self.max_decision_history:
            self.decisions = self.decisions[-self.max_decision_history:]
            
    def _trim_violation_history(self):
        """Trim violation history to maximum size."""
        if len(self.violations) > self.max_violation_history:
            self.violations = self.violations[-self.max_violation_history:]

# Global ethics overseer instance
ethics_overseer = EthicsOverseer()