"""
Archangel Authorization Validator
Advanced authorization validation and scope enforcement system

This module provides comprehensive authorization validation with multi-factor
authentication, role-based access control, and scope enforcement.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import hashlib
import secrets
import jwt
from pathlib import Path
import ipaddress

from .mcp_guardian_protocol import MCPTeamType, MCPToolCategory, DataClassification, LegalAuthorization

logger = logging.getLogger(__name__)

class AuthorizationLevel(Enum):
    """Authorization levels in ascending order of privilege"""
    GUEST = 1
    USER = 2
    ANALYST = 3
    SECURITY_ANALYST = 4
    SECURITY_ADMIN = 5
    PENETRATION_TESTER = 6
    INCIDENT_RESPONDER = 7
    SECURITY_MANAGER = 8
    PRIVACY_OFFICER = 9
    COMPLIANCE_OFFICER = 10
    CISO = 11
    SYSTEM_ADMIN = 12
    PRODUCTION_AUTHORIZED = 13
    FEDERAL_AUTHORIZED = 14
    EMERGENCY_RESPONDER = 15

class PermissionScope(Enum):
    """Permission scopes for different operations"""
    READ_ONLY = "read_only"
    NETWORK_SCAN = "network_scan"
    VULNERABILITY_SCAN = "vulnerability_scan"
    LOG_ANALYSIS = "log_analysis"
    INCIDENT_INVESTIGATION = "incident_investigation"
    PENETRATION_TESTING = "penetration_testing"
    SYSTEM_MODIFICATION = "system_modification"
    DATA_ACCESS = "data_access"
    PRODUCTION_ACCESS = "production_access"
    OFFENSIVE_OPERATIONS = "offensive_operations"
    EMERGENCY_RESPONSE = "emergency_response"
    COMPLIANCE_AUDIT = "compliance_audit"

class AccessControlRule(Enum):
    """Access control rule types"""
    ROLE_BASED = "role_based"
    TIME_BASED = "time_based"  
    LOCATION_BASED = "location_based"
    RESOURCE_BASED = "resource_based"
    RISK_BASED = "risk_based"
    MULTI_FACTOR = "multi_factor"

@dataclass
class AuthorizationContext:
    """Authorization context for validation"""
    user_id: str
    session_id: str
    authorization_level: AuthorizationLevel
    roles: List[str]
    permissions: Set[PermissionScope]
    source_ip: str
    user_agent: str
    timestamp: datetime
    mfa_verified: bool
    risk_score: float

@dataclass
class MCPAuthorizationContext(AuthorizationContext):
    """Extended authorization context for MCP operations"""
    mcp_agent_id: str
    team_type: MCPTeamType
    tool_category: MCPToolCategory
    target_systems: List[str]
    external_apis: List[str]
    data_classification: DataClassification
    legal_authorization: LegalAuthorization
    penetration_test_id: Optional[str] = None
    incident_id: Optional[str] = None
    
@dataclass
class ScopeDefinition:
    """Defines the scope of authorized operations"""
    scope_id: str
    name: str
    description: str
    allowed_targets: List[str]
    allowed_operations: List[str]
    time_restrictions: Optional[Dict[str, Any]]
    ip_restrictions: List[str]
    requires_approval: bool
    max_duration: Optional[timedelta]
    compliance_requirements: List[str]

@dataclass
class AuthorizationToken:
    """Secure authorization token"""
    token_id: str
    user_id: str
    issued_at: datetime
    expires_at: datetime
    scope: ScopeDefinition
    permissions: Set[PermissionScope]
    mfa_required: bool
    single_use: bool
    used: bool = False

@dataclass
class ValidationResult:
    """Authorization validation result"""
    authorized: bool
    authorization_level: AuthorizationLevel
    granted_permissions: Set[PermissionScope]
    scope_limitations: List[str]
    requires_elevation: bool
    requires_approval: bool
    validation_errors: List[str]
    token_id: Optional[str]
    expires_at: Optional[datetime]

class AuthorizationValidator:
    """
    Advanced authorization validation system
    
    Provides multi-layer authorization validation including:
    - Role-based access control (RBAC)
    - Scope enforcement and validation
    - Time-based access controls
    - Multi-factor authentication requirements
    - Risk-based authorization
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "/Users/samoakes/Desktop/Archangel/config/auth_config.json"
        self.authorization_rules: Dict[str, Dict[str, Any]] = {}
        self.scope_definitions: Dict[str, ScopeDefinition] = {}
        self.active_tokens: Dict[str, AuthorizationToken] = {}
        self.user_sessions: Dict[str, Dict[str, Any]] = {}
        self.failed_attempts: Dict[str, List[datetime]] = {}
        
        # JWT secret for token signing
        self.jwt_secret = secrets.token_hex(32)
        
        # Load configuration
        self._load_configuration()
        
        logger.info("Authorization Validator initialized with RBAC and scope enforcement")
    
    def _load_configuration(self):
        """Load authorization configuration"""
        try:
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                self._parse_configuration(config)
            else:
                logger.warning(f"Authorization config not found: {self.config_path}")
                self._load_default_configuration()
        except Exception as e:
            logger.error(f"Failed to load authorization config: {e}")
            self._load_default_configuration()
    
    def _load_default_configuration(self):
        """Load default authorization configuration"""
        # Default role permissions
        self.authorization_rules = {
            "guest": {
                "level": AuthorizationLevel.GUEST,
                "permissions": [PermissionScope.READ_ONLY],
                "max_risk_score": 1.0,
                "requires_mfa": False
            },
            "user": {
                "level": AuthorizationLevel.USER,
                "permissions": [PermissionScope.READ_ONLY, PermissionScope.LOG_ANALYSIS],
                "max_risk_score": 2.0,
                "requires_mfa": False
            },
            "analyst": {
                "level": AuthorizationLevel.ANALYST,
                "permissions": [
                    PermissionScope.READ_ONLY, PermissionScope.LOG_ANALYSIS,
                    PermissionScope.NETWORK_SCAN, PermissionScope.VULNERABILITY_SCAN
                ],
                "max_risk_score": 4.0,
                "requires_mfa": False
            },
            "security_analyst": {
                "level": AuthorizationLevel.SECURITY_ANALYST,
                "permissions": [
                    PermissionScope.READ_ONLY, PermissionScope.LOG_ANALYSIS,
                    PermissionScope.NETWORK_SCAN, PermissionScope.VULNERABILITY_SCAN,
                    PermissionScope.INCIDENT_INVESTIGATION
                ],
                "max_risk_score": 6.0,
                "requires_mfa": True
            },
            "security_admin": {
                "level": AuthorizationLevel.SECURITY_ADMIN,
                "permissions": [
                    PermissionScope.READ_ONLY, PermissionScope.LOG_ANALYSIS,
                    PermissionScope.NETWORK_SCAN, PermissionScope.VULNERABILITY_SCAN,
                    PermissionScope.INCIDENT_INVESTIGATION, PermissionScope.SYSTEM_MODIFICATION,
                    PermissionScope.DATA_ACCESS
                ],
                "max_risk_score": 8.0,
                "requires_mfa": True
            },
            "penetration_tester": {
                "level": AuthorizationLevel.PENETRATION_TESTER,
                "permissions": [
                    PermissionScope.READ_ONLY, PermissionScope.NETWORK_SCAN,
                    PermissionScope.VULNERABILITY_SCAN, PermissionScope.PENETRATION_TESTING
                ],
                "max_risk_score": 7.0,
                "requires_mfa": True,
                "requires_approval": True
            },
            "ciso": {
                "level": AuthorizationLevel.CISO,
                "permissions": list(PermissionScope),  # All permissions
                "max_risk_score": 10.0,
                "requires_mfa": True
            }
        }
        
        # Default scope definitions
        self.scope_definitions = {
            "dev_testing": ScopeDefinition(
                scope_id="dev_testing",
                name="Development Testing",
                description="Testing on development systems only",
                allowed_targets=["dev.*", "test.*", "staging.*"],
                allowed_operations=["scan", "test", "analyze"],
                time_restrictions=None,
                ip_restrictions=["10.0.0.0/8", "192.168.0.0/16"],
                requires_approval=False,
                max_duration=timedelta(hours=8),
                compliance_requirements=[]
            ),
            "production_readonly": ScopeDefinition(
                scope_id="production_readonly",
                name="Production Read-Only",
                description="Read-only access to production systems",
                allowed_targets=["prod.*"],
                allowed_operations=["read", "analyze", "monitor"],
                time_restrictions={"business_hours_only": True},
                ip_restrictions=["10.0.0.0/8"],
                requires_approval=True,
                max_duration=timedelta(hours=4),
                compliance_requirements=["SOC2", "SOX"]
            ),
            "penetration_testing": ScopeDefinition(
                scope_id="penetration_testing",
                name="Authorized Penetration Testing",
                description="Authorized penetration testing with defined scope",
                allowed_targets=[],  # Must be explicitly defined per engagement
                allowed_operations=["scan", "exploit", "test"],
                time_restrictions={"start_time": None, "end_time": None},
                ip_restrictions=[],  # Defined per engagement
                requires_approval=True,
                max_duration=timedelta(days=30),
                compliance_requirements=["legal_authorization", "scope_documentation"]
            ),
            "incident_response": ScopeDefinition(
                scope_id="incident_response",
                name="Incident Response",
                description="Emergency incident response operations",
                allowed_targets=["*"],  # Broad access during incidents
                allowed_operations=["*"],
                time_restrictions=None,  # 24/7 access during incidents
                ip_restrictions=[],
                requires_approval=False,  # Pre-approved for emergencies
                max_duration=timedelta(hours=72),
                compliance_requirements=["incident_documentation"]
            )
        }
        
        logger.info(f"Loaded default authorization config: {len(self.authorization_rules)} roles, "
                   f"{len(self.scope_definitions)} scopes")
    
    def _parse_configuration(self, config: Dict[str, Any]):
        """Parse configuration from JSON"""
        # Implementation for parsing complex authorization configuration
        pass
    
    async def validate_authorization(self, 
                                   user_id: str,
                                   requested_action: str,
                                   target_resource: str,
                                   context: AuthorizationContext) -> ValidationResult:
        """
        Validate user authorization for requested action
        
        Args:
            user_id: User identifier
            requested_action: Action being requested
            target_resource: Target resource/system
            context: Authorization context
            
        Returns:
            ValidationResult with authorization decision
        """
        start_time = time.time()
        
        try:
            # Step 1: Basic validation
            basic_result = await self._basic_validation(user_id, context)
            if not basic_result.authorized:
                return basic_result
            
            # Step 2: Role-based validation
            rbac_result = await self._rbac_validation(context, requested_action)
            if not rbac_result.authorized:
                return rbac_result
            
            # Step 3: Scope validation
            scope_result = await self._scope_validation(context, requested_action, target_resource)
            if not scope_result.authorized:
                return scope_result
            
            # Step 4: Risk-based validation
            risk_result = await self._risk_based_validation(context, requested_action, target_resource)
            if not risk_result.authorized:
                return risk_result
            
            # Step 5: Time and location validation
            temporal_result = await self._temporal_validation(context, requested_action)
            if not temporal_result.authorized:
                return temporal_result
            
            # Combine results
            final_result = self._combine_authorization_results([
                basic_result, rbac_result, scope_result, risk_result, temporal_result
            ])
            
            # Generate authorization token if successful
            if final_result.authorized:
                token = await self._generate_authorization_token(context, requested_action, target_resource)
                final_result.token_id = token.token_id
                final_result.expires_at = token.expires_at
            
            # Log authorization attempt
            validation_time = (time.time() - start_time) * 1000
            logger.info(f"Authorization validation complete: {final_result.authorized} "
                       f"for {user_id} -> {requested_action} (time: {validation_time:.2f}ms)")
            
            return final_result
            
        except Exception as e:
            logger.error(f"Authorization validation failed: {e}")
            return ValidationResult(
                authorized=False,
                authorization_level=AuthorizationLevel.GUEST,
                granted_permissions=set(),
                scope_limitations=[],
                requires_elevation=False,
                requires_approval=True,
                validation_errors=[f"System error: {str(e)}"],
                token_id=None,
                expires_at=None
            )
    
    async def _basic_validation(self, user_id: str, context: AuthorizationContext) -> ValidationResult:
        """Basic authorization validation"""
        errors = []
        
        # Check session validity
        if not self._is_session_valid(user_id, context.session_id):
            errors.append("Invalid or expired session")
        
        # Check for account lockout
        if self._is_account_locked(user_id):
            errors.append("Account is locked due to failed attempts")
        
        # Check MFA requirement
        user_role = self._get_user_primary_role(context.roles)
        if (user_role in self.authorization_rules and 
            self.authorization_rules[user_role].get("requires_mfa", False) and
            not context.mfa_verified):
            errors.append("Multi-factor authentication required")
        
        # Basic risk check
        if context.risk_score > 8.0:
            errors.append(f"Risk score too high: {context.risk_score}")
        
        return ValidationResult(
            authorized=len(errors) == 0,
            authorization_level=context.authorization_level,
            granted_permissions=context.permissions,
            scope_limitations=[],
            requires_elevation=False,
            requires_approval=False,
            validation_errors=errors,
            token_id=None,
            expires_at=None
        )
    
    async def _rbac_validation(self, context: AuthorizationContext, requested_action: str) -> ValidationResult:
        """Role-based access control validation"""
        errors = []
        requires_elevation = False
        requires_approval = False
        
        # Get primary role
        primary_role = self._get_user_primary_role(context.roles)
        
        if primary_role not in self.authorization_rules:
            errors.append(f"Unknown role: {primary_role}")
            return ValidationResult(
                authorized=False,
                authorization_level=AuthorizationLevel.GUEST,
                granted_permissions=set(),
                scope_limitations=[],
                requires_elevation=False,
                requires_approval=True,
                validation_errors=errors,
                token_id=None,
                expires_at=None
            )
        
        role_config = self.authorization_rules[primary_role]
        
        # Check authorization level
        required_level = self._get_required_authorization_level(requested_action)
        if context.authorization_level.value < required_level.value:
            requires_elevation = True
            errors.append(f"Insufficient authorization level: {context.authorization_level.name} < {required_level.name}")
        
        # Check permission requirements
        required_permissions = self._get_required_permissions(requested_action)
        missing_permissions = required_permissions - context.permissions
        
        if missing_permissions:
            errors.append(f"Missing permissions: {[p.value for p in missing_permissions]}")
        
        # Check if approval is required
        if role_config.get("requires_approval", False):
            requires_approval = True
        
        # Check maximum risk score for role
        max_risk = role_config.get("max_risk_score", 10.0)
        if context.risk_score > max_risk:
            errors.append(f"Risk score {context.risk_score} exceeds role maximum {max_risk}")
        
        return ValidationResult(
            authorized=len(errors) == 0,
            authorization_level=context.authorization_level,
            granted_permissions=context.permissions,
            scope_limitations=[],
            requires_elevation=requires_elevation,
            requires_approval=requires_approval,
            validation_errors=errors,
            token_id=None,
            expires_at=None
        )
    
    async def _scope_validation(self, context: AuthorizationContext, 
                              requested_action: str, target_resource: str) -> ValidationResult:
        """Scope enforcement validation"""
        errors = []
        scope_limitations = []
        
        # Determine applicable scope
        applicable_scope = self._determine_applicable_scope(context, requested_action, target_resource)
        
        if not applicable_scope:
            errors.append("No applicable scope found for requested action")
            return ValidationResult(
                authorized=False,
                authorization_level=context.authorization_level,
                granted_permissions=context.permissions,
                scope_limitations=scope_limitations,
                requires_elevation=False,
                requires_approval=True,
                validation_errors=errors,
                token_id=None,
                expires_at=None
            )
        
        # Validate target resource against allowed targets
        if not self._is_target_allowed(target_resource, applicable_scope.allowed_targets):
            errors.append(f"Target resource '{target_resource}' not in allowed scope")
            scope_limitations.append(f"Allowed targets: {applicable_scope.allowed_targets}")
        
        # Validate operation against allowed operations
        if not self._is_operation_allowed(requested_action, applicable_scope.allowed_operations):
            errors.append(f"Operation '{requested_action}' not allowed in scope")
            scope_limitations.append(f"Allowed operations: {applicable_scope.allowed_operations}")
        
        # Check IP restrictions
        if applicable_scope.ip_restrictions and not self._is_ip_allowed(context.source_ip, applicable_scope.ip_restrictions):
            errors.append(f"Source IP '{context.source_ip}' not in allowed IP ranges")
            scope_limitations.append(f"Allowed IP ranges: {applicable_scope.ip_restrictions}")
        
        return ValidationResult(
            authorized=len(errors) == 0,
            authorization_level=context.authorization_level,
            granted_permissions=context.permissions,
            scope_limitations=scope_limitations,
            requires_elevation=False,
            requires_approval=applicable_scope.requires_approval,
            validation_errors=errors,
            token_id=None,
            expires_at=None
        )
    
    async def _risk_based_validation(self, context: AuthorizationContext,
                                   requested_action: str, target_resource: str) -> ValidationResult:
        """Risk-based authorization validation"""
        errors = []
        requires_elevation = False
        
        # Calculate action-specific risk score
        action_risk = self._calculate_action_risk(requested_action, target_resource)
        
        # Combine with context risk
        total_risk = (context.risk_score + action_risk) / 2
        
        # Risk-based decision thresholds
        if total_risk >= 9.0:
            errors.append(f"Risk score too high for authorization: {total_risk:.2f}")
        elif total_risk >= 7.0:
            requires_elevation = True
        
        # High-risk operations require additional validation
        if action_risk >= 8.0:
            if not context.mfa_verified:
                errors.append("High-risk operation requires MFA verification")
            
            # Check for recent similar operations (potential abuse)
            if self._check_suspicious_pattern(context.user_id, requested_action):
                errors.append("Suspicious activity pattern detected")
        
        return ValidationResult(
            authorized=len(errors) == 0,
            authorization_level=context.authorization_level,
            granted_permissions=context.permissions,
            scope_limitations=[],
            requires_elevation=requires_elevation,
            requires_approval=total_risk >= 6.0,
            validation_errors=errors,
            token_id=None,
            expires_at=None
        )
    
    async def _temporal_validation(self, context: AuthorizationContext, requested_action: str) -> ValidationResult:
        """Time and location-based validation"""
        errors = []
        
        current_time = datetime.now()
        
        # Check business hours restrictions
        if self._requires_business_hours(requested_action):
            if not self._is_business_hours(current_time):
                errors.append("Operation only allowed during business hours (8 AM - 6 PM)")
        
        # Check weekend restrictions
        if self._requires_weekday(requested_action):
            if current_time.weekday() >= 5:  # Saturday = 5, Sunday = 6
                errors.append("Operation not allowed on weekends")
        
        # Check maintenance windows
        if self._is_maintenance_window(current_time):
            errors.append("Operations restricted during maintenance window")
        
        # Geolocation validation (simplified)
        if not self._is_location_authorized(context.source_ip):
            errors.append("Access from unauthorized geographic location")
        
        return ValidationResult(
            authorized=len(errors) == 0,
            authorization_level=context.authorization_level,
            granted_permissions=context.permissions,
            scope_limitations=[],
            requires_elevation=False,
            requires_approval=False,
            validation_errors=errors,
            token_id=None,
            expires_at=None
        )
    
    def _combine_authorization_results(self, results: List[ValidationResult]) -> ValidationResult:
        """Combine results from all validation layers"""
        # Authorization succeeds only if all layers pass
        all_authorized = all(result.authorized for result in results)
        
        # Combine authorization levels (use minimum)
        min_level = min(result.authorization_level for result in results)
        
        # Combine permissions (intersection)
        combined_permissions = set.intersection(*[result.granted_permissions for result in results])
        
        # Combine scope limitations
        all_limitations = []
        for result in results:
            all_limitations.extend(result.scope_limitations)
        
        # Combine errors
        all_errors = []
        for result in results:
            all_errors.extend(result.validation_errors)
        
        # Check if elevation or approval is required
        requires_elevation = any(result.requires_elevation for result in results)
        requires_approval = any(result.requires_approval for result in results)
        
        return ValidationResult(
            authorized=all_authorized,
            authorization_level=min_level,
            granted_permissions=combined_permissions,
            scope_limitations=list(set(all_limitations)),
            requires_elevation=requires_elevation,
            requires_approval=requires_approval,
            validation_errors=list(set(all_errors)),
            token_id=None,
            expires_at=None
        )
    
    async def _generate_authorization_token(self, context: AuthorizationContext,
                                          requested_action: str, target_resource: str) -> AuthorizationToken:
        """Generate secure authorization token"""
        token_id = secrets.token_hex(16)
        now = datetime.now()
        
        # Determine token lifetime based on action risk
        action_risk = self._calculate_action_risk(requested_action, target_resource)
        if action_risk >= 8.0:
            lifetime = timedelta(minutes=30)  # High-risk operations get short tokens
        elif action_risk >= 5.0:
            lifetime = timedelta(hours=2)
        else:
            lifetime = timedelta(hours=8)
        
        expires_at = now + lifetime
        
        # Find applicable scope
        applicable_scope = self._determine_applicable_scope(context, requested_action, target_resource)
        
        token = AuthorizationToken(
            token_id=token_id,
            user_id=context.user_id,
            issued_at=now,
            expires_at=expires_at,
            scope=applicable_scope,
            permissions=context.permissions,
            mfa_required=action_risk >= 7.0,
            single_use=action_risk >= 9.0
        )
        
        self.active_tokens[token_id] = token
        
        logger.info(f"Authorization token generated: {token_id} for {context.user_id}")
        return token
    
    # Helper methods
    def _is_session_valid(self, user_id: str, session_id: str) -> bool:
        """Check if user session is valid"""
        if user_id not in self.user_sessions:
            return False
        
        session = self.user_sessions[user_id].get(session_id)
        if not session:
            return False
        
        # Check expiration
        expires_at = datetime.fromisoformat(session.get("expires_at", ""))
        return datetime.now() < expires_at
    
    def _is_account_locked(self, user_id: str) -> bool:
        """Check if account is locked due to failed attempts"""
        if user_id not in self.failed_attempts:
            return False
        
        recent_failures = [
            attempt for attempt in self.failed_attempts[user_id]
            if datetime.now() - attempt < timedelta(minutes=30)
        ]
        
        return len(recent_failures) >= 5  # Lock after 5 failures in 30 minutes
    
    def _get_user_primary_role(self, roles: List[str]) -> str:
        """Get user's primary role (highest privilege)"""
        if not roles:
            return "guest"
        
        # Return role with highest authorization level
        role_levels = {
            role: self.authorization_rules.get(role, {}).get("level", AuthorizationLevel.GUEST)
            for role in roles if role in self.authorization_rules
        }
        
        if not role_levels:
            return "guest"
        
        return max(role_levels.keys(), key=lambda r: role_levels[r].value)
    
    def _get_required_authorization_level(self, action: str) -> AuthorizationLevel:
        """Get required authorization level for action"""
        action_lower = action.lower()
        
        if any(keyword in action_lower for keyword in ["production", "prod"]):
            return AuthorizationLevel.PRODUCTION_AUTHORIZED
        elif any(keyword in action_lower for keyword in ["penetration", "pentest", "exploit"]):
            return AuthorizationLevel.PENETRATION_TESTER
        elif any(keyword in action_lower for keyword in ["modify", "change", "delete", "admin"]):
            return AuthorizationLevel.SECURITY_ADMIN
        elif any(keyword in action_lower for keyword in ["investigate", "incident"]):
            return AuthorizationLevel.INCIDENT_RESPONDER
        elif any(keyword in action_lower for keyword in ["scan", "analyze"]):
            return AuthorizationLevel.SECURITY_ANALYST
        else:
            return AuthorizationLevel.USER
    
    def _get_required_permissions(self, action: str) -> Set[PermissionScope]:
        """Get required permissions for action"""
        action_lower = action.lower()
        required = set()
        
        if any(keyword in action_lower for keyword in ["scan", "network"]):
            required.add(PermissionScope.NETWORK_SCAN)
        if any(keyword in action_lower for keyword in ["vulnerability", "vuln"]):
            required.add(PermissionScope.VULNERABILITY_SCAN)
        if any(keyword in action_lower for keyword in ["penetration", "pentest", "exploit"]):
            required.add(PermissionScope.PENETRATION_TESTING)
        if any(keyword in action_lower for keyword in ["modify", "change", "admin"]):
            required.add(PermissionScope.SYSTEM_MODIFICATION)
        if any(keyword in action_lower for keyword in ["data", "database", "file"]):
            required.add(PermissionScope.DATA_ACCESS)
        if any(keyword in action_lower for keyword in ["production", "prod"]):
            required.add(PermissionScope.PRODUCTION_ACCESS)
        
        return required or {PermissionScope.READ_ONLY}
    
    def _determine_applicable_scope(self, context: AuthorizationContext, 
                                  action: str, target: str) -> Optional[ScopeDefinition]:
        """Determine which scope definition applies"""
        # Simple logic - in production this would be more sophisticated
        if "prod" in target.lower():
            return self.scope_definitions.get("production_readonly")
        elif any(keyword in action.lower() for keyword in ["penetration", "pentest"]):
            return self.scope_definitions.get("penetration_testing")
        elif "incident" in action.lower():
            return self.scope_definitions.get("incident_response")
        else:
            return self.scope_definitions.get("dev_testing")
    
    def _is_target_allowed(self, target: str, allowed_targets: List[str]) -> bool:
        """Check if target is allowed by scope"""
        if "*" in allowed_targets:
            return True
        
        for allowed in allowed_targets:
            if allowed.endswith("*"):
                if target.startswith(allowed[:-1]):
                    return True
            elif target == allowed:
                return True
        
        return False
    
    def _is_operation_allowed(self, operation: str, allowed_operations: List[str]) -> bool:
        """Check if operation is allowed by scope"""
        if "*" in allowed_operations:
            return True
        
        operation_lower = operation.lower()
        return any(allowed.lower() in operation_lower for allowed in allowed_operations)
    
    def _is_ip_allowed(self, source_ip: str, allowed_ranges: List[str]) -> bool:
        """Check if IP is in allowed ranges"""
        # Simplified IP range checking
        for allowed_range in allowed_ranges:
            if allowed_range.endswith("/8") and source_ip.startswith(allowed_range[:3]):
                return True
            elif allowed_range.endswith("/16") and source_ip.startswith(allowed_range[:7]):
                return True
        return False
    
    def _calculate_action_risk(self, action: str, target: str) -> float:
        """Calculate risk score for specific action and target"""
        risk_score = 1.0
        action_lower = action.lower()
        target_lower = target.lower()
        
        # High-risk actions
        if any(keyword in action_lower for keyword in ["exploit", "attack", "penetrate"]):
            risk_score += 4.0
        elif any(keyword in action_lower for keyword in ["modify", "delete", "change"]):
            risk_score += 3.0
        elif any(keyword in action_lower for keyword in ["scan", "probe"]):
            risk_score += 1.0
        
        # High-risk targets
        if any(keyword in target_lower for keyword in ["prod", "production", "live"]):
            risk_score += 3.0
        elif any(keyword in target_lower for keyword in ["database", "db", "sql"]):
            risk_score += 2.0
        elif any(keyword in target_lower for keyword in ["admin", "root", "system"]):
            risk_score += 2.0
        
        return min(risk_score, 10.0)
    
    def _check_suspicious_pattern(self, user_id: str, action: str) -> bool:
        """Check for suspicious activity patterns"""
        # Simplified pattern detection
        # In production, this would use ML/statistics
        return False
    
    def _requires_business_hours(self, action: str) -> bool:
        """Check if action requires business hours"""
        high_risk_keywords = ["production", "critical", "modify", "delete"]
        return any(keyword in action.lower() for keyword in high_risk_keywords)
    
    def _is_business_hours(self, timestamp: datetime) -> bool:
        """Check if timestamp is during business hours"""
        hour = timestamp.hour
        weekday = timestamp.weekday()
        return weekday < 5 and 8 <= hour <= 18  # Mon-Fri, 8 AM - 6 PM
    
    def _requires_weekday(self, action: str) -> bool:
        """Check if action requires weekday"""
        return "production" in action.lower()
    
    def _is_maintenance_window(self, timestamp: datetime) -> bool:
        """Check if timestamp is during maintenance window"""
        # Simplified: Sunday 2-4 AM is maintenance window
        return timestamp.weekday() == 6 and 2 <= timestamp.hour <= 4
    
    def _is_location_authorized(self, source_ip: str) -> bool:
        """Check if location is authorized (simplified)"""
        # Allow internal networks
        return (source_ip.startswith("10.") or 
                source_ip.startswith("192.168.") or 
                source_ip.startswith("127."))
    
    # Public API methods
    def validate_token(self, token_id: str) -> Tuple[bool, Optional[AuthorizationToken]]:
        """Validate authorization token"""
        if token_id not in self.active_tokens:
            return False, None
        
        token = self.active_tokens[token_id]
        
        # Check expiration
        if datetime.now() > token.expires_at:
            del self.active_tokens[token_id]
            return False, None
        
        # Check single-use tokens
        if token.single_use and token.used:
            del self.active_tokens[token_id]
            return False, None
        
        return True, token
    
    def revoke_token(self, token_id: str) -> bool:
        """Revoke authorization token"""
        if token_id in self.active_tokens:
            del self.active_tokens[token_id]
            logger.info(f"Authorization token revoked: {token_id}")
            return True
        return False
    
    def create_session(self, user_id: str, authorization_level: AuthorizationLevel,
                      roles: List[str], source_ip: str) -> str:
        """Create user session"""
        session_id = secrets.token_hex(16)
        expires_at = datetime.now() + timedelta(hours=12)
        
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = {}
        
        self.user_sessions[user_id][session_id] = {
            "authorization_level": authorization_level.name,
            "roles": roles,
            "source_ip": source_ip,
            "created_at": datetime.now().isoformat(),
            "expires_at": expires_at.isoformat()
        }
        
        logger.info(f"Session created: {session_id} for {user_id}")
        return session_id
    
    def get_authorization_status(self, user_id: str) -> Dict[str, Any]:
        """Get user authorization status"""
        return {
            "user_id": user_id,
            "active_sessions": len(self.user_sessions.get(user_id, {})),
            "active_tokens": len([t for t in self.active_tokens.values() if t.user_id == user_id]),
            "account_locked": self._is_account_locked(user_id),
            "failed_attempts": len(self.failed_attempts.get(user_id, [])),
            "last_activity": datetime.now().isoformat()
        }
    
    # MCP-specific authorization methods
    
    async def validate_mcp_authorization(self, 
                                       mcp_context: MCPAuthorizationContext,
                                       requested_action: str,
                                       target_resource: str) -> ValidationResult:
        """
        Validate MCP-specific authorization for AI agents
        
        Args:
            mcp_context: MCP authorization context
            requested_action: Action being requested
            target_resource: Target resource/system
            
        Returns:
            ValidationResult with MCP-specific validation
        """
        start_time = time.time()
        
        try:
            # Step 1: Base authorization validation
            base_result = await self.validate_authorization(
                mcp_context.user_id, requested_action, target_resource, mcp_context
            )
            
            if not base_result.authorized:
                return base_result
            
            # Step 2: MCP team authorization
            team_result = await self._validate_mcp_team_authorization(mcp_context, requested_action)
            if not team_result.authorized:
                return team_result
            
            # Step 3: Tool category authorization
            tool_result = await self._validate_mcp_tool_authorization(mcp_context, requested_action)
            if not tool_result.authorized:
                return tool_result
            
            # Step 4: External API authorization
            api_result = await self._validate_mcp_api_authorization(mcp_context, requested_action)
            if not api_result.authorized:
                return api_result
            
            # Step 5: Legal authorization validation
            legal_result = await self._validate_mcp_legal_authorization(mcp_context, requested_action, target_resource)
            if not legal_result.authorized:
                return legal_result
            
            # Step 6: Data classification authorization
            data_result = await self._validate_mcp_data_authorization(mcp_context, requested_action)
            if not data_result.authorized:
                return data_result
            
            # Combine MCP results
            all_results = [base_result, team_result, tool_result, api_result, legal_result, data_result]
            final_result = self._combine_authorization_results(all_results)
            
            # Generate MCP-specific token if successful
            if final_result.authorized:
                mcp_token = await self._generate_mcp_authorization_token(mcp_context, requested_action, target_resource)
                final_result.token_id = mcp_token.token_id
                final_result.expires_at = mcp_token.expires_at
            
            validation_time = (time.time() - start_time) * 1000
            logger.info(f"MCP authorization validation complete: {final_result.authorized} "
                       f"for {mcp_context.mcp_agent_id} ({mcp_context.team_type.value}) -> {requested_action} "
                       f"(time: {validation_time:.2f}ms)")
            
            return final_result
            
        except Exception as e:
            logger.error(f"MCP authorization validation failed: {e}")
            return ValidationResult(
                authorized=False,
                authorization_level=AuthorizationLevel.GUEST,
                granted_permissions=set(),
                scope_limitations=[],
                requires_elevation=False,
                requires_approval=True,
                validation_errors=[f"MCP system error: {str(e)}"],
                token_id=None,
                expires_at=None
            )
    
    async def _validate_mcp_team_authorization(self, mcp_context: MCPAuthorizationContext,
                                             requested_action: str) -> ValidationResult:
        """Validate MCP team-specific authorization"""
        errors = []
        
        # Team-specific constraints
        if mcp_context.team_type == MCPTeamType.RED_TEAM:
            # Red team constraints
            if not self._has_penetration_test_authorization(mcp_context):
                errors.append("Red team operations require valid penetration test authorization")
            
            if mcp_context.legal_authorization == LegalAuthorization.UNAUTHORIZED:
                errors.append("Red team operations require legal authorization")
            
            # Production system restrictions
            if self._targets_production_systems(mcp_context.target_systems):
                if mcp_context.legal_authorization != LegalAuthorization.PRODUCTION_APPROVED:
                    errors.append("Red team production access requires explicit production approval")
        
        elif mcp_context.team_type == MCPTeamType.BLUE_TEAM:
            # Blue team constraints
            if mcp_context.tool_category == MCPToolCategory.EXPLOITATION:
                errors.append("Blue team not authorized for exploitation tools")
            
            # Incident response authorization
            if (mcp_context.tool_category == MCPToolCategory.INCIDENT_RESPONSE and 
                not mcp_context.incident_id):
                errors.append("Incident response tools require valid incident ID")
        
        elif mcp_context.team_type == MCPTeamType.NEUTRAL:
            # Neutral team restrictions
            restricted_categories = [MCPToolCategory.EXPLOITATION, MCPToolCategory.VULNERABILITY_SCANNING]
            if mcp_context.tool_category in restricted_categories:
                errors.append(f"Neutral team not authorized for {mcp_context.tool_category.value}")
        
        # Cross-team operation restrictions
        if self._is_cross_team_operation(mcp_context, requested_action):
            errors.append("Cross-team operations require elevated authorization")
        
        return ValidationResult(
            authorized=len(errors) == 0,
            authorization_level=mcp_context.authorization_level,
            granted_permissions=mcp_context.permissions,
            scope_limitations=[],
            requires_elevation=False,
            requires_approval=len(errors) > 0,
            validation_errors=errors,
            token_id=None,
            expires_at=None
        )
    
    async def _validate_mcp_tool_authorization(self, mcp_context: MCPAuthorizationContext,
                                             requested_action: str) -> ValidationResult:
        """Validate tool-specific authorization"""
        errors = []
        requires_elevation = False
        
        # High-risk tool categories
        high_risk_tools = [
            MCPToolCategory.EXPLOITATION,
            MCPToolCategory.VULNERABILITY_SCANNING,
            MCPToolCategory.MEMORY_ANALYSIS
        ]
        
        if mcp_context.tool_category in high_risk_tools:
            # Require elevated privileges
            if mcp_context.authorization_level.value < AuthorizationLevel.SECURITY_ADMIN.value:
                requires_elevation = True
                errors.append(f"Tool category {mcp_context.tool_category.value} requires security admin privileges")
            
            # Require MFA for high-risk tools
            if not mcp_context.mfa_verified:
                errors.append(f"High-risk tool {mcp_context.tool_category.value} requires MFA verification")
        
        # Tool-specific authorization checks
        if mcp_context.tool_category == MCPToolCategory.EXPLOITATION:
            # Exploitation tools have strict requirements
            if mcp_context.legal_authorization not in [
                LegalAuthorization.AUTHORIZED_TESTING,
                LegalAuthorization.PRODUCTION_APPROVED
            ]:
                errors.append("Exploitation tools require authorized testing or production approval")
            
            if not mcp_context.penetration_test_id:
                errors.append("Exploitation tools require valid penetration test ID")
        
        elif mcp_context.tool_category == MCPToolCategory.MEMORY_ANALYSIS:
            # Memory analysis can access sensitive data
            if mcp_context.data_classification in [DataClassification.RESTRICTED, DataClassification.TOP_SECRET]:
                if mcp_context.authorization_level.value < AuthorizationLevel.PRIVACY_OFFICER.value:
                    errors.append("Memory analysis of classified data requires privacy officer privileges")
        
        # Time-based tool restrictions
        if self._has_time_restrictions(mcp_context.tool_category):
            if not self._is_authorized_time(mcp_context.timestamp, mcp_context.tool_category):
                errors.append(f"Tool {mcp_context.tool_category.value} not authorized at current time")
        
        return ValidationResult(
            authorized=len(errors) == 0,
            authorization_level=mcp_context.authorization_level,
            granted_permissions=mcp_context.permissions,
            scope_limitations=[],
            requires_elevation=requires_elevation,
            requires_approval=requires_elevation or len(errors) > 0,
            validation_errors=errors,
            token_id=None,
            expires_at=None
        )
    
    async def _validate_mcp_api_authorization(self, mcp_context: MCPAuthorizationContext,
                                            requested_action: str) -> ValidationResult:
        """Validate external API authorization"""
        errors = []
        scope_limitations = []
        
        # API-specific authorization
        for api_name in mcp_context.external_apis:
            # Team restrictions for APIs
            if not self._is_api_authorized_for_team(api_name, mcp_context.team_type):
                errors.append(f"Team {mcp_context.team_type.value} not authorized for {api_name} API")
            
            # Rate limiting validation
            if not self._check_api_rate_limits_for_user(mcp_context.user_id, api_name):
                errors.append(f"Rate limit exceeded for {api_name} API")
            
            # Data classification restrictions
            if not self._is_api_authorized_for_data_class(api_name, mcp_context.data_classification):
                errors.append(f"API {api_name} not authorized for {mcp_context.data_classification.value} data")
            
            # Geographic restrictions
            if not self._check_api_geographic_compliance(api_name, mcp_context.source_ip):
                errors.append(f"Geographic restrictions prevent {api_name} API access")
            
            # Add API-specific scope limitations
            api_limitations = self._get_api_scope_limitations(api_name, mcp_context.team_type)
            scope_limitations.extend(api_limitations)
        
        return ValidationResult(
            authorized=len(errors) == 0,
            authorization_level=mcp_context.authorization_level,
            granted_permissions=mcp_context.permissions,
            scope_limitations=scope_limitations,
            requires_elevation=False,
            requires_approval=len(errors) > 0,
            validation_errors=errors,
            token_id=None,
            expires_at=None
        )
    
    async def _validate_mcp_legal_authorization(self, mcp_context: MCPAuthorizationContext,
                                              requested_action: str, target_resource: str) -> ValidationResult:
        """Validate legal authorization for MCP operations"""
        errors = []
        requires_approval = False
        
        # Legal authorization level validation
        action_risk = self._calculate_action_risk(requested_action, target_resource)
        
        if action_risk >= 8.0:  # High-risk operations
            if mcp_context.legal_authorization != LegalAuthorization.PRODUCTION_APPROVED:
                errors.append("High-risk operations require production approval")
        
        elif action_risk >= 6.0:  # Medium-high risk
            if mcp_context.legal_authorization == LegalAuthorization.UNAUTHORIZED:
                errors.append("Medium-high risk operations require legal authorization")
        
        elif action_risk >= 4.0:  # Medium risk
            if mcp_context.legal_authorization == LegalAuthorization.UNAUTHORIZED:
                requires_approval = True
        
        # Penetration testing specific validation
        if mcp_context.team_type == MCPTeamType.RED_TEAM:
            if mcp_context.tool_category in [MCPToolCategory.EXPLOITATION, MCPToolCategory.VULNERABILITY_SCANNING]:
                if not self._has_valid_penetration_test_scope(mcp_context, target_resource):
                    errors.append("Target not within authorized penetration test scope")
        
        # Production system legal requirements
        if self._targets_production_systems(mcp_context.target_systems):
            if mcp_context.legal_authorization not in [
                LegalAuthorization.PRODUCTION_APPROVED,
                LegalAuthorization.EMERGENCY_RESPONSE
            ]:
                errors.append("Production system access requires production approval or emergency authorization")
        
        # Emergency response validation
        if mcp_context.legal_authorization == LegalAuthorization.EMERGENCY_RESPONSE:
            if not mcp_context.incident_id:
                errors.append("Emergency response authorization requires valid incident ID")
            
            # Emergency authorizations are time-limited
            if not self._is_within_emergency_timeframe(mcp_context):
                errors.append("Emergency authorization has expired")
        
        return ValidationResult(
            authorized=len(errors) == 0,
            authorization_level=mcp_context.authorization_level,
            granted_permissions=mcp_context.permissions,
            scope_limitations=[],
            requires_elevation=False,
            requires_approval=requires_approval or len(errors) > 0,
            validation_errors=errors,
            token_id=None,
            expires_at=None
        )
    
    async def _validate_mcp_data_authorization(self, mcp_context: MCPAuthorizationContext,
                                             requested_action: str) -> ValidationResult:
        """Validate data classification authorization"""
        errors = []
        requires_elevation = False
        
        # Data classification authorization levels
        data_auth_requirements = {
            DataClassification.PUBLIC: AuthorizationLevel.USER,
            DataClassification.INTERNAL: AuthorizationLevel.ANALYST,
            DataClassification.CONFIDENTIAL: AuthorizationLevel.SECURITY_ANALYST,
            DataClassification.RESTRICTED: AuthorizationLevel.PRIVACY_OFFICER,
            DataClassification.TOP_SECRET: AuthorizationLevel.FEDERAL_AUTHORIZED
        }
        
        required_level = data_auth_requirements.get(mcp_context.data_classification, AuthorizationLevel.USER)
        
        if mcp_context.authorization_level.value < required_level.value:
            requires_elevation = True
            errors.append(f"Data classification {mcp_context.data_classification.value} requires "
                         f"{required_level.name} authorization (current: {mcp_context.authorization_level.name})")
        
        # Sensitive data handling requirements
        if mcp_context.data_classification in [DataClassification.RESTRICTED, DataClassification.TOP_SECRET]:
            # Require additional safeguards
            if not mcp_context.mfa_verified:
                errors.append("Classified data access requires MFA verification")
            
            # Geographic restrictions for classified data
            if not self._is_classified_data_location_authorized(mcp_context.source_ip):
                errors.append("Geographic location not authorized for classified data access")
            
            # Time restrictions for classified data
            if not self._is_classified_data_time_authorized(mcp_context.timestamp):
                errors.append("Current time not authorized for classified data access")
        
        # Cross-border data transfer restrictions
        if self._involves_cross_border_data_transfer(mcp_context):
            if mcp_context.data_classification != DataClassification.PUBLIC:
                errors.append("Cross-border transfer of non-public data requires additional authorization")
        
        return ValidationResult(
            authorized=len(errors) == 0,
            authorization_level=mcp_context.authorization_level,
            granted_permissions=mcp_context.permissions,
            scope_limitations=[],
            requires_elevation=requires_elevation,
            requires_approval=requires_elevation or len(errors) > 0,
            validation_errors=errors,
            token_id=None,
            expires_at=None
        )
    
    async def _generate_mcp_authorization_token(self, mcp_context: MCPAuthorizationContext,
                                              requested_action: str, target_resource: str) -> AuthorizationToken:
        """Generate MCP-specific authorization token"""
        token_id = f"mcp_{secrets.token_hex(12)}"
        now = datetime.now()
        
        # Determine token lifetime based on risk and team type
        action_risk = self._calculate_action_risk(requested_action, target_resource)
        
        if mcp_context.team_type == MCPTeamType.RED_TEAM:
            # Red team tokens are shorter-lived
            if action_risk >= 8.0:
                lifetime = timedelta(minutes=15)
            elif action_risk >= 5.0:
                lifetime = timedelta(minutes=30)
            else:
                lifetime = timedelta(hours=1)
        
        elif mcp_context.team_type == MCPTeamType.BLUE_TEAM:
            # Blue team tokens can be longer
            if action_risk >= 8.0:
                lifetime = timedelta(minutes=30)
            elif action_risk >= 5.0:
                lifetime = timedelta(hours=1)
            else:
                lifetime = timedelta(hours=4)
        
        else:
            # Neutral/admin teams
            lifetime = timedelta(hours=2)
        
        # Classified data reduces token lifetime
        if mcp_context.data_classification in [DataClassification.RESTRICTED, DataClassification.TOP_SECRET]:
            lifetime = lifetime / 2
        
        expires_at = now + lifetime
        
        # Create MCP scope definition
        mcp_scope = ScopeDefinition(
            scope_id=f"mcp_{mcp_context.team_type.value}_{mcp_context.tool_category.value}",
            name=f"MCP {mcp_context.team_type.value} {mcp_context.tool_category.value}",
            description=f"MCP authorization for {mcp_context.team_type.value} using {mcp_context.tool_category.value}",
            allowed_targets=mcp_context.target_systems,
            allowed_operations=[requested_action],
            time_restrictions=None,
            ip_restrictions=[mcp_context.source_ip],
            requires_approval=mcp_context.legal_authorization == LegalAuthorization.RESEARCH_ONLY,
            max_duration=lifetime,
            compliance_requirements=[mcp_context.data_classification.value]
        )
        
        token = AuthorizationToken(
            token_id=token_id,
            user_id=mcp_context.user_id,
            issued_at=now,
            expires_at=expires_at,
            scope=mcp_scope,
            permissions=mcp_context.permissions,
            mfa_required=action_risk >= 7.0 or mcp_context.data_classification.value >= 3,
            single_use=action_risk >= 9.0 or mcp_context.tool_category == MCPToolCategory.EXPLOITATION
        )
        
        self.active_tokens[token_id] = token
        
        logger.info(f"MCP authorization token generated: {token_id} for {mcp_context.mcp_agent_id} "
                   f"({mcp_context.team_type.value}, expires: {expires_at})")
        return token
    
    # MCP-specific helper methods
    
    def _has_penetration_test_authorization(self, mcp_context: MCPAuthorizationContext) -> bool:
        """Check if user has valid penetration test authorization"""
        if mcp_context.penetration_test_id:
            # In production, would validate against active penetration test registry
            return True
        
        # Check if user has general penetration testing permissions
        return PermissionScope.PENETRATION_TESTING in mcp_context.permissions
    
    def _targets_production_systems(self, target_systems: List[str]) -> bool:
        """Check if any target systems are production systems"""
        production_indicators = ["prod", "production", "live", "prd"]
        
        for target in target_systems:
            if any(indicator in target.lower() for indicator in production_indicators):
                return True
        
        return False
    
    def _is_cross_team_operation(self, mcp_context: MCPAuthorizationContext, action: str) -> bool:
        """Check if operation involves cross-team resources"""
        # Simple heuristic - in production would be more sophisticated
        cross_team_keywords = ["share", "transfer", "collaborate", "handoff"]
        return any(keyword in action.lower() for keyword in cross_team_keywords)
    
    def _has_time_restrictions(self, tool_category: MCPToolCategory) -> bool:
        """Check if tool category has time restrictions"""
        restricted_tools = [
            MCPToolCategory.EXPLOITATION,
            MCPToolCategory.VULNERABILITY_SCANNING,
            MCPToolCategory.MEMORY_ANALYSIS
        ]
        return tool_category in restricted_tools
    
    def _is_authorized_time(self, timestamp: datetime, tool_category: MCPToolCategory) -> bool:
        """Check if tool usage is authorized at given time"""
        if tool_category == MCPToolCategory.EXPLOITATION:
            # Exploitation tools only during business hours
            return self._is_business_hours(timestamp)
        
        if tool_category == MCPToolCategory.VULNERABILITY_SCANNING:
            # Vulnerability scanning not during peak hours
            hour = timestamp.hour
            return not (9 <= hour <= 17)  # Avoid peak business hours
        
        return True
    
    def _is_api_authorized_for_team(self, api_name: str, team_type: MCPTeamType) -> bool:
        """Check if API is authorized for team type"""
        api_team_restrictions = {
            "shodan": [MCPTeamType.RED_TEAM, MCPTeamType.BLUE_TEAM],
            "virustotal": [MCPTeamType.BLUE_TEAM],
            "exploit_db": [MCPTeamType.RED_TEAM],
            "metasploit": [MCPTeamType.RED_TEAM],
            "misp": [MCPTeamType.BLUE_TEAM, MCPTeamType.NEUTRAL],
            "osquery": [MCPTeamType.BLUE_TEAM],
            "yara": [MCPTeamType.BLUE_TEAM],
            "volatility": [MCPTeamType.BLUE_TEAM],
            "suricata": [MCPTeamType.BLUE_TEAM]
        }
        
        allowed_teams = api_team_restrictions.get(api_name, [])
        return team_type in allowed_teams or team_type == MCPTeamType.ADMIN
    
    def _check_api_rate_limits_for_user(self, user_id: str, api_name: str) -> bool:
        """Check API rate limits for specific user"""
        # Simplified rate limiting - in production would use time-series data
        api_rate_limits = {
            "shodan": {"requests_per_hour": 100},
            "virustotal": {"requests_per_hour": 50},
            "exploit_db": {"requests_per_hour": 25},
            "metasploit": {"requests_per_hour": 10}
        }
        
        # For now, always return True - would implement actual rate limiting in production
        return True
    
    def _is_api_authorized_for_data_class(self, api_name: str, data_classification: DataClassification) -> bool:
        """Check if API is authorized for data classification level"""
        api_data_restrictions = {
            "shodan": [DataClassification.PUBLIC, DataClassification.INTERNAL],
            "virustotal": [DataClassification.PUBLIC, DataClassification.INTERNAL, DataClassification.CONFIDENTIAL],
            "exploit_db": [DataClassification.PUBLIC, DataClassification.INTERNAL],
            "metasploit": [DataClassification.INTERNAL, DataClassification.CONFIDENTIAL]
        }
        
        allowed_classifications = api_data_restrictions.get(api_name, list(DataClassification))
        return data_classification in allowed_classifications
    
    def _check_api_geographic_compliance(self, api_name: str, source_ip: str) -> bool:
        """Check API geographic compliance"""
        # Simplified geographic compliance
        try:
            ip_addr = ipaddress.ip_address(source_ip)
            
            # Block private IPs for external APIs (simplified check)
            if ip_addr.is_private:
                return True  # Allow private IPs for testing
            
            # In production, would check actual geolocation restrictions
            return True
            
        except ValueError:
            return False
    
    def _get_api_scope_limitations(self, api_name: str, team_type: MCPTeamType) -> List[str]:
        """Get API-specific scope limitations"""
        limitations = []
        
        if api_name == "shodan" and team_type == MCPTeamType.RED_TEAM:
            limitations.append("Shodan queries limited to authorized IP ranges")
        
        if api_name == "metasploit":
            limitations.append("Metasploit limited to authorized test environments")
        
        if api_name == "virustotal" and team_type == MCPTeamType.BLUE_TEAM:
            limitations.append("VirusTotal limited to security analysis purposes")
        
        return limitations
    
    def _has_valid_penetration_test_scope(self, mcp_context: MCPAuthorizationContext, target_resource: str) -> bool:
        """Check if target is within penetration test scope"""
        if not mcp_context.penetration_test_id:
            return False
        
        # In production, would validate against penetration test scope registry
        # For now, assume valid if penetration_test_id is provided
        return True
    
    def _is_within_emergency_timeframe(self, mcp_context: MCPAuthorizationContext) -> bool:
        """Check if emergency authorization is within valid timeframe"""
        if not mcp_context.incident_id:
            return False
        
        # Emergency authorizations typically valid for 72 hours
        # In production, would check against incident creation time
        return True
    
    def _is_classified_data_location_authorized(self, source_ip: str) -> bool:
        """Check if location is authorized for classified data access"""
        # Simplified location check for classified data
        try:
            ip_addr = ipaddress.ip_address(source_ip)
            
            # Only allow from specific secure networks
            authorized_networks = [
                ipaddress.ip_network("10.0.0.0/8"),
                ipaddress.ip_network("192.168.0.0/16"),
                ipaddress.ip_network("127.0.0.0/8")
            ]
            
            return any(ip_addr in network for network in authorized_networks)
            
        except ValueError:
            return False
    
    def _is_classified_data_time_authorized(self, timestamp: datetime) -> bool:
        """Check if time is authorized for classified data access"""
        # Classified data access only during extended business hours
        hour = timestamp.hour
        weekday = timestamp.weekday()
        
        # Monday-Friday, 6 AM - 10 PM
        return weekday < 5 and 6 <= hour <= 22
    
    def _involves_cross_border_data_transfer(self, mcp_context: MCPAuthorizationContext) -> bool:
        """Check if operation involves cross-border data transfer"""
        # Check if any external APIs are international
        international_apis = ["shodan", "virustotal", "exploit_db"]
        return any(api in international_apis for api in mcp_context.external_apis)