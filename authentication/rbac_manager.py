"""
Role-Based Access Control (RBAC) Manager for agent scope limitation.

This module provides RBAC functionality with:
- Role and permission management
- Agent scope limitation
- Resource access control
- Policy enforcement
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Union, Tuple, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import fnmatch


class PermissionType(Enum):
    """Types of permissions."""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    DELETE = "delete"
    ADMIN = "admin"


class ResourceType(Enum):
    """Types of resources that can be protected."""
    AGENT = "agent"
    COORDINATION = "coordination"
    INTELLIGENCE = "intelligence"
    MONITORING = "monitoring"
    SYSTEM = "system"
    NETWORK = "network"
    FILE = "file"
    DATABASE = "database"
    SERVICE = "service"


@dataclass
class Permission:
    """Individual permission definition."""
    name: str
    resource_type: ResourceType
    permission_type: PermissionType
    resource_pattern: str = "*"  # Glob pattern for resource matching
    description: str = ""
    
    def matches_resource(self, resource_name: str) -> bool:
        """Check if permission applies to a specific resource."""
        return fnmatch.fnmatch(resource_name, self.resource_pattern)


@dataclass
class Role:
    """Role definition with permissions."""
    name: str
    description: str
    permissions: Set[str] = field(default_factory=set)  # Permission names
    parent_roles: Set[str] = field(default_factory=set)  # Inherited roles
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def add_permission(self, permission_name: str) -> None:
        """Add permission to role."""
        self.permissions.add(permission_name)
        self.updated_at = datetime.now()
    
    def remove_permission(self, permission_name: str) -> None:
        """Remove permission from role."""
        self.permissions.discard(permission_name)
        self.updated_at = datetime.now()
    
    def add_parent_role(self, role_name: str) -> None:
        """Add parent role for inheritance."""
        self.parent_roles.add(role_name)
        self.updated_at = datetime.now()


@dataclass
class Subject:
    """Subject (user/agent) with role assignments."""
    subject_id: str
    subject_type: str  # "user", "agent", "service"
    display_name: str
    roles: Set[str] = field(default_factory=set)
    direct_permissions: Set[str] = field(default_factory=set)
    attributes: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_access: Optional[datetime] = None
    is_active: bool = True
    
    def add_role(self, role_name: str) -> None:
        """Add role to subject."""
        self.roles.add(role_name)
    
    def remove_role(self, role_name: str) -> None:
        """Remove role from subject."""
        self.roles.discard(role_name)
    
    def add_direct_permission(self, permission_name: str) -> None:
        """Add direct permission to subject."""
        self.direct_permissions.add(permission_name)
    
    def update_last_access(self) -> None:
        """Update last access timestamp."""
        self.last_access = datetime.now()


@dataclass
class AccessRequest:
    """Access request for authorization."""
    subject_id: str
    resource_type: ResourceType
    resource_name: str
    permission_type: PermissionType
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AccessDecision:
    """Access control decision."""
    request: AccessRequest
    granted: bool
    reason: str
    applicable_permissions: List[str] = field(default_factory=list)
    decision_time: datetime = field(default_factory=datetime.now)


@dataclass
class PolicyRule:
    """Policy rule for access control."""
    name: str
    description: str
    subject_pattern: str = "*"  # Pattern for subject matching
    resource_pattern: str = "*"  # Pattern for resource matching
    permission_pattern: str = "*"  # Pattern for permission matching
    effect: str = "allow"  # "allow" or "deny"
    conditions: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # Higher priority rules are evaluated first
    
    def matches_request(self, request: AccessRequest, subject: Subject) -> bool:
        """Check if policy rule matches the access request."""
        
        # Check subject pattern
        if not fnmatch.fnmatch(request.subject_id, self.subject_pattern):
            return False
        
        # Check resource pattern
        resource_full_name = f"{request.resource_type.value}:{request.resource_name}"
        if not fnmatch.fnmatch(resource_full_name, self.resource_pattern):
            return False
        
        # Check permission pattern
        if not fnmatch.fnmatch(request.permission_type.value, self.permission_pattern):
            return False
        
        # Check conditions
        if self.conditions:
            return self._evaluate_conditions(request, subject)
        
        return True
    
    def _evaluate_conditions(self, request: AccessRequest, subject: Subject) -> bool:
        """Evaluate policy conditions."""
        
        for condition_key, condition_value in self.conditions.items():
            if condition_key == "time_range":
                # Check if current time is within allowed range
                current_hour = datetime.now().hour
                start_hour, end_hour = condition_value
                if not (start_hour <= current_hour <= end_hour):
                    return False
            
            elif condition_key == "subject_type":
                if subject.subject_type != condition_value:
                    return False
            
            elif condition_key == "has_attribute":
                attr_name, attr_value = condition_value
                if subject.attributes.get(attr_name) != attr_value:
                    return False
            
            elif condition_key == "context_contains":
                if condition_value not in request.context:
                    return False
        
        return True


class RBACManager:
    """
    Role-Based Access Control Manager.
    
    Provides comprehensive RBAC functionality including:
    - Role and permission management
    - Subject (user/agent) management
    - Access control decisions
    - Policy enforcement
    """
    
    def __init__(self):
        self.permissions: Dict[str, Permission] = {}
        self.roles: Dict[str, Role] = {}
        self.subjects: Dict[str, Subject] = {}
        self.policy_rules: List[PolicyRule] = []
        self.access_log: List[AccessDecision] = []
        self.logger = logging.getLogger(__name__)
        
        # Initialize default RBAC structure
        self._initialize_default_rbac()
    
    def _initialize_default_rbac(self) -> None:
        """Initialize default roles, permissions, and policies."""
        
        # Create default permissions
        default_permissions = [
            # Agent permissions
            ("agent:read", ResourceType.AGENT, PermissionType.READ, "*", "Read agent information"),
            ("agent:write", ResourceType.AGENT, PermissionType.WRITE, "*", "Modify agent configuration"),
            ("agent:execute", ResourceType.AGENT, PermissionType.EXECUTE, "*", "Execute agent actions"),
            ("agent:admin", ResourceType.AGENT, PermissionType.ADMIN, "*", "Full agent administration"),
            
            # Coordination permissions
            ("coordination:participate", ResourceType.COORDINATION, PermissionType.EXECUTE, "*", "Participate in coordination"),
            ("coordination:manage", ResourceType.COORDINATION, PermissionType.ADMIN, "*", "Manage coordination system"),
            
            # Intelligence permissions
            ("intelligence:read", ResourceType.INTELLIGENCE, PermissionType.READ, "*", "Read intelligence data"),
            ("intelligence:share", ResourceType.INTELLIGENCE, PermissionType.WRITE, "*", "Share intelligence"),
            ("intelligence:admin", ResourceType.INTELLIGENCE, PermissionType.ADMIN, "*", "Manage intelligence system"),
            
            # Monitoring permissions
            ("monitoring:read", ResourceType.MONITORING, PermissionType.READ, "*", "Read monitoring data"),
            ("monitoring:access", ResourceType.MONITORING, PermissionType.EXECUTE, "*", "Access monitoring systems"),
            ("monitoring:admin", ResourceType.MONITORING, PermissionType.ADMIN, "*", "Administer monitoring"),
            
            # System permissions
            ("system:read", ResourceType.SYSTEM, PermissionType.READ, "*", "Read system information"),
            ("system:admin", ResourceType.SYSTEM, PermissionType.ADMIN, "*", "System administration"),
            
            # Defense permissions
            ("defense:read", ResourceType.SERVICE, PermissionType.READ, "defense:*", "Read defense information"),
            ("defense:manage", ResourceType.SERVICE, PermissionType.ADMIN, "defense:*", "Manage defense systems"),
            
            # Network permissions
            ("network:scan", ResourceType.NETWORK, PermissionType.READ, "*", "Network scanning"),
            ("network:attack", ResourceType.NETWORK, PermissionType.EXECUTE, "*", "Network attacks"),
            ("network:defend", ResourceType.NETWORK, PermissionType.WRITE, "*", "Network defense"),
        ]
        
        for perm_name, resource_type, perm_type, pattern, description in default_permissions:
            permission = Permission(
                name=perm_name,
                resource_type=resource_type,
                permission_type=perm_type,
                resource_pattern=pattern,
                description=description
            )
            self.permissions[perm_name] = permission
        
        # Create default roles
        self._create_default_roles()
        
        # Create default subjects
        self._create_default_subjects()
        
        # Create default policies
        self._create_default_policies()
    
    def _create_default_roles(self) -> None:
        """Create default roles with appropriate permissions."""
        
        # Red Team Agent role
        red_team_role = Role(
            name="red_team_agent",
            description="Red Team Agent with offensive capabilities"
        )
        red_team_permissions = [
            "agent:read", "agent:write", "agent:execute",
            "coordination:participate", "intelligence:read", "intelligence:share",
            "network:scan", "network:attack", "monitoring:read"
        ]
        for perm in red_team_permissions:
            red_team_role.add_permission(perm)
        self.roles[red_team_role.name] = red_team_role
        
        # Blue Team Agent role
        blue_team_role = Role(
            name="blue_team_agent",
            description="Blue Team Agent with defensive capabilities"
        )
        blue_team_permissions = [
            "agent:read", "agent:write", "agent:execute",
            "coordination:participate", "intelligence:read", "intelligence:share",
            "defense:read", "defense:manage", "network:defend",
            "monitoring:read", "monitoring:access"
        ]
        for perm in blue_team_permissions:
            blue_team_role.add_permission(perm)
        self.roles[blue_team_role.name] = blue_team_role
        
        # Coordinator role
        coordinator_role = Role(
            name="coordinator",
            description="System Coordinator with management capabilities"
        )
        coordinator_permissions = [
            "agent:admin", "coordination:manage", "intelligence:admin",
            "monitoring:admin", "system:read", "system:admin"
        ]
        for perm in coordinator_permissions:
            coordinator_role.add_permission(perm)
        self.roles[coordinator_role.name] = coordinator_role
        
        # Monitoring System role
        monitoring_role = Role(
            name="monitoring_system",
            description="Monitoring System with read-only access"
        )
        monitoring_permissions = [
            "agent:read", "intelligence:read", "monitoring:read",
            "system:read", "defense:read"
        ]
        for perm in monitoring_permissions:
            monitoring_role.add_permission(perm)
        self.roles[monitoring_role.name] = monitoring_role
        
        # Observer role (limited access)
        observer_role = Role(
            name="observer",
            description="Observer with read-only access"
        )
        observer_permissions = ["agent:read", "intelligence:read", "monitoring:read"]
        for perm in observer_permissions:
            observer_role.add_permission(perm)
        self.roles[observer_role.name] = observer_role
    
    def _create_default_subjects(self) -> None:
        """Create default subjects (agents and services)."""
        
        # Red Team agents
        red_agents = ["recon_agent", "exploit_agent", "persistence_agent", "exfiltration_agent"]
        for agent_id in red_agents:
            subject = Subject(
                subject_id=agent_id,
                subject_type="agent",
                display_name=agent_id.replace("_", " ").title(),
                attributes={"team": "red", "agent_type": agent_id.split("_")[0]}
            )
            subject.add_role("red_team_agent")
            self.subjects[agent_id] = subject
        
        # Blue Team agents
        blue_agents = ["soc_analyst", "firewall_config", "siem_integrator", "compliance_audit"]
        for agent_id in blue_agents:
            subject = Subject(
                subject_id=agent_id,
                subject_type="agent",
                display_name=agent_id.replace("_", " ").title(),
                attributes={"team": "blue", "agent_type": agent_id.split("_")[0]}
            )
            subject.add_role("blue_team_agent")
            self.subjects[agent_id] = subject
        
        # System services
        coordinator_subject = Subject(
            subject_id="archangel_coordinator",
            subject_type="service",
            display_name="Archangel Coordinator",
            attributes={"service_type": "coordination"}
        )
        coordinator_subject.add_role("coordinator")
        self.subjects["archangel_coordinator"] = coordinator_subject
        
        monitoring_subject = Subject(
            subject_id="monitoring_system",
            subject_type="service",
            display_name="Monitoring System",
            attributes={"service_type": "monitoring"}
        )
        monitoring_subject.add_role("monitoring_system")
        self.subjects["monitoring_system"] = monitoring_subject
    
    def _create_default_policies(self) -> None:
        """Create default policy rules."""
        
        # Time-based restrictions
        business_hours_policy = PolicyRule(
            name="business_hours_only",
            description="Restrict certain operations to business hours",
            resource_pattern="system:*",
            permission_pattern="admin",
            effect="deny",
            conditions={"time_range": (9, 17)},  # 9 AM to 5 PM
            priority=100
        )
        self.policy_rules.append(business_hours_policy)
        
        # Team isolation policy
        red_team_isolation = PolicyRule(
            name="red_team_isolation",
            description="Red team agents cannot access blue team resources",
            subject_pattern="*",
            resource_pattern="defense:*",
            effect="deny",
            conditions={"has_attribute": ("team", "red")},
            priority=90
        )
        self.policy_rules.append(red_team_isolation)
        
        blue_team_isolation = PolicyRule(
            name="blue_team_isolation",
            description="Blue team agents cannot perform attacks",
            subject_pattern="*",
            resource_pattern="network:*",
            permission_pattern="attack",
            effect="deny",
            conditions={"has_attribute": ("team", "blue")},
            priority=90
        )
        self.policy_rules.append(blue_team_isolation)
        
        # Admin protection policy
        admin_protection = PolicyRule(
            name="admin_protection",
            description="Only coordinators can perform admin operations",
            resource_pattern="*",
            permission_pattern="admin",
            effect="deny",
            priority=80
        )
        self.policy_rules.append(admin_protection)
        
        # Allow coordinator admin access
        coordinator_admin = PolicyRule(
            name="coordinator_admin_access",
            description="Coordinators have admin access",
            subject_pattern="*coordinator*",
            resource_pattern="*",
            permission_pattern="admin",
            effect="allow",
            priority=85
        )
        self.policy_rules.append(coordinator_admin)
    
    def create_permission(self, name: str, resource_type: ResourceType,
                         permission_type: PermissionType, resource_pattern: str = "*",
                         description: str = "") -> Permission:
        """Create a new permission."""
        
        if name in self.permissions:
            raise ValueError(f"Permission {name} already exists")
        
        permission = Permission(
            name=name,
            resource_type=resource_type,
            permission_type=permission_type,
            resource_pattern=resource_pattern,
            description=description
        )
        
        self.permissions[name] = permission
        self.logger.info(f"Created permission: {name}")
        return permission
    
    def create_role(self, name: str, description: str,
                   permissions: Optional[List[str]] = None,
                   parent_roles: Optional[List[str]] = None) -> Role:
        """Create a new role."""
        
        if name in self.roles:
            raise ValueError(f"Role {name} already exists")
        
        role = Role(name=name, description=description)
        
        if permissions:
            for perm_name in permissions:
                if perm_name in self.permissions:
                    role.add_permission(perm_name)
                else:
                    self.logger.warning(f"Unknown permission: {perm_name}")
        
        if parent_roles:
            for parent_name in parent_roles:
                if parent_name in self.roles:
                    role.add_parent_role(parent_name)
                else:
                    self.logger.warning(f"Unknown parent role: {parent_name}")
        
        self.roles[name] = role
        self.logger.info(f"Created role: {name}")
        return role
    
    def create_subject(self, subject_id: str, subject_type: str, display_name: str,
                      roles: Optional[List[str]] = None,
                      attributes: Optional[Dict[str, Any]] = None) -> Subject:
        """Create a new subject."""
        
        if subject_id in self.subjects:
            raise ValueError(f"Subject {subject_id} already exists")
        
        subject = Subject(
            subject_id=subject_id,
            subject_type=subject_type,
            display_name=display_name,
            attributes=attributes or {}
        )
        
        if roles:
            for role_name in roles:
                if role_name in self.roles:
                    subject.add_role(role_name)
                else:
                    self.logger.warning(f"Unknown role: {role_name}")
        
        self.subjects[subject_id] = subject
        self.logger.info(f"Created subject: {subject_id}")
        return subject
    
    def assign_role(self, subject_id: str, role_name: str) -> bool:
        """Assign role to subject."""
        
        if subject_id not in self.subjects:
            self.logger.error(f"Unknown subject: {subject_id}")
            return False
        
        if role_name not in self.roles:
            self.logger.error(f"Unknown role: {role_name}")
            return False
        
        self.subjects[subject_id].add_role(role_name)
        self.logger.info(f"Assigned role {role_name} to subject {subject_id}")
        return True
    
    def revoke_role(self, subject_id: str, role_name: str) -> bool:
        """Revoke role from subject."""
        
        if subject_id not in self.subjects:
            self.logger.error(f"Unknown subject: {subject_id}")
            return False
        
        self.subjects[subject_id].remove_role(role_name)
        self.logger.info(f"Revoked role {role_name} from subject {subject_id}")
        return True
    
    def check_access(self, subject_id: str, resource_type: ResourceType,
                    resource_name: str, permission_type: PermissionType,
                    context: Optional[Dict[str, Any]] = None) -> AccessDecision:
        """
        Check if subject has access to perform operation on resource.
        
        Args:
            subject_id: Subject requesting access
            resource_type: Type of resource being accessed
            resource_name: Name of specific resource
            permission_type: Type of permission required
            context: Additional context for decision
            
        Returns:
            Access decision with details
        """
        
        request = AccessRequest(
            subject_id=subject_id,
            resource_type=resource_type,
            resource_name=resource_name,
            permission_type=permission_type,
            context=context or {}
        )
        
        # Check if subject exists
        if subject_id not in self.subjects:
            decision = AccessDecision(
                request=request,
                granted=False,
                reason=f"Unknown subject: {subject_id}"
            )
            self.access_log.append(decision)
            return decision
        
        subject = self.subjects[subject_id]
        
        # Update last access time
        subject.update_last_access()
        
        # Check if subject is active
        if not subject.is_active:
            decision = AccessDecision(
                request=request,
                granted=False,
                reason="Subject is inactive"
            )
            self.access_log.append(decision)
            return decision
        
        # Evaluate policy rules first (they can override permissions)
        policy_decision = self._evaluate_policies(request, subject)
        if policy_decision is not None:
            decision = AccessDecision(
                request=request,
                granted=policy_decision[0],
                reason=policy_decision[1]
            )
            self.access_log.append(decision)
            return decision
        
        # Get all permissions for subject (including inherited)
        all_permissions = self._get_subject_permissions(subject)
        
        # Check if any permission grants access
        applicable_permissions = []
        for perm_name in all_permissions:
            if perm_name in self.permissions:
                permission = self.permissions[perm_name]
                
                # Check if permission applies to this request
                if (permission.resource_type == resource_type and
                    permission.permission_type == permission_type and
                    permission.matches_resource(resource_name)):
                    
                    applicable_permissions.append(perm_name)
        
        # Grant access if any applicable permission found
        granted = len(applicable_permissions) > 0
        reason = f"Access {'granted' if granted else 'denied'} based on permissions"
        
        if granted:
            reason += f" (applicable: {', '.join(applicable_permissions)})"
        
        decision = AccessDecision(
            request=request,
            granted=granted,
            reason=reason,
            applicable_permissions=applicable_permissions
        )
        
        self.access_log.append(decision)
        
        if granted:
            self.logger.debug(f"Access granted: {subject_id} -> {resource_type.value}:{resource_name} ({permission_type.value})")
        else:
            self.logger.warning(f"Access denied: {subject_id} -> {resource_type.value}:{resource_name} ({permission_type.value})")
        
        return decision
    
    def _evaluate_policies(self, request: AccessRequest, subject: Subject) -> Optional[Tuple[bool, str]]:
        """Evaluate policy rules for access request."""
        
        # Sort policies by priority (higher first)
        sorted_policies = sorted(self.policy_rules, key=lambda p: p.priority, reverse=True)
        
        for policy in sorted_policies:
            if policy.matches_request(request, subject):
                if policy.effect == "deny":
                    return False, f"Denied by policy: {policy.name}"
                elif policy.effect == "allow":
                    return True, f"Allowed by policy: {policy.name}"
        
        return None  # No policy matched, continue with permission check
    
    def _get_subject_permissions(self, subject: Subject) -> Set[str]:
        """Get all permissions for subject including inherited from roles."""
        
        all_permissions = subject.direct_permissions.copy()
        
        # Add permissions from roles (including inherited roles)
        processed_roles = set()
        roles_to_process = list(subject.roles)
        
        while roles_to_process:
            role_name = roles_to_process.pop(0)
            
            if role_name in processed_roles or role_name not in self.roles:
                continue
            
            processed_roles.add(role_name)
            role = self.roles[role_name]
            
            # Add role permissions
            all_permissions.update(role.permissions)
            
            # Add parent roles to processing queue
            roles_to_process.extend(role.parent_roles)
        
        return all_permissions
    
    def get_subject_info(self, subject_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a subject."""
        
        if subject_id not in self.subjects:
            return None
        
        subject = self.subjects[subject_id]
        all_permissions = self._get_subject_permissions(subject)
        
        return {
            "subject_id": subject.subject_id,
            "subject_type": subject.subject_type,
            "display_name": subject.display_name,
            "roles": list(subject.roles),
            "direct_permissions": list(subject.direct_permissions),
            "all_permissions": list(all_permissions),
            "attributes": subject.attributes,
            "created_at": subject.created_at.isoformat(),
            "last_access": subject.last_access.isoformat() if subject.last_access else None,
            "is_active": subject.is_active
        }
    
    def get_role_info(self, role_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a role."""
        
        if role_name not in self.roles:
            return None
        
        role = self.roles[role_name]
        
        return {
            "name": role.name,
            "description": role.description,
            "permissions": list(role.permissions),
            "parent_roles": list(role.parent_roles),
            "created_at": role.created_at.isoformat(),
            "updated_at": role.updated_at.isoformat()
        }
    
    def get_access_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get access control statistics."""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_decisions = [decision for decision in self.access_log if decision.decision_time >= cutoff_time]
        
        if not recent_decisions:
            return {
                "total_requests": 0,
                "granted_requests": 0,
                "denied_requests": 0,
                "top_subjects": [],
                "top_resources": [],
                "denial_reasons": {}
            }
        
        granted_count = len([d for d in recent_decisions if d.granted])
        denied_count = len(recent_decisions) - granted_count
        
        # Top subjects by request count
        subject_counts = {}
        for decision in recent_decisions:
            subject_id = decision.request.subject_id
            subject_counts[subject_id] = subject_counts.get(subject_id, 0) + 1
        
        top_subjects = sorted(subject_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Top resources by request count
        resource_counts = {}
        for decision in recent_decisions:
            resource_key = f"{decision.request.resource_type.value}:{decision.request.resource_name}"
            resource_counts[resource_key] = resource_counts.get(resource_key, 0) + 1
        
        top_resources = sorted(resource_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Denial reasons
        denial_reasons = {}
        for decision in recent_decisions:
            if not decision.granted:
                reason = decision.reason
                denial_reasons[reason] = denial_reasons.get(reason, 0) + 1
        
        return {
            "total_requests": len(recent_decisions),
            "granted_requests": granted_count,
            "denied_requests": denied_count,
            "top_subjects": top_subjects,
            "top_resources": top_resources,
            "denial_reasons": denial_reasons,
            "total_subjects": len(self.subjects),
            "total_roles": len(self.roles),
            "total_permissions": len(self.permissions),
            "total_policies": len(self.policy_rules)
        }