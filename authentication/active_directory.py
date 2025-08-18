"""
Active Directory simulation with realistic domain structure and policies.

This module simulates an enterprise Active Directory environment with:
- Domain controllers and organizational units
- User accounts with realistic attributes
- Group policies and security groups
- Authentication and authorization services
"""

import json
import hashlib
import secrets
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
try:
    import ldap3
    LDAP3_AVAILABLE = True
except ImportError:
    LDAP3_AVAILABLE = False

try:
    from cryptography.fernet import Fernet
    FERNET_AVAILABLE = True
except ImportError:
    FERNET_AVAILABLE = False


class UserAccountControl(Enum):
    """User Account Control flags for AD users."""
    NORMAL_ACCOUNT = 0x0200
    DISABLED_ACCOUNT = 0x0002
    PASSWORD_NEVER_EXPIRES = 0x10000
    ACCOUNT_LOCKED = 0x0010
    MUST_CHANGE_PASSWORD = 0x800000


class GroupType(Enum):
    """Active Directory group types."""
    SECURITY_GLOBAL = "security_global"
    SECURITY_DOMAIN_LOCAL = "security_domain_local"
    SECURITY_UNIVERSAL = "security_universal"
    DISTRIBUTION_GLOBAL = "distribution_global"


@dataclass
class ADUser:
    """Active Directory user object."""
    username: str
    display_name: str
    email: str
    department: str
    title: str
    manager: Optional[str] = None
    password_hash: str = ""
    account_control: UserAccountControl = UserAccountControl.NORMAL_ACCOUNT
    groups: Set[str] = field(default_factory=set)
    last_logon: Optional[datetime] = None
    password_last_set: datetime = field(default_factory=datetime.now)
    login_count: int = 0
    failed_login_count: int = 0
    account_locked_until: Optional[datetime] = None
    
    def __post_init__(self):
        if not self.password_hash:
            # Generate a default password hash
            self.password_hash = self._hash_password("Password123!")
    
    def _hash_password(self, password: str) -> str:
        """Hash password using SHA-256 with salt."""
        salt = secrets.token_hex(16)
        return hashlib.sha256((password + salt).encode()).hexdigest() + ":" + salt
    
    def verify_password(self, password: str) -> bool:
        """Verify password against stored hash."""
        try:
            stored_hash, salt = self.password_hash.split(":")
            test_hash = hashlib.sha256((password + salt).encode()).hexdigest()
            return stored_hash == test_hash
        except ValueError:
            return False
    
    def is_account_locked(self) -> bool:
        """Check if account is currently locked."""
        if self.account_locked_until:
            return datetime.now() < self.account_locked_until
        return self.account_control == UserAccountControl.ACCOUNT_LOCKED
    
    def is_account_disabled(self) -> bool:
        """Check if account is disabled."""
        return self.account_control == UserAccountControl.DISABLED_ACCOUNT


@dataclass
class ADGroup:
    """Active Directory group object."""
    name: str
    description: str
    group_type: GroupType
    members: Set[str] = field(default_factory=set)
    nested_groups: Set[str] = field(default_factory=set)
    
    def add_member(self, username: str) -> None:
        """Add user to group."""
        self.members.add(username)
    
    def remove_member(self, username: str) -> None:
        """Remove user from group."""
        self.members.discard(username)
    
    def add_nested_group(self, group_name: str) -> None:
        """Add nested group."""
        self.nested_groups.add(group_name)


@dataclass
class OrganizationalUnit:
    """Active Directory Organizational Unit."""
    name: str
    description: str
    parent_ou: Optional[str] = None
    users: Set[str] = field(default_factory=set)
    groups: Set[str] = field(default_factory=set)
    child_ous: Set[str] = field(default_factory=set)


@dataclass
class GroupPolicy:
    """Group Policy Object."""
    name: str
    description: str
    settings: Dict[str, Any] = field(default_factory=dict)
    linked_ous: Set[str] = field(default_factory=set)
    enabled: bool = True


class ActiveDirectorySimulator:
    """
    Simulates an enterprise Active Directory environment.
    
    Provides realistic domain structure, user management, authentication,
    and policy enforcement for cybersecurity training scenarios.
    """
    
    def __init__(self, domain_name: str = "archangel.local"):
        self.domain_name = domain_name
        self.domain_controller = f"dc01.{domain_name}"
        self.users: Dict[str, ADUser] = {}
        self.groups: Dict[str, ADGroup] = {}
        self.organizational_units: Dict[str, OrganizationalUnit] = {}
        self.group_policies: Dict[str, GroupPolicy] = {}
        self.authentication_log: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)
        
        # Initialize default domain structure
        self._initialize_default_structure()
    
    def _initialize_default_structure(self) -> None:
        """Initialize default AD structure with realistic OUs and groups."""
        
        # Create default Organizational Units
        self.create_organizational_unit("Users", "Default Users Container")
        self.create_organizational_unit("Computers", "Default Computers Container")
        self.create_organizational_unit("IT", "Information Technology Department")
        self.create_organizational_unit("HR", "Human Resources Department")
        self.create_organizational_unit("Finance", "Finance Department")
        self.create_organizational_unit("Sales", "Sales Department")
        self.create_organizational_unit("ServiceAccounts", "Service Accounts")
        
        # Create default security groups
        self.create_group("Domain Admins", "Domain Administrators", GroupType.SECURITY_GLOBAL)
        self.create_group("Enterprise Admins", "Enterprise Administrators", GroupType.SECURITY_UNIVERSAL)
        self.create_group("IT Support", "IT Support Staff", GroupType.SECURITY_GLOBAL)
        self.create_group("HR Staff", "Human Resources Staff", GroupType.SECURITY_GLOBAL)
        self.create_group("Finance Users", "Finance Department Users", GroupType.SECURITY_GLOBAL)
        self.create_group("Sales Team", "Sales Department Team", GroupType.SECURITY_GLOBAL)
        self.create_group("Remote Users", "Remote Access Users", GroupType.SECURITY_GLOBAL)
        
        # Create default users
        self._create_default_users()
        
        # Create default group policies
        self._create_default_policies()
    
    def _create_default_users(self) -> None:
        """Create realistic default users for simulation."""
        
        default_users = [
            # IT Department
            ("admin", "Domain Administrator", "admin@archangel.local", "IT", "Domain Administrator", None, ["Domain Admins", "Enterprise Admins"]),
            ("jsmith", "John Smith", "jsmith@archangel.local", "IT", "IT Manager", None, ["IT Support"]),
            ("mwilson", "Mary Wilson", "mwilson@archangel.local", "IT", "System Administrator", "jsmith", ["IT Support"]),
            ("backup_svc", "Backup Service", "backup@archangel.local", "ServiceAccounts", "Service Account", None, []),
            
            # HR Department
            ("hjones", "Helen Jones", "hjones@archangel.local", "HR", "HR Director", None, ["HR Staff"]),
            ("rbrown", "Robert Brown", "rbrown@archangel.local", "HR", "HR Specialist", "hjones", ["HR Staff"]),
            
            # Finance Department
            ("slee", "Sarah Lee", "slee@archangel.local", "Finance", "CFO", None, ["Finance Users"]),
            ("dchen", "David Chen", "dchen@archangel.local", "Finance", "Accountant", "slee", ["Finance Users"]),
            
            # Sales Department
            ("tgarcia", "Tom Garcia", "tgarcia@archangel.local", "Sales", "Sales Manager", None, ["Sales Team"]),
            ("kpatel", "Kim Patel", "kpatel@archangel.local", "Sales", "Sales Rep", "tgarcia", ["Sales Team", "Remote Users"]),
        ]
        
        for username, display_name, email, dept, title, manager, groups in default_users:
            user = self.create_user(username, display_name, email, dept, title, manager)
            for group in groups:
                self.add_user_to_group(username, group)
    
    def _create_default_policies(self) -> None:
        """Create default group policies."""
        
        # Password Policy
        password_policy = GroupPolicy(
            name="Default Domain Password Policy",
            description="Default password complexity requirements",
            settings={
                "minimum_password_length": 8,
                "password_complexity": True,
                "maximum_password_age": 90,
                "minimum_password_age": 1,
                "password_history": 12,
                "account_lockout_threshold": 5,
                "account_lockout_duration": 30
            }
        )
        self.group_policies["password_policy"] = password_policy
        
        # Security Policy
        security_policy = GroupPolicy(
            name="Default Security Policy",
            description="Default security settings",
            settings={
                "require_secure_logon": True,
                "disable_guest_account": True,
                "audit_logon_events": True,
                "audit_privilege_use": True,
                "minimum_session_security": "NTLMv2"
            }
        )
        self.group_policies["security_policy"] = security_policy
    
    def create_user(self, username: str, display_name: str, email: str, 
                   department: str, title: str, manager: Optional[str] = None) -> ADUser:
        """Create a new Active Directory user."""
        
        if username in self.users:
            raise ValueError(f"User {username} already exists")
        
        user = ADUser(
            username=username,
            display_name=display_name,
            email=email,
            department=department,
            title=title,
            manager=manager
        )
        
        self.users[username] = user
        
        # Add to appropriate OU based on department
        ou_name = department if department in self.organizational_units else "Users"
        self.organizational_units[ou_name].users.add(username)
        
        self.logger.info(f"Created user: {username} in department: {department}")
        return user
    
    def create_group(self, name: str, description: str, group_type: GroupType) -> ADGroup:
        """Create a new Active Directory group."""
        
        if name in self.groups:
            raise ValueError(f"Group {name} already exists")
        
        group = ADGroup(
            name=name,
            description=description,
            group_type=group_type
        )
        
        self.groups[name] = group
        self.logger.info(f"Created group: {name}")
        return group
    
    def create_organizational_unit(self, name: str, description: str, 
                                 parent_ou: Optional[str] = None) -> OrganizationalUnit:
        """Create a new Organizational Unit."""
        
        if name in self.organizational_units:
            raise ValueError(f"OU {name} already exists")
        
        ou = OrganizationalUnit(
            name=name,
            description=description,
            parent_ou=parent_ou
        )
        
        self.organizational_units[name] = ou
        
        if parent_ou and parent_ou in self.organizational_units:
            self.organizational_units[parent_ou].child_ous.add(name)
        
        self.logger.info(f"Created OU: {name}")
        return ou
    
    def add_user_to_group(self, username: str, group_name: str) -> bool:
        """Add user to a group."""
        
        if username not in self.users:
            self.logger.error(f"User {username} not found")
            return False
        
        if group_name not in self.groups:
            self.logger.error(f"Group {group_name} not found")
            return False
        
        self.groups[group_name].add_member(username)
        self.users[username].groups.add(group_name)
        
        self.logger.info(f"Added user {username} to group {group_name}")
        return True
    
    def authenticate_user(self, username: str, password: str, 
                         source_ip: str = "127.0.0.1") -> Dict[str, Any]:
        """
        Authenticate user credentials.
        
        Returns authentication result with success status and user info.
        """
        
        auth_result = {
            "success": False,
            "username": username,
            "timestamp": datetime.now(),
            "source_ip": source_ip,
            "reason": "",
            "user_info": None
        }
        
        # Check if user exists
        if username not in self.users:
            auth_result["reason"] = "User not found"
            self._log_authentication_attempt(auth_result)
            return auth_result
        
        user = self.users[username]
        
        # Check if account is disabled
        if user.is_account_disabled():
            auth_result["reason"] = "Account disabled"
            self._log_authentication_attempt(auth_result)
            return auth_result
        
        # Check if account is locked
        if user.is_account_locked():
            auth_result["reason"] = "Account locked"
            self._log_authentication_attempt(auth_result)
            return auth_result
        
        # Verify password
        if not user.verify_password(password):
            user.failed_login_count += 1
            
            # Check for account lockout
            lockout_threshold = self.group_policies.get("password_policy", {}).settings.get("account_lockout_threshold", 5)
            if user.failed_login_count >= lockout_threshold:
                lockout_duration = self.group_policies.get("password_policy", {}).settings.get("account_lockout_duration", 30)
                user.account_locked_until = datetime.now() + timedelta(minutes=lockout_duration)
                auth_result["reason"] = "Account locked due to failed attempts"
            else:
                auth_result["reason"] = "Invalid password"
            
            self._log_authentication_attempt(auth_result)
            return auth_result
        
        # Successful authentication
        user.last_logon = datetime.now()
        user.login_count += 1
        user.failed_login_count = 0  # Reset failed attempts
        
        auth_result.update({
            "success": True,
            "reason": "Authentication successful",
            "user_info": {
                "username": user.username,
                "display_name": user.display_name,
                "email": user.email,
                "department": user.department,
                "title": user.title,
                "groups": list(user.groups),
                "last_logon": user.last_logon,
                "login_count": user.login_count
            }
        })
        
        self._log_authentication_attempt(auth_result)
        return auth_result
    
    def _log_authentication_attempt(self, auth_result: Dict[str, Any]) -> None:
        """Log authentication attempt for audit purposes."""
        
        log_entry = {
            "event_id": len(self.authentication_log) + 1,
            "timestamp": auth_result["timestamp"],
            "event_type": "authentication",
            "username": auth_result["username"],
            "source_ip": auth_result["source_ip"],
            "success": auth_result["success"],
            "reason": auth_result["reason"]
        }
        
        self.authentication_log.append(log_entry)
        
        if auth_result["success"]:
            self.logger.info(f"Successful authentication: {auth_result['username']} from {auth_result['source_ip']}")
        else:
            self.logger.warning(f"Failed authentication: {auth_result['username']} from {auth_result['source_ip']} - {auth_result['reason']}")
    
    def get_user_groups(self, username: str, recursive: bool = True) -> Set[str]:
        """Get all groups for a user, optionally including nested groups."""
        
        if username not in self.users:
            return set()
        
        user_groups = self.users[username].groups.copy()
        
        if recursive:
            # Add nested groups
            groups_to_check = list(user_groups)
            while groups_to_check:
                group_name = groups_to_check.pop()
                if group_name in self.groups:
                    nested_groups = self.groups[group_name].nested_groups
                    for nested_group in nested_groups:
                        if nested_group not in user_groups:
                            user_groups.add(nested_group)
                            groups_to_check.append(nested_group)
        
        return user_groups
    
    def check_user_permission(self, username: str, required_groups: List[str]) -> bool:
        """Check if user has required group membership."""
        
        user_groups = self.get_user_groups(username)
        return any(group in user_groups for group in required_groups)
    
    def get_domain_info(self) -> Dict[str, Any]:
        """Get domain information summary."""
        
        return {
            "domain_name": self.domain_name,
            "domain_controller": self.domain_controller,
            "total_users": len(self.users),
            "total_groups": len(self.groups),
            "total_ous": len(self.organizational_units),
            "total_policies": len(self.group_policies),
            "authentication_events": len(self.authentication_log)
        }
    
    def export_configuration(self) -> Dict[str, Any]:
        """Export AD configuration for backup or analysis."""
        
        return {
            "domain_name": self.domain_name,
            "users": {username: {
                "display_name": user.display_name,
                "email": user.email,
                "department": user.department,
                "title": user.title,
                "groups": list(user.groups)
            } for username, user in self.users.items()},
            "groups": {name: {
                "description": group.description,
                "group_type": group.group_type.value,
                "members": list(group.members)
            } for name, group in self.groups.items()},
            "organizational_units": {name: {
                "description": ou.description,
                "users": list(ou.users),
                "groups": list(ou.groups)
            } for name, ou in self.organizational_units.items()}
        }
    
    def get_authentication_logs(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get authentication logs for audit purposes."""
        
        logs = self.authentication_log.copy()
        if limit:
            logs = logs[-limit:]
        
        return logs