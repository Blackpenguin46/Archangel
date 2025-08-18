"""
Boundary enforcement system to prevent simulation escape.
"""

import time
import threading
import logging
import ipaddress
import socket
import subprocess
import re
import os
from typing import Dict, List, Optional, Set, Union, Pattern, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)

class BoundaryType(Enum):
    """Types of simulation boundaries."""
    NETWORK = "network"              # Network/IP address boundaries
    FILESYSTEM = "filesystem"        # File system path boundaries  
    PROCESS = "process"              # Process and system call boundaries
    COMMAND = "command"              # Command execution boundaries
    API = "api"                     # External API access boundaries
    TIME = "time"                   # Time-based boundaries (simulation duration)
    RESOURCE = "resource"           # Resource usage boundaries

class ViolationSeverity(Enum):
    """Severity levels for boundary violations."""
    LOW = "low"                     # Minor boundary crossing attempt
    MEDIUM = "medium"               # Significant boundary violation
    HIGH = "high"                   # Major simulation escape attempt
    CRITICAL = "critical"           # Imminent system compromise risk

@dataclass
class SimulationBoundary:
    """Definition of a simulation boundary."""
    boundary_id: str
    boundary_type: BoundaryType
    description: str
    allowed_patterns: List[str] = field(default_factory=list)
    blocked_patterns: List[str] = field(default_factory=list)
    allowed_ranges: List[str] = field(default_factory=list)  # For IP ranges, file paths, etc.
    blocked_ranges: List[str] = field(default_factory=list)
    max_violations: int = 5
    violation_window: float = 300.0  # 5 minutes
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass 
class BoundaryViolation:
    """Record of a boundary violation."""
    violation_id: str
    agent_id: str
    boundary_id: str
    boundary_type: BoundaryType
    severity: ViolationSeverity
    attempted_action: str
    blocked_content: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    automatic_response: str = ""
    resolved: bool = False

class BoundaryEnforcer:
    """
    Comprehensive boundary enforcement system to prevent simulation escape.
    
    This system provides:
    - Network boundary enforcement (IP ranges, domains)
    - Filesystem boundary enforcement (path restrictions)
    - Command execution boundaries
    - Process and system call monitoring
    - API access restrictions
    - Real-time violation detection and response
    - Automatic containment measures
    """
    
    def __init__(self):
        """Initialize the boundary enforcer."""
        self.boundaries: Dict[str, SimulationBoundary] = {}
        self.violations: List[BoundaryViolation] = []
        self.agent_violation_counts: Dict[str, Dict[str, int]] = {}  # agent_id -> boundary_id -> count
        self._lock = threading.RLock()
        
        # Compiled patterns for performance
        self._compiled_patterns: Dict[str, List[Pattern]] = {}
        
        # Callbacks for boundary events
        self._violation_callbacks: List[Callable[[BoundaryViolation], None]] = []
        self._enforcement_callbacks: List[Callable[[str, str], None]] = []  # action, reason
        
        # State tracking
        self.emergency_mode = False
        self.global_lockdown = False
        
        # Initialize default boundaries
        self._initialize_default_boundaries()
        
    def check_network_boundary(self, agent_id: str, target: str, port: Optional[int] = None) -> bool:
        """Check if network access is within boundaries.
        
        Args:
            agent_id: ID of the agent making the request
            target: Target hostname or IP address
            port: Optional port number
            
        Returns:
            True if access is allowed
        """
        try:
            # Resolve hostname to IP if needed
            if not self._is_ip_address(target):
                try:
                    target_ip = socket.gethostbyname(target)
                except socket.gaierror:
                    # Can't resolve - block by default
                    self._record_violation(
                        agent_id, "network_boundary", BoundaryType.NETWORK,
                        ViolationSeverity.MEDIUM,
                        f"DNS resolution of {target}",
                        target,
                        {"reason": "unresolvable_hostname", "port": port}
                    )
                    return False
            else:
                target_ip = target
                
            # Check against network boundaries
            for boundary in self._get_boundaries_by_type(BoundaryType.NETWORK):
                if not boundary.enabled:
                    continue
                    
                # Check allowed ranges
                if boundary.allowed_ranges:
                    if not any(self._ip_in_range(target_ip, range_spec) for range_spec in boundary.allowed_ranges):
                        self._record_violation(
                            agent_id, boundary.boundary_id, BoundaryType.NETWORK,
                            ViolationSeverity.HIGH,
                            f"Network access to {target}:{port}",
                            target_ip,
                            {"port": port, "reason": "not_in_allowed_ranges"}
                        )
                        return False
                        
                # Check blocked ranges  
                if boundary.blocked_ranges:
                    if any(self._ip_in_range(target_ip, range_spec) for range_spec in boundary.blocked_ranges):
                        self._record_violation(
                            agent_id, boundary.boundary_id, BoundaryType.NETWORK,
                            ViolationSeverity.CRITICAL,
                            f"Network access to blocked range {target}:{port}",
                            target_ip,
                            {"port": port, "reason": "in_blocked_range"}
                        )
                        return False
                        
                # Check hostname patterns
                if not self._check_patterns(target, boundary):
                    self._record_violation(
                        agent_id, boundary.boundary_id, BoundaryType.NETWORK,
                        ViolationSeverity.HIGH,
                        f"Network access to {target}:{port}",
                        target,
                        {"port": port, "reason": "hostname_pattern_violation"}
                    )
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Error checking network boundary: {e}")
            # Fail secure - block on error
            return False
            
    def check_filesystem_boundary(self, agent_id: str, path: str, operation: str) -> bool:
        """Check if filesystem access is within boundaries.
        
        Args:
            agent_id: ID of the agent making the request
            path: File or directory path being accessed
            operation: Type of operation (read, write, execute, delete)
            
        Returns:
            True if access is allowed
        """
        try:
            # Normalize path
            normalized_path = os.path.abspath(path)
            
            for boundary in self._get_boundaries_by_type(BoundaryType.FILESYSTEM):
                if not boundary.enabled:
                    continue
                    
                # Check allowed paths
                if boundary.allowed_ranges:
                    if not any(normalized_path.startswith(allowed_path) for allowed_path in boundary.allowed_ranges):
                        self._record_violation(
                            agent_id, boundary.boundary_id, BoundaryType.FILESYSTEM,
                            ViolationSeverity.HIGH,
                            f"Filesystem {operation} access to {path}",
                            normalized_path,
                            {"operation": operation, "reason": "path_not_allowed"}
                        )
                        return False
                        
                # Check blocked paths
                if boundary.blocked_ranges:
                    if any(normalized_path.startswith(blocked_path) for blocked_path in boundary.blocked_ranges):
                        self._record_violation(
                            agent_id, boundary.boundary_id, BoundaryType.FILESYSTEM,
                            ViolationSeverity.CRITICAL,
                            f"Filesystem {operation} access to blocked path {path}",
                            normalized_path,
                            {"operation": operation, "reason": "path_blocked"}
                        )
                        return False
                        
                # Check path patterns
                if not self._check_patterns(normalized_path, boundary):
                    self._record_violation(
                        agent_id, boundary.boundary_id, BoundaryType.FILESYSTEM,
                        ViolationSeverity.MEDIUM,
                        f"Filesystem {operation} access to {path}",
                        normalized_path,
                        {"operation": operation, "reason": "path_pattern_violation"}
                    )
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Error checking filesystem boundary: {e}")
            return False
            
    def check_command_boundary(self, agent_id: str, command: str, args: List[str]) -> bool:
        """Check if command execution is within boundaries.
        
        Args:
            agent_id: ID of the agent making the request
            command: Command being executed
            args: Command arguments
            
        Returns:
            True if execution is allowed
        """
        try:
            full_command = f"{command} {' '.join(args)}" if args else command
            
            for boundary in self._get_boundaries_by_type(BoundaryType.COMMAND):
                if not boundary.enabled:
                    continue
                    
                # Check blocked commands
                if boundary.blocked_patterns:
                    for pattern in self._get_compiled_patterns(boundary.boundary_id, boundary.blocked_patterns):
                        if pattern.search(full_command):
                            self._record_violation(
                                agent_id, boundary.boundary_id, BoundaryType.COMMAND,
                                ViolationSeverity.CRITICAL,
                                f"Blocked command execution: {command}",
                                full_command,
                                {"command": command, "args": args, "reason": "blocked_command"}
                            )
                            return False
                            
                # Check allowed commands (if specified)
                if boundary.allowed_patterns:
                    allowed = False
                    for pattern in self._get_compiled_patterns(boundary.boundary_id, boundary.allowed_patterns):
                        if pattern.search(full_command):
                            allowed = True
                            break
                    if not allowed:
                        self._record_violation(
                            agent_id, boundary.boundary_id, BoundaryType.COMMAND,
                            ViolationSeverity.HIGH,
                            f"Command not in allowed list: {command}",
                            full_command,
                            {"command": command, "args": args, "reason": "command_not_allowed"}
                        )
                        return False
                        
            return True
            
        except Exception as e:
            logger.error(f"Error checking command boundary: {e}")
            return False
            
    def check_api_boundary(self, agent_id: str, url: str, method: str) -> bool:
        """Check if API access is within boundaries.
        
        Args:
            agent_id: ID of the agent making the request
            url: API URL being accessed
            method: HTTP method (GET, POST, etc.)
            
        Returns:
            True if access is allowed
        """
        try:
            for boundary in self._get_boundaries_by_type(BoundaryType.API):
                if not boundary.enabled:
                    continue
                    
                # Extract domain from URL
                domain = self._extract_domain(url)
                
                # Check blocked domains
                if boundary.blocked_patterns:
                    for pattern in self._get_compiled_patterns(boundary.boundary_id, boundary.blocked_patterns):
                        if pattern.search(url) or pattern.search(domain):
                            self._record_violation(
                                agent_id, boundary.boundary_id, BoundaryType.API,
                                ViolationSeverity.HIGH,
                                f"Blocked API access: {method} {url}",
                                url,
                                {"method": method, "domain": domain, "reason": "blocked_api"}
                            )
                            return False
                            
                # Check allowed domains (if specified)
                if boundary.allowed_patterns:
                    allowed = False
                    for pattern in self._get_compiled_patterns(boundary.boundary_id, boundary.allowed_patterns):
                        if pattern.search(url) or pattern.search(domain):
                            allowed = True
                            break
                    if not allowed:
                        self._record_violation(
                            agent_id, boundary.boundary_id, BoundaryType.API,
                            ViolationSeverity.MEDIUM,
                            f"API not in allowed list: {method} {url}",
                            url,
                            {"method": method, "domain": domain, "reason": "api_not_allowed"}
                        )
                        return False
                        
            return True
            
        except Exception as e:
            logger.error(f"Error checking API boundary: {e}")
            return False
            
    def add_boundary(self, boundary: SimulationBoundary):
        """Add a simulation boundary.
        
        Args:
            boundary: SimulationBoundary to add
        """
        with self._lock:
            self.boundaries[boundary.boundary_id] = boundary
            # Clear compiled patterns cache for this boundary
            if boundary.boundary_id in self._compiled_patterns:
                del self._compiled_patterns[boundary.boundary_id]
            logger.info(f"Added simulation boundary: {boundary.boundary_id}")
            
    def remove_boundary(self, boundary_id: str):
        """Remove a simulation boundary.
        
        Args:
            boundary_id: ID of boundary to remove
        """
        with self._lock:
            if boundary_id in self.boundaries:
                del self.boundaries[boundary_id]
                if boundary_id in self._compiled_patterns:
                    del self._compiled_patterns[boundary_id]
                logger.info(f"Removed simulation boundary: {boundary_id}")
                
    def enable_boundary(self, boundary_id: str):
        """Enable a simulation boundary.
        
        Args:
            boundary_id: ID of boundary to enable
        """
        with self._lock:
            if boundary_id in self.boundaries:
                self.boundaries[boundary_id].enabled = True
                logger.info(f"Enabled boundary: {boundary_id}")
                
    def disable_boundary(self, boundary_id: str):
        """Disable a simulation boundary.
        
        Args:
            boundary_id: ID of boundary to disable
        """
        with self._lock:
            if boundary_id in self.boundaries:
                self.boundaries[boundary_id].enabled = False
                logger.warning(f"Disabled boundary: {boundary_id}")
                
    def emergency_lockdown(self, reason: str):
        """Activate emergency lockdown mode.
        
        Args:
            reason: Reason for emergency lockdown
        """
        with self._lock:
            self.emergency_mode = True
            self.global_lockdown = True
            logger.critical(f"EMERGENCY LOCKDOWN ACTIVATED: {reason}")
            
            # Notify enforcement callbacks
            for callback in self._enforcement_callbacks:
                try:
                    callback("emergency_lockdown", reason)
                except Exception as e:
                    logger.error(f"Error in enforcement callback: {e}")
                    
    def lift_emergency_lockdown(self, authorization_code: str):
        """Lift emergency lockdown (requires authorization).
        
        Args:
            authorization_code: Authorization code to lift lockdown
        """
        # In production, this would verify the authorization code
        expected_code = hashlib.sha256(b"archangel_emergency_override").hexdigest()[:16]
        
        if authorization_code == expected_code:
            with self._lock:
                self.emergency_mode = False
                self.global_lockdown = False
                logger.info("Emergency lockdown lifted")
        else:
            logger.error("Invalid authorization code for lockdown lift")
            
    def get_violation_summary(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get summary of boundary violations.
        
        Args:
            agent_id: Optional agent ID to filter by
            
        Returns:
            Dictionary with violation statistics
        """
        with self._lock:
            violations = self.violations
            if agent_id:
                violations = [v for v in violations if v.agent_id == agent_id]
                
            if not violations:
                return {"total_violations": 0, "by_severity": {}, "by_type": {}}
                
            by_severity = {}
            by_type = {}
            
            for violation in violations:
                # Count by severity
                severity = violation.severity.value
                by_severity[severity] = by_severity.get(severity, 0) + 1
                
                # Count by type
                boundary_type = violation.boundary_type.value
                by_type[boundary_type] = by_type.get(boundary_type, 0) + 1
                
            return {
                "total_violations": len(violations),
                "by_severity": by_severity,
                "by_type": by_type,
                "recent_violations": len([v for v in violations if time.time() - v.timestamp < 3600]),
                "emergency_mode": self.emergency_mode,
                "global_lockdown": self.global_lockdown
            }
            
    def _initialize_default_boundaries(self):
        """Initialize default simulation boundaries."""
        # Network boundary - restrict to simulation networks
        network_boundary = SimulationBoundary(
            boundary_id="default_network",
            boundary_type=BoundaryType.NETWORK,
            description="Default network boundary for simulation containment",
            allowed_ranges=[
                "192.168.10.0/24",  # DMZ network
                "192.168.20.0/24",  # Internal network
                "192.168.40.0/24",  # Management network
                "192.168.50.0/24",  # Deception network
                "127.0.0.1/32",     # Localhost
                "172.17.0.0/16"     # Docker default network
            ],
            blocked_ranges=[
                "10.0.0.0/8",       # Common private networks
                "172.16.0.0/12",    # Private networks (except Docker)
                "192.168.0.0/16"    # Private networks (except simulation)
            ],
            blocked_patterns=[
                r".*\.google\.com",
                r".*\.github\.com", 
                r".*\.amazon\.com",
                r".*\.microsoft\.com",
                r"production.*",
                r"prod.*"
            ]
        )
        
        # Filesystem boundary - restrict to simulation directories
        filesystem_boundary = SimulationBoundary(
            boundary_id="default_filesystem",
            boundary_type=BoundaryType.FILESYSTEM,
            description="Default filesystem boundary for simulation containment",
            allowed_ranges=[
                "/tmp/archangel",
                "/var/lib/archangel",
                "/opt/archangel",
                "/home/archangel"
            ],
            blocked_ranges=[
                "/",
                "/root",
                "/etc",
                "/usr/bin",
                "/usr/sbin", 
                "/sbin",
                "/bin",
                "/boot",
                "/sys",
                "/proc"
            ],
            blocked_patterns=[
                r".*\.ssh.*",
                r".*password.*",
                r".*shadow.*",
                r".*private.*",
                r".*\.key$"
            ]
        )
        
        # Command boundary - restrict dangerous commands
        command_boundary = SimulationBoundary(
            boundary_id="default_command",
            boundary_type=BoundaryType.COMMAND,
            description="Default command execution boundary",
            blocked_patterns=[
                r"rm\s+-rf\s+/",
                r".*shutdown.*",
                r".*reboot.*",
                r".*halt.*",
                r".*mkfs.*",
                r".*format.*",
                r"dd\s+.*of=/dev/",
                r".*iptables.*",
                r".*ufw.*",
                r".*firewall.*",
                r".*passwd.*",
                r".*useradd.*",
                r".*userdel.*",
                r".*sudo.*",
                r".*su\s+.*"
            ]
        )
        
        # API boundary - restrict external API access
        api_boundary = SimulationBoundary(
            boundary_id="default_api",
            boundary_type=BoundaryType.API,
            description="Default API access boundary",
            blocked_patterns=[
                r"https?://.*\.google\.com.*",
                r"https?://.*\.github\.com.*",
                r"https?://.*\.amazon\.com.*",
                r"https?://.*\.microsoft\.com.*",
                r"https?://api\..*",
                r"https?://.*production.*",
                r"https?://.*prod.*"
            ],
            allowed_patterns=[
                r"http://localhost.*",
                r"http://127\.0\.0\.1.*",
                r"http://192\.168\..*"
            ]
        )
        
        # Add default boundaries
        self.add_boundary(network_boundary)
        self.add_boundary(filesystem_boundary)
        self.add_boundary(command_boundary)
        self.add_boundary(api_boundary)
        
    def _get_boundaries_by_type(self, boundary_type: BoundaryType) -> List[SimulationBoundary]:
        """Get boundaries of a specific type.
        
        Args:
            boundary_type: Type of boundary to get
            
        Returns:
            List of matching boundaries
        """
        return [b for b in self.boundaries.values() if b.boundary_type == boundary_type]
        
    def _is_ip_address(self, address: str) -> bool:
        """Check if string is an IP address.
        
        Args:
            address: String to check
            
        Returns:
            True if it's an IP address
        """
        try:
            ipaddress.ip_address(address)
            return True
        except ValueError:
            return False
            
    def _ip_in_range(self, ip: str, range_spec: str) -> bool:
        """Check if IP is in specified range.
        
        Args:
            ip: IP address to check
            range_spec: IP range specification (CIDR notation)
            
        Returns:
            True if IP is in range
        """
        try:
            return ipaddress.ip_address(ip) in ipaddress.ip_network(range_spec, strict=False)
        except ValueError:
            return False
            
    def _check_patterns(self, text: str, boundary: SimulationBoundary) -> bool:
        """Check text against boundary patterns.
        
        Args:
            text: Text to check
            boundary: Boundary with patterns
            
        Returns:
            True if patterns allow the text
        """
        # Check blocked patterns first
        if boundary.blocked_patterns:
            blocked_patterns = self._get_compiled_patterns(boundary.boundary_id, boundary.blocked_patterns)
            if any(pattern.search(text) for pattern in blocked_patterns):
                return False
                
        # Check allowed patterns (if specified)
        if boundary.allowed_patterns:
            allowed_patterns = self._get_compiled_patterns(boundary.boundary_id, boundary.allowed_patterns)
            return any(pattern.search(text) for pattern in allowed_patterns)
            
        return True
        
    def _get_compiled_patterns(self, boundary_id: str, patterns: List[str]) -> List[Pattern]:
        """Get compiled regex patterns for a boundary.
        
        Args:
            boundary_id: ID of the boundary
            patterns: List of pattern strings
            
        Returns:
            List of compiled regex patterns
        """
        cache_key = f"{boundary_id}_{hashlib.md5('|'.join(patterns).encode()).hexdigest()}"
        
        if cache_key not in self._compiled_patterns:
            compiled = []
            for pattern in patterns:
                try:
                    compiled.append(re.compile(pattern, re.IGNORECASE))
                except re.error as e:
                    logger.error(f"Invalid regex pattern '{pattern}': {e}")
            self._compiled_patterns[cache_key] = compiled
            
        return self._compiled_patterns[cache_key]
        
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL.
        
        Args:
            url: URL to extract domain from
            
        Returns:
            Domain name
        """
        try:
            # Simple domain extraction
            if "://" in url:
                domain_part = url.split("://")[1]
            else:
                domain_part = url
                
            domain = domain_part.split("/")[0].split("?")[0].split("#")[0]
            return domain
        except:
            return url
            
    def _record_violation(self, agent_id: str, boundary_id: str, boundary_type: BoundaryType,
                         severity: ViolationSeverity, attempted_action: str, blocked_content: str,
                         evidence: Dict[str, Any]):
        """Record a boundary violation.
        
        Args:
            agent_id: ID of the agent
            boundary_id: ID of the violated boundary
            boundary_type: Type of boundary
            severity: Severity of violation
            attempted_action: What the agent tried to do
            blocked_content: What was blocked
            evidence: Additional evidence
        """
        violation_id = hashlib.md5(f"{agent_id}_{boundary_id}_{time.time()}".encode()).hexdigest()[:12]
        
        violation = BoundaryViolation(
            violation_id=violation_id,
            agent_id=agent_id,
            boundary_id=boundary_id,
            boundary_type=boundary_type,
            severity=severity,
            attempted_action=attempted_action,
            blocked_content=blocked_content,
            evidence=evidence,
            automatic_response=self._determine_automatic_response(severity)
        )
        
        with self._lock:
            self.violations.append(violation)
            
            # Update agent violation counts
            if agent_id not in self.agent_violation_counts:
                self.agent_violation_counts[agent_id] = {}
            if boundary_id not in self.agent_violation_counts[agent_id]:
                self.agent_violation_counts[agent_id][boundary_id] = 0
            self.agent_violation_counts[agent_id][boundary_id] += 1
            
            # Check for violation threshold breaches
            if self._check_violation_threshold(agent_id, boundary_id):
                self._escalate_violation(agent_id, boundary_id, violation)
                
        # Notify callbacks
        for callback in self._violation_callbacks:
            try:
                callback(violation)
            except Exception as e:
                logger.error(f"Error in violation callback: {e}")
                
        logger.warning(f"Boundary violation: {violation_id} - {attempted_action}")
        
    def _determine_automatic_response(self, severity: ViolationSeverity) -> str:
        """Determine automatic response to violation.
        
        Args:
            severity: Severity of the violation
            
        Returns:
            Description of automatic response
        """
        if severity == ViolationSeverity.CRITICAL:
            return "Agent quarantined, emergency protocols activated"
        elif severity == ViolationSeverity.HIGH:
            return "Enhanced monitoring enabled, action blocked"
        elif severity == ViolationSeverity.MEDIUM:
            return "Action blocked, warning logged"
        else:
            return "Action blocked, informational log"
            
    def _check_violation_threshold(self, agent_id: str, boundary_id: str) -> bool:
        """Check if agent has exceeded violation threshold.
        
        Args:
            agent_id: ID of the agent
            boundary_id: ID of the boundary
            
        Returns:
            True if threshold exceeded
        """
        boundary = self.boundaries.get(boundary_id)
        if not boundary:
            return False
            
        violation_count = self.agent_violation_counts.get(agent_id, {}).get(boundary_id, 0)
        return violation_count >= boundary.max_violations
        
    def _escalate_violation(self, agent_id: str, boundary_id: str, violation: BoundaryViolation):
        """Escalate violation due to threshold breach.
        
        Args:
            agent_id: ID of the agent
            boundary_id: ID of the boundary
            violation: The violation that triggered escalation
        """
        logger.critical(f"Violation threshold exceeded for agent {agent_id} on boundary {boundary_id}")
        
        # Activate emergency measures for critical violations
        if violation.severity == ViolationSeverity.CRITICAL:
            self.emergency_lockdown(f"Critical boundary violation by agent {agent_id}")
        
        # Notify enforcement callbacks
        for callback in self._enforcement_callbacks:
            try:
                callback("violation_threshold_exceeded", f"Agent {agent_id}, boundary {boundary_id}")
            except Exception as e:
                logger.error(f"Error in enforcement callback: {e}")

# Global boundary enforcer instance
boundary_enforcer = BoundaryEnforcer()