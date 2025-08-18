#!/usr/bin/env python3
"""
Comprehensive Security Controls and Monitoring System
Production-grade security hardening with real-time monitoring and alerting
"""

import os
import sys
import json
import time
import psutil
import hashlib
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Set
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import sqlite3
from collections import defaultdict
import subprocess
import socket
import ssl
import requests
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


class SecurityLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high" 
    CRITICAL = "critical"


class ControlStatus(Enum):
    ENABLED = "enabled"
    DISABLED = "disabled"
    FAILED = "failed"
    MONITORING = "monitoring"


@dataclass
class SecurityEvent:
    event_id: str
    timestamp: datetime
    event_type: str
    severity: SecurityLevel
    source: str
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolution_time: Optional[datetime] = None


@dataclass
class SecurityControl:
    control_id: str
    name: str
    description: str
    category: str
    severity: SecurityLevel
    status: ControlStatus
    last_check: Optional[datetime] = None
    check_interval: int = 300  # seconds
    enabled: bool = True
    remediation_actions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SecurityControlsMonitor:
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "security/config/controls.json"
        self.controls: Dict[str, SecurityControl] = {}
        self.events: List[SecurityEvent] = []
        self.monitors: Dict[str, threading.Thread] = {}
        self.running = False
        self.lock = threading.Lock()
        
        # Database for event logging
        self.db_path = "security/data/security_events.db"
        self._init_database()
        
        # Encryption for sensitive data
        self.cipher_suite = self._init_encryption()
        
        # Alert handlers
        self.alert_handlers: List[Callable[[SecurityEvent], None]] = []
        
        # System baselines
        self.system_baseline = self._establish_baseline()
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _init_database(self):
        """Initialize SQLite database for event logging"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS security_events (
                event_id TEXT PRIMARY KEY,
                timestamp TEXT,
                event_type TEXT,
                severity TEXT,
                source TEXT,
                description TEXT,
                metadata TEXT,
                resolved BOOLEAN,
                resolution_time TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS control_status (
                control_id TEXT PRIMARY KEY,
                name TEXT,
                status TEXT,
                last_check TEXT,
                metadata TEXT
            )
        ''')
        
        conn.commit()
        conn.close()

    def _init_encryption(self) -> Fernet:
        """Initialize encryption for sensitive data"""
        key_file = Path("security/keys/master.key")
        
        if key_file.exists():
            with open(key_file, 'rb') as f:
                key = f.read()
        else:
            # Generate new key
            key = Fernet.generate_key()
            os.makedirs(key_file.parent, exist_ok=True)
            with open(key_file, 'wb') as f:
                f.write(key)
            os.chmod(key_file, 0o600)
        
        return Fernet(key)

    def _establish_baseline(self) -> Dict[str, Any]:
        """Establish system baseline for anomaly detection"""
        baseline = {
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total,
            'disk_usage': {},
            'network_interfaces': [],
            'running_processes': set(),
            'listening_ports': set(),
            'startup_time': time.time()
        }
        
        # Disk usage baseline
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                baseline['disk_usage'][partition.mountpoint] = {
                    'total': usage.total,
                    'used': usage.used,
                    'free': usage.free
                }
            except PermissionError:
                continue
        
        # Network interfaces
        for interface, addrs in psutil.net_if_addrs().items():
            baseline['network_interfaces'].append({
                'interface': interface,
                'addresses': [addr.address for addr in addrs]
            })
        
        # Process baseline
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                baseline['running_processes'].add(proc.info['name'])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        # Network connections baseline
        for conn in psutil.net_connections(kind='inet'):
            if conn.status == psutil.CONN_LISTEN:
                baseline['listening_ports'].add(conn.laddr.port)
        
        return baseline

    def load_controls(self) -> None:
        """Load security controls from configuration"""
        default_controls = self._get_default_controls()
        
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                
            for control_data in config.get('controls', []):
                control = SecurityControl(**control_data)
                self.controls[control.control_id] = control
        else:
            # Use default controls
            for control in default_controls:
                self.controls[control.control_id] = control
            
            # Save default configuration
            self.save_controls()

    def _get_default_controls(self) -> List[SecurityControl]:
        """Get default security controls"""
        return [
            SecurityControl(
                control_id="SYS_001",
                name="System Resource Monitoring",
                description="Monitor CPU, memory, and disk usage for anomalies",
                category="system",
                severity=SecurityLevel.HIGH,
                status=ControlStatus.ENABLED,
                check_interval=60,
                remediation_actions=["alert_administrator", "resource_cleanup"]
            ),
            SecurityControl(
                control_id="NET_001", 
                name="Network Connection Monitoring",
                description="Monitor network connections for suspicious activity",
                category="network",
                severity=SecurityLevel.CRITICAL,
                status=ControlStatus.ENABLED,
                check_interval=30,
                remediation_actions=["block_connection", "alert_security_team"]
            ),
            SecurityControl(
                control_id="PROC_001",
                name="Process Integrity Monitoring",
                description="Monitor process creation and execution patterns",
                category="process",
                severity=SecurityLevel.HIGH,
                status=ControlStatus.ENABLED,
                check_interval=45,
                remediation_actions=["terminate_process", "quarantine_binary"]
            ),
            SecurityControl(
                control_id="FILE_001",
                name="File System Integrity",
                description="Monitor critical files and directories for changes",
                category="filesystem",
                severity=SecurityLevel.CRITICAL,
                status=ControlStatus.ENABLED,
                check_interval=300,
                remediation_actions=["restore_backup", "alert_administrator"]
            ),
            SecurityControl(
                control_id="AUTH_001",
                name="Authentication Monitoring",
                description="Monitor authentication attempts and failures",
                category="authentication",
                severity=SecurityLevel.CRITICAL,
                status=ControlStatus.ENABLED,
                check_interval=15,
                remediation_actions=["account_lockout", "rate_limiting"]
            ),
            SecurityControl(
                control_id="CRYPTO_001",
                name="Cryptographic Operations",
                description="Monitor cryptographic operations and key usage",
                category="cryptography",
                severity=SecurityLevel.HIGH,
                status=ControlStatus.ENABLED,
                check_interval=120,
                remediation_actions=["key_rotation", "audit_crypto_usage"]
            )
        ]

    def save_controls(self) -> None:
        """Save security controls to configuration file"""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        config = {
            'controls': [
                {
                    'control_id': control.control_id,
                    'name': control.name,
                    'description': control.description,
                    'category': control.category,
                    'severity': control.severity.value,
                    'status': control.status.value,
                    'check_interval': control.check_interval,
                    'enabled': control.enabled,
                    'remediation_actions': control.remediation_actions,
                    'metadata': control.metadata
                }
                for control in self.controls.values()
            ]
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)

    def start_monitoring(self) -> None:
        """Start all security control monitors"""
        self.running = True
        self.load_controls()
        
        for control in self.controls.values():
            if control.enabled and control.status == ControlStatus.ENABLED:
                monitor_thread = threading.Thread(
                    target=self._monitor_control,
                    args=(control,),
                    daemon=True
                )
                monitor_thread.start()
                self.monitors[control.control_id] = monitor_thread
        
        self.logger.info(f"Started monitoring {len(self.monitors)} security controls")

    def stop_monitoring(self) -> None:
        """Stop all security control monitors"""
        self.running = False
        
        # Wait for threads to finish
        for thread in self.monitors.values():
            thread.join(timeout=5)
        
        self.monitors.clear()
        self.logger.info("Stopped all security control monitors")

    def _monitor_control(self, control: SecurityControl) -> None:
        """Monitor a specific security control"""
        while self.running:
            try:
                # Update last check time
                control.last_check = datetime.now()
                
                # Execute control check based on category
                if control.category == "system":
                    self._check_system_resources(control)
                elif control.category == "network":
                    self._check_network_connections(control)
                elif control.category == "process":
                    self._check_process_integrity(control)
                elif control.category == "filesystem":
                    self._check_file_integrity(control)
                elif control.category == "authentication":
                    self._check_authentication_events(control)
                elif control.category == "cryptography":
                    self._check_crypto_operations(control)
                
                # Update control status in database
                self._update_control_status(control)
                
            except Exception as e:
                self.logger.error(f"Error monitoring control {control.control_id}: {e}")
                control.status = ControlStatus.FAILED
                
                # Create error event
                event = SecurityEvent(
                    event_id=f"ERR_{int(time.time())}_{control.control_id}",
                    timestamp=datetime.now(),
                    event_type="control_error",
                    severity=SecurityLevel.HIGH,
                    source=control.control_id,
                    description=f"Control monitoring failed: {str(e)}",
                    metadata={'control': control.name, 'error': str(e)}
                )
                self._handle_security_event(event)
            
            # Sleep until next check
            time.sleep(control.check_interval)

    def _check_system_resources(self, control: SecurityControl) -> None:
        """Check system resource usage"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # CPU threshold check
        if cpu_percent > 90:
            event = SecurityEvent(
                event_id=f"SYS_{int(time.time())}_CPU",
                timestamp=datetime.now(),
                event_type="resource_threshold",
                severity=SecurityLevel.HIGH,
                source="system_monitor",
                description=f"High CPU usage detected: {cpu_percent}%",
                metadata={'cpu_percent': cpu_percent, 'threshold': 90}
            )
            self._handle_security_event(event)
        
        # Memory threshold check
        if memory.percent > 95:
            event = SecurityEvent(
                event_id=f"SYS_{int(time.time())}_MEM",
                timestamp=datetime.now(),
                event_type="resource_threshold",
                severity=SecurityLevel.HIGH,
                source="system_monitor",
                description=f"High memory usage detected: {memory.percent}%",
                metadata={'memory_percent': memory.percent, 'threshold': 95}
            )
            self._handle_security_event(event)

    def _check_network_connections(self, control: SecurityControl) -> None:
        """Check network connections for anomalies"""
        current_connections = set()
        suspicious_connections = []
        
        for conn in psutil.net_connections(kind='inet'):
            if conn.raddr:
                current_connections.add((conn.raddr.ip, conn.raddr.port))
                
                # Check for suspicious ports
                if conn.raddr.port in [4444, 5555, 6666, 7777, 8888, 9999]:
                    suspicious_connections.append(conn)
        
        # Alert on suspicious connections
        for conn in suspicious_connections:
            event = SecurityEvent(
                event_id=f"NET_{int(time.time())}_{conn.raddr.ip}_{conn.raddr.port}",
                timestamp=datetime.now(),
                event_type="suspicious_connection",
                severity=SecurityLevel.CRITICAL,
                source="network_monitor",
                description=f"Suspicious connection to {conn.raddr.ip}:{conn.raddr.port}",
                metadata={'connection': {'ip': conn.raddr.ip, 'port': conn.raddr.port, 'status': conn.status}}
            )
            self._handle_security_event(event)

    def _check_process_integrity(self, control: SecurityControl) -> None:
        """Check process integrity and detect anomalies"""
        current_processes = set()
        new_processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
            try:
                process_name = proc.info['name']
                current_processes.add(process_name)
                
                # Check for new processes not in baseline
                if process_name not in self.system_baseline['running_processes']:
                    new_processes.append(proc.info)
                
                # Check for suspicious process names
                suspicious_names = ['nc', 'netcat', 'nmap', 'john', 'hashcat', 'metasploit']
                if any(name in process_name.lower() for name in suspicious_names):
                    event = SecurityEvent(
                        event_id=f"PROC_{int(time.time())}_{proc.info['pid']}",
                        timestamp=datetime.now(),
                        event_type="suspicious_process",
                        severity=SecurityLevel.HIGH,
                        source="process_monitor",
                        description=f"Suspicious process detected: {process_name}",
                        metadata={'process': proc.info}
                    )
                    self._handle_security_event(event)
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        # Alert on significant number of new processes
        if len(new_processes) > 10:
            event = SecurityEvent(
                event_id=f"PROC_{int(time.time())}_MASS_SPAWN",
                timestamp=datetime.now(),
                event_type="mass_process_creation",
                severity=SecurityLevel.HIGH,
                source="process_monitor",
                description=f"{len(new_processes)} new processes detected",
                metadata={'new_process_count': len(new_processes), 'processes': new_processes[:5]}
            )
            self._handle_security_event(event)

    def _check_file_integrity(self, control: SecurityControl) -> None:
        """Check file system integrity"""
        critical_files = [
            '/etc/passwd',
            '/etc/shadow',
            '/etc/sudoers',
            '/etc/hosts',
            '/etc/ssh/sshd_config'
        ]
        
        for file_path in critical_files:
            if os.path.exists(file_path):
                try:
                    # Calculate file hash
                    with open(file_path, 'rb') as f:
                        file_hash = hashlib.sha256(f.read()).hexdigest()
                    
                    # Check against stored hash
                    stored_hash_key = f"file_hash_{file_path.replace('/', '_')}"
                    stored_hash = control.metadata.get(stored_hash_key)
                    
                    if stored_hash is None:
                        # First time - store hash
                        control.metadata[stored_hash_key] = file_hash
                    elif stored_hash != file_hash:
                        # File has been modified
                        event = SecurityEvent(
                            event_id=f"FILE_{int(time.time())}_{hashlib.md5(file_path.encode()).hexdigest()[:8]}",
                            timestamp=datetime.now(),
                            event_type="file_modification",
                            severity=SecurityLevel.CRITICAL,
                            source="file_monitor",
                            description=f"Critical file modified: {file_path}",
                            metadata={'file_path': file_path, 'old_hash': stored_hash, 'new_hash': file_hash}
                        )
                        self._handle_security_event(event)
                        
                        # Update stored hash
                        control.metadata[stored_hash_key] = file_hash
                        
                except Exception as e:
                    self.logger.error(f"Error checking file integrity for {file_path}: {e}")

    def _check_authentication_events(self, control: SecurityControl) -> None:
        """Check authentication events (placeholder for system log integration)"""
        # This would integrate with system logs, PAM, etc.
        # For now, we'll simulate some basic checks
        
        # Check for multiple failed login attempts (would read from auth logs)
        failed_attempts = control.metadata.get('failed_attempts', 0)
        
        # Simulate detection of failed attempts
        if failed_attempts > 5:
            event = SecurityEvent(
                event_id=f"AUTH_{int(time.time())}_BRUTE_FORCE",
                timestamp=datetime.now(),
                event_type="authentication_failure",
                severity=SecurityLevel.CRITICAL,
                source="auth_monitor",
                description=f"Multiple authentication failures detected: {failed_attempts}",
                metadata={'failed_attempts': failed_attempts, 'threshold': 5}
            )
            self._handle_security_event(event)
            
            # Reset counter after alerting
            control.metadata['failed_attempts'] = 0

    def _check_crypto_operations(self, control: SecurityControl) -> None:
        """Check cryptographic operations"""
        # Monitor crypto library usage, key operations, etc.
        # This is a simplified implementation
        
        crypto_events = control.metadata.get('crypto_events', [])
        
        # Check for unusual crypto activity patterns
        if len(crypto_events) > 100:  # Threshold for high crypto activity
            event = SecurityEvent(
                event_id=f"CRYPTO_{int(time.time())}_HIGH_ACTIVITY",
                timestamp=datetime.now(),
                event_type="crypto_anomaly",
                severity=SecurityLevel.MEDIUM,
                source="crypto_monitor",
                description="High cryptographic activity detected",
                metadata={'event_count': len(crypto_events), 'threshold': 100}
            )
            self._handle_security_event(event)
            
            # Clear old events
            control.metadata['crypto_events'] = []

    def _handle_security_event(self, event: SecurityEvent) -> None:
        """Handle a security event"""
        with self.lock:
            self.events.append(event)
            
            # Log to database
            self._log_event_to_database(event)
            
            # Execute alert handlers
            for handler in self.alert_handlers:
                try:
                    handler(event)
                except Exception as e:
                    self.logger.error(f"Error in alert handler: {e}")
            
            # Log event
            self.logger.warning(f"Security Event: {event.event_type} - {event.description}")

    def _log_event_to_database(self, event: SecurityEvent) -> None:
        """Log security event to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO security_events 
            (event_id, timestamp, event_type, severity, source, description, metadata, resolved, resolution_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            event.event_id,
            event.timestamp.isoformat(),
            event.event_type,
            event.severity.value,
            event.source,
            event.description,
            json.dumps(event.metadata),
            event.resolved,
            event.resolution_time.isoformat() if event.resolution_time else None
        ))
        
        conn.commit()
        conn.close()

    def _update_control_status(self, control: SecurityControl) -> None:
        """Update control status in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO control_status 
            (control_id, name, status, last_check, metadata)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            control.control_id,
            control.name,
            control.status.value,
            control.last_check.isoformat() if control.last_check else None,
            json.dumps(control.metadata)
        ))
        
        conn.commit()
        conn.close()

    def add_alert_handler(self, handler: Callable[[SecurityEvent], None]) -> None:
        """Add an alert handler function"""
        self.alert_handlers.append(handler)

    def get_events(self, hours: int = 24, severity: Optional[SecurityLevel] = None) -> List[SecurityEvent]:
        """Get security events from the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        filtered_events = [
            event for event in self.events
            if event.timestamp >= cutoff_time
        ]
        
        if severity:
            filtered_events = [
                event for event in filtered_events
                if event.severity == severity
            ]
        
        return sorted(filtered_events, key=lambda x: x.timestamp, reverse=True)

    def get_control_status(self) -> Dict[str, Dict[str, Any]]:
        """Get current status of all controls"""
        status = {}
        
        for control in self.controls.values():
            status[control.control_id] = {
                'name': control.name,
                'category': control.category,
                'status': control.status.value,
                'severity': control.severity.value,
                'last_check': control.last_check.isoformat() if control.last_check else None,
                'enabled': control.enabled
            }
        
        return status

    def resolve_event(self, event_id: str, resolution_note: str = "") -> bool:
        """Mark a security event as resolved"""
        for event in self.events:
            if event.event_id == event_id:
                event.resolved = True
                event.resolution_time = datetime.now()
                event.metadata['resolution_note'] = resolution_note
                
                # Update database
                self._log_event_to_database(event)
                
                self.logger.info(f"Resolved security event {event_id}: {resolution_note}")
                return True
        
        return False

    def generate_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report"""
        recent_events = self.get_events(hours=24)
        control_status = self.get_control_status()
        
        # Event statistics
        event_stats = defaultdict(int)
        severity_stats = defaultdict(int)
        
        for event in recent_events:
            event_stats[event.event_type] += 1
            severity_stats[event.severity.value] += 1
        
        # Control health
        control_health = {
            'total': len(self.controls),
            'enabled': len([c for c in self.controls.values() if c.enabled]),
            'failed': len([c for c in self.controls.values() if c.status == ControlStatus.FAILED])
        }
        
        return {
            'report_timestamp': datetime.now().isoformat(),
            'event_summary': {
                'total_events_24h': len(recent_events),
                'by_type': dict(event_stats),
                'by_severity': dict(severity_stats),
                'unresolved': len([e for e in recent_events if not e.resolved])
            },
            'control_health': control_health,
            'control_status': control_status,
            'system_baseline': self.system_baseline,
            'recommendations': self._generate_recommendations(recent_events)
        }

    def _generate_recommendations(self, events: List[SecurityEvent]) -> List[str]:
        """Generate security recommendations based on events"""
        recommendations = []
        
        # High CPU/Memory usage
        resource_events = [e for e in events if e.event_type == "resource_threshold"]
        if len(resource_events) > 5:
            recommendations.append("Consider implementing resource limits and monitoring")
        
        # Suspicious connections
        network_events = [e for e in events if e.event_type == "suspicious_connection"]
        if network_events:
            recommendations.append("Review firewall rules and network access controls")
        
        # Authentication failures
        auth_events = [e for e in events if e.event_type == "authentication_failure"]
        if auth_events:
            recommendations.append("Implement account lockout policies and MFA")
        
        # File modifications
        file_events = [e for e in events if e.event_type == "file_modification"]
        if file_events:
            recommendations.append("Review file integrity monitoring and backup policies")
        
        return recommendations


# Example usage and testing
if __name__ == "__main__":
    # Create security monitor
    monitor = SecurityControlsMonitor()
    
    # Add sample alert handler
    def console_alert_handler(event: SecurityEvent):
        print(f"SECURITY ALERT: {event.event_type} - {event.description}")
    
    monitor.add_alert_handler(console_alert_handler)
    
    # Start monitoring
    print("Starting security monitoring...")
    monitor.start_monitoring()
    
    try:
        # Run for demo period
        time.sleep(30)
        
        # Generate report
        report = monitor.generate_security_report()
        print(f"\nSecurity Report Generated:")
        print(f"Total Events (24h): {report['event_summary']['total_events_24h']}")
        print(f"Control Health: {report['control_health']}")
        
    except KeyboardInterrupt:
        print("\nShutting down security monitoring...")
    finally:
        monitor.stop_monitoring()