#!/usr/bin/env python3
"""
Legacy Systems Simulation Module

This module simulates legacy systems with outdated protocols, vulnerabilities,
and realistic failure modes commonly found in enterprise environments.
"""

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import threading
import socket
import struct

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LegacyProtocol(Enum):
    """Legacy communication protocols"""
    TELNET = "telnet"
    FTP = "ftp"
    RLOGIN = "rlogin"
    RSH = "rsh"
    TFTP = "tftp"
    SNMP_V1 = "snmp_v1"
    SNMP_V2C = "snmp_v2c"
    HTTP_BASIC = "http_basic"
    CLEAR_TEXT_LDAP = "clear_text_ldap"
    POP3 = "pop3"
    IMAP = "imap"
    SMTP = "smtp"
    MODBUS_TCP = "modbus_tcp"
    DNP3 = "dnp3"
    IEC_61850 = "iec_61850"
    BACNET = "bacnet"
    X25 = "x25"
    SNA = "sna"
    DECNET = "decnet"


class LegacySystemType(Enum):
    """Types of legacy systems"""
    MAINFRAME = "mainframe"
    MINICOMPUTER = "minicomputer"
    UNIX_WORKSTATION = "unix_workstation"
    DOS_SYSTEM = "dos_system"
    WINDOWS_NT = "windows_nt"
    WINDOWS_2000 = "windows_2000"
    WINDOWS_XP = "windows_xp"
    SCADA_HMI = "scada_hmi"
    PLC = "plc"
    DCS = "dcs"
    RTU = "rtu"
    LEGACY_DATABASE = "legacy_database"
    TERMINAL_SERVER = "terminal_server"
    PRINT_SERVER = "print_server"
    FILE_SERVER = "file_server"


class VulnerabilityCategory(Enum):
    """Categories of vulnerabilities in legacy systems"""
    AUTHENTICATION_BYPASS = "authentication_bypass"
    BUFFER_OVERFLOW = "buffer_overflow"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    REMOTE_CODE_EXECUTION = "remote_code_execution"
    INFORMATION_DISCLOSURE = "information_disclosure"
    DENIAL_OF_SERVICE = "denial_of_service"
    WEAK_CRYPTOGRAPHY = "weak_cryptography"
    DEFAULT_CREDENTIALS = "default_credentials"
    UNPATCHED_VULNERABILITY = "unpatched_vulnerability"
    PROTOCOL_WEAKNESS = "protocol_weakness"


@dataclass
class LegacyVulnerability:
    """Represents a vulnerability in a legacy system"""
    cve_id: str
    category: VulnerabilityCategory
    severity: str  # Critical, High, Medium, Low
    description: str
    exploit_available: bool = False
    patch_available: bool = False
    workaround_available: bool = False
    first_discovered: datetime = field(default_factory=datetime.now)
    
    def get_cvss_score(self) -> float:
        """Get CVSS score based on severity"""
        severity_scores = {
            "Critical": random.uniform(9.0, 10.0),
            "High": random.uniform(7.0, 8.9),
            "Medium": random.uniform(4.0, 6.9),
            "Low": random.uniform(0.1, 3.9)
        }
        return severity_scores.get(self.severity, 5.0)


@dataclass
class LegacyService:
    """Represents a service running on a legacy system"""
    service_name: str
    protocol: LegacyProtocol
    port: int
    version: str
    banner: str
    is_encrypted: bool = False
    authentication_required: bool = True
    default_credentials: Optional[Tuple[str, str]] = None
    vulnerabilities: List[LegacyVulnerability] = field(default_factory=list)
    uptime: timedelta = field(default_factory=lambda: timedelta(days=random.randint(1, 3650)))
    failure_rate: float = 0.05  # 5% chance of failure
    last_restart: datetime = field(default_factory=datetime.now)
    
    def is_vulnerable(self) -> bool:
        """Check if service has vulnerabilities"""
        return len(self.vulnerabilities) > 0
    
    def simulate_service_failure(self) -> bool:
        """Simulate service failure"""
        return random.random() < self.failure_rate
    
    def get_authentication_methods(self) -> List[str]:
        """Get available authentication methods"""
        if not self.authentication_required:
            return ["anonymous"]
        
        methods = []
        if self.default_credentials:
            methods.append("default_credentials")
        
        # Legacy systems often have weak auth
        if self.protocol in [LegacyProtocol.TELNET, LegacyProtocol.FTP, LegacyProtocol.RLOGIN]:
            methods.extend(["cleartext_password", "challenge_response"])
        elif self.protocol == LegacyProtocol.SNMP_V1:
            methods.append("community_string")
        elif self.protocol == LegacyProtocol.HTTP_BASIC:
            methods.append("basic_auth")
        
        return methods if methods else ["password"]


@dataclass
class LegacySystem:
    """Represents a legacy system"""
    system_id: str
    system_type: LegacySystemType
    hostname: str
    ip_address: str
    mac_address: str
    os_name: str
    os_version: str
    architecture: str  # x86, x64, SPARC, PowerPC, etc.
    manufacturer: str
    model: str
    install_date: datetime
    last_patch_date: Optional[datetime] = None
    services: List[LegacyService] = field(default_factory=list)
    is_online: bool = True
    is_critical: bool = False
    business_function: str = "unknown"
    maintenance_window: Optional[str] = None
    eol_date: Optional[datetime] = None  # End of Life date
    support_status: str = "unsupported"
    network_segment: str = "legacy"
    
    def add_service(self, service: LegacyService):
        """Add a service to the system"""
        self.services.append(service)
    
    def get_open_ports(self) -> List[int]:
        """Get list of open ports"""
        return [service.port for service in self.services if self.is_online]
    
    def has_critical_vulnerabilities(self) -> bool:
        """Check if system has critical vulnerabilities"""
        for service in self.services:
            for vuln in service.vulnerabilities:
                if vuln.severity == "Critical":
                    return True
        return False
    
    def get_vulnerability_count(self) -> Dict[str, int]:
        """Get count of vulnerabilities by severity"""
        counts = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
        
        for service in self.services:
            for vuln in service.vulnerabilities:
                counts[vuln.severity] += 1
        
        return counts
    
    def calculate_risk_score(self) -> float:
        """Calculate overall risk score for the system"""
        base_score = 0.0
        
        # Age factor
        age_years = (datetime.now() - self.install_date).days / 365
        age_factor = min(age_years / 10, 1.0)  # Max factor of 1.0 for 10+ years
        
        # Patch status factor
        if self.last_patch_date:
            days_since_patch = (datetime.now() - self.last_patch_date).days
            patch_factor = min(days_since_patch / 365, 1.0)  # Max factor for 1+ year
        else:
            patch_factor = 1.0  # Never patched
        
        # Vulnerability factor
        vuln_counts = self.get_vulnerability_count()
        vuln_factor = (
            vuln_counts["Critical"] * 1.0 +
            vuln_counts["High"] * 0.7 +
            vuln_counts["Medium"] * 0.4 +
            vuln_counts["Low"] * 0.1
        ) / 10  # Normalize
        
        # Criticality factor
        criticality_factor = 1.0 if self.is_critical else 0.5
        
        # Support status factor
        support_factor = 1.0 if self.support_status == "unsupported" else 0.3
        
        # Calculate final score (0-10 scale)
        risk_score = (
            age_factor * 2 +
            patch_factor * 2 +
            vuln_factor * 3 +
            criticality_factor * 2 +
            support_factor * 1
        )
        
        return min(risk_score, 10.0)


class LegacySystemFactory:
    """Factory for creating various legacy systems"""
    
    @staticmethod
    def create_mainframe(system_id: str, ip_address: str) -> LegacySystem:
        """Create a mainframe system"""
        install_date = datetime.now() - timedelta(days=random.randint(3650, 14600))  # 10-40 years old
        
        system = LegacySystem(
            system_id=system_id,
            system_type=LegacySystemType.MAINFRAME,
            hostname=f"MAINFRAME-{system_id.upper()}",
            ip_address=ip_address,
            mac_address=LegacySystemFactory._generate_mac(),
            os_name="z/OS",
            os_version=f"{random.randint(1, 2)}.{random.randint(1, 4)}",
            architecture="System z",
            manufacturer="IBM",
            model=f"System z{random.randint(9, 15)}",
            install_date=install_date,
            last_patch_date=install_date + timedelta(days=random.randint(365, 1825)),
            is_critical=True,
            business_function="Core Business Processing",
            maintenance_window="Sunday 02:00-06:00",
            support_status="limited_support",
            eol_date=datetime.now() + timedelta(days=random.randint(365, 1825))
        )
        
        # Add typical mainframe services
        services = [
            LegacyService(
                service_name="TSO/ISPF",
                protocol=LegacyProtocol.TELNET,
                port=23,
                version="2.3",
                banner="z/OS TSO/E LOGON",
                default_credentials=("IBMUSER", "SYS1"),
                vulnerabilities=[
                    LegacyVulnerability(
                        cve_id="CVE-2019-4102",
                        category=VulnerabilityCategory.AUTHENTICATION_BYPASS,
                        severity="High",
                        description="TSO authentication bypass vulnerability"
                    )
                ]
            ),
            LegacyService(
                service_name="FTP Server",
                protocol=LegacyProtocol.FTP,
                port=21,
                version="1.13",
                banner="z/OS FTP Server Ready",
                default_credentials=("anonymous", "guest@mainframe.com"),
                vulnerabilities=[
                    LegacyVulnerability(
                        cve_id="CVE-2018-1234",
                        category=VulnerabilityCategory.BUFFER_OVERFLOW,
                        severity="Critical",
                        description="FTP server buffer overflow"
                    )
                ]
            )
        ]
        
        for service in services:
            system.add_service(service)
        
        return system
    
    @staticmethod
    def create_scada_hmi(system_id: str, ip_address: str) -> LegacySystem:
        """Create a SCADA HMI system"""
        install_date = datetime.now() - timedelta(days=random.randint(1825, 7300))  # 5-20 years old
        
        system = LegacySystem(
            system_id=system_id,
            system_type=LegacySystemType.SCADA_HMI,
            hostname=f"SCADA-HMI-{system_id.upper()}",
            ip_address=ip_address,
            mac_address=LegacySystemFactory._generate_mac(),
            os_name="Windows XP Embedded",
            os_version="SP3",
            architecture="x86",
            manufacturer=random.choice(["Wonderware", "GE", "Siemens", "ABB"]),
            model=f"HMI-{random.randint(1000, 9999)}",
            install_date=install_date,
            last_patch_date=None,  # Often never patched
            is_critical=True,
            business_function="Industrial Process Control",
            maintenance_window="Scheduled Shutdown Only",
            support_status="unsupported",
            eol_date=datetime.now() - timedelta(days=random.randint(365, 3650))  # Already EOL
        )
        
        # Add SCADA services
        services = [
            LegacyService(
                service_name="Modbus TCP",
                protocol=LegacyProtocol.MODBUS_TCP,
                port=502,
                version="1.0",
                banner="Modbus TCP Server",
                is_encrypted=False,
                authentication_required=False,
                vulnerabilities=[
                    LegacyVulnerability(
                        cve_id="CVE-2020-12345",
                        category=VulnerabilityCategory.PROTOCOL_WEAKNESS,
                        severity="High",
                        description="Modbus protocol lacks authentication"
                    )
                ]
            ),
            LegacyService(
                service_name="VNC Server",
                protocol=LegacyProtocol.HTTP_BASIC,
                port=5900,
                version="3.3.7",
                banner="RealVNC Server",
                default_credentials=("admin", "password"),
                vulnerabilities=[
                    LegacyVulnerability(
                        cve_id="CVE-2019-15681",
                        category=VulnerabilityCategory.AUTHENTICATION_BYPASS,
                        severity="Critical",
                        description="VNC authentication bypass"
                    )
                ]
            ),
            LegacyService(
                service_name="Web Interface",
                protocol=LegacyProtocol.HTTP_BASIC,
                port=80,
                version="IIS 5.1",
                banner="Microsoft-IIS/5.1",
                default_credentials=("admin", "admin"),
                vulnerabilities=[
                    LegacyVulnerability(
                        cve_id="CVE-2017-7269",
                        category=VulnerabilityCategory.BUFFER_OVERFLOW,
                        severity="Critical",
                        description="IIS 6.0 WebDAV buffer overflow"
                    )
                ]
            )
        ]
        
        for service in services:
            system.add_service(service)
        
        return system
    
    @staticmethod
    def create_legacy_plc(system_id: str, ip_address: str) -> LegacySystem:
        """Create a legacy PLC system"""
        install_date = datetime.now() - timedelta(days=random.randint(2555, 10950))  # 7-30 years old
        
        system = LegacySystem(
            system_id=system_id,
            system_type=LegacySystemType.PLC,
            hostname=f"PLC-{system_id.upper()}",
            ip_address=ip_address,
            mac_address=LegacySystemFactory._generate_mac(),
            os_name="Proprietary RTOS",
            os_version="1.2.3",
            architecture="Proprietary",
            manufacturer=random.choice(["Allen-Bradley", "Siemens", "Schneider", "Mitsubishi"]),
            model=f"PLC-{random.randint(1000, 9999)}",
            install_date=install_date,
            last_patch_date=None,  # PLCs rarely get firmware updates
            is_critical=True,
            business_function="Industrial Automation",
            maintenance_window="Production Shutdown Only",
            support_status="unsupported",
            eol_date=datetime.now() - timedelta(days=random.randint(0, 1825))
        )
        
        # Add PLC services
        services = [
            LegacyService(
                service_name="Modbus TCP",
                protocol=LegacyProtocol.MODBUS_TCP,
                port=502,
                version="1.0",
                banner="Modbus TCP Slave",
                is_encrypted=False,
                authentication_required=False,
                failure_rate=0.02,  # PLCs are generally reliable
                vulnerabilities=[
                    LegacyVulnerability(
                        cve_id="CVE-2021-22681",
                        category=VulnerabilityCategory.DENIAL_OF_SERVICE,
                        severity="Medium",
                        description="Modbus DoS vulnerability"
                    )
                ]
            ),
            LegacyService(
                service_name="Ethernet/IP",
                protocol=LegacyProtocol.HTTP_BASIC,
                port=44818,
                version="1.0",
                banner="EtherNet/IP Device",
                is_encrypted=False,
                authentication_required=False,
                vulnerabilities=[
                    LegacyVulnerability(
                        cve_id="CVE-2020-6998",
                        category=VulnerabilityCategory.REMOTE_CODE_EXECUTION,
                        severity="Critical",
                        description="EtherNet/IP stack buffer overflow"
                    )
                ]
            )
        ]
        
        for service in services:
            system.add_service(service)
        
        return system
    
    @staticmethod
    def create_windows_xp_system(system_id: str, ip_address: str) -> LegacySystem:
        """Create a Windows XP system"""
        install_date = datetime.now() - timedelta(days=random.randint(2555, 7300))  # 7-20 years old
        
        system = LegacySystem(
            system_id=system_id,
            system_type=LegacySystemType.WINDOWS_XP,
            hostname=f"WS-XP-{system_id.upper()}",
            ip_address=ip_address,
            mac_address=LegacySystemFactory._generate_mac(),
            os_name="Windows XP Professional",
            os_version="SP3",
            architecture="x86",
            manufacturer=random.choice(["Dell", "HP", "IBM", "Compaq"]),
            model=f"OptiPlex-{random.randint(100, 999)}",
            install_date=install_date,
            last_patch_date=datetime(2014, 4, 8),  # XP support ended
            is_critical=False,
            business_function="Legacy Application Support",
            maintenance_window="Weekends",
            support_status="unsupported",
            eol_date=datetime(2014, 4, 8)
        )
        
        # Add Windows XP services
        services = [
            LegacyService(
                service_name="SMB",
                protocol=LegacyProtocol.HTTP_BASIC,
                port=445,
                version="SMBv1",
                banner="Windows XP SMB Server",
                vulnerabilities=[
                    LegacyVulnerability(
                        cve_id="CVE-2017-0144",
                        category=VulnerabilityCategory.REMOTE_CODE_EXECUTION,
                        severity="Critical",
                        description="EternalBlue SMBv1 vulnerability",
                        exploit_available=True
                    ),
                    LegacyVulnerability(
                        cve_id="CVE-2008-4250",
                        category=VulnerabilityCategory.REMOTE_CODE_EXECUTION,
                        severity="Critical",
                        description="MS08-067 Server service vulnerability",
                        exploit_available=True
                    )
                ]
            ),
            LegacyService(
                service_name="RPC Endpoint Mapper",
                protocol=LegacyProtocol.HTTP_BASIC,
                port=135,
                version="5.1",
                banner="Windows RPC Endpoint Mapper",
                vulnerabilities=[
                    LegacyVulnerability(
                        cve_id="CVE-2003-0352",
                        category=VulnerabilityCategory.BUFFER_OVERFLOW,
                        severity="Critical",
                        description="RPC DCOM vulnerability"
                    )
                ]
            ),
            LegacyService(
                service_name="Telnet Server",
                protocol=LegacyProtocol.TELNET,
                port=23,
                version="5.1",
                banner="Microsoft Telnet Server",
                default_credentials=("administrator", "password"),
                vulnerabilities=[
                    LegacyVulnerability(
                        cve_id="CVE-2002-0020",
                        category=VulnerabilityCategory.PRIVILEGE_ESCALATION,
                        severity="High",
                        description="Telnet server privilege escalation"
                    )
                ]
            )
        ]
        
        for service in services:
            system.add_service(service)
        
        return system
    
    @staticmethod
    def create_legacy_unix_system(system_id: str, ip_address: str) -> LegacySystem:
        """Create a legacy UNIX system"""
        install_date = datetime.now() - timedelta(days=random.randint(3650, 12775))  # 10-35 years old
        
        unix_variants = [
            ("Solaris", "2.6", "SPARC", "Sun Microsystems"),
            ("AIX", "4.3", "PowerPC", "IBM"),
            ("HP-UX", "10.20", "PA-RISC", "Hewlett-Packard"),
            ("IRIX", "6.5", "MIPS", "Silicon Graphics")
        ]
        
        os_name, os_version, arch, manufacturer = random.choice(unix_variants)
        
        system = LegacySystem(
            system_id=system_id,
            system_type=LegacySystemType.UNIX_WORKSTATION,
            hostname=f"UNIX-{system_id.upper()}",
            ip_address=ip_address,
            mac_address=LegacySystemFactory._generate_mac(),
            os_name=os_name,
            os_version=os_version,
            architecture=arch,
            manufacturer=manufacturer,
            model=f"WS-{random.randint(1000, 9999)}",
            install_date=install_date,
            last_patch_date=install_date + timedelta(days=random.randint(365, 2555)),
            is_critical=True,
            business_function="Legacy Database Server",
            maintenance_window="Monthly",
            support_status="unsupported",
            eol_date=datetime.now() - timedelta(days=random.randint(1825, 7300))
        )
        
        # Add UNIX services
        services = [
            LegacyService(
                service_name="rlogin",
                protocol=LegacyProtocol.RLOGIN,
                port=513,
                version="1.0",
                banner="rlogind",
                is_encrypted=False,
                vulnerabilities=[
                    LegacyVulnerability(
                        cve_id="CVE-1999-0651",
                        category=VulnerabilityCategory.AUTHENTICATION_BYPASS,
                        severity="High",
                        description="rlogin authentication bypass"
                    )
                ]
            ),
            LegacyService(
                service_name="rsh",
                protocol=LegacyProtocol.RSH,
                port=514,
                version="1.0",
                banner="rshd",
                is_encrypted=False,
                authentication_required=False,
                vulnerabilities=[
                    LegacyVulnerability(
                        cve_id="CVE-1999-0183",
                        category=VulnerabilityCategory.REMOTE_CODE_EXECUTION,
                        severity="Critical",
                        description="rsh remote command execution"
                    )
                ]
            ),
            LegacyService(
                service_name="FTP",
                protocol=LegacyProtocol.FTP,
                port=21,
                version="wu-ftpd 2.6.0",
                banner="wu-ftpd 2.6.0 ready",
                default_credentials=("ftp", "ftp@domain.com"),
                vulnerabilities=[
                    LegacyVulnerability(
                        cve_id="CVE-2000-0573",
                        category=VulnerabilityCategory.BUFFER_OVERFLOW,
                        severity="Critical",
                        description="wu-ftpd format string vulnerability"
                    )
                ]
            )
        ]
        
        for service in services:
            system.add_service(service)
        
        return system
    
    @staticmethod
    def _generate_mac() -> str:
        """Generate a random MAC address"""
        return ":".join([f"{random.randint(0, 255):02x}" for _ in range(6)])


class LegacyProtocolSimulator:
    """Simulates legacy protocol communications"""
    
    def __init__(self):
        self.active_connections: Dict[str, List[Dict]] = {}
    
    def simulate_telnet_session(self, system: LegacySystem) -> Dict[str, Any]:
        """Simulate a Telnet session"""
        session = {
            "protocol": "TELNET",
            "target": system.hostname,
            "port": 23,
            "timestamp": datetime.now().isoformat(),
            "session_id": f"telnet_{random.randint(10000, 99999)}",
            "commands": [],
            "authentication": "cleartext",
            "encrypted": False
        }
        
        # Simulate common commands
        commands = [
            "login", "ls -la", "ps -ef", "netstat -an", "cat /etc/passwd",
            "uname -a", "whoami", "id", "history", "logout"
        ]
        
        for _ in range(random.randint(3, 8)):
            cmd = random.choice(commands)
            session["commands"].append({
                "command": cmd,
                "timestamp": datetime.now().isoformat(),
                "response_size": random.randint(100, 2000)
            })
        
        return session
    
    def simulate_ftp_session(self, system: LegacySystem) -> Dict[str, Any]:
        """Simulate an FTP session"""
        session = {
            "protocol": "FTP",
            "target": system.hostname,
            "port": 21,
            "timestamp": datetime.now().isoformat(),
            "session_id": f"ftp_{random.randint(10000, 99999)}",
            "operations": [],
            "authentication": "cleartext",
            "encrypted": False,
            "passive_mode": random.choice([True, False])
        }
        
        # Simulate FTP operations
        operations = [
            "USER anonymous", "PASS guest@domain.com", "PWD", "LIST",
            "CWD /pub", "RETR file.txt", "STOR upload.txt", "QUIT"
        ]
        
        for _ in range(random.randint(4, 10)):
            op = random.choice(operations)
            session["operations"].append({
                "operation": op,
                "timestamp": datetime.now().isoformat(),
                "response_code": random.choice([200, 220, 230, 250, 425, 500]),
                "data_size": random.randint(0, 10000) if "RETR" in op or "STOR" in op else 0
            })
        
        return session
    
    def simulate_snmp_query(self, system: LegacySystem) -> Dict[str, Any]:
        """Simulate an SNMP query"""
        query = {
            "protocol": "SNMP",
            "version": "v1" if random.choice([True, False]) else "v2c",
            "target": system.hostname,
            "port": 161,
            "timestamp": datetime.now().isoformat(),
            "community_string": random.choice(["public", "private", "admin"]),
            "operation": random.choice(["GET", "GETNEXT", "GETBULK", "SET"]),
            "oid": f"1.3.6.1.2.1.{random.randint(1, 25)}.{random.randint(1, 10)}",
            "encrypted": False
        }
        
        return query
    
    def simulate_modbus_communication(self, system: LegacySystem) -> Dict[str, Any]:
        """Simulate Modbus communication"""
        communication = {
            "protocol": "MODBUS_TCP",
            "target": system.hostname,
            "port": 502,
            "timestamp": datetime.now().isoformat(),
            "transaction_id": random.randint(1, 65535),
            "unit_id": random.randint(1, 247),
            "function_code": random.choice([1, 2, 3, 4, 5, 6, 15, 16]),
            "register_address": random.randint(0, 65535),
            "register_count": random.randint(1, 125),
            "encrypted": False,
            "authenticated": False
        }
        
        return communication


class LegacySystemManager:
    """Manages legacy systems simulation"""
    
    def __init__(self):
        self.systems: Dict[str, LegacySystem] = {}
        self.protocol_simulator = LegacyProtocolSimulator()
        self.simulation_running = False
        self.simulation_thread = None
        
    def add_system(self, system: LegacySystem):
        """Add a legacy system"""
        self.systems[system.system_id] = system
        logger.info(f"Added legacy system: {system.hostname} ({system.os_name})")
    
    def populate_legacy_network(self, base_ip: str = "192.168.40.", count: int = 10):
        """Populate the legacy network with various systems"""
        system_creators = [
            LegacySystemFactory.create_mainframe,
            LegacySystemFactory.create_scada_hmi,
            LegacySystemFactory.create_legacy_plc,
            LegacySystemFactory.create_windows_xp_system,
            LegacySystemFactory.create_legacy_unix_system
        ]
        
        for i in range(count):
            system_id = f"legacy_{i:03d}"
            ip_address = f"{base_ip}{i + 10}"
            creator = random.choice(system_creators)
            system = creator(system_id, ip_address)
            self.add_system(system)
    
    def start_simulation(self):
        """Start the legacy systems simulation"""
        if self.simulation_running:
            logger.warning("Simulation already running")
            return
        
        logger.info("Starting legacy systems simulation")
        self.simulation_running = True
        
        # Start background simulation thread
        self.simulation_thread = threading.Thread(target=self._simulation_loop)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
        
        logger.info("Legacy systems simulation started successfully")
    
    def stop_simulation(self):
        """Stop the simulation"""
        logger.info("Stopping legacy systems simulation")
        self.simulation_running = False
        
        if self.simulation_thread:
            self.simulation_thread.join(timeout=5)
    
    def _simulation_loop(self):
        """Main simulation loop"""
        while self.simulation_running:
            try:
                # Simulate system activities
                self._simulate_system_activities()
                
                # Simulate service failures
                self._simulate_service_failures()
                
                # Simulate protocol communications
                self._simulate_protocol_communications()
                
                # Sleep for simulation interval
                time.sleep(15)
                
            except Exception as e:
                logger.error(f"Error in simulation loop: {e}")
    
    def _simulate_system_activities(self):
        """Simulate various system activities"""
        for system in self.systems.values():
            if not system.is_online:
                # Chance for system to come back online
                if random.random() < 0.1:  # 10% chance
                    system.is_online = True
                    logger.info(f"Legacy system {system.hostname} came back online")
                continue
            
            # Simulate system going offline (rare for critical systems)
            offline_probability = 0.001 if system.is_critical else 0.005
            if random.random() < offline_probability:
                system.is_online = False
                logger.warning(f"Legacy system {system.hostname} went offline")
    
    def _simulate_service_failures(self):
        """Simulate service failures"""
        for system in self.systems.values():
            if not system.is_online:
                continue
            
            for service in system.services:
                if service.simulate_service_failure():
                    logger.warning(f"Service failure: {service.service_name} on {system.hostname}")
                    service.last_restart = datetime.now()
    
    def _simulate_protocol_communications(self):
        """Simulate legacy protocol communications"""
        active_systems = [s for s in self.systems.values() if s.is_online]
        
        if not active_systems:
            return
        
        # Simulate some protocol activity
        for _ in range(random.randint(1, 5)):
            system = random.choice(active_systems)
            
            # Choose a random service to simulate
            if system.services:
                service = random.choice(system.services)
                
                if service.protocol == LegacyProtocol.TELNET:
                    session = self.protocol_simulator.simulate_telnet_session(system)
                    logger.debug(f"Simulated Telnet session: {session['session_id']}")
                
                elif service.protocol == LegacyProtocol.FTP:
                    session = self.protocol_simulator.simulate_ftp_session(system)
                    logger.debug(f"Simulated FTP session: {session['session_id']}")
                
                elif service.protocol in [LegacyProtocol.SNMP_V1, LegacyProtocol.SNMP_V2C]:
                    query = self.protocol_simulator.simulate_snmp_query(system)
                    logger.debug(f"Simulated SNMP query: {query['oid']}")
                
                elif service.protocol == LegacyProtocol.MODBUS_TCP:
                    comm = self.protocol_simulator.simulate_modbus_communication(system)
                    logger.debug(f"Simulated Modbus communication: Function {comm['function_code']}")
    
    def get_legacy_statistics(self) -> Dict[str, Any]:
        """Get legacy systems statistics"""
        total_systems = len(self.systems)
        online_systems = sum(1 for s in self.systems.values() if s.is_online)
        critical_systems = sum(1 for s in self.systems.values() if s.is_critical)
        unsupported_systems = sum(1 for s in self.systems.values() if s.support_status == "unsupported")
        
        # Vulnerability statistics
        total_vulns = 0
        critical_vulns = 0
        systems_with_critical_vulns = 0
        
        system_types = {}
        os_types = {}
        
        for system in self.systems.values():
            # Count vulnerabilities
            vuln_counts = system.get_vulnerability_count()
            total_vulns += sum(vuln_counts.values())
            critical_vulns += vuln_counts["Critical"]
            
            if system.has_critical_vulnerabilities():
                systems_with_critical_vulns += 1
            
            # Count system types
            sys_type = system.system_type.value
            system_types[sys_type] = system_types.get(sys_type, 0) + 1
            
            # Count OS types
            os_type = system.os_name
            os_types[os_type] = os_types.get(os_type, 0) + 1
        
        # Calculate average risk score
        risk_scores = [system.calculate_risk_score() for system in self.systems.values()]
        avg_risk_score = sum(risk_scores) / len(risk_scores) if risk_scores else 0
        
        return {
            "total_systems": total_systems,
            "online_systems": online_systems,
            "critical_systems": critical_systems,
            "unsupported_systems": unsupported_systems,
            "total_vulnerabilities": total_vulns,
            "critical_vulnerabilities": critical_vulns,
            "systems_with_critical_vulns": systems_with_critical_vulns,
            "system_types": system_types,
            "operating_systems": os_types,
            "average_risk_score": round(avg_risk_score, 2),
            "unsupported_percentage": (unsupported_systems / total_systems * 100) if total_systems > 0 else 0
        }
    
    def get_vulnerability_report(self) -> Dict[str, Any]:
        """Generate a comprehensive vulnerability report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "systems": {},
            "summary": {
                "total_systems_scanned": len(self.systems),
                "vulnerable_systems": 0,
                "total_vulnerabilities": 0,
                "by_severity": {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
            }
        }
        
        for system in self.systems.values():
            system_vulns = []
            vuln_counts = system.get_vulnerability_count()
            
            for service in system.services:
                for vuln in service.vulnerabilities:
                    system_vulns.append({
                        "cve_id": vuln.cve_id,
                        "service": service.service_name,
                        "port": service.port,
                        "severity": vuln.severity,
                        "category": vuln.category.value,
                        "description": vuln.description,
                        "cvss_score": vuln.get_cvss_score(),
                        "exploit_available": vuln.exploit_available,
                        "patch_available": vuln.patch_available
                    })
            
            if system_vulns:
                report["summary"]["vulnerable_systems"] += 1
            
            report["summary"]["total_vulnerabilities"] += len(system_vulns)
            
            for severity, count in vuln_counts.items():
                report["summary"]["by_severity"][severity] += count
            
            report["systems"][system.system_id] = {
                "hostname": system.hostname,
                "ip_address": system.ip_address,
                "system_type": system.system_type.value,
                "os_name": system.os_name,
                "os_version": system.os_version,
                "risk_score": system.calculate_risk_score(),
                "is_critical": system.is_critical,
                "support_status": system.support_status,
                "vulnerabilities": system_vulns,
                "vulnerability_counts": vuln_counts
            }
        
        return report
    
    def export_legacy_inventory(self, filename: str):
        """Export legacy systems inventory"""
        inventory = {
            "timestamp": datetime.now().isoformat(),
            "systems": {
                system_id: {
                    "hostname": system.hostname,
                    "ip_address": system.ip_address,
                    "system_type": system.system_type.value,
                    "os_name": system.os_name,
                    "os_version": system.os_version,
                    "architecture": system.architecture,
                    "manufacturer": system.manufacturer,
                    "model": system.model,
                    "install_date": system.install_date.isoformat(),
                    "last_patch_date": system.last_patch_date.isoformat() if system.last_patch_date else None,
                    "eol_date": system.eol_date.isoformat() if system.eol_date else None,
                    "support_status": system.support_status,
                    "is_critical": system.is_critical,
                    "business_function": system.business_function,
                    "risk_score": system.calculate_risk_score(),
                    "services": [
                        {
                            "name": service.service_name,
                            "protocol": service.protocol.value,
                            "port": service.port,
                            "version": service.version,
                            "is_encrypted": service.is_encrypted,
                            "default_credentials": service.default_credentials is not None,
                            "vulnerability_count": len(service.vulnerabilities)
                        }
                        for service in system.services
                    ]
                }
                for system_id, system in self.systems.items()
            },
            "statistics": self.get_legacy_statistics()
        }
        
        with open(filename, 'w') as f:
            json.dump(inventory, f, indent=2, default=str)
        
        logger.info(f"Legacy systems inventory exported to {filename}")


def main():
    """Main function for testing the legacy systems simulation"""
    manager = LegacySystemManager()
    
    try:
        # Populate legacy network
        manager.populate_legacy_network(count=8)
        
        # Start simulation
        manager.start_simulation()
        
        # Let it run for a bit
        time.sleep(30)
        
        # Get statistics
        stats = manager.get_legacy_statistics()
        print(f"Legacy Systems Statistics: {stats}")
        
        # Generate vulnerability report
        vuln_report = manager.get_vulnerability_report()
        print(f"Critical Vulnerabilities: {vuln_report['summary']['by_severity']['Critical']}")
        
        # Export inventory
        manager.export_legacy_inventory("legacy_systems_inventory.json")
        
    finally:
        manager.stop_simulation()


if __name__ == "__main__":
    main()