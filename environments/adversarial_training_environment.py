#!/usr/bin/env python3
"""
Adversarial Training Environment
Red Team vs Blue Team Containerized Cyber Range

This system creates a safe, isolated environment where:
- Red Team agents run in Kali Linux containers with real attack tools
- Blue Team agents run in monitoring containers with real defense tools  
- Both teams learn from actual tool outputs and interactions
- All operations are contained and safe for training purposes
"""

import asyncio
import json
import logging
import subprocess
import uuid
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import os

# Import our container management
from scripts.apple_container_setup import AppleContainerManager

class TeamRole(Enum):
    RED_TEAM = "red_team"
    BLUE_TEAM = "blue_team"
    PURPLE_TEAM = "purple_team"
    TARGET_SYSTEM = "target_system"

class ExercisePhase(Enum):
    SETUP = "setup"
    RECONNAISSANCE = "reconnaissance"
    INITIAL_ACCESS = "initial_access"
    PERSISTENCE = "persistence"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DEFENSE_EVASION = "defense_evasion"
    CREDENTIAL_ACCESS = "credential_access"
    DISCOVERY = "discovery"
    LATERAL_MOVEMENT = "lateral_movement"
    COLLECTION = "collection"
    COMMAND_CONTROL = "command_control"
    EXFILTRATION = "exfiltration"
    IMPACT = "impact"
    DETECTION = "detection"
    RESPONSE = "response"
    RECOVERY = "recovery"

@dataclass
class ToolExecution:
    """Represents execution of a security tool"""
    tool_name: str
    command: str
    args: List[str]
    target: str
    execution_time: datetime
    exit_code: int
    stdout: str
    stderr: str
    duration: float
    success: bool
    findings: List[str] = field(default_factory=list)
    iocs: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AdversarialExercise:
    """Represents a complete red vs blue training exercise"""
    exercise_id: str
    exercise_name: str
    scenario: str
    start_time: datetime
    end_time: Optional[datetime] = None
    red_team_actions: List[ToolExecution] = field(default_factory=list)
    blue_team_actions: List[ToolExecution] = field(default_factory=list)
    detection_events: List[Dict[str, Any]] = field(default_factory=list)
    learning_outcomes: Dict[str, Any] = field(default_factory=dict)
    success_metrics: Dict[str, float] = field(default_factory=dict)

class AdversarialTrainingEnvironment:
    """
    Complete adversarial training environment for red vs blue team exercises
    
    Features:
    - Isolated Kali Linux containers for red team operations
    - Monitoring containers for blue team defense
    - Target system containers for realistic attack scenarios
    - Real security tool integration and automation
    - Comprehensive logging and learning data collection
    """
    
    def __init__(self, container_manager: Optional[AppleContainerManager] = None):
        self.container_manager = container_manager or AppleContainerManager()
        self.logger = logging.getLogger(__name__)
        
        # Environment state
        self.active_exercises: Dict[str, AdversarialExercise] = {}
        self.container_registry: Dict[str, Dict[str, Any]] = {}
        self.network_topology: Dict[str, List[str]] = {}
        
        # Tool configurations
        self.red_team_tools = self._initialize_red_team_tools()
        self.blue_team_tools = self._initialize_blue_team_tools()
        
        # Training data collection
        self.training_data: List[Dict[str, Any]] = []
        self.exercise_history: List[AdversarialExercise] = []
        
    async def initialize_environment(self) -> bool:
        """Initialize the complete adversarial training environment"""
        self.logger.info("🏗️ Initializing Adversarial Training Environment...")
        
        try:
            # Initialize container manager
            if not await self.container_manager.initialize():
                self.logger.error("❌ Container manager initialization failed")
                return False
            
            self.logger.info("✅ Adversarial Training Environment ready")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Environment initialization failed: {e}")
            return False
    
    def _initialize_red_team_tools(self) -> Dict[str, Dict[str, Any]]:
        """Initialize red team tool configurations"""
        return {
            # Reconnaissance Tools
            "nmap": {
                "category": "reconnaissance",
                "description": "Network discovery and security auditing",
                "binary": "/usr/bin/nmap",
                "common_args": ["-sS", "-sV", "-O", "-A"],
                "output_parsers": ["nmap_xml", "nmap_normal"],
                "safety_checks": ["target_validation", "rate_limiting"]
            },
            "masscan": {
                "category": "reconnaissance", 
                "description": "High-speed port scanner",
                "binary": "/usr/bin/masscan",
                "common_args": ["-p1-65535", "--rate=1000"],
                "output_parsers": ["masscan_xml"],
                "safety_checks": ["target_validation", "rate_limiting"]
            },
            "dirb": {
                "category": "reconnaissance",
                "description": "Web content scanner",
                "binary": "/usr/bin/dirb",
                "common_args": ["-r", "-S", "-w"],
                "output_parsers": ["dirb_text"],
                "safety_checks": ["target_validation", "request_limiting"]
            },
            "nikto": {
                "category": "reconnaissance",
                "description": "Web server vulnerability scanner",
                "binary": "/usr/bin/nikto",
                "common_args": ["-h", "-Format", "xml"],
                "output_parsers": ["nikto_xml"],
                "safety_checks": ["target_validation", "request_limiting"]
            },
            
            # Exploitation Tools
            "metasploit": {
                "category": "exploitation",
                "description": "Penetration testing framework",
                "binary": "/usr/bin/msfconsole",
                "common_args": ["-q", "-r"],
                "output_parsers": ["msf_output"],
                "safety_checks": ["target_validation", "payload_restrictions", "sandbox_only"]
            },
            "sqlmap": {
                "category": "exploitation",
                "description": "SQL injection testing tool",
                "binary": "/usr/bin/sqlmap",
                "common_args": ["-u", "--batch", "--random-agent"],
                "output_parsers": ["sqlmap_output"],
                "safety_checks": ["target_validation", "safe_payloads_only"]
            },
            "burpsuite": {
                "category": "exploitation",
                "description": "Web application security testing",
                "binary": "/usr/bin/burpsuite",
                "common_args": ["--headless", "--project-file"],
                "output_parsers": ["burp_xml"],
                "safety_checks": ["target_validation", "scope_limiting"]
            },
            
            # Post-Exploitation Tools
            "empire": {
                "category": "post_exploitation",
                "description": "PowerShell post-exploitation framework",
                "binary": "/opt/Empire/empire",
                "common_args": ["--headless"],
                "output_parsers": ["empire_output"],
                "safety_checks": ["sandbox_only", "no_lateral_movement"]
            },
            "covenant": {
                "category": "post_exploitation", 
                "description": ".NET command and control framework",
                "binary": "/opt/Covenant/Covenant",
                "common_args": ["--headless"],
                "output_parsers": ["covenant_output"],
                "safety_checks": ["sandbox_only", "no_persistence"]
            }
        }
    
    def _initialize_blue_team_tools(self) -> Dict[str, Dict[str, Any]]:
        """Initialize blue team tool configurations"""
        return {
            # Network Monitoring
            "suricata": {
                "category": "network_monitoring",
                "description": "Network intrusion detection system",
                "binary": "/usr/bin/suricata",
                "common_args": ["-c", "/etc/suricata/suricata.yaml", "-i"],
                "output_parsers": ["suricata_eve_json"],
                "monitoring_type": "real_time"
            },
            "zeek": {
                "category": "network_monitoring",
                "description": "Network security monitor",
                "binary": "/usr/local/zeek/bin/zeek",
                "common_args": ["-i", "-C"],
                "output_parsers": ["zeek_logs"],
                "monitoring_type": "real_time"
            },
            "wireshark": {
                "category": "network_analysis",
                "description": "Network protocol analyzer",
                "binary": "/usr/bin/tshark",
                "common_args": ["-i", "-w", "-f"],
                "output_parsers": ["pcap", "wireshark_json"],
                "monitoring_type": "capture_analyze"
            },
            
            # Endpoint Monitoring
            "osquery": {
                "category": "endpoint_monitoring",
                "description": "Operating system instrumentation framework",
                "binary": "/usr/bin/osqueryi",
                "common_args": ["--json"],
                "output_parsers": ["osquery_json"],
                "monitoring_type": "endpoint_telemetry"
            },
            "sysmon": {
                "category": "endpoint_monitoring",
                "description": "System monitoring for Windows",
                "binary": "/opt/sysmon/sysmon",
                "common_args": ["-accepteula", "-i"],
                "output_parsers": ["sysmon_xml"],
                "monitoring_type": "system_events"
            },
            
            # Log Analysis
            "splunk": {
                "category": "log_analysis",
                "description": "Security information and event management",
                "binary": "/opt/splunk/bin/splunk",
                "common_args": ["search"],
                "output_parsers": ["splunk_json"],
                "monitoring_type": "siem"
            },
            "elastic": {
                "category": "log_analysis",
                "description": "Elasticsearch log analysis",
                "binary": "/usr/share/elasticsearch/bin/elasticsearch",
                "common_args": [],
                "output_parsers": ["elasticsearch_json"],
                "monitoring_type": "siem"
            },
            
            # Forensics Tools
            "volatility": {
                "category": "forensics",
                "description": "Memory forensics framework",
                "binary": "/usr/bin/volatility",
                "common_args": ["-f", "--profile"],
                "output_parsers": ["volatility_output"],
                "monitoring_type": "forensic_analysis"
            },
            "autopsy": {
                "category": "forensics",
                "description": "Digital forensics platform",
                "binary": "/opt/autopsy/bin/autopsy",
                "common_args": ["--headless"],
                "output_parsers": ["autopsy_xml"],
                "monitoring_type": "forensic_analysis"
            }
        }
    
    async def create_adversarial_exercise(self,
                                        exercise_name: str,
                                        scenario: str,
                                        red_team_objectives: List[str],
                                        blue_team_objectives: List[str]) -> str:
        """Create a new adversarial training exercise"""
        
        exercise_id = f"exercise_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        self.logger.info(f"🎮 Creating adversarial exercise: {exercise_name}")
        
        try:
            # Create exercise record
            exercise = AdversarialExercise(
                exercise_id=exercise_id,
                exercise_name=exercise_name,
                scenario=scenario,
                start_time=datetime.now()
            )
            
            self.active_exercises[exercise_id] = exercise
            
            # Setup environment topology
            await self._setup_exercise_topology(exercise_id, scenario)
            
            # Deploy containers
            containers = await self._deploy_exercise_containers(exercise_id)
            
            # Configure networking
            await self._configure_exercise_networking(exercise_id, containers)
            
            # Initialize monitoring
            await self._initialize_exercise_monitoring(exercise_id)
            
            self.logger.info(f"✅ Adversarial exercise ready: {exercise_id}")
            return exercise_id
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create exercise: {e}")
            raise
    
    async def _setup_exercise_topology(self, exercise_id: str, scenario: str):
        """Setup network topology for the exercise"""
        
        # Define topology based on scenario
        if scenario == "enterprise_breach":
            topology = {
                "external_zone": ["red_team_kali"],
                "dmz_zone": ["web_server", "email_server"],
                "internal_zone": ["workstation", "database_server", "domain_controller"],
                "monitoring_zone": ["blue_team_soc", "siem_server"]
            }
        elif scenario == "cloud_compromise":
            topology = {
                "external_zone": ["red_team_kali"],
                "cloud_zone": ["web_app", "api_server", "database"],
                "management_zone": ["cloud_console", "monitoring_server"],
                "security_zone": ["blue_team_cloud_soc"]
            }
        else:
            # Default topology
            topology = {
                "attack_zone": ["red_team_kali"],
                "target_zone": ["target_server", "target_workstation"],
                "defense_zone": ["blue_team_soc"]
            }
        
        self.network_topology[exercise_id] = topology
    
    async def _deploy_exercise_containers(self, exercise_id: str) -> Dict[str, str]:
        """Deploy containers for the exercise"""
        containers = {}
        topology = self.network_topology[exercise_id]
        
        try:
            # Deploy red team containers
            for zone, systems in topology.items():
                for system in systems:
                    if system.startswith("red_team"):
                        container_name = f"{exercise_id}_{system}"
                        
                        # Create Kali Linux container for red team
                        red_team_tools = list(self.red_team_tools.keys())
                        result = await self.container_manager.create_kali_container(
                            container_name,
                            red_team_tools
                        )
                        
                        if result.get("success"):
                            containers[system] = container_name
                            self.logger.info(f"✅ Red team container deployed: {container_name}")
                    
                    elif system.startswith("blue_team"):
                        container_name = f"{exercise_id}_{system}"
                        
                        # Create monitoring container for blue team
                        result = await self.container_manager.create_monitoring_container(
                            container_name
                        )
                        
                        if result.get("success"):
                            containers[system] = container_name
                            self.logger.info(f"✅ Blue team container deployed: {container_name}")
                    
                    else:
                        # Create target system containers
                        container_name = f"{exercise_id}_{system}"
                        target_config = await self._create_target_system_container(
                            container_name, system
                        )
                        
                        if target_config.get("success"):
                            containers[system] = container_name
                            self.logger.info(f"✅ Target system deployed: {container_name}")
            
            self.container_registry[exercise_id] = containers
            return containers
            
        except Exception as e:
            self.logger.error(f"Container deployment failed: {e}")
            raise
    
    async def _create_target_system_container(self, container_name: str, system_type: str) -> Dict[str, Any]:
        """Create target system container with vulnerabilities for training"""
        
        # Define vulnerable system configurations
        system_configs = {
            "web_server": {
                "base_image": "ubuntu:20.04",
                "services": ["apache2", "mysql-server", "php"],
                "vulnerabilities": ["sql_injection", "xss", "directory_traversal"],
                "network_config": {"ports": ["80:80", "443:443", "22:22"]}
            },
            "workstation": {
                "base_image": "ubuntu:20.04", 
                "services": ["ssh", "rdp", "smb"],
                "vulnerabilities": ["weak_passwords", "unpatched_services", "misconfigurations"],
                "network_config": {"ports": ["22:22", "3389:3389", "445:445"]}
            },
            "database_server": {
                "base_image": "ubuntu:20.04",
                "services": ["mysql", "postgresql"],
                "vulnerabilities": ["weak_authentication", "privilege_escalation", "data_exposure"],
                "network_config": {"ports": ["3306:3306", "5432:5432"]}
            }
        }
        
        config = system_configs.get(system_type, system_configs["workstation"])
        
        # Create container with intentional vulnerabilities for training
        container_config = {
            "name": container_name,
            "image": config["base_image"],
            "services": config["services"],
            "vulnerabilities": config["vulnerabilities"],
            "network": config["network_config"],
            "purpose": "training_target"
        }
        
        # This would create the actual vulnerable container
        # For safety, we'll simulate the creation
        return {"success": True, "config": container_config}
    
    async def execute_red_team_operation(self,
                                       exercise_id: str,
                                       phase: ExercisePhase,
                                       target: str,
                                       tools: List[str]) -> List[ToolExecution]:
        """Execute red team operation with real tools"""
        
        self.logger.info(f"🔴 Executing red team {phase.value} phase")
        
        if exercise_id not in self.active_exercises:
            raise ValueError(f"Exercise {exercise_id} not found")
        
        exercise = self.active_exercises[exercise_id]
        executions = []
        
        try:
            containers = self.container_registry[exercise_id]
            red_team_container = None
            
            # Find red team container
            for system, container_name in containers.items():
                if system.startswith("red_team"):
                    red_team_container = container_name
                    break
            
            if not red_team_container:
                raise ValueError("No red team container found")
            
            # Execute tools in sequence
            for tool_name in tools:
                if tool_name not in self.red_team_tools:
                    self.logger.warning(f"Unknown red team tool: {tool_name}")
                    continue
                
                tool_config = self.red_team_tools[tool_name]
                
                # Generate tool command based on phase and target
                command, args = await self._generate_tool_command(
                    tool_name, tool_config, phase, target
                )
                
                # Execute tool in container
                execution = await self._execute_tool_in_container(
                    red_team_container, tool_name, command, args
                )
                
                # Parse tool output for findings
                execution.findings = await self._parse_tool_output(
                    tool_name, execution.stdout, execution.stderr
                )
                
                executions.append(execution)
                exercise.red_team_actions.append(execution)
                
                # Generate learning data
                await self._generate_training_data_from_execution(
                    exercise_id, "red_team", execution
                )
                
                self.logger.info(f"✅ Executed {tool_name}: {execution.success}")
        
        except Exception as e:
            self.logger.error(f"Red team operation failed: {e}")
            raise
        
        return executions
    
    async def execute_blue_team_operation(self,
                                        exercise_id: str,
                                        detection_phase: str,
                                        monitoring_scope: str,
                                        tools: List[str]) -> List[ToolExecution]:
        """Execute blue team defensive operations"""
        
        self.logger.info(f"🔵 Executing blue team {detection_phase} phase")
        
        if exercise_id not in self.active_exercises:
            raise ValueError(f"Exercise {exercise_id} not found")
        
        exercise = self.active_exercises[exercise_id]
        executions = []
        
        try:
            containers = self.container_registry[exercise_id]
            blue_team_container = None
            
            # Find blue team container
            for system, container_name in containers.items():
                if system.startswith("blue_team"):
                    blue_team_container = container_name
                    break
            
            if not blue_team_container:
                raise ValueError("No blue team container found")
            
            # Execute defensive tools
            for tool_name in tools:
                if tool_name not in self.blue_team_tools:
                    self.logger.warning(f"Unknown blue team tool: {tool_name}")
                    continue
                
                tool_config = self.blue_team_tools[tool_name]
                
                # Generate monitoring/analysis command
                command, args = await self._generate_blue_team_command(
                    tool_name, tool_config, detection_phase, monitoring_scope
                )
                
                # Execute tool in container
                execution = await self._execute_tool_in_container(
                    blue_team_container, tool_name, command, args
                )
                
                # Parse defensive tool output
                execution.findings = await self._parse_defensive_output(
                    tool_name, execution.stdout, execution.stderr
                )
                
                executions.append(execution)
                exercise.blue_team_actions.append(execution)
                
                # Generate learning data
                await self._generate_training_data_from_execution(
                    exercise_id, "blue_team", execution
                )
                
                self.logger.info(f"✅ Executed {tool_name}: {execution.success}")
        
        except Exception as e:
            self.logger.error(f"Blue team operation failed: {e}")
            raise
        
        return executions
    
    async def _generate_tool_command(self,
                                   tool_name: str,
                                   tool_config: Dict[str, Any],
                                   phase: ExercisePhase,
                                   target: str) -> Tuple[str, List[str]]:
        """Generate appropriate tool command for the phase and target"""
        
        base_cmd = tool_config["binary"]
        args = []
        
        # Phase-specific command generation
        if tool_name == "nmap":
            if phase == ExercisePhase.RECONNAISSANCE:
                args = ["-sS", "-sV", "-O", target]
            elif phase == ExercisePhase.DISCOVERY:
                args = ["-sU", "-sT", "-p-", target]
            else:
                args = ["-sS", target]
                
        elif tool_name == "dirb":
            if phase == ExercisePhase.RECONNAISSANCE:
                args = [f"http://{target}/", "/usr/share/dirb/wordlists/common.txt"]
            else:
                args = [f"http://{target}/"]
                
        elif tool_name == "nikto":
            args = ["-h", target, "-Format", "xml"]
            
        elif tool_name == "sqlmap":
            if phase == ExercisePhase.INITIAL_ACCESS:
                args = ["-u", f"http://{target}/login.php", "--batch", "--dbs"]
            else:
                args = ["-u", f"http://{target}/", "--batch"]
                
        elif tool_name == "metasploit":
            # Create metasploit resource script for automated execution
            args = ["-q", "-r", "/tmp/msf_script.rc"]
            
        # Add safety checks
        if "safety_checks" in tool_config:
            args = await self._apply_safety_checks(args, tool_config["safety_checks"])
        
        return base_cmd, args
    
    async def _generate_blue_team_command(self,
                                        tool_name: str,
                                        tool_config: Dict[str, Any],
                                        detection_phase: str,
                                        monitoring_scope: str) -> Tuple[str, List[str]]:
        """Generate blue team monitoring/analysis commands"""
        
        base_cmd = tool_config["binary"]
        args = []
        
        if tool_name == "suricata":
            args = ["-c", "/etc/suricata/suricata.yaml", "-i", "eth0", "-l", "/var/log/suricata/"]
            
        elif tool_name == "osquery":
            if detection_phase == "process_monitoring":
                args = ["--json", "SELECT * FROM processes WHERE name LIKE '%suspicious%';"]
            elif detection_phase == "network_monitoring": 
                args = ["--json", "SELECT * FROM process_open_sockets;"]
            else:
                args = ["--json", "SELECT * FROM system_info;"]
                
        elif tool_name == "wireshark":
            args = ["-i", "eth0", "-w", "/tmp/capture.pcap", "-c", "1000"]
            
        elif tool_name == "splunk":
            if detection_phase == "log_analysis":
                args = ["search", f"index=main earliest=-1h {monitoring_scope}"]
            else:
                args = ["search", "index=main earliest=-15m"]
        
        return base_cmd, args
    
    async def _execute_tool_in_container(self,
                                       container_name: str,
                                       tool_name: str,
                                       command: str,
                                       args: List[str]) -> ToolExecution:
        """Execute security tool in container and capture results"""
        
        start_time = datetime.now()
        full_command = f"{command} {' '.join(args)}"
        
        try:
            # Execute command in container
            result = await self.container_manager.execute_in_container(
                container_name,
                full_command,
                timeout=300  # 5 minute timeout
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            execution = ToolExecution(
                tool_name=tool_name,
                command=command,
                args=args,
                target=args[-1] if args else "unknown",
                execution_time=start_time,
                exit_code=result.get("exit_code", -1),
                stdout=result.get("stdout", ""),
                stderr=result.get("stderr", ""),
                duration=duration,
                success=result.get("success", False)
            )
            
            return execution
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return ToolExecution(
                tool_name=tool_name,
                command=command,
                args=args,
                target="error",
                execution_time=start_time,
                exit_code=-1,
                stdout="",
                stderr=str(e),
                duration=duration,
                success=False
            )
    
    async def _parse_tool_output(self,
                               tool_name: str,
                               stdout: str,
                               stderr: str) -> List[str]:
        """Parse security tool output to extract findings"""
        
        findings = []
        
        if tool_name == "nmap":
            # Parse nmap output for open ports and services
            lines = stdout.split('\n')
            for line in lines:
                if "/tcp" in line and "open" in line:
                    findings.append(f"Open port found: {line.strip()}")
                elif "OS:" in line:
                    findings.append(f"OS detection: {line.strip()}")
                    
        elif tool_name == "dirb":
            # Parse dirb output for discovered directories
            lines = stdout.split('\n')
            for line in lines:
                if "==> DIRECTORY:" in line:
                    findings.append(f"Directory found: {line.strip()}")
                elif "CODE:" in line and ("200" in line or "301" in line):
                    findings.append(f"Interesting response: {line.strip()}")
                    
        elif tool_name == "nikto":
            # Parse nikto XML output for vulnerabilities
            if "OSVDB" in stdout or "CVE" in stdout:
                findings.append("Potential vulnerabilities found")
                
        elif tool_name == "sqlmap":
            # Parse sqlmap output for SQL injection findings
            if "vulnerable" in stdout.lower():
                findings.append("SQL injection vulnerability detected")
            if "database" in stdout.lower():
                findings.append("Database access achieved")
                
        # Add error findings
        if stderr and "error" in stderr.lower():
            findings.append(f"Tool error: {stderr[:100]}")
        
        return findings
    
    async def _parse_defensive_output(self,
                                    tool_name: str,
                                    stdout: str,
                                    stderr: str) -> List[str]:
        """Parse blue team tool output for security events"""
        
        findings = []
        
        if tool_name == "suricata":
            # Parse Suricata alerts
            lines = stdout.split('\n')
            for line in lines:
                if '"event_type":"alert"' in line:
                    findings.append(f"Security alert: {line[:100]}")
                    
        elif tool_name == "osquery":
            # Parse osquery JSON output
            try:
                import json
                data = json.loads(stdout)
                if isinstance(data, list) and data:
                    findings.append(f"Query returned {len(data)} results")
            except:
                pass
                
        elif tool_name == "wireshark":
            # Parse tshark output
            if "packets captured" in stdout:
                findings.append(f"Network capture: {stdout.strip()}")
                
        elif tool_name == "splunk":
            # Parse Splunk search results
            if "events" in stdout:
                findings.append(f"Log analysis: {stdout[:100]}")
        
        return findings
    
    async def _generate_training_data_from_execution(self,
                                                   exercise_id: str,
                                                   team: str,
                                                   execution: ToolExecution):
        """Generate training data from tool execution"""
        
        training_example = {
            "exercise_id": exercise_id,
            "team": team,
            "tool": execution.tool_name,
            "command": f"{execution.command} {' '.join(execution.args)}",
            "target": execution.target,
            "success": execution.success,
            "duration": execution.duration,
            "findings": execution.findings,
            "timestamp": execution.execution_time.isoformat(),
            "training_context": {
                "scenario": "adversarial_training",
                "tool_category": self._get_tool_category(execution.tool_name, team),
                "execution_phase": self._determine_execution_phase(execution),
                "effectiveness": len(execution.findings) / max(execution.duration, 1)
            }
        }
        
        self.training_data.append(training_example)
    
    def _get_tool_category(self, tool_name: str, team: str) -> str:
        """Get tool category for training context"""
        if team == "red_team" and tool_name in self.red_team_tools:
            return self.red_team_tools[tool_name]["category"]
        elif team == "blue_team" and tool_name in self.blue_team_tools:
            return self.blue_team_tools[tool_name]["category"]
        return "unknown"
    
    def _determine_execution_phase(self, execution: ToolExecution) -> str:
        """Determine execution phase from tool and findings"""
        tool_name = execution.tool_name
        
        if tool_name in ["nmap", "masscan", "dirb", "nikto"]:
            return "reconnaissance"
        elif tool_name in ["metasploit", "sqlmap", "burpsuite"]:
            return "exploitation"
        elif tool_name in ["empire", "covenant"]:
            return "post_exploitation"
        elif tool_name in ["suricata", "zeek", "wireshark"]:
            return "detection"
        elif tool_name in ["osquery", "sysmon"]:
            return "monitoring"
        elif tool_name in ["splunk", "elastic"]:
            return "analysis"
        else:
            return "unknown"
    
    async def run_complete_adversarial_exercise(self,
                                              exercise_name: str,
                                              scenario: str,
                                              duration_minutes: int = 60) -> AdversarialExercise:
        """Run a complete red vs blue adversarial training exercise"""
        
        self.logger.info(f"🎯 Starting complete adversarial exercise: {exercise_name}")
        
        # Create exercise
        exercise_id = await self.create_adversarial_exercise(
            exercise_name,
            scenario,
            red_team_objectives=["reconnaissance", "exploitation", "persistence"],
            blue_team_objectives=["detection", "analysis", "response"]
        )
        
        exercise = self.active_exercises[exercise_id]
        
        try:
            # Phase 1: Red Team Reconnaissance
            self.logger.info("🔴 Phase 1: Red Team Reconnaissance")
            red_recon = await self.execute_red_team_operation(
                exercise_id,
                ExercisePhase.RECONNAISSANCE,
                "target_server",
                ["nmap", "dirb", "nikto"]
            )
            
            # Phase 2: Blue Team Detection
            self.logger.info("🔵 Phase 2: Blue Team Detection")
            blue_detection = await self.execute_blue_team_operation(
                exercise_id,
                "network_monitoring",
                "external_traffic",
                ["suricata", "wireshark"]
            )
            
            # Phase 3: Red Team Exploitation
            self.logger.info("🔴 Phase 3: Red Team Exploitation")
            red_exploit = await self.execute_red_team_operation(
                exercise_id,
                ExercisePhase.INITIAL_ACCESS,
                "target_server",
                ["sqlmap", "metasploit"]
            )
            
            # Phase 4: Blue Team Response
            self.logger.info("🔵 Phase 4: Blue Team Analysis and Response")
            blue_response = await self.execute_blue_team_operation(
                exercise_id,
                "incident_analysis",
                "security_events",
                ["osquery", "splunk"]
            )
            
            # Calculate exercise metrics
            exercise.success_metrics = await self._calculate_exercise_metrics(exercise)
            exercise.learning_outcomes = await self._extract_learning_outcomes(exercise)
            exercise.end_time = datetime.now()
            
            # Move to completed exercises
            self.exercise_history.append(exercise)
            del self.active_exercises[exercise_id]
            
            self.logger.info(f"✅ Adversarial exercise completed: {exercise_name}")
            return exercise
            
        except Exception as e:
            self.logger.error(f"❌ Adversarial exercise failed: {e}")
            exercise.end_time = datetime.now()
            exercise.success_metrics["status"] = "failed"
            exercise.success_metrics["error"] = str(e)
            return exercise
    
    async def _calculate_exercise_metrics(self, exercise: AdversarialExercise) -> Dict[str, float]:
        """Calculate success metrics for the exercise"""
        
        metrics = {}
        
        # Red team metrics
        red_tools_executed = len(exercise.red_team_actions)
        red_successful_actions = len([a for a in exercise.red_team_actions if a.success])
        metrics["red_team_success_rate"] = red_successful_actions / max(red_tools_executed, 1)
        
        total_red_findings = sum(len(a.findings) for a in exercise.red_team_actions)
        metrics["red_team_findings_rate"] = total_red_findings / max(red_tools_executed, 1)
        
        # Blue team metrics
        blue_tools_executed = len(exercise.blue_team_actions)
        blue_successful_actions = len([a for a in exercise.blue_team_actions if a.success])
        metrics["blue_team_success_rate"] = blue_successful_actions / max(blue_tools_executed, 1)
        
        total_blue_findings = sum(len(a.findings) for a in exercise.blue_team_actions)
        metrics["blue_team_detection_rate"] = total_blue_findings / max(blue_tools_executed, 1)
        
        # Overall exercise metrics
        total_duration = (exercise.end_time - exercise.start_time).total_seconds() if exercise.end_time else 0
        metrics["exercise_duration_minutes"] = total_duration / 60
        
        metrics["total_actions"] = red_tools_executed + blue_tools_executed
        metrics["overall_success_rate"] = (red_successful_actions + blue_successful_actions) / max(metrics["total_actions"], 1)
        
        return metrics
    
    async def _extract_learning_outcomes(self, exercise: AdversarialExercise) -> Dict[str, Any]:
        """Extract learning outcomes from the exercise"""
        
        outcomes = {
            "red_team_learnings": [],
            "blue_team_learnings": [],
            "cross_team_insights": [],
            "tool_effectiveness": {},
            "improvement_recommendations": []
        }
        
        # Analyze red team tool effectiveness
        for action in exercise.red_team_actions:
            tool_name = action.tool_name
            effectiveness = len(action.findings) / max(action.duration, 1)
            
            if tool_name not in outcomes["tool_effectiveness"]:
                outcomes["tool_effectiveness"][tool_name] = []
            outcomes["tool_effectiveness"][tool_name].append(effectiveness)
            
            # Generate learnings
            if action.success and action.findings:
                outcomes["red_team_learnings"].append(
                    f"{tool_name} successfully identified {len(action.findings)} findings"
                )
        
        # Analyze blue team detection capabilities
        for action in exercise.blue_team_actions:
            tool_name = action.tool_name
            detection_rate = len(action.findings) / max(action.duration, 1)
            
            if action.success and action.findings:
                outcomes["blue_team_learnings"].append(
                    f"{tool_name} detected {len(action.findings)} security events"
                )
        
        # Generate improvement recommendations
        red_success_rate = exercise.success_metrics.get("red_team_success_rate", 0)
        blue_detection_rate = exercise.success_metrics.get("blue_team_detection_rate", 0)
        
        if red_success_rate > 0.8:
            outcomes["improvement_recommendations"].append("Blue team defenses need strengthening")
        if blue_detection_rate < 0.5:
            outcomes["improvement_recommendations"].append("Blue team detection capabilities need improvement")
        
        return outcomes
    
    async def get_training_dataset(self) -> List[Dict[str, Any]]:
        """Get collected training data from all exercises"""
        return self.training_data
    
    async def export_training_data(self, filename: str):
        """Export training data to file for model training"""
        try:
            with open(filename, 'w') as f:
                json.dump(self.training_data, f, indent=2, default=str)
            
            self.logger.info(f"✅ Training data exported to: {filename}")
            
        except Exception as e:
            self.logger.error(f"Failed to export training data: {e}")
    
    async def cleanup_exercise(self, exercise_id: str):
        """Cleanup exercise containers and resources"""
        if exercise_id in self.container_registry:
            containers = self.container_registry[exercise_id]
            
            for system, container_name in containers.items():
                await self.container_manager.stop_container(container_name)
            
            del self.container_registry[exercise_id]
            
        if exercise_id in self.active_exercises:
            del self.active_exercises[exercise_id]
            
        self.logger.info(f"✅ Exercise {exercise_id} cleaned up")
    
    async def cleanup_all(self):
        """Cleanup all exercises and resources"""
        for exercise_id in list(self.active_exercises.keys()):
            await self.cleanup_exercise(exercise_id)
        
        await self.container_manager.cleanup_all_containers()
        self.logger.info("✅ All adversarial training environment resources cleaned up")


# Factory functions

def create_adversarial_training_environment() -> AdversarialTrainingEnvironment:
    """Create adversarial training environment"""
    return AdversarialTrainingEnvironment()

async def run_adversarial_training_demo():
    """Demonstrate adversarial training capabilities"""
    print("🎯 Adversarial Training Environment Demo")
    print("=" * 50)
    
    # Create environment
    env = create_adversarial_training_environment()
    
    if await env.initialize_environment():
        print("✅ Adversarial training environment ready")
        
        # Run demo exercise
        exercise = await env.run_complete_adversarial_exercise(
            "Demo Red vs Blue Exercise",
            "enterprise_breach",
            duration_minutes=10
        )
        
        # Show results
        print(f"\n📊 Exercise Results:")
        print(f"Red Team Success Rate: {exercise.success_metrics.get('red_team_success_rate', 0):.2f}")
        print(f"Blue Team Detection Rate: {exercise.success_metrics.get('blue_team_detection_rate', 0):.2f}")
        print(f"Total Actions: {exercise.success_metrics.get('total_actions', 0)}")
        print(f"Duration: {exercise.success_metrics.get('exercise_duration_minutes', 0):.1f} minutes")
        
        # Export training data
        training_data = await env.get_training_dataset()
        print(f"\n🎓 Training Data Generated: {len(training_data)} examples")
        
        await env.export_training_data("adversarial_training_data.json")
        
        # Cleanup
        await env.cleanup_all()
        
    else:
        print("❌ Failed to initialize adversarial training environment")

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    sys.exit(asyncio.run(run_adversarial_training_demo()))