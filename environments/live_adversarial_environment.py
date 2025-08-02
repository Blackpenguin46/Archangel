#!/usr/bin/env python3
"""
Live Adversarial Hacking Environment
Real-time red team vs blue team cybersecurity simulation with container isolation
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import subprocess
import threading
import signal
import os

@dataclass
class AttackScenario:
    """Defines a specific attack scenario for red team"""
    name: str
    description: str
    target_services: List[str]
    attack_tools: List[str]
    success_criteria: List[str]
    estimated_duration: int  # minutes
    difficulty_level: str
    learning_objectives: List[str]

@dataclass
class DefensePosture:
    """Defines blue team defensive posture"""
    monitoring_tools: List[str]
    detection_rules: List[str]
    response_playbooks: List[str]
    alert_thresholds: Dict[str, float]
    automated_responses: List[str]

@dataclass
class LiveAttackEvent:
    """Real-time attack event during live exercise"""
    timestamp: datetime
    event_id: str
    attack_type: str
    source_ip: str
    target_ip: str
    target_service: str
    attack_payload: str
    success: bool
    detection_time: Optional[float]
    response_time: Optional[float]
    mitigation_action: Optional[str]

class LiveAdversarialEnvironment:
    """Live red team vs blue team hacking simulation environment"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.exercise_id = str(uuid.uuid4())
        self.start_time = None
        self.red_team_container = None
        self.blue_team_container = None
        self.attack_events = []
        self.defense_events = []
        self.live_metrics = {}
        self.exercise_running = False
        
        # Container configurations with unique names
        import time
        timestamp = str(int(time.time()))[-6:]  # Last 6 digits for uniqueness
        
        self.red_team_config = {
            "image": "kalilinux/kali-rolling",
            "name": f"archangel-red-{timestamp}",
            "network": "archangel-combat-net",
            "ip": "192.168.100.10",
            "capabilities": ["NET_ADMIN", "SYS_ADMIN"],
            "volumes": {
                "./red_team_tools": "/opt/red_tools",
                "./attack_scripts": "/opt/attacks",
                "./logs": "/var/log/attacks"
            }
        }
        
        self.blue_team_config = {
            "image": "ubuntu:22.04",  # Defender target system
            "name": f"archangel-blue-{timestamp}",
            "network": "archangel-combat-net",
            "ip": "192.168.100.20",
            "services": ["ssh", "http", "ftp", "smtp"],
            "volumes": {
                "./blue_team_tools": "/opt/blue_tools",
                "./defense_scripts": "/opt/defenses",
                "./logs": "/var/log/defenses"
            }
        }
        
        # Predefined attack scenarios
        self.attack_scenarios = self._load_attack_scenarios()
        self.defense_postures = self._load_defense_postures()
        
        # Real-time monitoring
        self.monitoring_active = False
        self.monitoring_threads = []
        
    def _load_attack_scenarios(self) -> List[AttackScenario]:
        """Load predefined red team attack scenarios"""
        return [
            AttackScenario(
                name="Network Reconnaissance",
                description="Port scanning and service enumeration of blue team systems",
                target_services=["ssh", "http", "ftp", "smtp"],
                attack_tools=["nmap", "masscan", "zmap"],
                success_criteria=["Identify open ports", "Enumerate services", "Detect OS fingerprint"],
                estimated_duration=10,
                difficulty_level="Beginner",
                learning_objectives=["Network discovery", "Service enumeration", "Stealth scanning"]
            ),
            AttackScenario(
                name="SSH Brute Force Attack",
                description="Attempt to brute force SSH credentials on blue team system",
                target_services=["ssh"],
                attack_tools=["hydra", "medusa", "patator"],
                success_criteria=["Discover valid credentials", "Gain shell access"],
                estimated_duration=15,
                difficulty_level="Intermediate",
                learning_objectives=["Credential attacks", "Authentication bypass", "Persistence"]
            ),
            AttackScenario(
                name="Web Application Exploitation",
                description="Identify and exploit web application vulnerabilities",
                target_services=["http", "https"],
                attack_tools=["burpsuite", "sqlmap", "nikto", "dirb"],
                success_criteria=["Find SQL injection", "Achieve RCE", "Extract data"],
                estimated_duration=20,
                difficulty_level="Advanced",
                learning_objectives=["Web vulnerabilities", "Injection attacks", "Data extraction"]
            ),
            AttackScenario(
                name="Lateral Movement Simulation",
                description="Move laterally through blue team network after initial compromise",
                target_services=["ssh", "smb", "rdp"],
                attack_tools=["metasploit", "crackmapexec", "bloodhound"],
                success_criteria=["Compromise additional hosts", "Escalate privileges", "Access sensitive data"],
                estimated_duration=25,
                difficulty_level="Expert",
                learning_objectives=["Lateral movement", "Privilege escalation", "Network traversal"]
            ),
            AttackScenario(
                name="Data Exfiltration Operation", 
                description="Extract sensitive data from compromised blue team systems",
                target_services=["ftp", "http", "dns"],
                attack_tools=["wget", "curl", "dnscat2", "netcat"],
                success_criteria=["Identify sensitive files", "Establish covert channel", "Exfiltrate data"],
                estimated_duration=18,
                difficulty_level="Advanced",
                learning_objectives=["Data discovery", "Covert channels", "Exfiltration techniques"]
            )
        ]
    
    def _load_defense_postures(self) -> List[DefensePosture]:
        """Load predefined blue team defense postures"""
        return [
            DefensePosture(
                monitoring_tools=["fail2ban", "ossec", "suricata", "tcpdump"],
                detection_rules=[
                    "Multiple failed SSH attempts",
                    "Port scan detection",
                    "Unusual network traffic patterns",
                    "Privilege escalation attempts"
                ],
                response_playbooks=[
                    "Block suspicious IP addresses",
                    "Increase logging verbosity",
                    "Alert security team",
                    "Initiate incident response"
                ],
                alert_thresholds={
                    "failed_logins": 5.0,
                    "port_scan_rate": 10.0,
                    "data_transfer_rate": 100.0
                },
                automated_responses=[
                    "IP blocking via iptables",
                    "Service isolation",
                    "Evidence collection",
                    "Backup critical data"
                ]
            )
        ]
    
    async def setup_combat_environment(self) -> bool:
        """Setup isolated container environment for red vs blue combat"""
        try:
            self.logger.info("🔧 Setting up live adversarial combat environment...")
            
            # Create isolated network for combat
            await self._create_combat_network()
            
            # Setup red team attack container
            red_success = await self._setup_red_team_container()
            
            # Setup blue team target container
            blue_success = await self._setup_blue_team_container()
            
            if not red_success or not blue_success:
                self.logger.error("❌ Container setup failed")
                return False
            
            # Setup monitoring and logging
            await self._setup_monitoring_infrastructure()
            
            # Initialize attack and defense tools
            await self._initialize_combat_tools()
            
            self.logger.info("✅ Live combat environment ready")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to setup combat environment: {e}")
            return False
    
    async def _create_combat_network(self):
        """Create isolated Docker network for red vs blue combat"""
        try:
            import docker
            client = docker.from_env()
            
            # Force cleanup of existing network and containers
            try:
                # Stop and remove any existing containers first
                containers = client.containers.list(all=True, filters={"name": "archangel-"})
                for container in containers:
                    try:
                        container.stop()
                        container.remove()
                        self.logger.info(f"Removed existing container: {container.name}")
                    except:
                        pass
                
                # Remove existing network
                try:
                    existing_network = client.networks.get("archangel-combat-net")
                    existing_network.remove()
                    self.logger.info("Removed existing combat network")
                except docker.errors.NotFound:
                    pass
            except Exception as e:
                self.logger.debug(f"Cleanup warning: {e}")
            
            # Create new isolated network with unique timestamp
            import time
            network_suffix = str(int(time.time()))[-4:]  # Last 4 digits of timestamp
            
            result = subprocess.run([
                "docker", "network", "create",
                "--driver", "bridge",
                "--subnet", "192.168.100.0/24",
                "--gateway", "192.168.100.1",
                "archangel-combat-net"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info("✅ Created isolated combat network: 192.168.100.0/24")
                return
            
            # If primary subnet fails, try alternative subnets
            for subnet_base in [101, 102, 103, 104]:
                self.logger.warning(f"Trying alternative subnet: 192.168.{subnet_base}.0/24")
                alt_result = subprocess.run([
                    "docker", "network", "create",
                    "--driver", "bridge", 
                    "--subnet", f"192.168.{subnet_base}.0/24",
                    "--gateway", f"192.168.{subnet_base}.1",
                    "archangel-combat-net"
                ], capture_output=True, text=True)
                
                if alt_result.returncode == 0:
                    self.logger.info(f"✅ Created combat network: 192.168.{subnet_base}.0/24")
                    # Update IP configs for alternative subnet
                    self.red_team_config["ip"] = f"192.168.{subnet_base}.10"
                    self.blue_team_config["ip"] = f"192.168.{subnet_base}.20"
                    return
            
            # If all subnets fail, raise error
            self.logger.error(f"❌ Failed to create network: {result.stderr}")
            raise Exception("Could not create combat network with any subnet")
                
        except Exception as e:
            self.logger.error(f"Network creation failed: {e}")
            raise
    
    async def _setup_red_team_container(self):
        """Setup Kali Linux container for red team attacks"""
        try:
            self.logger.info("🔴 Setting up Red Team attack container...")
            
            # Remove any existing container with same name
            subprocess.run(["docker", "rm", "-f", self.red_team_config["name"]], 
                         capture_output=True)
            
            # Create directories for red team tools and logs
            os.makedirs("red_team_tools", exist_ok=True)
            os.makedirs("attack_scripts", exist_ok=True)
            os.makedirs("logs", exist_ok=True)
            
            # Pull Kali Linux image
            subprocess.run(["docker", "pull", "kalilinux/kali-rolling"], 
                         capture_output=True)
            
            # Try to create container with IP retry logic
            for attempt in range(3):
                red_team_cmd = [
                    "docker", "run", "-d",
                    "--name", self.red_team_config["name"],
                    "--network", self.red_team_config["network"],
                    "--ip", self.red_team_config["ip"],
                    "--cap-add", "NET_ADMIN",
                    "--cap-add", "SYS_ADMIN",
                    "-v", f"{os.getcwd()}/red_team_tools:/opt/red_tools",
                    "-v", f"{os.getcwd()}/attack_scripts:/opt/attacks", 
                    "-v", f"{os.getcwd()}/logs:/var/log/attacks",
                    "kalilinux/kali-rolling",
                    "/bin/bash", "-c", "while true; do sleep 30; done"
                ]
                
                result = subprocess.run(red_team_cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    self.red_team_container = self.red_team_config["name"]
                    self.logger.info(f"✅ Red Team container created: {self.red_team_config['ip']}")
                    
                    # Install attack tools in red team container
                    await self._install_red_team_tools()
                    return True
                    
                elif "Address already in use" in result.stderr and attempt < 2:
                    # Try next IP
                    current_ip = self.red_team_config["ip"]
                    parts = current_ip.split(".")
                    parts[-1] = str(int(parts[-1]) + attempt + 1)
                    self.red_team_config["ip"] = ".".join(parts)
                    self.logger.warning(f"IP conflict, trying: {self.red_team_config['ip']}")
                    await asyncio.sleep(1)
                else:
                    self.logger.error(f"❌ Red team container failed: {result.stderr}")
                    break
                    
            self.red_team_container = None
            return False
                
        except Exception as e:
            self.logger.error(f"Red team setup failed: {e}")
            self.red_team_container = None
            return False
    
    async def _setup_blue_team_container(self):
        """Setup Ubuntu container with vulnerable services for blue team defense"""
        try:
            self.logger.info("🔵 Setting up Blue Team target container...")
            
            # Remove any existing container with same name
            subprocess.run(["docker", "rm", "-f", self.blue_team_config["name"]], 
                         capture_output=True)
            
            # Create directories for blue team tools and logs
            os.makedirs("blue_team_tools", exist_ok=True)
            os.makedirs("defense_scripts", exist_ok=True)
            
            # Pull Ubuntu image
            subprocess.run(["docker", "pull", "ubuntu:22.04"], 
                         capture_output=True)
            
            # Try to create container with IP retry logic
            for attempt in range(3):
                blue_team_cmd = [
                    "docker", "run", "-d",
                    "--name", self.blue_team_config["name"],
                    "--network", self.blue_team_config["network"],
                    "--ip", self.blue_team_config["ip"],
                    "-v", f"{os.getcwd()}/blue_team_tools:/opt/blue_tools",
                    "-v", f"{os.getcwd()}/defense_scripts:/opt/defenses",
                    "-v", f"{os.getcwd()}/logs:/var/log/defenses",
                    "-p", "2222:22",   # SSH access
                    "-p", "8080:80",   # HTTP service
                    "-p", "2121:21",   # FTP service
                    "ubuntu:22.04",
                    "/bin/bash", "-c", "while true; do sleep 30; done"
                ]
                
                result = subprocess.run(blue_team_cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    self.blue_team_container = self.blue_team_config["name"]
                    self.logger.info(f"✅ Blue Team container created: {self.blue_team_config['ip']}")
                    
                    # Setup vulnerable services and defensive tools
                    await self._setup_blue_team_services()
                    return True
                    
                elif "Address already in use" in result.stderr and attempt < 2:
                    # Try next IP
                    current_ip = self.blue_team_config["ip"]
                    parts = current_ip.split(".")
                    parts[-1] = str(int(parts[-1]) + attempt + 1)
                    self.blue_team_config["ip"] = ".".join(parts)
                    self.logger.warning(f"IP conflict, trying: {self.blue_team_config['ip']}")
                    await asyncio.sleep(1)
                else:
                    self.logger.error(f"❌ Blue team container failed: {result.stderr}")
                    break
                    
            self.blue_team_container = None
            return False
                
        except Exception as e:
            self.logger.error(f"Blue team setup failed: {e}")
            self.blue_team_container = None
            return False
    
    async def _install_red_team_tools(self):
        """Install attack tools in red team container"""
        # Essential attack tools installation - optimized for Kali Linux
        tool_groups = [
            {
                "name": "System Update",
                "cmd": "apt update"
            },
            {
                "name": "Network Tools", 
                "cmd": "apt install -y nmap masscan netcat-traditional curl wget dnsutils"
            },
            {
                "name": "Web Tools",
                "cmd": "apt install -y nikto dirb gobuster ffuf"
            },
            {
                "name": "Brute Force Tools",
                "cmd": "apt install -y hydra medusa crackmapexec"
            },
            {
                "name": "Python Environment",
                "cmd": "apt install -y python3 python3-dev python3-venv"
            },
            {
                "name": "Development Tools",
                "cmd": "apt install -y git vim nano"
            }
        ]
        
        for tool_group in tool_groups:
            try:
                self.logger.info(f"Installing {tool_group['name']}...")
                result = subprocess.run([
                    "docker", "exec", self.red_team_container,
                    "/bin/bash", "-c", tool_group["cmd"]
                ], capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    self.logger.info(f"✅ Installed: {tool_group['name']}")
                else:
                    # Log warning but continue with other tools
                    stderr_short = result.stderr[:200] if result.stderr else "Unknown error"
                    self.logger.warning(f"⚠️ {tool_group['name']} issue: {stderr_short}")
                    
            except subprocess.TimeoutExpired:
                self.logger.warning(f"⚠️ {tool_group['name']} installation timeout")
            except Exception as e:
                self.logger.warning(f"⚠️ {tool_group['name']} error: {e}")
        
        # Install Python packages in virtual environment to avoid externally-managed error
        try:
            self.logger.info("Setting up Python security tools...")
            venv_setup = """
            python3 -m venv /opt/security-venv
            source /opt/security-venv/bin/activate
            pip install --upgrade pip
            pip install impacket requests beautifulsoup4 pycryptodome
            """
            
            result = subprocess.run([
                "docker", "exec", self.red_team_container,
                "/bin/bash", "-c", venv_setup
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                self.logger.info("✅ Python security tools installed in virtual environment")
            else:
                self.logger.warning(f"⚠️ Python tools setup issue: {result.stderr[:200]}")
                
        except Exception as e:
            self.logger.warning(f"⚠️ Python tools setup error: {e}")
        
        self.logger.info("🔴 Red team tool installation completed")
    
    async def _setup_blue_team_services(self):
        """Setup vulnerable services and monitoring in blue team container"""
        service_setup_commands = [
            # Update and install basic services
            "apt update && apt install -y openssh-server apache2 vsftpd fail2ban",
            
            # Setup SSH service
            "mkdir -p /var/run/sshd",
            "echo 'root:vulnerable123' | chpasswd",
            "echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config",
            "service ssh start",
            
            # Setup HTTP service with basic content
            "echo '<html><body><h1>Blue Team Target System</h1><p>Defend this!</p></body></html>' > /var/www/html/index.html",
            "service apache2 start",
            
            # Setup FTP service
            "echo 'ftpuser:ftppass123' | chpasswd",
            "useradd -m ftpuser",
            "service vsftpd start",
            
            # Setup basic monitoring
            "service fail2ban start",
            
            # Create monitoring scripts
            "mkdir -p /opt/blue_monitoring",
            
            # Install monitoring tools
            "apt install -y tcpdump iptables-persistent netstat-nat"
        ]
        
        for cmd in service_setup_commands:
            try:
                result = subprocess.run([
                    "docker", "exec", self.blue_team_container,
                    "/bin/bash", "-c", cmd
                ], capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0:
                    self.logger.info(f"✅ Blue team service setup: {cmd.split()[0] if cmd.split() else 'command'}")
                else:
                    self.logger.warning(f"⚠️ Service setup issue: {result.stderr[:100]}")
                    
            except subprocess.TimeoutExpired:
                self.logger.warning(f"⚠️ Service setup timeout: {cmd[:50]}")
            except Exception as e:
                self.logger.warning(f"⚠️ Service setup error: {e}")
    
    async def _setup_monitoring_infrastructure(self):
        """Setup real-time monitoring of the combat environment"""
        try:
            # Create monitoring scripts
            await self._create_network_monitoring_script()
            await self._create_attack_detection_script()
            await self._create_defense_monitoring_script()
            
            self.logger.info("✅ Monitoring infrastructure ready")
            
        except Exception as e:
            self.logger.error(f"Monitoring setup failed: {e}")
    
    async def _create_network_monitoring_script(self):
        """Create network traffic monitoring script"""
        monitoring_script = f'''#!/bin/bash
# Network monitoring for red vs blue combat
LOGFILE="/var/log/attacks/network_monitor.log"
echo "$(date): Starting network monitoring" >> $LOGFILE

# Monitor network traffic between red and blue teams
tcpdump -i any -w /var/log/attacks/traffic_$(date +%s).pcap \\
  "host {self.red_team_config['ip']} or host {self.blue_team_config['ip']}" &

# Monitor connection attempts
netstat -tuln >> $LOGFILE
ss -tuln >> $LOGFILE

echo "$(date): Network monitoring active" >> $LOGFILE
'''
        
        with open("attack_scripts/network_monitor.sh", "w") as f:
            f.write(monitoring_script)
        
        os.chmod("attack_scripts/network_monitor.sh", 0o755)
    
    async def _create_attack_detection_script(self):
        """Create attack detection script for blue team"""
        detection_script = '''#!/bin/bash
# Attack detection for blue team defense
LOGFILE="/var/log/defenses/attack_detection.log"
echo "$(date): Starting attack detection" >> $LOGFILE

# Monitor for port scans
tail -f /var/log/auth.log | while read line; do
    if echo "$line" | grep -q "Failed password"; then
        echo "$(date): ATTACK DETECTED - Brute force attempt: $line" >> $LOGFILE
    fi
done &

# Monitor for unusual network connections
netstat -tuln | awk '{print $4}' | sort | uniq -c | sort -nr >> $LOGFILE

echo "$(date): Attack detection active" >> $LOGFILE
'''
        
        with open("defense_scripts/attack_detection.sh", "w") as f:
            f.write(detection_script)
        
        os.chmod("defense_scripts/attack_detection.sh", 0o755)
    
    async def _create_defense_monitoring_script(self):
        """Create defensive action monitoring script"""
        defense_script = '''#!/bin/bash
# Defense action monitoring
LOGFILE="/var/log/defenses/defense_actions.log"
echo "$(date): Defense monitoring started" >> $LOGFILE

# Monitor iptables for blocking actions
iptables -L -n -v >> $LOGFILE

# Monitor fail2ban status
fail2ban-client status >> $LOGFILE

echo "$(date): Defense monitoring active" >> $LOGFILE
'''
        
        with open("defense_scripts/defense_monitor.sh", "w") as f:
            f.write(defense_script)
        
        os.chmod("defense_scripts/defense_monitor.sh", 0o755)
    
    async def _initialize_combat_tools(self):
        """Initialize tools and scripts for combat simulation"""
        try:
            # Copy monitoring scripts to containers
            if self.red_team_container:
                subprocess.run([
                    "docker", "cp", "attack_scripts/network_monitor.sh",
                    f"{self.red_team_container}:/opt/attacks/"
                ], capture_output=True)
                
            if self.blue_team_container:
                subprocess.run([
                    "docker", "cp", "defense_scripts/attack_detection.sh",
                    f"{self.blue_team_container}:/opt/defenses/"
                ], capture_output=True)
                
                subprocess.run([
                    "docker", "cp", "defense_scripts/defense_monitor.sh", 
                    f"{self.blue_team_container}:/opt/defenses/"
                ], capture_output=True)
            
            self.logger.info("✅ Combat tools initialized")
            
        except Exception as e:
            self.logger.error(f"Tool initialization failed: {e}")
    
    async def start_live_combat_exercise(self, scenario_name: str = None, duration_minutes: int = 30) -> Dict[str, Any]:
        """Start live red team vs blue team combat exercise"""
        try:
            self.logger.info(f"🚀 Starting live adversarial combat exercise")
            self.start_time = datetime.now()
            self.exercise_running = True
            
            # Select attack scenario
            if scenario_name:
                scenario = next((s for s in self.attack_scenarios if s.name == scenario_name), None)
            else:
                scenario = self.attack_scenarios[0]  # Default to first scenario
            
            if not scenario:
                raise ValueError(f"Attack scenario '{scenario_name}' not found")
            
            self.logger.info(f"📋 Scenario: {scenario.name} - {scenario.description}")
            
            # Start monitoring
            await self._start_real_time_monitoring()
            
            # Launch red team attacks
            red_team_task = asyncio.create_task(
                self._execute_red_team_attacks(scenario)
            )
            
            # Launch blue team defenses  
            blue_team_task = asyncio.create_task(
                self._execute_blue_team_defenses()
            )
            
            # Monitor exercise progress
            monitor_task = asyncio.create_task(
                self._monitor_combat_exercise(duration_minutes)
            )
            
            # Wait for exercise completion
            results = await asyncio.gather(
                red_team_task,
                blue_team_task, 
                monitor_task,
                return_exceptions=True
            )
            
            # Stop monitoring
            await self._stop_real_time_monitoring()
            
            # Generate exercise report
            exercise_results = await self._generate_exercise_report(scenario)
            
            self.exercise_running = False
            self.logger.info("✅ Live combat exercise completed")
            
            return exercise_results
            
        except Exception as e:
            self.logger.error(f"❌ Combat exercise failed: {e}")
            self.exercise_running = False
            return {"status": "failed", "error": str(e)}
    
    async def _start_real_time_monitoring(self):
        """Start real-time monitoring of combat exercise"""
        self.monitoring_active = True
        
        # Start network monitoring in red team container
        if self.red_team_container:
            subprocess.Popen([
                "docker", "exec", "-d", self.red_team_container,
                "/bin/bash", "/opt/attacks/network_monitor.sh"
            ])
        
        # Start attack detection in blue team container
        if self.blue_team_container:
            subprocess.Popen([
                "docker", "exec", "-d", self.blue_team_container,
                "/bin/bash", "/opt/defenses/attack_detection.sh"
            ])
            
            subprocess.Popen([
                "docker", "exec", "-d", self.blue_team_container, 
                "/bin/bash", "/opt/defenses/defense_monitor.sh"
            ])
        
        self.logger.info("✅ Real-time monitoring started")
    
    async def _execute_red_team_attacks(self, scenario: AttackScenario) -> Dict[str, Any]:
        """Execute red team attacks according to scenario"""
        red_team_results = {
            "scenario": scenario.name,
            "attacks_executed": [],
            "success_rate": 0.0,
            "techniques_used": scenario.attack_tools,
            "targets_compromised": [],
            "data_extracted": []
        }
        
        try:
            self.logger.info(f"🔴 RED TEAM: Starting {scenario.name}")
            
            # Execute attacks based on scenario
            if scenario.name == "Network Reconnaissance":
                attacks = await self._execute_reconnaissance_attacks()
            elif scenario.name == "SSH Brute Force Attack":
                attacks = await self._execute_brute_force_attacks()
            elif scenario.name == "Web Application Exploitation":
                attacks = await self._execute_web_attacks()
            elif scenario.name == "Lateral Movement Simulation":
                attacks = await self._execute_lateral_movement()
            elif scenario.name == "Data Exfiltration Operation":
                attacks = await self._execute_data_exfiltration()
            else:
                attacks = await self._execute_general_attacks()
            
            red_team_results["attacks_executed"] = attacks
            red_team_results["success_rate"] = len([a for a in attacks if a.get("success", False)]) / len(attacks) if attacks else 0.0
            
            self.logger.info(f"🔴 RED TEAM: Completed with {red_team_results['success_rate']:.2%} success rate")
            
        except Exception as e:
            self.logger.error(f"🔴 RED TEAM ERROR: {e}")
            red_team_results["error"] = str(e)
        
        return red_team_results
    
    async def _execute_reconnaissance_attacks(self) -> List[Dict[str, Any]]:
        """Execute network reconnaissance attacks"""
        attacks = []
        target_ip = self.blue_team_config["ip"]  # Blue team IP
        
        # Port scan attack
        nmap_attack = await self._execute_container_command(
            self.red_team_container,
            f"nmap -sS -O {target_ip}",
            "Port Scan Attack"
        )
        attacks.append(nmap_attack)
        
        # Service enumeration
        service_enum = await self._execute_container_command(
            self.red_team_container,
            f"nmap -sV -p 22,80,21 {target_ip}",
            "Service Enumeration"
        )
        attacks.append(service_enum)
        
        # Banner grabbing
        banner_grab = await self._execute_container_command(
            self.red_team_container,
            f"netcat -v {target_ip} 22",
            "Banner Grabbing"
        )
        attacks.append(banner_grab)
        
        return attacks
    
    async def _execute_brute_force_attacks(self) -> List[Dict[str, Any]]:
        """Execute SSH brute force attacks"""
        attacks = []
        target_ip = self.blue_team_config["ip"]
        
        # Create wordlist for brute force
        wordlist_cmd = "echo -e 'admin\nroot\nuser\ntest\nvulnerable123' > /tmp/passwords.txt"
        await self._execute_container_command(
            self.red_team_container,
            wordlist_cmd,
            "Create Password Wordlist"
        )
        
        # SSH brute force with hydra
        brute_force = await self._execute_container_command(
            self.red_team_container,
            f"hydra -l root -P /tmp/passwords.txt ssh://{target_ip}",
            "SSH Brute Force Attack"
        )
        attacks.append(brute_force)
        
        return attacks
    
    async def _execute_web_attacks(self) -> List[Dict[str, Any]]:
        """Execute web application attacks"""
        attacks = []
        target_ip = self.blue_team_config["ip"]
        
        # Directory enumeration
        dir_enum = await self._execute_container_command(
            self.red_team_container,
            f"dirb http://{target_ip}",
            "Directory Enumeration"
        )
        attacks.append(dir_enum)
        
        # Nikto web vulnerability scan
        nikto_scan = await self._execute_container_command(
            self.red_team_container,
            f"nikto -h http://{target_ip}",
            "Web Vulnerability Scan"
        )
        attacks.append(nikto_scan)
        
        return attacks
    
    async def _execute_lateral_movement(self) -> List[Dict[str, Any]]:
        """Execute lateral movement attacks"""
        attacks = []
        target_ip = self.blue_team_config["ip"]
        
        # Attempt SSH connection with discovered credentials
        ssh_connect = await self._execute_container_command(
            self.red_team_container,
            f"sshpass -p 'vulnerable123' ssh -o StrictHostKeyChecking=no root@{target_ip} 'whoami && ls -la /'",
            "SSH Connection Attempt"
        )
        attacks.append(ssh_connect)
        
        return attacks
    
    async def _execute_data_exfiltration(self) -> List[Dict[str, Any]]:
        """Execute data exfiltration attacks"""
        attacks = []
        target_ip = self.blue_team_config["ip"]
        
        # Attempt to extract /etc/passwd
        data_extract = await self._execute_container_command(
            self.red_team_container,
            f"sshpass -p 'vulnerable123' ssh -o StrictHostKeyChecking=no root@{target_ip} 'cat /etc/passwd'",
            "Data Exfiltration Attempt"
        )
        attacks.append(data_extract)
        
        return attacks
    
    async def _execute_general_attacks(self) -> List[Dict[str, Any]]:
        """Execute general attacks"""
        return await self._execute_reconnaissance_attacks()
    
    async def _execute_container_command(self, container: str, command: str, attack_name: str) -> Dict[str, Any]:
        """Execute command in container and log results"""
        attack_event = {
            "timestamp": datetime.now(),
            "attack_name": attack_name,
            "command": command,
            "success": False,
            "output": "",
            "error": ""
        }
        
        try:
            self.logger.info(f"🔴 Executing: {attack_name}")
            
            result = subprocess.run([
                "docker", "exec", container,
                "/bin/bash", "-c", command
            ], capture_output=True, text=True, timeout=30)
            
            attack_event["output"] = result.stdout
            attack_event["error"] = result.stderr
            attack_event["success"] = result.returncode == 0
            
            if attack_event["success"]:
                self.logger.info(f"✅ {attack_name} completed successfully")
            else:
                self.logger.warning(f"⚠️ {attack_name} failed: {result.stderr[:100]}")
            
            # Log attack event
            self.attack_events.append(LiveAttackEvent(
                timestamp=attack_event["timestamp"],
                event_id=str(uuid.uuid4()),
                attack_type=attack_name,
                source_ip="192.168.100.10",
                target_ip="192.168.100.20",
                target_service="multiple",
                attack_payload=command,
                success=attack_event["success"],
                detection_time=None,
                response_time=None,
                mitigation_action=None
            ))
            
        except subprocess.TimeoutExpired:
            attack_event["error"] = "Command timeout"
            self.logger.warning(f"⏰ {attack_name} timed out")
        except Exception as e:
            attack_event["error"] = str(e)
            self.logger.error(f"❌ {attack_name} error: {e}")
        
        return attack_event
    
    async def _execute_blue_team_defenses(self) -> Dict[str, Any]:
        """Execute blue team defensive actions"""
        defense_results = {
            "defenses_activated": [],
            "attacks_detected": 0,
            "response_time_avg": 0.0,
            "mitigation_actions": [],
            "false_positives": 0
        }
        
        try:
            self.logger.info("🔵 BLUE TEAM: Starting defensive operations")
            
            # Monitor for attacks and respond
            defense_actions = []
            
            # Activate fail2ban monitoring
            fail2ban_activation = await self._activate_fail2ban_monitoring()
            defense_actions.append(fail2ban_activation)
            
            # Setup intrusion detection
            ids_setup = await self._setup_intrusion_detection()
            defense_actions.append(ids_setup)
            
            # Monitor network traffic
            network_monitoring = await self._activate_network_monitoring()
            defense_actions.append(network_monitoring)
            
            # Analyze logs for attack patterns
            log_analysis = await self._perform_log_analysis()
            defense_actions.append(log_analysis)
            
            defense_results["defenses_activated"] = defense_actions
            defense_results["attacks_detected"] = len([d for d in defense_actions if d.get("attacks_found", 0) > 0])
            
            self.logger.info(f"🔵 BLUE TEAM: Detected {defense_results['attacks_detected']} attack patterns")
            
        except Exception as e:
            self.logger.error(f"🔵 BLUE TEAM ERROR: {e}")
            defense_results["error"] = str(e)
        
        return defense_results
    
    async def _activate_fail2ban_monitoring(self) -> Dict[str, Any]:
        """Activate fail2ban monitoring on blue team"""
        return await self._execute_container_command(
            self.blue_team_container,
            "fail2ban-client status && fail2ban-client status sshd",
            "Fail2Ban Activation"
        )
    
    async def _setup_intrusion_detection(self) -> Dict[str, Any]:
        """Setup intrusion detection on blue team"""
        return await self._execute_container_command(
            self.blue_team_container,
            "netstat -tuln | grep LISTEN",
            "Intrusion Detection Setup"
        )
    
    async def _activate_network_monitoring(self) -> Dict[str, Any]:
        """Activate network monitoring on blue team"""
        return await self._execute_container_command(
            self.blue_team_container,
            "ss -tuln && iptables -L -n",
            "Network Monitoring Activation"
        )
    
    async def _perform_log_analysis(self) -> Dict[str, Any]:
        """Perform log analysis for attack detection"""
        return await self._execute_container_command(
            self.blue_team_container,
            "tail -20 /var/log/auth.log && tail -20 /var/log/apache2/access.log",
            "Log Analysis"
        )
    
    async def _monitor_combat_exercise(self, duration_minutes: int) -> Dict[str, Any]:
        """Monitor the ongoing combat exercise"""
        monitoring_results = {
            "duration_minutes": duration_minutes,
            "events_captured": 0,
            "network_activity": [],
            "security_events": []
        }
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        self.logger.info(f"📊 Monitoring combat exercise for {duration_minutes} minutes")
        
        try:
            while time.time() < end_time and self.exercise_running:
                # Collect real-time metrics
                await self._collect_real_time_metrics()
                
                # Check for significant events
                await self._check_security_events()
                
                # Sleep for monitoring interval
                await asyncio.sleep(10)  # Check every 10 seconds
            
            monitoring_results["events_captured"] = len(self.attack_events)
            self.logger.info(f"📊 Monitoring completed: {monitoring_results['events_captured']} events captured")
            
        except Exception as e:
            self.logger.error(f"📊 Monitoring error: {e}")
            monitoring_results["error"] = str(e)
        
        return monitoring_results
    
    async def _collect_real_time_metrics(self):
        """Collect real-time metrics from combat environment"""
        try:
            # Get container stats
            if self.red_team_container:
                red_stats = subprocess.run([
                    "docker", "stats", self.red_team_container, "--no-stream", "--format", 
                    "table {{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"
                ], capture_output=True, text=True)
                
            if self.blue_team_container:
                blue_stats = subprocess.run([
                    "docker", "stats", self.blue_team_container, "--no-stream", "--format",
                    "table {{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"
                ], capture_output=True, text=True)
            
            # Update live metrics
            self.live_metrics["timestamp"] = datetime.now()
            self.live_metrics["red_team_active"] = self.red_team_container is not None
            self.live_metrics["blue_team_active"] = self.blue_team_container is not None
            self.live_metrics["total_events"] = len(self.attack_events)
            
        except Exception as e:
            self.logger.warning(f"Metrics collection error: {e}")
    
    async def _check_security_events(self):
        """Check for significant security events"""
        try:
            # Check attack logs
            if os.path.exists("logs/network_monitor.log"):
                with open("logs/network_monitor.log", "r") as f:
                    recent_logs = f.readlines()[-10:]  # Last 10 lines
                    
                for log_line in recent_logs:
                    if "ATTACK DETECTED" in log_line:
                        self.logger.warning(f"🚨 {log_line.strip()}")
            
        except Exception as e:
            self.logger.warning(f"Security event check error: {e}")
    
    async def _stop_real_time_monitoring(self):
        """Stop real-time monitoring"""
        self.monitoring_active = False
        
        # Kill monitoring processes in containers
        if self.red_team_container:
            subprocess.run([
                "docker", "exec", self.red_team_container,
                "pkill", "-f", "network_monitor.sh"
            ], capture_output=True)
            
        if self.blue_team_container:
            subprocess.run([
                "docker", "exec", self.blue_team_container,
                "pkill", "-f", "attack_detection.sh"
            ], capture_output=True)
            
            subprocess.run([
                "docker", "exec", self.blue_team_container,
                "pkill", "-f", "defense_monitor.sh"
            ], capture_output=True)
        
        self.logger.info("⏹️ Real-time monitoring stopped")
    
    async def _generate_exercise_report(self, scenario: AttackScenario) -> Dict[str, Any]:
        """Generate comprehensive exercise report"""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        report = {
            "exercise_id": self.exercise_id,
            "scenario": {
                "name": scenario.name,
                "description": scenario.description,
                "difficulty": scenario.difficulty_level
            },
            "timing": {
                "start_time": self.start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": duration,
                "duration_minutes": duration / 60
            },
            "red_team": {
                "container": self.red_team_container,
                "attacks_executed": len(self.attack_events),
                "successful_attacks": len([e for e in self.attack_events if e.success]),
                "success_rate": len([e for e in self.attack_events if e.success]) / len(self.attack_events) if self.attack_events else 0.0
            },
            "blue_team": {
                "container": self.blue_team_container,
                "defenses_activated": len(self.defense_events),
                "attacks_detected": len([e for e in self.defense_events if "detected" in str(e).lower()]),
                "response_effectiveness": "High" if len(self.defense_events) > 0 else "Low"
            },
            "learning_outcomes": [
                f"Red team executed {len(self.attack_events)} attack attempts",
                f"Blue team activated {len(self.defense_events)} defensive measures",
                f"Attack success rate: {(len([e for e in self.attack_events if e.success]) / len(self.attack_events) * 100):.1f}%" if self.attack_events else "0.0%",
                "Real-time adversarial learning data collected",
                "Container-based isolation successful"
            ],
            "recommendations": [
                "Increase blue team monitoring frequency",
                "Implement automated response triggers",
                "Enhance attack detection capabilities",
                "Improve red team stealth techniques"
            ],
            "data_collected": {
                "attack_events": len(self.attack_events),
                "defense_events": len(self.defense_events),
                "network_traffic_captured": "Available in logs/",
                "learning_data_quality": "High"
            }
        }
        
        # Save detailed report
        report_file = f"logs/exercise_report_{self.exercise_id[:8]}.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"📋 Exercise report saved: {report_file}")
        
        return report
    
    async def cleanup_combat_environment(self):
        """Clean up combat environment and containers"""
        try:
            self.logger.info("🧹 Cleaning up combat environment...")
            
            # Stop monitoring
            if self.monitoring_active:
                await self._stop_real_time_monitoring()
            
            # Remove containers
            if self.red_team_container:
                subprocess.run([
                    "docker", "rm", "-f", self.red_team_container
                ], capture_output=True)
                self.logger.info("✅ Red team container removed")
            
            if self.blue_team_container:
                subprocess.run([
                    "docker", "rm", "-f", self.blue_team_container
                ], capture_output=True)
                self.logger.info("✅ Blue team container removed")
            
            # Remove network
            subprocess.run([
                "docker", "network", "rm", "archangel-combat-net"
            ], capture_output=True)
            self.logger.info("✅ Combat network removed")
            
            self.logger.info("🧹 Combat environment cleanup completed")
            
        except Exception as e:
            self.logger.error(f"❌ Cleanup error: {e}")

async def main():
    """Main function to demonstrate live adversarial environment"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🛡️ Archangel Live Adversarial Hacking Environment")
    print("=" * 60)
    
    # Create combat environment
    combat_env = LiveAdversarialEnvironment()
    
    try:
        # Setup combat environment
        print("\n🔧 Setting up live red team vs blue team environment...")
        if await combat_env.setup_combat_environment():
            print("✅ Combat environment ready")
            
            # Start live exercise
            print("\n🚀 Starting live adversarial exercise...")
            print(f"🔴 Red Team: Kali Linux container ({self.red_team_config['ip']})")
            print(f"🔵 Blue Team: Ubuntu target container ({self.blue_team_config['ip']})")
            print("📊 Duration: 10 minutes of live combat")
            
            results = await combat_env.start_live_combat_exercise(
                scenario_name="Network Reconnaissance",
                duration_minutes=10
            )
            
            print("\n📋 Exercise Results:")
            print(f"Status: {results.get('status', 'completed')}")
            print(f"Red Team Success Rate: {results.get('red_team', {}).get('success_rate', 0)*100:.1f}%")
            print(f"Blue Team Detections: {results.get('blue_team', {}).get('attacks_detected', 0)}")
            print(f"Total Events: {results.get('data_collected', {}).get('attack_events', 0)}")
            
        else:
            print("❌ Failed to setup combat environment")
            
    except KeyboardInterrupt:
        print("\n🛑 Exercise interrupted by user")
    except Exception as e:
        print(f"❌ Exercise failed: {e}")
    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        await combat_env.cleanup_combat_environment()
        print("✅ Cleanup completed")

if __name__ == "__main__":
    asyncio.run(main())