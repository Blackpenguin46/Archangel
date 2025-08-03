#!/usr/bin/env python3
"""
AI-Controlled Blue Team Agent running inside Ubuntu SOC container
Monitors, detects, and responds to red team activities using enterprise security tools
"""

import asyncio
import subprocess
import json
import time
import os
import sys
import psutil
from datetime import datetime
from typing import Dict, List, Any
import logging
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - BLUE TEAM - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/app/logs/blue_team_container.log')
    ]
)
logger = logging.getLogger(__name__)

class UbuntuBlueTeamAgent:
    """AI-controlled blue team agent with enterprise security tools"""
    
    def __init__(self):
        self.agent_id = "ubuntu_blue_team_001"
        self.monitored_network = "172.20.0.0/24"
        self.detected_threats = []
        self.active_incidents = []
        self.security_tools = self._detect_security_tools()
        
        logger.info(f"🔵 Blue Team Agent {self.agent_id} initialized in Ubuntu SOC")
        logger.info(f"Available security tools: {', '.join(self.security_tools)}")
    
    def _detect_security_tools(self) -> List[str]:
        """Detect which security tools are available"""
        tools = []
        tool_commands = {
            'tcpdump': 'tcpdump --version',
            'netstat': 'netstat --version',
            'ss': 'ss --version',
            'iptables': 'iptables --version',
            'fail2ban': 'fail2ban-client --version',
            'rkhunter': 'rkhunter --version',
            'chkrootkit': 'chkrootkit -V',
            'clamav': 'clamscan --version'
        }
        
        for tool, cmd in tool_commands.items():
            try:
                subprocess.run(cmd.split(), capture_output=True, check=True, timeout=5)
                tools.append(tool)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
                pass
        
        return tools
    
    async def run_autonomous_operations(self):
        """Main loop for autonomous blue team operations"""
        logger.info("🚀 Starting autonomous blue team SOC operations...")
        
        # Start continuous monitoring
        monitoring_tasks = [
            self._monitor_network_connections(),
            self._monitor_system_logs(),
            self._monitor_process_activity(),
            self._monitor_file_integrity(),
            self._analyze_threat_intelligence()
        ]
        
        try:
            await asyncio.gather(*monitoring_tasks)
        except KeyboardInterrupt:
            logger.info("🛑 Blue team monitoring shutting down")
        except Exception as e:
            logger.error(f"❌ Monitoring error: {e}")
    
    async def _monitor_network_connections(self):
        """Monitor network connections for suspicious activity"""
        logger.info("🌐 Starting network connection monitoring")
        
        while True:
            try:
                # Get current network connections
                connections = self._get_network_connections()
                
                # Analyze for suspicious patterns
                threats = await self._analyze_network_threats(connections)
                
                if threats:
                    await self._handle_network_threats(threats)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Network monitoring error: {e}")
                await asyncio.sleep(60)
    
    def _get_network_connections(self) -> List[Dict]:
        """Get current network connections"""
        connections = []
        
        try:
            # Use netstat to get connections
            result = subprocess.run(
                ['netstat', '-tuln'], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            
            lines = result.stdout.split('\n')
            for line in lines[2:]:  # Skip headers
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 4:
                        connections.append({
                            'protocol': parts[0],
                            'local_address': parts[3],
                            'foreign_address': parts[4] if len(parts) > 4 else '',
                            'state': parts[5] if len(parts) > 5 else '',
                            'timestamp': datetime.now().isoformat()
                        })
        
        except Exception as e:
            logger.error(f"Failed to get network connections: {e}")
        
        return connections
    
    async def _analyze_network_threats(self, connections: List[Dict]) -> List[Dict]:
        """Analyze network connections for threats"""
        threats = []
        
        for conn in connections:
            # Check for suspicious port scanning patterns
            if self._detect_port_scan_pattern(conn):
                threats.append({
                    'type': 'port_scan',
                    'severity': 'high',
                    'source': conn['foreign_address'],
                    'target': conn['local_address'],
                    'evidence': conn,
                    'timestamp': datetime.now().isoformat()
                })
            
            # Check for suspicious connections
            if self._detect_suspicious_connection(conn):
                threats.append({
                    'type': 'suspicious_connection',
                    'severity': 'medium',
                    'connection': conn,
                    'timestamp': datetime.now().isoformat()
                })
        
        return threats
    
    def _detect_port_scan_pattern(self, connection: Dict) -> bool:
        """Detect potential port scanning activity"""
        # Simple heuristic: multiple connections from same source to different ports
        foreign_addr = connection.get('foreign_address', '')
        if foreign_addr and ':' in foreign_addr:
            source_ip = foreign_addr.split(':')[0]
            
            # Check if we've seen this IP connecting to multiple ports recently
            recent_connections = [t for t in self.detected_threats 
                               if t.get('source', '').startswith(source_ip)]
            
            return len(recent_connections) > 5  # Threshold for port scan detection
        
        return False
    
    def _detect_suspicious_connection(self, connection: Dict) -> bool:
        """Detect suspicious network connections"""
        foreign_addr = connection.get('foreign_address', '')
        local_addr = connection.get('local_address', '')
        
        # Check for connections on unusual ports
        if ':' in local_addr:
            port = local_addr.split(':')[-1]
            suspicious_ports = ['1234', '4444', '5555', '6666', '31337']
            if port in suspicious_ports:
                return True
        
        # Check for external connections during off-hours
        current_hour = datetime.now().hour
        if current_hour < 6 or current_hour > 22:  # Outside business hours
            if foreign_addr and not foreign_addr.startswith('127.') and not foreign_addr.startswith('172.20.'):
                return True
        
        return False
    
    async def _handle_network_threats(self, threats: List[Dict]):
        """Handle detected network threats"""
        for threat in threats:
            logger.warning(f"🚨 THREAT DETECTED: {threat['type']} - Severity: {threat['severity']}")
            
            # Add to detected threats
            self.detected_threats.append(threat)
            
            # Take defensive action based on threat type
            if threat['type'] == 'port_scan':
                await self._block_suspicious_ip(threat['source'])
            elif threat['type'] == 'suspicious_connection':
                await self._investigate_connection(threat['connection'])
            
            # Log to security events
            await self._log_security_event(threat)
    
    async def _block_suspicious_ip(self, source_ip: str):
        """Block suspicious IP using iptables"""
        try:
            if source_ip and ':' in source_ip:
                ip = source_ip.split(':')[0]
                
                # Add iptables rule to block the IP
                cmd = ['iptables', '-I', 'INPUT', '-s', ip, '-j', 'DROP']
                
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                await process.communicate()
                logger.info(f"🛡️ Blocked suspicious IP: {ip}")
                
        except Exception as e:
            logger.error(f"Failed to block IP {source_ip}: {e}")
    
    async def _monitor_system_logs(self):
        """Monitor system logs for security events"""
        logger.info("📋 Starting system log monitoring")
        
        while True:
            try:
                # Monitor auth logs for failed login attempts
                auth_events = await self._analyze_auth_logs()
                
                if auth_events:
                    await self._handle_auth_events(auth_events)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Log monitoring error: {e}")
                await asyncio.sleep(120)
    
    async def _analyze_auth_logs(self) -> List[Dict]:
        """Analyze authentication logs"""
        events = []
        
        try:
            # Read recent auth log entries
            if os.path.exists('/var/log/auth.log'):
                with open('/var/log/auth.log', 'r') as f:
                    lines = f.readlines()[-100:]  # Last 100 lines
                
                for line in lines:
                    if 'Failed password' in line or 'authentication failure' in line:
                        events.append({
                            'type': 'failed_authentication',
                            'severity': 'medium',
                            'log_entry': line.strip(),
                            'timestamp': datetime.now().isoformat()
                        })
            
        except Exception as e:
            logger.error(f"Failed to analyze auth logs: {e}")
        
        return events
    
    async def _handle_auth_events(self, events: List[Dict]):
        """Handle authentication security events"""
        for event in events:
            logger.warning(f"🔐 AUTH EVENT: {event['type']}")
            
            # Extract IP from log entry if possible
            ip_match = re.search(r'from (\d+\.\d+\.\d+\.\d+)', event['log_entry'])
            if ip_match:
                suspicious_ip = ip_match.group(1)
                await self._block_suspicious_ip(suspicious_ip)
            
            await self._log_security_event(event)
    
    async def _monitor_process_activity(self):
        """Monitor process activity for malicious behavior"""
        logger.info("⚙️ Starting process activity monitoring")
        
        known_processes = set()
        
        while True:
            try:
                current_processes = set()
                
                for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'username']):
                    try:
                        proc_info = proc.info
                        proc_signature = f"{proc_info['name']}_{proc_info['username']}"
                        current_processes.add(proc_signature)
                        
                        # Check for suspicious processes
                        if self._is_suspicious_process(proc_info):
                            threat = {
                                'type': 'suspicious_process',
                                'severity': 'high',
                                'process': proc_info,
                                'timestamp': datetime.now().isoformat()
                            }
                            await self._handle_process_threat(threat)
                    
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                
                # Detect new processes
                new_processes = current_processes - known_processes
                if new_processes:
                    logger.info(f"📈 New processes detected: {len(new_processes)}")
                
                known_processes = current_processes
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Process monitoring error: {e}")
                await asyncio.sleep(60)
    
    def _is_suspicious_process(self, proc_info: Dict) -> bool:
        """Check if a process is suspicious"""
        name = proc_info.get('name', '').lower()
        cmdline = ' '.join(proc_info.get('cmdline', [])).lower()
        
        # Check for known malicious process names
        suspicious_names = ['nc', 'netcat', 'ncat', 'socat', 'nmap', 'masscan', 'sqlmap']
        if any(sus_name in name for sus_name in suspicious_names):
            return True
        
        # Check for suspicious command line patterns
        suspicious_patterns = ['reverse shell', 'bind shell', '/bin/sh', '/bin/bash', 'perl -e', 'python -c']
        if any(pattern in cmdline for pattern in suspicious_patterns):
            return True
        
        return False
    
    async def _handle_process_threat(self, threat: Dict):
        """Handle suspicious process threat"""
        logger.warning(f"⚠️ SUSPICIOUS PROCESS: {threat['process']['name']}")
        
        # Kill suspicious process
        try:
            pid = threat['process']['pid']
            os.kill(pid, 9)  # SIGKILL
            logger.info(f"🔪 Terminated suspicious process PID {pid}")
        except Exception as e:
            logger.error(f"Failed to terminate process: {e}")
        
        await self._log_security_event(threat)
    
    async def _monitor_file_integrity(self):
        """Monitor file system for unauthorized changes"""
        logger.info("📁 Starting file integrity monitoring")
        
        # Monitor critical system directories
        critical_paths = ['/etc', '/bin', '/sbin', '/usr/bin', '/usr/sbin']
        
        while True:
            try:
                for path in critical_paths:
                    if os.path.exists(path):
                        # Simple file count check (in production would use proper FIM)
                        file_count = len(os.listdir(path))
                        logger.debug(f"📊 {path}: {file_count} files")
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"File integrity monitoring error: {e}")
                await asyncio.sleep(600)
    
    async def _analyze_threat_intelligence(self):
        """Analyze collected threat intelligence"""
        logger.info("🧠 Starting threat intelligence analysis")
        
        while True:
            try:
                # Analyze patterns in detected threats
                if len(self.detected_threats) > 10:
                    analysis = self._perform_threat_analysis()
                    if analysis:
                        logger.info(f"📊 Threat analysis: {analysis}")
                
                await asyncio.sleep(300)  # Analyze every 5 minutes
                
            except Exception as e:
                logger.error(f"Threat intelligence error: {e}")
                await asyncio.sleep(600)
    
    def _perform_threat_analysis(self) -> Dict:
        """Perform threat pattern analysis"""
        recent_threats = self.detected_threats[-50:]  # Last 50 threats
        
        threat_types = {}
        source_ips = {}
        
        for threat in recent_threats:
            threat_type = threat.get('type', 'unknown')
            threat_types[threat_type] = threat_types.get(threat_type, 0) + 1
            
            source = threat.get('source', '')
            if source:
                source_ips[source] = source_ips.get(source, 0) + 1
        
        return {
            'most_common_threat': max(threat_types, key=threat_types.get) if threat_types else None,
            'top_attacking_ip': max(source_ips, key=source_ips.get) if source_ips else None,
            'total_threats': len(recent_threats),
            'threat_types': threat_types
        }
    
    async def _log_security_event(self, event: Dict):
        """Log security event to file and console"""
        try:
            log_entry = {
                'agent_id': self.agent_id,
                'timestamp': datetime.now().isoformat(),
                'event': event
            }
            
            # Log to security events file
            with open('/var/log/security/blue_team_events.json', 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
            
            # Log to main log file
            with open('/app/logs/blue_team_security.json', 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
                
        except Exception as e:
            logger.error(f"Failed to log security event: {e}")
    
    async def _investigate_connection(self, connection: Dict):
        """Investigate suspicious connection"""
        logger.info(f"🔍 Investigating connection: {connection}")
        
        # Perform additional analysis
        foreign_addr = connection.get('foreign_address', '')
        if foreign_addr:
            # Could integrate with threat intelligence feeds here
            threat = {
                'type': 'connection_investigation',
                'severity': 'low',
                'connection': connection,
                'timestamp': datetime.now().isoformat()
            }
            await self._log_security_event(threat)

async def main():
    """Main entry point for blue team container agent"""
    logger.info("🔵 Starting Archangel Blue Team Container Agent")
    
    agent = UbuntuBlueTeamAgent()
    
    try:
        await agent.run_autonomous_operations()
    except KeyboardInterrupt:
        logger.info("🛑 Blue team agent shutting down")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())