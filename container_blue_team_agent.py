#!/usr/bin/env python3
"""
Blue Team AI Agent - Runs inside Ubuntu SOC container
Monitors, detects, and responds to red team activities
"""

import asyncio
import json
import subprocess
import time
import socket
import os
import sys
from datetime import datetime
from typing import Dict, List, Any
import re

class ContainerBlueTeamAgent:
    def __init__(self):
        self.agent_id = "blue-container-agent"
        self.container_name = "archangel-blue-team"
        self.orchestrator_url = "http://172.18.0.1:8000"  # Host machine
        self.status = "initializing"
        self.tools_available = []
        self.monitored_interfaces = ["eth0"]
        self.detected_threats = []
        self.blocked_ips = []
        self.active_connections = []
        
    async def initialize(self):
        """Initialize the blue team agent inside Ubuntu container"""
        print(f"🔵 Blue Team AI Agent initializing in container...")
        
        # Check available tools
        await self._check_available_tools()
        
        # Set up directories
        os.makedirs("/logs", exist_ok=True)
        os.makedirs("/var/log/security", exist_ok=True)
        
        # Initialize monitoring
        await self._setup_monitoring()
        
        self.status = "ready"
        print(f"✅ Blue Team Agent ready with {len(self.tools_available)} tools")
        
        # Report to orchestrator
        await self._report_status()
        
    async def _check_available_tools(self):
        """Check which security monitoring tools are available"""
        tools_to_check = ["python3", "iptables", "tcpdump", "ps", "netstat", "ss", "lsof"]
        
        for tool in tools_to_check:
            try:
                result = subprocess.run(["which", tool], capture_output=True, text=True)
                if result.returncode == 0:
                    self.tools_available.append(tool)
                    print(f"✓ Tool available: {tool}")
            except Exception as e:
                print(f"✗ Tool check failed for {tool}: {e}")
                
        # Check for ss as netstat alternative
        if "netstat" not in self.tools_available and "ss" in self.tools_available:
            self.tools_available.append("netstat")  # We'll use ss as netstat
    
    async def _setup_monitoring(self):
        """Set up security monitoring"""
        try:
            # Create security log directory
            os.makedirs("/var/log/security", exist_ok=True)
            
            # Initialize security log
            with open("/var/log/security/blue_team.log", "w") as f:
                f.write(f"Blue Team monitoring started at {datetime.now().isoformat()}\n")
            
            print(f"🛡️ Security monitoring initialized")
            
        except Exception as e:
            print(f"⚠️ Failed to setup monitoring: {e}")
    
    async def _report_status(self):
        """Report status to the main orchestrator"""
        try:
            status_data = {
                "agent_id": self.agent_id,
                "container": self.container_name,
                "status": self.status,
                "tools_available": self.tools_available,
                "timestamp": datetime.now().isoformat(),
                "detected_threats": len(self.detected_threats),
                "blocked_ips": len(self.blocked_ips),
                "active_connections": len(self.active_connections)
            }
            
            # Write to local log file
            with open("/logs/blue_team_status.log", "w") as f:
                json.dump(status_data, f, indent=2)
                
            print(f"📊 Status reported: {self.status} - {len(self.detected_threats)} threats detected")
            
        except Exception as e:
            print(f"⚠️ Failed to report status: {e}")
    
    async def monitor_network_activity(self):
        """Monitor network activity for threats"""
        try:
            print(f"🌐 Monitoring network activity...")
            
            # Get current network connections
            connections = await self._get_network_connections()
            
            # Analyze for suspicious patterns
            threats = await self._analyze_network_threats(connections)
            
            if threats:
                for threat in threats:
                    await self._handle_threat(threat)
            
            self.active_connections = connections
            
        except Exception as e:
            print(f"❌ Network monitoring error: {e}")
    
    async def _get_network_connections(self) -> List[Dict[str, Any]]:
        """Get current network connections"""
        connections = []
        
        try:
            # Use ss instead of netstat if available
            if "ss" in self.tools_available:
                cmd = ["ss", "-tuln"]
            else:
                cmd = ["netstat", "-tuln"]
                
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                for line in lines[1:]:  # Skip header
                    if line.strip():
                        parts = line.split()
                        if len(parts) >= 4:
                            connections.append({
                                'protocol': parts[0],
                                'local_address': parts[3] if len(parts) > 3 else 'unknown',
                                'state': parts[4] if len(parts) > 4 else 'unknown',
                                'timestamp': datetime.now().isoformat()
                            })
            
        except Exception as e:
            print(f"⚠️ Failed to get network connections: {e}")
        
        return connections
    
    async def _analyze_network_threats(self, connections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze network connections for threats"""
        threats = []
        
        try:
            # Look for suspicious patterns
            for conn in connections:
                local_addr = conn.get('local_address', '')
                
                # Check for suspicious ports
                if ':' in local_addr:
                    port = local_addr.split(':')[-1]
                    if port in ['22', '23', '80', '443', '8080']:  # Common target ports
                        # Check if this is a new connection
                        if not any(c.get('local_address') == local_addr for c in self.active_connections):
                            threats.append({
                                'type': 'new_service_connection',
                                'source': local_addr,
                                'description': f'New connection detected on port {port}',
                                'severity': 'medium',
                                'timestamp': datetime.now().isoformat()
                            })
                
                # Look for unusual protocols or states
                if conn.get('protocol') in ['tcp'] and conn.get('state') == 'LISTEN':
                    if not any(c.get('local_address') == local_addr for c in self.active_connections):
                        threats.append({
                            'type': 'new_listener',
                            'source': local_addr,
                            'description': f'New listening service detected: {local_addr}',
                            'severity': 'low',
                            'timestamp': datetime.now().isoformat()
                        })
        
        except Exception as e:
            print(f"⚠️ Threat analysis error: {e}")
        
        return threats
    
    async def _handle_threat(self, threat: Dict[str, Any]):
        """Handle detected threat"""
        try:
            threat_type = threat.get('type', 'unknown')
            source = threat.get('source', 'unknown')
            severity = threat.get('severity', 'low')
            
            print(f"🚨 THREAT DETECTED: {threat_type} from {source} (severity: {severity})")
            
            # Log the threat
            with open("/var/log/security/blue_team.log", "a") as f:
                f.write(f"{datetime.now().isoformat()} - THREAT: {json.dumps(threat)}\n")
            
            # Add to detected threats
            self.detected_threats.append(threat)
            
            # Take defensive action based on severity
            if severity in ['high', 'critical']:
                await self._block_threat_source(source)
            elif severity == 'medium':
                await self._increase_monitoring(source)
                
        except Exception as e:
            print(f"❌ Threat handling error: {e}")
    
    async def _block_threat_source(self, source: str):
        """Block threat source using iptables"""
        try:
            if ':' in source:
                ip = source.split(':')[0]
                
                if ip not in self.blocked_ips and ip not in ['127.0.0.1', '0.0.0.0']:
                    print(f"🛡️ Blocking IP: {ip}")
                    
                    # Use iptables to block the IP
                    if "iptables" in self.tools_available:
                        cmd = ["iptables", "-I", "INPUT", "-s", ip, "-j", "DROP"]
                        result = subprocess.run(cmd, capture_output=True, text=True)
                        
                        if result.returncode == 0:
                            self.blocked_ips.append(ip)
                            print(f"✅ Successfully blocked {ip}")
                        else:
                            print(f"⚠️ Failed to block {ip}: {result.stderr}")
                    else:
                        print(f"⚠️ iptables not available, cannot block {ip}")
                        
        except Exception as e:
            print(f"❌ IP blocking error: {e}")
    
    async def _increase_monitoring(self, source: str):
        """Increase monitoring for suspicious source"""
        try:
            print(f"👁️ Increasing monitoring for: {source}")
            
            # Log increased monitoring
            with open("/var/log/security/blue_team.log", "a") as f:
                f.write(f"{datetime.now().isoformat()} - MONITOR: Increased monitoring for {source}\n")
                
        except Exception as e:
            print(f"❌ Monitoring increase error: {e}")
    
    async def monitor_processes(self):
        """Monitor running processes for suspicious activity"""
        try:
            print(f"🔍 Monitoring processes...")
            
            if "ps" in self.tools_available:
                cmd = ["ps", "aux"]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    lines = result.stdout.split('\n')
                    suspicious_processes = []
                    
                    for line in lines[1:]:  # Skip header
                        if line.strip():
                            # Look for suspicious process names
                            if any(keyword in line.lower() for keyword in ['nmap', 'scan', 'attack', 'exploit']):
                                suspicious_processes.append(line)
                    
                    if suspicious_processes:
                        threat = {
                            'type': 'suspicious_process',
                            'source': 'local_system',
                            'description': f'Suspicious processes detected: {len(suspicious_processes)}',
                            'severity': 'high',
                            'details': suspicious_processes,
                            'timestamp': datetime.now().isoformat()
                        }
                        await self._handle_threat(threat)
                        
        except Exception as e:
            print(f"❌ Process monitoring error: {e}")
    
    async def execute_defensive_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a defensive action from the AI orchestrator"""
        try:
            action_type = action.get('type', 'unknown')
            target = action.get('target', 'unknown')
            
            print(f"🛡️ Executing defensive action: {action_type} for {target}")
            
            if action_type == 'block_ip':
                await self._block_threat_source(target)
                return {'success': True, 'action': 'ip_blocked'}
            elif action_type == 'monitor_increase':
                await self._increase_monitoring(target)
                return {'success': True, 'action': 'monitoring_increased'}
            elif action_type == 'threat_analysis':
                connections = await self._get_network_connections()
                threats = await self._analyze_network_threats(connections)
                return {'success': True, 'threats_found': len(threats), 'details': threats}
            else:
                return {
                    'success': False,
                    'error': f'Unknown action type: {action_type}',
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def run_continuous_monitoring(self):
        """Run continuous blue team monitoring operations"""
        print(f"🛡️ Starting continuous blue team monitoring...")
        
        monitor_cycle = 0
        while True:
            try:
                monitor_cycle += 1
                print(f"\n🔵 Blue Team Monitor Cycle #{monitor_cycle}")
                
                # Network monitoring
                await self.monitor_network_activity()
                
                # Process monitoring every 3rd cycle
                if monitor_cycle % 3 == 0:
                    await self.monitor_processes()
                
                # Report status every 5th cycle
                if monitor_cycle % 5 == 0:
                    await self._report_status()
                
                # Wait before next monitoring cycle
                await asyncio.sleep(8)
                
            except KeyboardInterrupt:
                print(f"\n🛑 Blue Team monitoring stopped")
                break
            except Exception as e:
                print(f"❌ Monitoring error: {e}")
                await asyncio.sleep(5)

async def main():
    """Main entry point for container blue team agent"""
    agent = ContainerBlueTeamAgent()
    
    try:
        await agent.initialize()
        await agent.run_continuous_monitoring()
    except KeyboardInterrupt:
        print(f"\n🛑 Blue Team Agent shutting down...")
    except Exception as e:
        print(f"❌ Fatal error: {e}")

if __name__ == "__main__":
    asyncio.run(main())