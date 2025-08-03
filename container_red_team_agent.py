#!/usr/bin/env python3
"""
Red Team AI Agent - Runs inside Kali Linux container
Controls actual penetration testing tools and reports to main orchestrator
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
import requests

class ContainerRedTeamAgent:
    def __init__(self):
        self.agent_id = "red-container-agent"
        self.container_name = "archangel-red-team"
        self.orchestrator_url = "http://172.18.0.1:8000"  # Host machine
        self.status = "initializing"
        self.tools_available = []
        self.target_networks = ["172.18.0.0/16"]  # Container network
        self.discovered_hosts = []
        
    async def initialize(self):
        """Initialize the red team agent inside Kali container"""
        print(f"🔴 Red Team AI Agent initializing in container...")
        
        # Check available tools
        await self._check_available_tools()
        
        # Set up directories
        os.makedirs("/logs", exist_ok=True)
        os.makedirs("/data", exist_ok=True)
        
        self.status = "ready"
        print(f"✅ Red Team Agent ready with {len(self.tools_available)} tools")
        
        # Report to orchestrator
        await self._report_status()
        
    async def _check_available_tools(self):
        """Check which penetration testing tools are available"""
        tools_to_check = ["nmap", "curl", "nc", "python3", "ping"]
        
        for tool in tools_to_check:
            try:
                result = subprocess.run(["which", tool], capture_output=True, text=True)
                if result.returncode == 0:
                    self.tools_available.append(tool)
                    print(f"✓ Tool available: {tool}")
            except Exception as e:
                print(f"✗ Tool check failed for {tool}: {e}")
    
    async def _report_status(self):
        """Report status to the main orchestrator"""
        try:
            status_data = {
                "agent_id": self.agent_id,
                "container": self.container_name,
                "status": self.status,
                "tools_available": self.tools_available,
                "timestamp": datetime.now().isoformat(),
                "discovered_hosts": self.discovered_hosts
            }
            
            # Write to local log file
            with open("/logs/red_team_status.log", "w") as f:
                json.dump(status_data, f, indent=2)
                
            print(f"📊 Status reported: {self.status}")
            
        except Exception as e:
            print(f"⚠️ Failed to report status: {e}")
    
    async def run_autonomous_scan(self):
        """Run autonomous network reconnaissance"""
        print(f"🔍 Starting autonomous reconnaissance...")
        
        for target_network in self.target_networks:
            await self._scan_network(target_network)
            
        await self._report_status()
    
    async def _scan_network(self, target: str):
        """Scan network for live hosts"""
        try:
            print(f"🎯 Scanning network: {target}")
            
            if "nmap" in self.tools_available:
                # Use nmap for host discovery
                cmd = ["nmap", "-sn", target]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    # Parse nmap results
                    lines = result.stdout.split('\n')
                    for line in lines:
                        if "Nmap scan report for" in line:
                            host = line.split()[-1]
                            if host not in self.discovered_hosts:
                                self.discovered_hosts.append(host)
                                print(f"🎯 Discovered host: {host}")
                else:
                    print(f"⚠️ Nmap scan failed: {result.stderr}")
            else:
                # Fallback to ping sweep
                print("📡 Using ping sweep fallback...")
                await self._ping_sweep(target)
                
        except Exception as e:
            print(f"❌ Network scan error: {e}")
    
    async def _ping_sweep(self, network: str):
        """Simple ping sweep for host discovery"""
        try:
            # Simple ping sweep for container network
            base_ip = "172.18.0"
            for i in range(1, 10):
                target_ip = f"{base_ip}.{i}"
                try:
                    result = subprocess.run(["ping", "-c", "1", "-W", "1", target_ip], 
                                          capture_output=True, timeout=2)
                    if result.returncode == 0:
                        if target_ip not in self.discovered_hosts:
                            self.discovered_hosts.append(target_ip)
                            print(f"🎯 Ping discovered: {target_ip}")
                except subprocess.TimeoutExpired:
                    pass
                except Exception:
                    pass
        except Exception as e:
            print(f"❌ Ping sweep error: {e}")
    
    async def execute_autonomous_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an autonomous action from the AI orchestrator"""
        try:
            action_type = action.get('type', 'unknown')
            target = action.get('target', 'unknown')
            
            print(f"🤖 Executing autonomous action: {action_type} on {target}")
            
            if action_type == 'port_scan':
                return await self._port_scan(target)
            elif action_type == 'service_enum':
                return await self._service_enumeration(target)
            elif action_type == 'vulnerability_scan':
                return await self._vulnerability_scan(target)
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
    
    async def _port_scan(self, target: str) -> Dict[str, Any]:
        """Perform port scan on target"""
        try:
            if "nmap" in self.tools_available:
                cmd = ["nmap", "-F", target]  # Fast scan
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                return {
                    'success': result.returncode == 0,
                    'output': result.stdout,
                    'error': result.stderr if result.returncode != 0 else None,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'success': False,
                    'error': 'nmap not available',
                    'timestamp': datetime.now().isoformat()
                }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _service_enumeration(self, target: str) -> Dict[str, Any]:
        """Enumerate services on target"""
        try:
            if "nmap" in self.tools_available:
                cmd = ["nmap", "-sV", "-F", target]  # Service version detection
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
                
                return {
                    'success': result.returncode == 0,
                    'output': result.stdout,
                    'error': result.stderr if result.returncode != 0 else None,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'success': False,
                    'error': 'nmap not available',
                    'timestamp': datetime.now().isoformat()
                }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _vulnerability_scan(self, target: str) -> Dict[str, Any]:
        """Basic vulnerability scanning"""
        try:
            if "nmap" in self.tools_available:
                cmd = ["nmap", "--script", "vuln", "-F", target]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                
                return {
                    'success': result.returncode == 0,
                    'output': result.stdout,
                    'error': result.stderr if result.returncode != 0 else None,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'success': False,
                    'error': 'nmap not available',
                    'timestamp': datetime.now().isoformat()
                }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def run_continuous_operations(self):
        """Run continuous red team operations"""
        print(f"🔥 Starting continuous red team operations...")
        
        operation_count = 0
        while True:
            try:
                operation_count += 1
                print(f"\n🔴 Red Team Operation #{operation_count}")
                
                # Periodic reconnaissance
                if operation_count % 3 == 1:
                    await self.run_autonomous_scan()
                
                # Attack discovered targets
                if self.discovered_hosts:
                    target = self.discovered_hosts[operation_count % len(self.discovered_hosts)]
                    
                    # Rotate through different attack types
                    if operation_count % 3 == 0:
                        await self._port_scan(target)
                    elif operation_count % 3 == 1:
                        await self._service_enumeration(target)
                    else:
                        await self._vulnerability_scan(target)
                
                # Report status periodically
                if operation_count % 5 == 0:
                    await self._report_status()
                
                # Wait before next operation
                await asyncio.sleep(10)
                
            except KeyboardInterrupt:
                print(f"\n🛑 Red Team operations stopped")
                break
            except Exception as e:
                print(f"❌ Operation error: {e}")
                await asyncio.sleep(5)

async def main():
    """Main entry point for container red team agent"""
    agent = ContainerRedTeamAgent()
    
    try:
        await agent.initialize()
        await agent.run_continuous_operations()
    except KeyboardInterrupt:
        print(f"\n🛑 Red Team Agent shutting down...")
    except Exception as e:
        print(f"❌ Fatal error: {e}")

if __name__ == "__main__":
    asyncio.run(main())